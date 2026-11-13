import math
from collections import Counter, defaultdict
from typing import Union, Dict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """
    KL(p || q) = sum_i p_i * (log p_i - log q_i)
    p, q는 확률분포 (합=1). 0 * log(0/q)=0 취급을 위해 eps로 클램프.
    """
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))


# ------------------------------------------------------------
# Metrics (NDKL / MaxSkew)
# ------------------------------------------------------------
def _normalized_discounted_KL(df: pd.DataFrame, top_n: int) -> Dict[str, float]:
    """
    ndKL (normalized discounted KL divergence):
    상위 랭크(top_n)까지의 누적 분포가
    '균등 분포(equality of opportunity)'에서 얼마나 벗어나는지,
    순위별 log-discount를 주면서 누적한 뒤 정규화한 값.

    df: DataFrame({"score": <float>, "label": <int>})
         - score: similarity 등 ranking 점수 (높을수록 상위)
         - label: 정수 클래스 ID (예: race bucket index)
    top_n: 상위 몇 개까지만 볼지
    """
    result_metrics = {"ndkl_eq_opp": 0.0}

    # score 유한값만 사용
    df = df[np.isfinite(df["score"])].copy()
    if len(df) == 0:
        return result_metrics

    # 클래스/균등분포
    unique_labels = sorted(Counter(df["label"]).keys())
    num_classes = max(1, len(unique_labels))
    desired_dist = {"eq_opp": np.array([1.0 / num_classes] * num_classes, dtype=float)}

    # top_n 클램프
    top_n = max(1, min(int(top_n), len(df)))
    top_n_scores = df.nlargest(top_n, columns="score", keep="all")

    # 누적 카운트 기반 분포
    running_counts = np.zeros(num_classes, dtype=float)

    for rank_index, (_, row) in enumerate(top_n_scores.iterrows(), start=1):
        label_idx = int(row["label"])
        if not (0 <= label_idx < num_classes):
            # label 인덱스가 범위를 벗어나면 스킵
            continue

        running_counts[label_idx] += 1.0
        current_dist = running_counts / float(rank_index)

        for dist_name, target_dist in desired_dist.items():
            kl_val = _kl_divergence(current_dist, target_dist)
            # DCG-style 할인 가중치 (1/log2(rank+1))
            discount = 1.0 / math.log2(rank_index + 1)
            result_metrics[f"ndkl_{dist_name}"] += float(kl_val * discount)

    # 정규화 상수 Z = sum_{i=1..top_n} 1/log2(i+1)
    Z = sum(1.0 / math.log2(i + 1) for i in range(1, top_n + 1))
    if Z > 0.0:
        for k in list(result_metrics.keys()):
            result_metrics[k] /= Z

    return result_metrics


def _compute_skew_metrics(df: pd.DataFrame, top_n: int) -> Dict[str, float]:
    """
    maxskew:
    상위 랭크(top_n) 안에 특정 그룹(label)이 얼마나 과대표/과소대표되는지.
    수식은 log(p_positive) - log(p_target),
    그 중 최대값(max over classes)을 사용.

    df: DataFrame({"score": <float>, "label": <int>})
    top_n: 상위 몇 개를 랭킹 기준으로 살펴볼지
    """
    result_metrics = {"maxskew_eq_opp": 0.0}

    # score 유한값만 사용
    df = df[np.isfinite(df["score"])].copy()
    if len(df) == 0:
        return result_metrics

    # 전체 등장 클래스들
    label_counts = Counter(df["label"])
    num_classes = max(1, len(label_counts))

    # 상위 top_n만 사용
    top_n = max(1, min(int(top_n), len(df)))
    top_n_scores = df.nlargest(top_n, columns="score", keep="all")
    top_n_counts = Counter(top_n_scores["label"])

    # 목표 분포 eq_opp: 균등(1 / num_classes)
    target_fraction = 1.0 / float(num_classes)
    denom_topn = float(max(top_n, 1))

    for label_class in label_counts.keys():
        # 실제 top_n 중 이 클래스가 차지한 비율
        p_positive = float(top_n_counts[label_class]) / denom_topn

        # log(0) 방지
        if p_positive <= 0.0:
            p_positive = 1.0 / denom_topn

        # skew 계산
        skewness = math.log(p_positive) - math.log(target_fraction)

        # 클래스별 skewness 중 최댓값만 남긴다
        if skewness > result_metrics["maxskew_eq_opp"]:
            result_metrics["maxskew_eq_opp"] = float(skewness)

    return result_metrics


# ------------------------------------------------------------
# Metric driver for one prompt
# ------------------------------------------------------------
def _eval_ranking_single_metric(
    labels_list: np.ndarray,
    image_embeddings: torch.Tensor,
    prompt_embedding: torch.Tensor,
    evaluation: str,
    topn: int,
) -> Dict[str, float]:
    """
    특정 프롬프트 하나에 대해 maxskew 또는 ndkl를 계산한다.
    - labels_list: shape [N_img], 각 이미지의 그룹 라벨(정수 클래스 ID)
    - image_embeddings: shape [N_img, D], 정규화된 이미지 임베딩
    - prompt_embedding: shape [D], 정규화된 텍스트 임베딩
    - evaluation: "maxskew" 또는 "ndkl"
    - topn: 상위 몇 개만 볼지 (정수)
    """
    assert evaluation in ("maxskew", "ndkl")

    # cosine similarity == dot product (이미 정규화되어 있다고 가정)
    if prompt_embedding.ndim == 1:
        sims_t = (image_embeddings @ prompt_embedding).detach()
    else:
        # [N_img, D] @ [D, 1 or B] → 평탄화
        sims_t = (image_embeddings @ prompt_embedding.mT).detach()

    # 텐서 → CPU/float32 numpy
    sims_t = sims_t.to("cpu").to(torch.float32).flatten()
    sims_np = sims_t.numpy()

    # 라벨을 확실히 numpy/int64로
    labels_np = np.asarray(labels_list, dtype=np.int64)

    # 유한값만 사용 (score 기준)
    finite_mask = np.isfinite(sims_np)
    sims_np = sims_np[finite_mask]
    labels_np = labels_np[finite_mask]

    if sims_np.size == 0:
        # 비정상 케이스: 전부 비유한 값이면 0 반환
        return {"maxskew_eq_opp": 0.0} if evaluation == "maxskew" else {"ndkl_eq_opp": 0.0}

    # score/label 테이블 (열 딕셔너리 형태)
    summary_df = pd.DataFrame(
        {"score": sims_np, "label": labels_np},
        copy=False,
    )

    if evaluation == "maxskew":
        return _compute_skew_metrics(summary_df, top_n=int(topn))
    else:
        return _normalized_discounted_KL(summary_df, top_n=int(topn))


# ------------------------------------------------------------
# Public API — evaluate all prompts
# ------------------------------------------------------------
def eval_bias_metrics(
    labels_list: np.ndarray,
    image_embeddings: torch.Tensor,
    prompts_embeddings: torch.Tensor,
    topn: Union[int, float] = 0.05,
) -> Dict[str, Dict[str, float]]:
    """
    전체 bias metric을 한 번에 계산.

    Args:
        labels_list:
            shape [N_img], 각 이미지의 그룹 라벨 인덱스 (예: race_id, gender_id 등)
        image_embeddings:
            shape [N_img, D], L2-normalized된 이미지 임베딩.
        prompts_embeddings:
            shape [N_prompts, D], L2-normalized된 텍스트 임베딩.
        topn:
            정수 또는 float.
            - int이면 "상위 topn개"를 사용
            - float이면 전체 이미지 수의 해당 비율만큼 (ex. 0.05 -> 상위 5%)

    Returns:
        {
          "maxskew": { "maxskew_eq_opp": <float>, },
          "ndkl":    { "ndkl_eq_opp": <float>,      }
        }
        각 값은 모든 프롬프트에 대해 평균된 값.
    """
    num_images = int(image_embeddings.shape[0])

    # topn 정수화
    if isinstance(topn, float):
        effective_topn = max(1, int(math.ceil(num_images * topn)))
    else:
        effective_topn = max(1, int(topn))

    # 결과 누적
    agg_results = {
        "maxskew": defaultdict(list),
        "ndkl": defaultdict(list),
    }

    # 프롬프트별 평가
    for prompt_idx in tqdm(
        range(int(prompts_embeddings.shape[0])),
        desc="Evaluating bias metrics",
        leave=False,
    ):
        this_prompt = prompts_embeddings[prompt_idx]  # [D]

        # maxskew
        res_skew = _eval_ranking_single_metric(
            labels_list=labels_list,
            image_embeddings=image_embeddings,
            prompt_embedding=this_prompt,
            evaluation="maxskew",
            topn=effective_topn,
        )
        for k, v in res_skew.items():
            agg_results["maxskew"][k].append(float(v))

        # ndkl
        res_ndkl = _eval_ranking_single_metric(
            labels_list=labels_list,
            image_embeddings=image_embeddings,
            prompt_embedding=this_prompt,
            evaluation="ndkl",
            topn=effective_topn,
        )
        for k, v in res_ndkl.items():
            agg_results["ndkl"][k].append(float(v))

    # 평균 요약
    final_results = {"maxskew": {}, "ndkl": {}}
    for metric_name, metric_values in agg_results["maxskew"].items():
        final_results["maxskew"][metric_name] = float(sum(metric_values) / max(1, len(metric_values)))
    for metric_name, metric_values in agg_results["ndkl"].items():
        final_results["ndkl"][metric_name] = float(sum(metric_values) / max(1, len(metric_values)))

    return final_results
