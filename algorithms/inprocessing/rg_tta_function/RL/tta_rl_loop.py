# RLCF_Debiasing/RL/tta_rl_loop.py

from typing import List, Optional, Tuple, Dict
import torch
import torch.nn.functional as F
from torch import amp
from tqdm import tqdm

# bias metric 유틸리티 (eval/measure_bias.py)
from ..eval.measure_bias import _eval_ranking_single_metric  # type: ignore


@torch.no_grad()
def _get_topk_indices(logits_per_text: torch.Tensor, sample_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    텍스트-로짓에서 상위 K개 인덱스를 선택.
    Returns:
        values [K], indices [K]
    """
    values, indices = torch.topk(logits_per_text, k=sample_k, dim=-1)
    return values[0], indices[0]  # batch dim 제거


def adapt_single_query(
    query_tokens: torch.Tensor,
    query_text: Optional[str],
    model,
    reward_model,
    optimizer,
    scaler,
    args,
    *,
    labels_list,
    q_idx: int = -1,
    outer_pbar=None,
) -> Tuple[List[float], List[float]]:
    """
    단일 쿼리에 대해 policy gradient TTA를 수행.
    - wandb 로깅은 하지 않음.
    - 각 epoch별 bias metric(maxskew, ndkl)을 리스트로 반환.

    Returns:
        epoch_maxskew_list: List[float] (len = args.tta_steps)
        epoch_ndkl_list:    List[float] (len = args.tta_steps)
    """
    model.train()

    # 리워드 모델 텍스트 세팅(리워드 CLIP 공간)
    if query_text is not None and hasattr(reward_model, "set_text_by_strings"):
        reward_model.set_text_by_strings(query_text)

    # 갤러리 임베딩(정책 공간) 캐시
    gallery_feats: torch.Tensor = model.image_features_cache  # [N_img, D]
    assert gallery_feats is not None, "model.set_image_features(...)가 먼저 호출되어야 합니다."
    num_images = int(gallery_feats.shape[0])
    eval_topn = int(min(1000, num_images))  # 평가 상한

    # per-epoch 결과 버퍼
    epoch_maxskew_list: List[float] = []
    epoch_ndkl_list: List[float] = []

    # Progress bar는 외부에서 관리하므로 여기서는 사용하지 않음
    query_desc = f"Query {q_idx + 1}" if q_idx >= 0 else "Query"
    tta_steps = int(args.tta_steps)

    for epoch in range(tta_steps):
        optimizer.zero_grad(set_to_none=True)

        # 혼합정밀 (PyTorch 최신 API)
        with torch.amp.autocast(device_type="cuda",enabled=False): #type : ignore
            # 1) 현재 정책으로 이미지 후보 점수 산출
            logits_per_text: torch.Tensor = model(query_tokens)  # [1, N_img]

            # 2) top-K 행동(이미지 후보) 선택
            _, topk_indices = _get_topk_indices(
                logits_per_text,
                sample_k=int(args.sample_k),
            )  # [K]

            # 3) (정책 공간) 현재 에포크 임베딩 추출
            policy_text_feat: torch.Tensor = model.encode_text(query_tokens)[0].detach()  # [D]
            policy_image_feats: torch.Tensor = model.image_features_cache.index_select(
                0, topk_indices.to(model.device)
            ).detach()  # [K, D]

            # 4) 총 리워드 계산 (리워드 CLIP + (옵션) 디바이싱)
            rewards: torch.Tensor = reward_model.total_reward(
                images_index=topk_indices,          # [K]
                policy_text_feat=policy_text_feat,   # [D]
                policy_image_feats=policy_image_feats,  # [K, D]
            )  # [K]

            # 5) 정책 손실 (REINFORCE 형태)
            rep_logits = logits_per_text.repeat_interleave(
                repeats=int(args.sample_k), dim=0
            )  # [K, N_img]
            per_candidate_loss = F.cross_entropy(
                rep_logits, topk_indices, reduction="none"
            )  # [K]
            pg_loss = torch.mean(rewards * per_candidate_loss)

        # 6) backward + optimizer step (AMP 스케일러)
        # 관심 파라미터 핸들
        param = model.clip_model.transformer.resblocks[0].attn.in_proj_weight


        # backward
        scaler.scale(pg_loss).backward()
        scaler.unscale_(optimizer)  # ← fp32 grad 검사 가능

        # step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


        # 7) (측정) 현재 정책 텍스트 임베딩으로 bias metric 계산
        with torch.no_grad():
            cur_text_feat: torch.Tensor = model.encode_text(query_tokens)[0]  # [D]

            res_skew = _eval_ranking_single_metric(
                labels_list=labels_list,
                image_embeddings=gallery_feats,
                prompt_embedding=cur_text_feat,
                evaluation="maxskew",
                topn=eval_topn,
            )
            res_ndkl = _eval_ranking_single_metric(
                labels_list=labels_list,
                image_embeddings=gallery_feats,
                prompt_embedding=cur_text_feat,
                evaluation="ndkl",
                topn=eval_topn,
            )

            maxskew_val = float(res_skew.get("maxskew_eq_opp", 0.0))
            ndkl_val = float(res_ndkl.get("ndkl_eq_opp", 0.0))
            epoch_maxskew_list.append(maxskew_val)
            epoch_ndkl_list.append(ndkl_val)

    # per-epoch 리스트 반환 (길이 = tta_steps)
    return epoch_maxskew_list, epoch_ndkl_list


def episodic_tta_loop(
    query_token_list: List[torch.Tensor],
    model,
    reward_model,
    optimizer,
    scaler,
    args,
    query_text_list: Optional[List[str]] = None,
    labels_list=None,
) -> Dict[str, object]:
    """
    여러 쿼리에 대해 TTA 수행.
    - 각 epoch마다 '모든 쿼리 평균' maxskew/ndkl을 1회 wandb.log.
    - best는 '최솟값' 기준 및 해당 epoch 인덱스로 기록.

    Returns(dict):
        {
            "avg_maxskew_per_epoch": List[float],
            "avg_ndkl_per_epoch": List[float],
            "best_avg_maxskew": float,
            "best_avg_maxskew_epoch": int,
            "best_avg_ndkl": float,
            "best_avg_ndkl_epoch": int,
        }
    """
    num_queries = len(query_token_list)
    tta_steps = int(args.tta_steps)

    # 누적 버퍼
    sum_maxskew = [0.0] * tta_steps
    sum_ndkl = [0.0] * tta_steps

    # 쿼리 기준 progress bar
    pbar_query = tqdm(
        total=num_queries,
        desc="Processing queries",
        position=0,
        dynamic_ncols=True,
        leave=True,
        unit="query",
        mininterval=0.1,  # 최소 업데이트 간격
    )
    
    for q_idx, query_tokens in enumerate(query_token_list):
        query_text = None
        if query_text_list is not None:
            query_text = query_text_list[q_idx]

        # Progress bar에 현재 쿼리 정보 업데이트
        pbar_query.set_description(f"Query {q_idx + 1}/{num_queries}")

        # 각 쿼리 시작 전: 초기 상태로 리셋 (momentum이 적용된 상태로)
        if hasattr(model, "reset_initial"):
            model.reset_initial()

        per_epoch_maxskew, per_epoch_ndkl = adapt_single_query(
            query_tokens=query_tokens,
            query_text=query_text,
            model=model,
            reward_model=reward_model,
            optimizer=optimizer,
            scaler=scaler,
            args=args,
            labels_list=labels_list,
            q_idx=q_idx,
            outer_pbar=None,  # 쿼리 기준으로 변경하므로 epoch별 업데이트 제거
        )

        for e in range(tta_steps):
            sum_maxskew[e] += per_epoch_maxskew[e]
            sum_ndkl[e] += per_epoch_ndkl[e]
        
        # 쿼리별 epoch별 결과 출력
        if len(per_epoch_maxskew) > 0 and len(per_epoch_ndkl) > 0:
            print(f"\n[Query {q_idx + 1}/{num_queries}] Epoch별 결과:")
            for epoch in range(tta_steps):
                print(f"  Epoch {epoch + 1}/{tta_steps}: maxskew={per_epoch_maxskew[epoch]:.4f}, ndkl={per_epoch_ndkl[epoch]:.4f}")
        
        # 쿼리 완료 후: momentum buffer 업데이트 (설정된 간격에 따라)
        momentum_update_freq = getattr(args, "momentum_update_freq", 1)
        if hasattr(model, "momentum_update_model") and (q_idx + 1) % momentum_update_freq == 0:
            model.momentum_update_model()
        
        # 쿼리 완료 시 progress bar 업데이트
        pbar_query.update(1)
        if len(per_epoch_maxskew) > 0 and len(per_epoch_ndkl) > 0:
            # 마지막 epoch의 값 표시
            last_maxskew = per_epoch_maxskew[-1]
            last_ndkl = per_epoch_ndkl[-1]
            pbar_query.set_postfix({
                "maxskew": f"{last_maxskew:.4f}",
                "ndkl": f"{last_ndkl:.4f}",
            })

    # 평균 계산
    denom = float(max(1, num_queries))
    avg_maxskew = [v / denom for v in sum_maxskew]
    avg_ndkl = [v / denom for v in sum_ndkl]

    # best는 '최솟값' 기준
    best_avg_maxskew = min(avg_maxskew) if len(avg_maxskew) > 0 else float("inf")
    best_epoch_maxskew = int(avg_maxskew.index(best_avg_maxskew)) if len(avg_maxskew) > 0 else -1

    best_avg_ndkl = min(avg_ndkl) if len(avg_ndkl) > 0 else float("inf")
    best_epoch_ndkl = int(avg_ndkl.index(best_avg_ndkl)) if len(avg_ndkl) > 0 else -1

    # wandb 로깅 (epoch당 1회, 평균 + best)
    if getattr(args, "use_wandb", False) and tta_steps > 0:
        import wandb

        # metric 정의: epoch 값을 기준으로 step을 명시적으로 관리
        wandb.define_metric("bias/epoch", summary="last")
        wandb.define_metric("maxskew", step_metric="bias/epoch")
        wandb.define_metric("ndkl", step_metric="bias/epoch")

        epoch_table = wandb.Table(columns=["epoch", "avg_maxskew", "avg_ndkl"])

        for epoch in range(tta_steps):
            epoch_step = epoch + 1
            wandb.log(
                {
                    "bias/epoch": epoch_step,
                    "maxskew": avg_maxskew[epoch],
                    "ndkl": avg_ndkl[epoch],
                },
                step=epoch_step,
            )
            epoch_table.add_data(epoch_step, avg_maxskew[epoch], avg_ndkl[epoch])

        best_ndkl_at_maxskew = (
            avg_ndkl[best_epoch_maxskew] if best_epoch_maxskew >= 0 else None
        )
        wandb.log(
            {
                "best_maxskew": best_avg_maxskew,
                "best_maxskew_epoch": best_epoch_maxskew + 1
                if best_epoch_maxskew >= 0
                else -1,
                "best_maxskew_ndkl": best_ndkl_at_maxskew,
            },
            step=tta_steps + 1,
        )

        wandb.log(
            {
                "bias/epoch_table": epoch_table,
            },
            step=tta_steps + 1,
        )

    return {
        "avg_maxskew_per_epoch": avg_maxskew,
        "avg_ndkl_per_epoch": avg_ndkl,
        "best_avg_maxskew": best_avg_maxskew,
        "best_avg_maxskew_epoch": best_epoch_maxskew,
        "best_avg_ndkl": best_avg_ndkl,
        "best_avg_ndkl_epoch": best_epoch_ndkl,
    }
