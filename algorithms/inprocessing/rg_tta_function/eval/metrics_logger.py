# eval/metrics_logger.py

import os
import json
import torch
import numpy as np
from typing import Dict, Any


def _to_serializable(obj):
    """
    Helper: JSON으로 바로 못 던지는 애들(torch.Tensor 등)을
    최대한 깔끔하게 바꿔준다.
    - torch.Tensor -> .tolist()
    - numpy types  -> .tolist()
    - 나머지 기본형(str/int/float/bool/list/dict/None)은 그대로 둔다.
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    return obj


def save_results(
    output_dir: str,
    run_name: str,
    score_matrix: torch.Tensor,
    meta: Dict[str, Any],
    args_dict: Dict[str, Any],
):
    """
    실험 1회 돌린 결과를 디스크에 저장한다.

    저장되는 것:
    - {output_dir}/{run_name}/scores_summary.json
         -> epoch별 avg_maxskew/avg_ndkl를 사람이 읽기 좋은 JSON 목록으로 저장

    - {output_dir}/{run_name}/meta.json
         -> prompts(쿼리 텍스트들),
            dataset_meta(index별 race/gender/age 등),
            run_meta(num_queries, num_gallery_images 등)

    - {output_dir}/{run_name}/config.json
         -> args_dict (tta_steps, sample_k, lr, dataset 등 재현성에 필요한 설정 전부)

    Args:
        output_dir: 상위 출력 디렉토리 (예: "./outputs")
        run_name:   실험 러닝 태그 (예: "fairface_race_cliponly_run01")
        score_matrix: torch.Tensor [num_queries, num_gallery_images]
        meta: {
           "dataset_meta": {
               "dataset_name": ...,
               "attribute": ...,
               "split": ...,
               "num_images": int,
               "meta_attr_names": [...],
               "index_to_path": [...],
               "index_to_attr": [...],   # list of dicts with race/gender/age/... per image
           },
           "prompts": [...],             # list of query strings (len = num_queries)
           "run_meta": {
               "num_queries": ...,
               "num_gallery_images": ...,
           }
        }
        args_dict: dict
           재현용 실험 설정 (seed, lr, tta_steps, sample_k, etc.)
    """

    # 1. 디렉토리 준비
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # 2. 사람이 읽기 쉬운 요약 저장
    score_matrix_cpu = score_matrix.detach().cpu()
    if score_matrix_cpu.ndim != 2 or score_matrix_cpu.shape[0] < 2:
        raise ValueError("score_matrix는 [2, T] 형태여야 합니다.")

    maxskew_per_epoch = score_matrix_cpu[0].tolist()
    ndkl_per_epoch = score_matrix_cpu[1].tolist()

    summary = []
    for idx, (maxskew, ndkl) in enumerate(zip(maxskew_per_epoch, ndkl_per_epoch), start=1):
        summary.append(
            {
                "epoch": idx,
                "avg_maxskew": float(maxskew),
                "avg_ndkl": float(ndkl),
            }
        )

    summary_path = os.path.join(run_dir, "scores_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 3. meta 저장 (json-friendly로 변환)
    #    meta 안에는 리스트/문자열/딕셔너리 위주라 대부분 바로 직렬화 가능하지만
    #    안전하게 _to_serializable로 한 번 정리해 준다
    meta_path = os.path.join(run_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(meta), f, indent=2, ensure_ascii=False)

    # 4. args_dict 저장
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(args_dict), f, indent=2, ensure_ascii=False)

    # 5. return 경로들 (원하면 caller에서 wandb 업로드 등에 사용 가능)
    return {
        "summary_path": summary_path,
        "meta_path": meta_path,
        "config_path": config_path,
        "run_dir": run_dir,
    }
