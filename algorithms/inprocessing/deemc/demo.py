"""DEE-MC 데모: λ_dee=0 과 λ_dee=0.5 두 arm 의 Test RMSE·DEE·MAF 18지표.

runner 가 세 arm(MF 기준선 / glocal_lambda0 / dee_mc)을 평가한 뒤 데모에 쓰는 두 arm 만 돌려준다.
λ_dee=0 체크포인트는 model/deemc/lambda0/{model_001.pt, model_002.pt} 에 미리 둔다(공급자 제공, git 미포함).

실행: python -m MAF.algorithms.inprocessing.deemc.demo [--device auto|cpu|cuda:0]
"""

import argparse
import json
import threading
from pathlib import Path

from MAF.algorithms.inprocessing.deemc.DEEMC import DEEMC
from MAF.algorithms.inprocessing.deemc.runner import (
    DEFAULT_MODEL_DIR,
    LAMBDA0_CHECKPOINTS,
    json_safe,
)

ARMS = ("glocal_lambda0", "dee_mc")
ARM_LABELS = {"glocal_lambda0": "λ_dee = 0", "dee_mc": "λ_dee = 0.5"}
MAF_METRIC_KEYS = (
    "true_negative_rate", "true_positive_rate", "false_positive_rate", "false_negative_rate",
    "precision", "false_discovery_rate", "false_omission_rate", "accuracy", "Balanced accuracy",
    "error_rate", "average_odds_difference", "average_abs_odds_difference", "selection_rate",
    "disparate_impact", "statistical_parity_difference", "generalized_entropy_index", "theil_index",
    "equal_opportunity_difference",
)  # fmt: skip
MISSING_HINT = "model/deemc/lambda0/ 에 λ_dee=0 체크포인트 model_001.pt·model_002.pt 를 넣어 주세요"

# 요청이 겹치면 직렬화: 체크포인트 5개를 GPU 에 동시에 두 번 올리지 않는다
_LOCK = threading.Lock()


class Lambda0Missing(FileNotFoundError):
    """λ_dee=0 체크포인트 부재(상대경로만 담아 화면에 그대로 보여도 된다)."""


def _present(path):
    return path.is_file() and path.stat().st_size > 0


def check_lambda0_assets(model_dir=DEFAULT_MODEL_DIR):
    """체크포인트 2개가 있는지(runner 호출 전 사전 검사). 설정·구조 검사는 runner 의 load_glocal 이 한다."""
    missing = [n for n in LAMBDA0_CHECKPOINTS if not _present(Path(model_dir) / n)]
    if missing:
        raise Lambda0Missing(
            f"λ_dee=0 체크포인트가 없습니다({', '.join(missing)}). {MISSING_HINT}"
        )
    return [str(Path(model_dir) / n) for n in LAMBDA0_CHECKPOINTS]


def _arm(report, name):
    native = report["native"][name]["metrics"]
    return {
        "label": ARM_LABELS[name],
        "test_rmse": float(native["test_rmse"]),
        "dee": float(native["dee"]),
        "maf": {k: json_safe(report["maf"][name][k]) for k in MAF_METRIC_KEYS},
    }


def compare_lambda(device="auto", model_dir=DEFAULT_MODEL_DIR):
    with _LOCK:
        check_lambda0_assets(model_dir)
        algorithm = DEEMC(device=device, model_dir=model_dir)
        algorithm.run()
        report = algorithm.report
    if not report["ablation"]["present"]:
        raise RuntimeError(
            "runner skipped the ablation arm although lambda0 assets exist"
        )
    evaluation = report["evaluation"]
    return json_safe(
        {
            "arms": {name: _arm(report, name) for name in ARMS},
            "metric_keys": list(MAF_METRIC_KEYS),
            "evaluation": {
                "test_ratings": evaluation["test_ratings"],
                "users": evaluation["users"],
                "movies": evaluation["movies"],
                "device": report["environment"]["device"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            "criteria_met": bool(report["native_criteria_met"]),
        }
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args(argv)
    result = compare_lambda(device=args.device, model_dir=args.model_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
