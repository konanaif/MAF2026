"""SAE 데모: 요청마다 합성 예제로 네 기법을 새로 학습·평가해 summary() 를 dict 로 돌려준다.

실행: python -m MAF.algorithms.inprocessing.sae.demo [--device auto|cuda:N]
"""

import argparse
import json
import logging
import math
import shutil
import tempfile
import threading
from pathlib import Path
from time import perf_counter

import torch
from MAF.algorithms.inprocessing.sae.SAE import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_ROOT,
    SAE,
)

log = logging.getLogger(__name__)

MODEL = "roberta-base"
METHODS = ("base", "base_swa", "fairbatch", "leace_swa")
METHOD_LABELS = {
    "base": "Base",
    "base_swa": "Base+SWA",
    "fairbatch": "FairBatch",
    "leace_swa": "LEACE+SWA",
}
COUNT_KEYS = ("tn", "fp", "fn", "tp")
METRIC_KEYS = ("accuracy", "toxic_f1", "macro_f1", "precision", "recall", "fpr")
HIGHER_BETTER = frozenset({"accuracy", "toxic_f1", "macro_f1", "precision", "recall"})
SYNTHETIC_DIR = DEFAULT_DATA_DIR / "synthetic"
DEMO_TMP_ROOT = DEFAULT_OUTPUT_ROOT / "demo_tmp"

# 요청이 겹치면 직렬화: GPU 학습 두 개를 동시에 돌리지 않는다
_LOCK = threading.Lock()


def _number(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    value = float(value)
    return value if math.isfinite(value) else None


def _block(metrics):
    return {k: _number(metrics.get(k)) for k in ("n", *COUNT_KEYS, *METRIC_KEYS)}


def run_demo(device="auto", data_dir=SYNTHETIC_DIR):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "demo needs a CUDA device (CPU training takes about 5 minutes)"
        )
    with _LOCK:
        started = perf_counter()
        DEMO_TMP_ROOT.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix="run_", dir=DEMO_TMP_ROOT))
        try:
            algorithm = SAE(
                data_dir=Path(data_dir),
                model=MODEL,
                method="all",
                output=tmp / "run",
                device=device,
            )
            plan = algorithm.plan()
            algorithm.run()
            summary = algorithm.summary()[MODEL]
        finally:
            try:
                shutil.rmtree(tmp)
            except OSError:
                log.warning("temp output not removed: %s", tmp)
        elapsed = perf_counter() - started
    splits = plan["data"]["splits"]
    methods = [m for m in METHODS if m in summary]
    return {
        "model": MODEL,
        "methods": methods,
        "labels": dict(METHOD_LABELS),
        "data": {s: int(splits[s]["n"]) for s in ("train", "dev", "test")},
        "argmax": {m: _block(summary[m]["argmax"]) for m in methods},
        "calibrated": {m: _block(summary[m]["calibrated"]) for m in methods},
        "threshold": {m: _number(summary[m]["threshold"]) for m in methods},
        "elapsed_seconds": elapsed,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto or cuda:N")
    args = parser.parse_args(argv)
    result = run_demo(device=args.device)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
