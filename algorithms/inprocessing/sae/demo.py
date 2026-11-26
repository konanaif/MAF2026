"""SAE 데모: 저장된 결과(result/sae/demo_result.json)가 있으면 그대로, 없으면 네 기법을 1회 학습·평가해 저장한다.

실행: python -m MAF.algorithms.inprocessing.sae.demo [--device auto|cuda:N]
"""

import argparse
import json
import logging
import math
import os
import shutil
import tempfile
import threading
from datetime import datetime, timedelta, timezone
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
DEMO_TMP_ROOT = DEFAULT_OUTPUT_ROOT / "demo_tmp"
DEMO_RESULT = DEFAULT_OUTPUT_ROOT / "demo_result.json"
RESULT_KEYS = ("methods", "labels", "data", "argmax", "calibrated", "threshold")
KST = timezone(timedelta(hours=9))

# 요청이 겹치면 직렬화: 첫 학습 중 들어온 요청은 기다렸다가 저장된 결과를 읽는다
_LOCK = threading.Lock()


def _load_saved():
    try:
        saved = json.loads(DEMO_RESULT.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        log.warning("saved demo result unreadable, training again: %s", DEMO_RESULT)
        return None
    if not isinstance(saved, dict) or any(k not in saved for k in RESULT_KEYS):
        log.warning("saved demo result incomplete, training again: %s", DEMO_RESULT)
        return None
    return saved


def _save(result):
    try:
        DEMO_RESULT.parent.mkdir(parents=True, exist_ok=True)
        tmp = DEMO_RESULT.with_name(DEMO_RESULT.name + ".tmp")
        tmp.write_text(
            json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(tmp, DEMO_RESULT)
    except OSError:
        log.exception("demo result not saved, next request trains again")


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


def run_demo(device="auto", data_dir=DEFAULT_DATA_DIR):
    with _LOCK:
        saved = _load_saved()
        if saved is not None:
            return saved
        if not torch.cuda.is_available():
            raise RuntimeError("demo needs a CUDA device for the first training run")
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
        splits = plan["data"]["splits"]
        methods = [m for m in METHODS if m in summary]
        result = {
            "model": MODEL,
            "methods": methods,
            "labels": dict(METHOD_LABELS),
            "data": {s: int(splits[s]["n"]) for s in ("train", "dev", "test")},
            "argmax": {m: _block(summary[m]["argmax"]) for m in methods},
            "calibrated": {m: _block(summary[m]["calibrated"]) for m in methods},
            "threshold": {m: _number(summary[m]["threshold"]) for m in methods},
            "elapsed_seconds": perf_counter() - started,
            "trained_at": datetime.now(KST).strftime("%Y-%m-%d %H:%M KST"),
        }
        _save(result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto or cuda:N")
    args = parser.parse_args(argv)
    result = run_demo(device=args.device)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
