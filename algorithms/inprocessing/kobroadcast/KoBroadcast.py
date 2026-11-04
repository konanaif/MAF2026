"""KoBroadcast 진입점: Base / Base+SWA / FairBatch / LEACE+SWA 로 방송 문장 유해성을 분류한다.

실행: python -m MAF.algorithms.inprocessing.kobroadcast.KoBroadcast [--data-dir DIR] [--model M] [--method X] [--device auto|cpu|cuda:N] [--output DIR]
데이터는 data/kobroadcast 의 train/dev/test.jsonl (또는 --data-dir). 모델 가중치는 --model-path 가 없으면
datamodule/kobroadcast/download_data.ensure_model() 이 model/kobroadcast/hf 에 받는다(없을 때만).
"""
# ruff: noqa: N999  모듈명 KoBroadcast는 제출분 진입점 이름 유지

import argparse
import json
from pathlib import Path

import torch
from MAF.algorithms.inprocessing.kobroadcast.constants import (
    METHODS,
    MODEL_REVISIONS,
    MODELS,
)
from MAF.algorithms.inprocessing.kobroadcast.experiment import (
    DEFAULT_OUTPUT_ROOT,
    MAF_ROOT,
    execute,
)
from MAF.datamodule.kobroadcast.download_data import ensure_model
from MAF.datamodule.kobroadcast.sample import audit_splits, split_paths

__all__ = ["DEFAULT_DATA_DIR", "DEFAULT_OUTPUT_ROOT", "KoBroadcast", "main"]

DEFAULT_DATA_DIR = MAF_ROOT / "data" / "kobroadcast"
PROMPT = Path(__file__).resolve().parent / "prompts" / "toxicity.md"


class KoBroadcast:
    """제출분 run.py 의 인자·검증을 그대로 받는 MAF 알고리즘 클래스."""

    def __init__(
        self,
        data_dir=None,
        model="all",
        method="all",
        output=None,
        device="auto",
        checkpoint=None,
        model_path=None,
        revision=None,
        offline=False,
        seed=20260625,
        epochs=None,
        batch_size=None,
        accumulation=None,
        learning_rate=None,
        max_length=None,
        swa_start=3,
        leace_rank=128,
        leace_strength=0.8,
    ):
        if model not in (*MODELS, "all") or method not in (*METHODS, "all"):
            raise ValueError(f"model must be one of {MODELS}, method one of {METHODS}")
        if (model_path or revision) and model == "all":
            raise ValueError("--model-path/--revision require one --model")
        if checkpoint and (model == "all" or method != "leace_swa"):
            raise ValueError("--checkpoint requires one --model and --method leace_swa")
        for name, value in (
            ("epochs", epochs),
            ("batch-size", batch_size),
            ("accumulation", accumulation),
            ("max-length", max_length),
            ("leace-rank", leace_rank),
        ):
            if value is not None and value < 1:
                raise ValueError(f"--{name} must be positive")
        if not 0 < leace_strength <= 1 or (
            learning_rate is not None and learning_rate <= 0
        ):
            raise ValueError("Use a positive learning rate and LEACE strength in (0,1]")
        if swa_start < 0:
            raise ValueError("--swa-start must be nonnegative")
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
        self.model, self.method, self.device = model, method, device
        self.output = Path(output) if output else None
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.model_path, self.revision, self.offline = model_path, revision, offline
        self.seed, self.swa_start = seed, swa_start
        self.epochs, self.batch_size, self.accumulation = (
            epochs,
            batch_size,
            accumulation,
        )
        self.learning_rate, self.max_length = learning_rate, max_length
        self.leace_rank, self.leace_strength = leace_rank, leace_strength
        self.results = None

    def _config(self, name):
        qwen = name == "qwen3-4b"
        cfg = {
            "model_path": self.model_path,
            "revision": self.revision
            or (None if self.model_path else MODEL_REVISIONS[name]),
            "offline": self.offline,
            "epochs": self.epochs or (3 if qwen else 5),
            "batch_size": self.batch_size or (2 if qwen else 8),
            "eval_batch_size": 1 if qwen else 8,
            "accumulation": self.accumulation or (8 if qwen else 1),
            "lr": self.learning_rate or (1e-4 if qwen else 2e-5),
            "max_length": self.max_length or (1024 if qwen else 512),
            "swa_start": self.swa_start,
            "swa_lr": 2e-5 if qwen else (self.learning_rate or 2e-5) * 0.1,
            "prompt": str(PROMPT),
        }
        needs_swa = (
            self.method in ("all", "base_swa", "leace_swa") and not self.checkpoint
        )
        if not qwen and needs_swa and cfg["swa_start"] >= cfg["epochs"]:
            raise ValueError(
                "--swa-start must be earlier than --epochs for RoBERTa SWA"
            )
        return cfg

    def _prepare(self):
        paths = split_paths(self.data_dir)
        audit = audit_splits(paths)
        names = MODELS if self.model == "all" else (self.model,)
        configs = {name: self._config(name) for name in names}
        return paths, audit, configs

    def plan(self):
        """모델을 로드하지 않고 분할 검증 결과와 실행 설정만 돌려준다(--dry-run)."""
        _, audit, configs = self._prepare()
        methods = list(METHODS) if self.method == "all" else [self.method]
        return {
            "models": list(configs),
            "methods": methods,
            "data": audit,
            "configs": configs,
        }

    def run(self):
        paths, audit, configs = self._prepare()
        # execute()와 같은 검사를 다운로드 전에 먼저 한다(수 GB를 받고 나서 멈추지 않도록).
        device = torch.device(self.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                "CUDA is unavailable; use --dry-run or a CPU RoBERTa smoke check"
            )
        if "qwen3-4b" in configs and device.type != "cuda":
            raise ValueError(
                "Qwen3-4B requires CUDA; select one RoBERTa model for CPU execution"
            )
        if self.output is not None and self.output.exists():
            raise FileExistsError(f"Output directory already exists: {self.output}")
        for name, cfg in configs.items():
            # --checkpoint 는 체크포인트 안의 cfg·가중치를 쓰므로 기본 모델이 필요 없다.
            if cfg["model_path"] is None and self.checkpoint is None:
                cfg["model_path"] = str(
                    ensure_model(name, self.revision, offline=self.offline)
                )
                cfg["revision"] = None
        args = argparse.Namespace(
            output_dir=self.output,
            device=self.device,
            seed=self.seed,
            method=self.method,
            checkpoint=self.checkpoint,
            leace_rank=self.leace_rank,
            leace_strength=self.leace_strength,
        )
        self.output = execute(args, paths, audit, configs)
        self.results = json.loads(
            (self.output / "all_results.json").read_text(encoding="utf-8")
        )
        return self.results

    def summary(self):
        """run() 이후 모델·기법별 TEST 지표(argmax / calibrated)와 임계값."""
        if self.results is None:
            raise RuntimeError("run() must finish before summary()")
        out = {}
        for r in self.results:
            out.setdefault(r["model"], {})[r["method"]] = {
                "argmax": r["argmax"]["overall"],
                "calibrated": r["calibrated"]["overall"],
                "threshold": r["threshold"],
            }
        return out


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--model", choices=(*MODELS, "all"), default="all")
    p.add_argument("--method", choices=(*METHODS, "all"), default="all")
    p.add_argument(
        "--output",
        type=Path,
        help="New output folder; default result/kobroadcast/<UTC>",
    )
    p.add_argument("--device", default="auto", help="auto, cpu, or cuda:N")
    p.add_argument(
        "--checkpoint",
        type=Path,
        help="Existing base_swa checkpoint, for leace_swa only",
    )
    p.add_argument(
        "--model-path", help="Local pretrained model/tokenizer path for one --model"
    )
    p.add_argument("--revision", help="Pin the pretrained Hugging Face model revision")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--seed", type=int, default=20260625)
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--accumulation", type=int)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--max-length", type=int)
    p.add_argument("--swa-start", type=int, default=3)
    p.add_argument("--leace-rank", type=int, default=128)
    p.add_argument("--leace-strength", type=float, default=0.8)
    p.add_argument(
        "--dry-run", action="store_true", help="Validate splits and show the plan only"
    )
    return p


def main(argv=None):
    p = parser()
    a = p.parse_args(argv)
    options = {k: v for k, v in vars(a).items() if k != "dry_run"}
    try:
        algorithm = KoBroadcast(**options)
        plan = algorithm.plan()
    except ValueError as error:
        p.error(str(error))
    if a.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    try:
        algorithm.run()
    except OSError as error:
        p.exit(1, f"KoBroadcast failed: {error}\n")
    print(json.dumps(algorithm.summary(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
