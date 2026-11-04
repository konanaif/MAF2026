"""KoBroadcast 사전학습 모델을 HF Hub에서 model/kobroadcast/hf 로 받는다(없을 때만).

실행: python -m MAF.datamodule.kobroadcast.download_data [--model roberta-base|roberta-large|qwen3-4b|all]
KoBroadcast.run()이 --model-path 없이 실행될 때 같은 함수(ensure_model)를 먼저 부른다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download
from MAF.algorithms.inprocessing.kobroadcast.constants import (
    MODEL_IDS,
    MODEL_REVISIONS,
    MODELS,
)

MAF_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = MAF_ROOT / "model" / "kobroadcast" / "hf"
# 고정 revision 3개 모두 safetensors를 제공한다. 다른 형식까지 받으면 용량이 2배다.
IGNORE_PATTERNS = ["*.bin", "*.h5", "*.msgpack", "*.ot", "*.onnx"]


def ensure_model(
    model_name: str,
    revision: str | None = None,
    cache_dir: Path = DEFAULT_MODEL_DIR,
    offline: bool = False,
) -> Path:
    """캐시(cache_dir)에 없으면 받고, 로컬 스냅샷 경로를 돌려준다."""
    if model_name not in MODEL_IDS:
        raise ValueError(f"Unknown model {model_name!r}; choose one of {MODELS}")
    repo_id = MODEL_IDS[model_name]
    revision = revision or MODEL_REVISIONS[model_name]
    try:
        snapshot = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=str(cache_dir),
            local_files_only=offline,
            ignore_patterns=IGNORE_PATTERNS,
        )
    except OSError as error:
        raise OSError(
            f"Could not prepare {repo_id}@{revision} in {cache_dir} ({error}). "
            "Without network access, pass a local snapshot with "
            "--model-path <local snapshot> --offline."
        ) from error
    return Path(snapshot)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*MODELS, "all"), default="all")
    parser.add_argument("--revision", help="Hub revision; default is the pinned one")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    names = MODELS if args.model == "all" else (args.model,)
    try:
        for name in names:
            print(f"{name}: {ensure_model(name, args.revision, offline=args.offline)}")
        return 0
    except (OSError, ValueError) as error:
        parser.exit(1, f"KoBroadcast model setup failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
