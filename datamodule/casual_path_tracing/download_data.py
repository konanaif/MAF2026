"""CPT가 쓰는 모델(pythia-14m)·nltk stopwords를 MAF/model 아래로 자동 수신한다.

timm 가중치는 라이브러리 `get_model`이 `create_model(cache_dir=MODEL_DIR)`로 직접 받는다.
"""

import argparse
import os
from pathlib import Path

import nltk
from huggingface_hub import snapshot_download
from MAF.datamodule.casual_path_tracing.sample import MAF_ROOT

MODEL_DIR = MAF_ROOT / "model" / "casual_path_tracing" / "hf"
NLTK_DIR = MAF_ROOT / "model" / "casual_path_tracing" / "nltk_data"

MODEL_IDS = {"pythia-14m": "EleutherAI/pythia-14m"}
MODEL_REVISIONS = {"pythia-14m": "cf967c0a9a04383db6f7b1108d86b2962634b4ac"}


def ensure_model(
    model: str,
    revision: str | None = None,
    model_dir: Path = MODEL_DIR,
    offline: bool = False,
) -> Path:
    """이름(`pythia-14m`)·repo id·로컬 디렉터리 중 무엇이 와도 로컬 모델 경로를 돌려준다."""
    if os.path.isdir(model):
        # 라이브러리가 경로 문자열의 'pythia'로 forward 함수를 고르므로 이름 없는 폴더는 거부한다
        if "pythia" not in model:
            raise ValueError(
                f"Local model dir {model!r} must contain 'pythia' in its path "
                f"(e.g. {MODEL_DIR / 'pythia-14m'})"
            )
        return Path(model).resolve()
    by_repo = {repo: name for name, repo in MODEL_IDS.items()}
    name = model if model in MODEL_IDS else by_repo.get(model)
    if name is None:
        raise ValueError(f"Unknown model {model!r}; supported: {sorted(MODEL_IDS)}")
    repo_id = MODEL_IDS[name]
    revision = revision or MODEL_REVISIONS[name]
    local_dir = str(model_dir / name)
    try:
        path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=local_dir,
            local_files_only=offline,
        )
    except OSError as error:
        raise OSError(
            f"Could not prepare {repo_id}@{revision} in {local_dir} ({error}). "
            "Check network/DNS, or pass --model <local dir> pointing at a "
            "previously downloaded copy."
        ) from error
    return Path(path)


def ensure_stopwords(nltk_dir: Path = NLTK_DIR) -> None:
    """stopwords 말뭉치가 없으면 nltk_dir로 받는다. 검색 경로 맨 앞에 nltk_dir를 둔다."""
    where = str(nltk_dir)
    if where in nltk.data.path:
        nltk.data.path.remove(where)
    nltk.data.path.insert(0, where)
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        if not nltk.download("stopwords", download_dir=where, quiet=True):
            raise OSError(f"Could not download nltk stopwords into {where}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="CPT 모델·nltk 데이터 선반입")
    parser.add_argument("--model", default="pythia-14m", choices=sorted(MODEL_IDS))
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    print(ensure_model(args.model, offline=args.offline))
    ensure_stopwords()
    print(NLTK_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
