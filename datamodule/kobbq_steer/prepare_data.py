"""KoBBQ-Steer 데이터 준비
실행: python -m MAF.datamodule.kobbq_steer.prepare_data [--data-dir DIR] [--model-dir DIR] [--output DIR]
입력 data/kobbq_steer/{ext/KoBBQ, raw/} 은 직접 준비한다(SETUP_HINT). 산출은 data/kobbq_steer/processed/ 와 <output>/splits/.
"""

import argparse
import sys
from pathlib import Path

import yaml
from MAF.algorithms.inprocessing.kobbq_steer.common import (
    get_logger,
    load_config,
    save_json,
    seed_all,
)
from MAF.algorithms.inprocessing.kobbq_steer.data_benign import build_benign
from MAF.algorithms.inprocessing.kobbq_steer.data_kobbq import (
    load_kobbq,
    template_kfold,
    template_split,
)
from MAF.algorithms.inprocessing.kobbq_steer.eval_kmmlu import load_split, make_val
from MAF.algorithms.inprocessing.kobbq_steer.prompts import assign_option_order

MAF_ROOT = Path(__file__).resolve().parents[2]
PKG = MAF_ROOT / "algorithms" / "inprocessing" / "kobbq_steer"
DEFAULT_CONFIG = PKG / "configs" / "default.yaml"
SELECTED_JSON = PKG / "configs" / "selected_alphasteer.json"
DEFAULT_DATA_DIR = MAF_ROOT / "data" / "kobbq_steer"
DEFAULT_MODEL_DIR = MAF_ROOT / "model" / "kobbq_steer"
DEFAULT_OUTPUT = MAF_ROOT / "result" / "kobbq_steer"
HF_DATASETS = (
    "HAERAE-HUB/KMMLU",
    "nlpai-lab/kullm-v2",
    "kyujinpy/KOpen-platypus",
    "kuotient/gsm8k-ko",
    "CarrotAI/ko-code-alpaca-QA",
)
SETUP_HINT = (
    "Prepare the inputs first (run from the MAF2026 root):\n"
    "  git clone https://github.com/naver-ai/KoBBQ data/kobbq_steer/ext/KoBBQ"
    " && git -C data/kobbq_steer/ext/KoBBQ checkout 0dd165a\n"
    "  huggingface-cli download <repo> --repo-type dataset"
    " --local-dir data/kobbq_steer/raw/<owner>__<name>   for each of: "
    + ", ".join(HF_DATASETS)
    + "\nSee algorithms/inprocessing/kobbq_steer/README.md."
)


def make_config(data_dir=None, model_dir=None, output=None):
    data_dir = Path(data_dir or DEFAULT_DATA_DIR)
    model_dir = Path(model_dir or DEFAULT_MODEL_DIR)
    output = Path(output or DEFAULT_OUTPUT)
    cfg = load_config(str(DEFAULT_CONFIG))
    cfg["paths"] = {
        "raw": str(data_dir / "raw"),
        "kobbq_dir": str(data_dir / "ext" / "KoBBQ" / "data"),
        "processed": str(data_dir / "processed"),
        "splits": str(output / "splits"),
        "results": str(output),
        "hf_home": str(model_dir / "hf"),
        "acts": str(model_dir / "acts"),
        "steer": str(model_dir / "steer"),
    }
    return cfg


def write_config(cfg, output):
    path = Path(output) / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump({"base": str(DEFAULT_CONFIG), "paths": cfg["paths"]}, f)
    return path


def prepare_data(cfg):
    P = cfg["paths"]
    required = [Path(P["raw"]) / r.replace("/", "__") for r in HF_DATASETS]
    required.append(Path(P["kobbq_dir"]) / "KoBBQ_all_samples.tsv")
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"{p}: not found. {SETUP_HINT}")
    seed_all(cfg["seed"])
    proc = Path(P["processed"])
    proc.mkdir(parents=True, exist_ok=True)
    log = get_logger("prep", Path(P["results"]) / "logs/01_prepare_data.log")

    # ---- KoBBQ
    kb = load_kobbq(P["kobbq_dir"], cfg["kobbq"]["unknown_text"])
    split = template_split(kb, cfg["kobbq"]["split_ratio"], cfg["seed"])
    folds = template_kfold(kb, cfg["kobbq"]["n_folds"], cfg["seed"])
    save_json(split, Path(P["splits"]) / "kobbq_template_split.json")
    save_json(folds, Path(P["splits"]) / "kobbq_5fold.json")
    kb["split"] = kb.template_id.map(split)
    kb["fold"] = kb.template_id.map(folds)
    kb = assign_option_order(kb, cfg["seed"])
    kb.to_parquet(proc / "kobbq.parquet")
    log.info(f"KoBBQ samples={len(kb)} templates={kb.template_id.nunique()}")
    log.info(
        "templates per split x category:\n"
        + kb.drop_duplicates("template_id")
        .pivot_table(
            index="category", columns="split", values="template_id", aggfunc="count"
        )
        .to_string()
    )
    log.info(
        "samples per split:\n"
        + kb.groupby(["split", "context_type"]).size().to_string()
    )
    for r in ["unknown", "stereo", "anti"]:
        log.info(
            f"{r} letter dist: {kb[f'{r}_option_idx'].value_counts().sort_index().to_dict()}"
        )

    # ---- KMMLU
    dev, train = load_split(P["raw"], "dev"), load_split(P["raw"], "train")
    val = make_val(train, cfg["kmmlu"]["val_size"], cfg["seed"])
    dev.to_parquet(proc / "kmmlu_dev.parquet")
    val.to_parquet(proc / "kmmlu_val.parquet")
    log.info(
        f"KMMLU dev={len(dev)} train={len(train)} val(from train)={len(val)} subjects={val.subject.nunique()}"
    )
    log.info(f"KMMLU-val per group: {val.group.value_counts().to_dict()}")

    test_q = load_split(P["raw"], "test").question.astype(str)

    # ---- benign
    excl = (
        set(test_q)
        | set(train.question.astype(str))
        | set(dev.question.astype(str))
        | set(kb.context)
        | set(kb.question)
    )
    ben, stats = build_benign(cfg, excl, log.info)
    ben.to_parquet(proc / "benign.parquet")
    save_json(stats, Path(P["results"]) / cfg["run_name"] / "benign_stats.json")
    log.info(f"benign total={len(ben)} split={ben.split.value_counts().to_dict()}")
    assert len(ben) >= 14000, "benign < 14k: STOP and report"


def main(argv=None):
    ap = argparse.ArgumentParser(description="KoBBQ-Steer data preparation")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)
    cfg = make_config(args.data_dir, args.model_dir, args.output)
    write_config(cfg, cfg["paths"]["results"])
    try:
        prepare_data(cfg)
    except FileNotFoundError as e:
        print(f"prepare_data: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
