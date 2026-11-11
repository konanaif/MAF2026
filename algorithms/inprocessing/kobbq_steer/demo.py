"""KoBBQ-Steer 데모: 한 문항을 base 와 alphasteer 로 비교한다.

실행: python -m MAF.algorithms.inprocessing.kobbq_steer.demo --context C --question Q --choices "a,b,c"
"""

import argparse
import json
import sys
import threading
from pathlib import Path

import torch
from MAF.algorithms.inprocessing.kobbq_steer.common import load_json, load_model
from MAF.algorithms.inprocessing.kobbq_steer.hooks import Steerer
from MAF.algorithms.inprocessing.kobbq_steer.prompts import kobbq_prompt
from MAF.algorithms.inprocessing.kobbq_steer.scoring import score_texts
from MAF.algorithms.inprocessing.kobbq_steer.steer_eval import build_spec

__all__ = ["compare_sample", "main"]

# GPU 한 장에 7B 모델을 동시에 두 번 올리지 않도록 호출을 직렬화한다
_LOCK = threading.Lock()


def _index_of(value, options, name):
    if value is None:
        return None
    v = value.strip()
    if v not in options:
        raise ValueError(f"{name} {value!r} is not one of the choices {options}")
    return options.index(v)


def _probs(logits):
    e = torch.softmax(torch.as_tensor(logits, dtype=torch.float64), dim=-1)
    return [float(x) for x in e]


def compare_sample(
    context,
    question,
    choices,
    answer=None,
    biased_answer=None,
    model_dir=None,
    device="auto",
):
    # MAF.datamodule.kobbq_steer 는 그 모듈들이 이 패키지를 import 하므로 함수 안에서 import 한다
    from MAF.datamodule.kobbq_steer.prepare_data import SELECTED_JSON, make_config

    options = [str(c).strip() for c in choices]
    if len(options) != 3 or not all(options):
        raise ValueError(f"choices must be exactly 3 non-empty strings: {choices!r}")
    answer_idx = _index_of(answer, options, "answer")
    biased_idx = _index_of(biased_answer, options, "biased_answer")

    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    with _LOCK:
        if not str(device).startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("demo needs a CUDA device (device='auto' or 'cuda:N')")
        cfg = make_config(model_dir=model_dir)
        s = load_json(SELECTED_JSON)["selected"]["alphasteer"]
        steer_dir = Path(cfg["paths"]["steer"]) / cfg["run_name"]
        files = [steer_dir / "directions.pt"] + [
            steer_dir
            / "alphasteer"
            / f"L{layer}_p{s['p']}_a{s['alpha']}_{s['r_kind']}.pt"
            for layer in s["layers"]
        ]
        for f in files:
            if not f.exists():
                raise FileNotFoundError(
                    f"{f}: not found. Run KoBBQSteer first or place the files under "
                    "model/kobbq_steer/steer/qwen2.5-7b/"
                )
        hf_dir = (
            Path(cfg["paths"]["hf_home"])
            / "hub"
            / f"models--{cfg['model']['name'].replace('/', '--')}"
        )
        if not hf_dir.exists():
            raise FileNotFoundError(
                f"{hf_dir}: HF model cache not found. Mount "
                "/data/nayoung.han/models/maf2026 or pass --model-dir "
                "(see algorithms/inprocessing/kobbq_steer/README.md)"
            )
        model, tok = load_model(cfg, device)
        try:
            row = {"options": options, "context": context, "question": question}
            text = kobbq_prompt(tok, row, cfg)
            spec = build_spec(
                "alphasteer",
                steer_dir,
                s["layers"],
                s["r_kind"],
                s["p"],
                s["alpha"],
                "alphasteer",
            )
            out = {}
            for name, cond_spec, lam in (
                ("base", {}, 0.0),
                ("alphasteer", spec, s["lam"]),
            ):
                with Steerer(model, cond_spec, lam, cfg["alphasteer"]["mode"]):
                    res = score_texts(model, tok, [text], "ABC")
                idx = int(res["pred_idx"][0])
                out[name] = {
                    "pred_idx": idx,
                    "pred_text": options[idx],
                    "probs": _probs(res["letter_logits"][0]),
                }
        finally:
            del model
            torch.cuda.empty_cache()
    return {
        "options": options,
        "answer_idx": answer_idx,
        "biased_idx": biased_idx,
        **out,
        "selected": {k: s[k] for k in ("layers", "r_kind", "p", "alpha", "lam")},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="KoBBQ-Steer: base vs alphasteer on one sample"
    )
    ap.add_argument("--context", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--choices", required=True, help="three choices, comma separated")
    ap.add_argument("--answer", default=None)
    ap.add_argument("--biased-answer", default=None)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    try:
        result = compare_sample(
            args.context,
            args.question,
            [c.strip() for c in args.choices.split(",")],
            args.answer,
            args.biased_answer,
            args.model_dir,
            args.device,
        )
    except (ValueError, OSError, RuntimeError) as e:
        print(f"demo: {e}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
