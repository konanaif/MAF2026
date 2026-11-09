"""KoBBQ-Steer 진입점: AlphaSteer null-space 활성화 스티어링으로 Qwen2.5-7B-Instruct 의 KoBBQ 편향을 완화한다.

실행: python -m MAF.algorithms.inprocessing.kobbq_steer.KoBBQSteer [--data-dir DIR] [--model-dir DIR] [--output DIR]
      [--device auto|cuda:N] [--kmmlu] [--rebuild-steer]
run() -> (base 지표, alphasteer 지표): KoBBQ test 템플릿 전량을 두 조건으로 평가한다(--kmmlu 면 KMMLU test 도).
데이터는 data/kobbq_steer/{ext/KoBBQ, raw/} 를 직접 두고(datamodule/kobbq_steer/prepare_data.SETUP_HINT), 모델은
model/kobbq_steer/hf 로 받는다. 스티어링 행렬 2개가 model/kobbq_steer/steer/<run>/ 에 없으면 build_steer 가 만든다
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from MAF.algorithms.inprocessing.kobbq_steer.common import (
    get_logger,
    load_json,
    load_model,
    run_dir,
    save_json,
    seed_all,
)
from MAF.algorithms.inprocessing.kobbq_steer.eval_kmmlu import (
    load_split,
    shots_by_subject,
)
from MAF.algorithms.inprocessing.kobbq_steer.eval_kobbq import metrics, official_check
from MAF.algorithms.inprocessing.kobbq_steer.steer_eval import (
    build_spec,
    eval_kmmlu,
    eval_kobbq,
)

# MAF.datamodule.kobbq_steer 는 메서드 안에서 import 한다: 그 모듈들이 이 패키지를 import 하고
__all__ = ["METRIC_KEYS", "KoBBQSteer", "main", "to_row"]

METRIC_KEYS = (
    "kobbq_amb_acc",
    "kobbq_dis_acc",
    "bias_amb",
    "bias_dis",
    "amb_stereo_rate",
    "amb_anti_rate",
    "amb_anti_share_of_errors",
)
_RENAME = {"acc_amb": "kobbq_amb_acc", "acc_dis": "kobbq_dis_acc"}
PREPARED = ("kobbq", "kmmlu_dev", "benign")


def to_row(condition, kb_metrics, kmmlu_acc=None, kmmlu_base_acc=None):
    row = {"condition": condition}
    for k, v in kb_metrics.items():
        k = _RENAME.get(k, k)
        if k in METRIC_KEYS:
            row[k] = v
    if kmmlu_acc is not None:
        row["kmmlu_acc"] = kmmlu_acc
        row["rel_drop"] = (kmmlu_base_acc - kmmlu_acc) / kmmlu_base_acc * 100
    return row


class KoBBQSteer:
    """base(무개입) 와 alphasteer(선택 설정 L16·caa·p0.3·α100·λ0.1) 두 조건을 평가해 지표 쌍을 돌려준다."""

    def __init__(
        self,
        data_dir=None,
        model_dir=None,
        output=None,
        device="auto",
        kmmlu=False,
        rebuild_steer=False,
    ):
        from MAF.datamodule.kobbq_steer.prepare_data import SELECTED_JSON, make_config

        self.cfg = make_config(data_dir, model_dir, output)
        self.official_py = (
            Path(self.cfg["paths"]["kobbq_dir"]).parent
            / "evaluation"
            / "5_evaluation.py"
        )
        self.selected = load_json(SELECTED_JSON)["selected"]["alphasteer"]
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.kmmlu = kmmlu
        self.rebuild_steer = rebuild_steer
        s = self.selected
        self.steer_dir = Path(self.cfg["paths"]["steer"]) / self.cfg["run_name"]
        self.steer_files = [
            self.steer_dir / "directions.pt",
            self.steer_dir
            / "alphasteer"
            / f"L{s['layers'][0]}_p{s['p']}_a{s['alpha']}_{s['r_kind']}.pt",
        ]
        self.results = None

    def ensure_prepared(self):
        proc = Path(self.cfg["paths"]["processed"])
        if all((proc / f"{f}.parquet").exists() for f in PREPARED):
            return False
        from MAF.datamodule.kobbq_steer.prepare_data import prepare_data

        prepare_data(self.cfg)
        return True

    def ensure_steer(self):
        if not self.rebuild_steer and all(f.exists() for f in self.steer_files):
            return False
        from MAF.datamodule.kobbq_steer.build_steer import build_steer

        build_steer(self.cfg, self.device)
        return True

    def run(self):
        from MAF.datamodule.kobbq_steer.prepare_data import SETUP_HINT, write_config

        if not str(self.device).startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError(
                "KoBBQSteer needs a CUDA device (device='auto' or 'cuda:N')"
            )
        cfg = self.cfg
        write_config(cfg, cfg["paths"]["results"])
        self.ensure_prepared()
        if not self.official_py.is_file():
            raise FileNotFoundError(f"{self.official_py}: not found. {SETUP_HINT}")
        self.ensure_steer()
        seed_all(cfg["seed"])
        model, tok = load_model(cfg, self.device)
        try:
            self._evaluate(model, tok)
        finally:
            del model
            torch.cuda.empty_cache()
        results = self.summary()
        save_json(results, run_dir(cfg, "results") / "summary.json")
        return results["base"], results["alphasteer"]

    def _evaluate(self, model, tok):
        cfg = self.cfg
        P = cfg["paths"]
        R = run_dir(cfg, "results")
        T = R / "test"
        T.mkdir(exist_ok=True)
        log = get_logger("KoBBQSteer", Path(P["results"]) / "logs/KoBBQSteer.log")
        max_tokens = cfg["extract"]["max_tokens_per_batch"]

        kb = pd.read_parquet(Path(P["processed"]) / "kobbq.parquet")
        kb_test = kb[kb.split == "test"].reset_index(drop=True)
        km_test = shots = None
        if self.kmmlu:
            km_test = load_split(P["raw"], "test")
            dev = pd.read_parquet(Path(P["processed"]) / "kmmlu_dev.parquet")
            shots = shots_by_subject(dev, cfg["kmmlu"]["n_shot"])
        log.info(
            f"KoBBQ test n={len(kb_test)} ({kb_test.template_id.nunique()} templates)"
            + (f", KMMLU test n={len(km_test)}" if km_test is not None else "")
        )

        s = self.selected
        conds = {
            "base": ({}, 0.0),
            "alphasteer": (
                build_spec(
                    "alphasteer",
                    self.steer_dir,
                    s["layers"],
                    s["r_kind"],
                    s["p"],
                    s["alpha"],
                    "alphasteer",
                ),
                s["lam"],
            ),
        }
        results, km_base = {}, None
        for name, (spec, lam) in conds.items():
            kbp = eval_kobbq(model, tok, kb_test, cfg, name, spec, lam, max_tokens)
            kbp.to_parquet(T / f"kobbq_{name}.parquet")
            m = metrics(kbp)
            off = official_check(kbp, self.official_py)
            for k in ("acc_amb", "acc_dis", "bias_amb", "bias_dis"):
                assert abs(off[k] - m[k]) < 1e-9, (name, k, off[k], m[k])
            log.info(f"{name} KoBBQ test: {m}")
            acc = None
            if km_test is not None:
                kmp = eval_kmmlu(
                    model, tok, km_test, shots, cfg, name, spec, lam, max_tokens
                )
                kmp.to_parquet(T / f"kmmlu_{name}.parquet")
                acc = float((kmp.pred_idx == kmp.answer - 1).mean())
                km_base = acc if name == "base" else km_base
                log.info(f"{name} KMMLU test acc={acc:.4f}")
            results[name] = to_row(name, m, acc, km_base)
        self.results = results

    def summary(self):
        if self.results is None:
            raise RuntimeError("call run() first")
        return {
            "base": self.results["base"],
            "alphasteer": self.results["alphasteer"],
            "selected": self.selected,
            "kmmlu": self.kmmlu,
        }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="KoBBQ-Steer: base vs alphasteer on KoBBQ test"
    )
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--kmmlu",
        action="store_true",
        help="also evaluate KMMLU test (~1h per condition)",
    )
    ap.add_argument(
        "--rebuild-steer", action="store_true", help="regenerate steering matrices"
    )
    args = ap.parse_args(argv)
    try:
        k = KoBBQSteer(
            args.data_dir,
            args.model_dir,
            args.output,
            args.device,
            args.kmmlu,
            args.rebuild_steer,
        )
        k.run()
    except (RuntimeError, FileNotFoundError, AssertionError) as e:
        print(f"KoBBQSteer: {e}", file=sys.stderr)
        return 1
    print(json.dumps(k.summary(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
