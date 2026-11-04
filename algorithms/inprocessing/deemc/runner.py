"""MAF interface for the unchanged AI_DEP_minimal final prediction pipeline."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import torch

from . import runtime
from .integrity import sha256, verify_original
from .maf_metrics import compute_metrics, prediction_dataset, to_aif360_dataset

MAF_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = MAF_ROOT / "data" / "deemc"
DEFAULT_MODEL_DIR = MAF_ROOT / "model" / "deemc"


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


class DEEMCRunner:
    """Return MF-reference and final-model metric dictionaries for MAF.

    This comparison uses different backbones; it is not a matched DEE ablation.
    The supplied fair checkpoints are never retrained or modified by run().
    """

    def __init__(self, dataset_name="movielens1m", protected="sex", *,
                 project_root=None, config_path=None, device="auto",
                 data_dir=None, model_dir=None):
        if dataset_name.lower().replace("_", "").replace("-", "") not in {"movielens1m", "ml1m"}:
            raise ValueError("This submission supports the bundled MovieLens 1M split only")
        if protected != "sex":
            raise ValueError("The supported protected attribute is sex (M=1, F=0)")
        self.dataset_name, self.protected = dataset_name, protected
        self.project_root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parent
        self.data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
        self.model_dir = Path(model_dir) if model_dir is not None else DEFAULT_MODEL_DIR
        config_path = Path(config_path) if config_path else Path("config.json")
        if not config_path.is_absolute():
            config_path = self.project_root / config_path
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing config: {config_path}")
        self.config = runtime.read_json(config_path)
        if self.config["baseline"] != "mf_reference":
            raise ValueError("Only the included MF reference baseline is supported")
        self.inference_config = self.config["inference"]
        if self.inference_config["blend_weight"] != 0.15 or len(self.inference_config["glocal_checkpoints"]) != 2:
            raise ValueError("Preserve the original two-model ensemble and 15% MF blend")
        self.threshold = float(self.config["classification_threshold"])
        if not np.isfinite(self.threshold) or not 1 <= self.threshold <= 5:
            raise ValueError("classification_threshold must be on the 1–5 rating scale")
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("Use cpu or cuda:N; MPS is not part of the reference implementation")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self.privileged_groups = [{protected: 1}]
        self.unprivileged_groups = [{protected: 0}]
        self.report = None
        self.dataset_orig_test = None

    def _aif_test(self, inputs):
        _, _, test_r, test_m, user_ids, movie_ids, groups = inputs
        mi, ui = np.nonzero(test_m)
        user_groups, item_groups = groups
        if not np.isin(user_groups, ["M", "F"]).all():
            raise ValueError("Unexpected MovieLens gender labels")
        features = np.column_stack([
            item_groups[mi] == "male_preferred",
            item_groups[mi] == "female_preferred",
        ]).astype(np.float64)
        names = [f"user_{int(user_ids[u])}:movie_{int(movie_ids[m])}" for m, u in zip(mi, ui)]
        truth = to_aif360_dataset(
            features, ["male_preferred_genre", "female_preferred_genre"],
            (test_r[mi, ui] >= self.threshold).astype(int),
            (user_groups[ui] == "F").astype(int), names, self.protected,
        )
        return truth, mi, ui

    def compute_metrics(self, dataset):
        if self.dataset_orig_test is None:
            raise RuntimeError("run() must create the held-out dataset first")
        return compute_metrics(self.dataset_orig_test, dataset,
                               self.unprivileged_groups, self.privileged_groups)

    def run(self):
        started = perf_counter()
        threads = int(self.config["threads"])
        if threads < 1:
            raise ValueError("threads must be positive")
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(threads)
        try:
            return self._run(started, threads)
        finally:
            torch.set_num_threads(previous_threads)

    def _run(self, started, threads):
        checksums = verify_original(self.project_root, self.data_dir, self.model_dir)
        if sha256(Path(runtime.original.__file__)) != checksums["util/GLocal_K_torch.py"]:
            raise ValueError("Packaged GLocal-K source differs from the original")
        config = self.inference_config
        inputs = runtime.load_inputs(config, self.data_dir)
        train_r, train_m, test_r, test_m, user_ids, movie_ids, groups = inputs
        tr = torch.from_numpy(train_r).to(self.device)
        tm = torch.from_numpy(train_m).to(self.device)
        predictions, loads = [], []
        for relative in config["glocal_checkpoints"]:
            model, check = runtime.load_glocal(
                self.model_dir / relative, self.config["training"],
                len(movie_ids), len(user_ids), self.device)
            predictions.append(runtime.glocal_predict(model, tr, tm))
            loads.append(check)
            del model
        if len(predictions) != 2:
            raise ValueError("The supplied final model requires exactly two GLocal-K checkpoints")
        mf, mf_origin = runtime.mf_predict(self.model_dir / config["mf_checkpoint"],
                                         user_ids, movie_ids, config, self.device)
        final = runtime.blend(predictions, mf, config["blend_weight"])
        del predictions, tr, tm
        self.dataset_orig_test, mi, ui = self._aif_test(inputs)
        result = {}
        native = {}
        for name, scores in (("mf_reference", mf), ("dee_mc", final)):
            native_metrics, details = runtime.metrics(scores, inputs, config["dee_threshold"])
            pred = prediction_dataset(self.dataset_orig_test, np.clip(scores[mi, ui], 1, 5), self.threshold)
            result[name] = self.compute_metrics(pred)
            native[name] = {"metrics": native_metrics, "dee_groups": details}
            if name == "mf_reference":
                self.dataset_orig_pred = pred
            else:
                self.dataset_transf_pred = pred
        fair = native["dee_mc"]["metrics"]
        criteria = self.config["criteria"]
        self.report = {
            "algorithm": "DEE-MC", "evaluated_utc": datetime.now(timezone.utc).isoformat(),
            "comparison": {"baseline": "MF reference (no fairness regularization)",
                           "final": "DEE-regularized GLocal-K SWA ensemble + 15% MF",
                           "matched_dee_ablation": False,
                           "interpretation": "Different backbones; differences are not attributable solely to DEE."},
            "formula": "0.85 * mean(GLocal-K_1, GLocal-K_2) + 0.15 * MF; original float64 accumulation/float32 boundaries; final clip [1,5]",
            "config": self.config,
            "native": native, "maf": result,
            "native_criteria": criteria,
            "native_criteria_met": bool(fair["test_rmse"] < criteria["test_rmse_lt"] and fair["dee"] < criteria["dee_lt"]),
            "evaluation": {"train_ratings": int(train_m.sum()), "test_ratings": len(mi),
                           "users": len(user_ids), "movies": len(movie_ids),
                           "maf_population": "Observed held-out ratings only; same rows for both models",
                           "maf_label_threshold": self.threshold,
                           "native_dee_population": "All users x sensitive-genre movies, including unobserved entries",
                           "native_dee_threshold": config["dee_threshold"],
                           "protected_encoding": {"M": 1, "F": 0},
                           "rating_features_used_by_maf": False,
                           "undefined_maf_metrics": {k: [n for n, v in d.items() if not np.isfinite(v)] for k, d in result.items()}},
            "checks": {"original_file_hashes_verified": len(checksums), "checkpoint_loads": loads,
                       "original_split_sha256_verified": True, "training_calls": 0,
                       "core_source_sha256": checksums["util/GLocal_K_torch.py"]},
            "mf_origin": mf_origin,
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                            "device": str(self.device), "threads": threads,
                            "packages": {p: version(p) for p in ("torch", "numpy", "pandas", "scipy", "scikit-learn", "aif360")}},
            "elapsed_seconds": perf_counter() - started,
        }
        return result["mf_reference"], result["dee_mc"]

    def write_report(self, output_dir):
        if self.report is None:
            raise RuntimeError("run() must finish before saving a report")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=False)
        (output / "report.json").write_text(json.dumps(json_safe(self.report), indent=2, allow_nan=False) + "\n")
        with (output / "metrics.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["metric_family", "metric", "mf_reference", "dee_mc"])
            for key in self.report["maf"]["mf_reference"]:
                writer.writerow(["maf", key, self.report["maf"]["mf_reference"][key], self.report["maf"]["dee_mc"][key]])
            for key in self.report["native"]["mf_reference"]["metrics"]:
                writer.writerow(["native", key, self.report["native"]["mf_reference"]["metrics"][key],
                                 self.report["native"]["dee_mc"]["metrics"][key]])
        return output / "report.json"
