import json
import os
from pathlib import Path
import subprocess
import sys
import time

CONFIG_DIR = Path(__file__).resolve().parent / "rg_tta_function" / "configs"


class RG_TTA:
    """RG-TTA (Selective Test-Time Debiasing for CLIP via Reward Gating).

    dataset_name : 'utkface' (MAF/data/utkface)
    protected    : 'gender' or 'race'
    seed, device : run settings; other keyword options override configs/<protected>.json

    run() returns (metrics_orig, metrics_transf): MaxSkew@1000 and NDKL@1000 of the
    original CLIP and of RG-TTA (lower is fairer).
    """

    def __init__(self, dataset_name="utkface", protected="gender", *,
                 seed=1, device="cuda", output_dir=None, **kwargs):
        if (dataset_name, protected) not in {("utkface", "gender"), ("utkface", "race")}:
            raise ValueError(f"Unsupported dataset/protected pair: {dataset_name}/{protected}")
        self.dataset_name = dataset_name
        self.protected = protected
        self.data_dir = Path(os.environ["PYTHONPATH"]) / "MAF" / "data" / dataset_name
        self.config = json.loads((CONFIG_DIR / f"{protected}.json").read_text())
        allowed = {"tta_steps", "report_epoch", "debias_lambda", "sample_k",
                   "soft_alignment_gamma", "momentum", "momentum_update_freq",
                   "debias_apply_threshold", "lr"}
        if set(kwargs) - allowed:
            raise ValueError(f"Unsupported options: {sorted(set(kwargs) - allowed)}")
        self.config.update(kwargs, seed=seed, device=device)
        if not 1 <= self.config["report_epoch"] <= self.config["tta_steps"]:
            raise ValueError("report_epoch must be between 1 and tta_steps")
        self.output_dir = Path(output_dir).resolve() if output_dir is not None else None
        self.load_and_preprocess_data()

    def load_and_preprocess_data(self):
        self.config["prompt_csv"] = str(CONFIG_DIR / "prompt_templates.csv")
        self.config["cache_data_dir"] = str(self.data_dir / "cache" / self.protected)
        for split, model in [("test", "ViT-B-16"), ("test", "ViT-L-14"), ("train", "ViT-B-16")]:
            path = self.data_dir / "cache" / self.protected / self.dataset_name / split / f"{model}.pt"
            if not path.is_file():
                raise FileNotFoundError(path)

    @staticmethod
    def compute_metrics(result):
        return {"maxskew": float(result["maxskew"]), "ndkl": float(result["ndkl"])}

    def run(self):
        # Each run writes its config, log and metrics to a new folder.
        if self.output_dir is None or self.output_dir.exists():
            name = f"{self.protected}_seed{self.config['seed']}_{time.strftime('%Y%m%d_%H%M%S')}"
            self.output_dir = self.data_dir / "results" / name
        self.output_dir.mkdir(parents=True, exist_ok=False)
        config = dict(self.config, output_dir=str(self.output_dir), run_name="evaluation")
        request = self.output_dir / "config.json"
        request.write_text(json.dumps(config, indent=2) + "\n")
        # A separate process keeps RG-TTA's torch settings out of the caller's process.
        with (self.output_dir / "execution.log").open("w") as log:
            subprocess.run([sys.executable, "-m", "MAF.algorithms.inprocessing.rg_tta_function.worker",
                            str(request)], cwd=self.data_dir, stdout=log, stderr=subprocess.STDOUT,
                           check=True)
        result = json.loads((self.output_dir / "evaluation" / "metrics.json").read_text())
        metrics_orig = self.compute_metrics({"maxskew": result["baseline_maxskew"],
                                             "ndkl": result["baseline_ndkl"]})
        metrics_transf = self.compute_metrics(result)
        (self.output_dir / "metrics.json").write_text(json.dumps(
            {"original": metrics_orig, "transformed": metrics_transf}, indent=2) + "\n")
        return metrics_orig, metrics_transf


if __name__ == "__main__":
    for protected in ("gender", "race"):
        metrics_orig, metrics_transf = RG_TTA(dataset_name="utkface", protected=protected).run()
        print(f"[{protected}] Metrics for original CLIP:")
        print(metrics_orig)
        print(f"[{protected}] Metrics for RG-TTA:")
        print(metrics_transf)
