"""DEE-MC 진입점: 저장된 체크포인트로 MF 기준선과 DEE-MC를 평가한다(재학습 없음).

실행: python -m MAF.algorithms.inprocessing.deemc.DEEMC [--device cpu] [--output DIR]
data/deemc 에 MovieLens 파일이 없으면 GroupLens 에서 내려받는다. 네트워크가 없는 환경에서는
movielens_1m_dataset.dat, users.dat, movies.dat 를 data/deemc 에 직접 넣고 실행한다.
"""
# ruff: noqa: N999  모듈명 DEEMC는 사용자 지정(제출분 진입점 이름 유지)

import argparse
import json
from pathlib import Path

from MAF.algorithms.inprocessing.deemc.runner import MAF_ROOT, DEEMCRunner, json_safe
from MAF.datamodule.deemc.download_data import ensure_data

DEFAULT_OUTPUT = MAF_ROOT / "result" / "deemc" / "run"


class DEEMC(DEEMCRunner):
    """Equal-Experience Regularization for Matrix Completion Recommendation."""

    def run(self):
        ensure_data(self.data_dir)
        return super().run()

    def summary(self):
        """run() 이후 native 지표와 자체 기준 충족 여부(JSON 직렬화 가능)."""
        if self.report is None:
            raise RuntimeError("run() must finish before summary()")
        return json_safe(
            {
                "native": self.report["native"],
                "native_criteria_met": self.report["native_criteria_met"],
            }
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda:N")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output directory already exists; choose a new directory")
    algorithm = DEEMC(config_path=args.config, device=args.device)
    _, fair = algorithm.run()
    report_path = algorithm.write_report(args.output)
    summary = {
        "algorithm": "DEE-MC",
        "report": report_path,
        "native": algorithm.report["native"],
        "maf_metrics_per_model": len(fair),
        "native_criteria_met": algorithm.report["native_criteria_met"],
    }
    print(json.dumps(json_safe(summary), indent=2))
    return 0 if algorithm.report["native_criteria_met"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
