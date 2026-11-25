"""Run one RG-TTA evaluation from a config file (started by rg_tta.RG_TTA.run in its own process).

Adapted from the provider's worker.py: the experiment code is imported from this package and
CLIP checkpoints are loaded from clip's default cache, which clip verifies by SHA256.
"""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace


def run_task(request):
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    from MAF.algorithms.inprocessing.rg_tta_function import run_tta_experiment as experiment

    config = json.loads(Path(request).read_text())
    task = config['attribute']
    report_epoch = config.pop('report_epoch')
    experiment.get_args = lambda: SimpleNamespace(**config)
    original_eval = experiment.evaluate_bias
    baseline = {}

    def save_baseline(*args, **kwargs):
        value = original_eval(*args, **kwargs)
        baseline.update(value)
        return value

    experiment.evaluate_bias = save_baseline
    experiment.main()
    output = Path(config['output_dir']) / config['run_name']
    rows = json.loads((output / 'scores_summary.json').read_text())
    meta = json.loads((output / 'meta.json').read_text())
    row = next(x for x in rows if x['epoch'] == report_epoch)
    result = {
        'dataset': {'fairface': 'FairFace', 'utkface': 'UTKFace'}[config['dataset']],
        'attribute': task,
        'num_images': meta['dataset_meta']['num_images'],
        'num_queries': len(meta['prompts']),
        'adaptation_steps': config['tta_steps'],
        'report_epoch': report_epoch,
        'maxskew': row['avg_maxskew'],
        'ndkl': row['avg_ndkl'],
        'baseline_maxskew': baseline['maxskew']['maxskew_eq_opp'],
        'baseline_ndkl': baseline['ndkl']['ndkl_eq_opp'],
    }
    (output / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    (output / 'epoch_metrics.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    run_task(sys.argv[1])
