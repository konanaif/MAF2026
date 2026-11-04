"""Toxicity, group-FPR and auxiliary metrics, with DEV-only threshold selection."""
import math
import numpy as np


def probabilities(margins):
    x = np.asarray(margins, dtype=np.float64)
    # Avoid overflow without merging distinct extreme scores through clipping.
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1 / (1 + np.exp(-x[positive]))
    exp = np.exp(x[~positive])
    out[~positive] = exp / (1 + exp)
    return out


def binary(y, pred):
    y, pred = np.asarray(y, bool), np.asarray(pred, bool)
    tn, fp, fn, tp = (int(np.sum(mask)) for mask in (~y & ~pred, ~y & pred, y & ~pred, y & pred))
    ratio = lambda a, b: a / b if b else 0.
    f1 = ratio(2 * tp, 2 * tp + fp + fn)
    return dict(n=len(y), tn=tn, fp=fp, fn=fn, tp=tp, accuracy=ratio(tn + tp, len(y)), toxic_f1=f1,
                macro_f1=(f1 + ratio(2 * tn, 2 * tn + fp + fn)) / 2,
                precision=ratio(tp, tp + fp), recall=ratio(tp, tp + fn), fpr=fp / (tn + fp) if tn + fp else None)


def evaluate(data, logits, threshold=None):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape != (len(data), 2) or not np.isfinite(logits).all():
        raise ValueError('Expected finite [n, 2] logits')
    margin = logits[:, 1] - logits[:, 0]
    pred = margin > 0 if threshold is None else probabilities(margin) >= threshold
    y = np.array([r['toxicity_label'] == 'toxic' for r in data])
    groups = {}
    for group in sorted({g for r in data for g in r['protected_attributes_mentioned']}):
        mask = np.array([group in r['protected_attributes_mentioned'] for r in data]) & ~y
        n, fp = int(mask.sum()), int(pred[mask].sum())
        groups[group] = dict(n_negative=n, fp=fp, fpr=fp / n if n else None)
    supported = [v['fpr'] for v in groups.values() if v['n_negative'] >= 20]
    unsupported = [v['fpr'] for v in groups.values() if 0 < v['n_negative'] < 20]
    categories = {}
    for category in sorted({g for r in data for g in r['bias_categories']}):
        mask = np.array([category in r['bias_categories'] for r in data])
        categories[category] = float(np.mean(y[mask] == pred[mask]))
    proxy = binary([r['broadcast_risk_decision'] == 'problematic' for r in data], pred)['toxic_f1']
    return dict(overall=binary(y, pred), groups=groups, max_supported_fpr=max(supported, default=None),
                max_unsupported_fpr=max(unsupported, default=None),
                auxiliary=dict(problematic_f1_toxicity_proxy=proxy, bias_category_group_accuracy=categories,
                               worst_bias_category_accuracy=min(categories.values(), default=None),
                               bias_category_accuracy_gap=max(categories.values()) - min(categories.values()) if categories else None))


def calibrate(dev, logits):
    margins = np.asarray(logits, float)[:, 1] - np.asarray(logits, float)[:, 0]
    candidates = sorted({0., .5, math.nextafter(1., math.inf), *probabilities(margins).tolist()})
    best = None
    for threshold in candidates:
        result = evaluate(dev, logits, threshold)
        o = result['overall']
        fprs = [result[k] for k in ('max_supported_fpr', 'max_unsupported_fpr') if result[k] is not None]
        key = (-o['toxic_f1'], -o['macro_f1'], -o['recall'], max(fprs, default=0), sum(fprs), abs(threshold - .5), threshold)
        if best is None or key < best[0]:
            best = (key, threshold, result)
    return dict(threshold=best[1], metrics=best[2], selection_split='dev', n=len(dev), test_consulted=False,
                rule='max Toxic-F1, Macro-F1, recall; min maxFPR,sumFPR,abs(theta-.5),theta')
