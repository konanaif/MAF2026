"""TRAIN-only conditional LEACE with an unchanged original readout."""
import numpy as np
import torch
from concept_erasure import LeaceEraser


def features(hidden, logits, transform):
    hidden, logits = np.asarray(hidden, np.float64), np.asarray(logits, np.float64)
    pca = (hidden - transform['center']) @ transform['components'].T / transform['scale']
    margin = logits[:, 1] - logits[:, 0]
    return np.column_stack([pca, (margin - transform['margin_mean']) / transform['margin_scale']])


def fit(train_hidden, train_logits, train, rank=128, strength=.8):
    if not 0 < strength <= 1:
        raise ValueError('LEACE strength must be in (0, 1]')
    hidden = np.asarray(train_hidden, np.float64)
    center = hidden.mean(0)
    _, singular, vt = np.linalg.svd(hidden - center, full_matrices=False)
    rank = min(rank, int(np.sum(singular > 1e-10)), len(hidden) - 1)
    if rank < 1:
        raise ValueError('TRAIN representations have no nonconstant directions')
    margin = np.asarray(train_logits, np.float64)[:, 1] - np.asarray(train_logits, np.float64)[:, 0]
    if margin.std() < 1e-10:
        raise ValueError('TRAIN logits have a constant margin; LEACE cannot fit this preset')
    transform = dict(center=center, components=vt[:rank], scale=singular[:rank] / np.sqrt(len(hidden) - 1), margin_mean=np.array(margin.mean()), margin_scale=np.array(margin.std()))
    names = sorted({g for row in train for g in row['protected_attributes_mentioned']})
    selected = np.array([r['toxicity_label'] == 'nontoxic' for r in train])
    if not names or selected.sum() < 2:
        raise ValueError('LEACE requires concept labels and at least two TRAIN negatives')
    z = np.array([[int(g in r['protected_attributes_mentioned']) for g in names] for r in train])
    x = features(hidden, train_logits, transform)
    eraser = LeaceEraser.fit(torch.as_tensor(x[selected], dtype=torch.float64), torch.as_tensor(z[selected], dtype=torch.float64), affine=True, constrain_cov_trace=False, shrinkage=False, svd_tol=1e-8)
    full_p = eraser.P.numpy()
    p = (1 - strength) * np.eye(x.shape[1]) + strength * full_p
    w = np.zeros(x.shape[1]); w[-1] = float(transform['margin_scale'])
    model = {**{'transform_' + k: v for k, v in transform.items()}, 'P': p, 'P_full': full_p, 'mean': eraser.bias.numpy(), 'w': w, 'b': transform['margin_mean']}
    return model, dict(implementation='official concept-erasure', group_names=names, fit_rows=int(selected.sum()), pca_rank=rank, strength=strength, fit_split='TRAIN only', classifier_retrained=False, caveat='strength<1 has no perfect linear-erasure guarantee; fitted only on negative TRAIN distribution')


def predict(hidden, logits, model):
    transform = {k[len('transform_'):]: v for k, v in model.items() if k.startswith('transform_')}
    x = features(hidden, logits, transform)
    margin = ((x - model['mean']) @ model['P'].T + model['mean']) @ model['w'] + model['b']
    return np.column_stack([np.zeros(len(x)), margin])
