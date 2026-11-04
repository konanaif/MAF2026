"""AIF360 adapters and the 18 classification metrics in the MAF 2024 guide.

These metrics describe binary preference on *observed held-out ratings* for
DEE-MC. The default favorable label is rating >= 3. They do not replace the
native matrix-completion RMSE or joint gender-by-genre DEE metrics, which use
a different evaluation population.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd
from aif360.datasets import BinaryLabelDataset, StandardDataset
from aif360.metrics import ClassificationMetric


def _vector(values: np.ndarray, name: str, size: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape not in {(size,), (size, 1)}:
        raise ValueError(f"{name} must have one entry per row ({size})")
    array = array.reshape(-1)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_rows(dataset: BinaryLabelDataset, name: str) -> int:
    if not isinstance(dataset, BinaryLabelDataset):
        raise TypeError(f"{name} must be an AIF360 BinaryLabelDataset")
    features = np.asarray(dataset.features)
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError(f"{name} must contain a nonempty feature matrix")
    size = features.shape[0]
    if not np.isfinite(features).all():
        raise ValueError(f"{name} features must contain only finite values")
    if np.asarray(dataset.labels).shape != (size, 1):
        raise ValueError(f"{name} labels must have shape ({size}, 1)")
    labels = _vector(dataset.labels, f"{name} labels", size)
    if not np.isin(labels, [dataset.unfavorable_label, dataset.favorable_label]).all():
        raise ValueError(f"{name} labels must use the declared binary values")
    names = list(dataset.instance_names)
    if len(names) != size or len(set(names)) != size:
        raise ValueError(f"{name} instance_names must uniquely identify every row")
    protected = np.asarray(dataset.protected_attributes)
    if protected.ndim != 2 or protected.shape[0] != size:
        raise ValueError(f"{name} protected_attributes must align with every row")
    if not np.isfinite(protected).all():
        raise ValueError(f"{name} protected_attributes must be finite")
    _vector(dataset.scores, f"{name} scores", size)
    weights = _vector(dataset.instance_weights, f"{name} instance_weights", size)
    if (weights < 0).any() or weights.sum() <= 0:
        raise ValueError(f"{name} instance_weights must be nonnegative with positive sum")
    return size


def to_aif360_dataset(
    features: np.ndarray,
    feature_names: list[str],
    labels: np.ndarray,
    genders: np.ndarray,
    instance_names: list[str],
    protected: str = "sex",
) -> StandardDataset:
    """Create aligned binary relevance data without dropping or reordering rows.

    ``labels`` must already be 0/1, normally ``observed_rating >= 3``.
    Observed and predicted ratings must not be supplied as features. Gender
    codes 0=M, 1=F become MAF protected values 1=M (privileged), 0=F
    (unprivileged); this orientation determines the signs of group differences.
    """
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("features must be a nonempty two-dimensional array")
    if not np.isfinite(matrix).all():
        raise ValueError("features must contain only finite values")
    size = matrix.shape[0]
    if not isinstance(protected, str) or not protected or protected == "liked":
        raise ValueError("protected must be a nonempty name distinct from liked")
    if len(feature_names) != matrix.shape[1]:
        raise ValueError("feature_names must match the number of feature columns")
    if any(not isinstance(name, str) or not name for name in feature_names):
        raise ValueError("feature_names must be nonempty strings")
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("feature_names must be unique")
    forbidden = {
        protected, "liked", "rating", "ratings", "query_rating",
        "target_rating", "predicted_rating",
    }
    if forbidden.intersection(feature_names):
        raise ValueError("feature_names must not include protected, liked, or query/predicted ratings")
    if len(instance_names) != size:
        raise ValueError("instance_names must identify every row")
    if any(not isinstance(name, str) or not name for name in instance_names):
        raise ValueError("instance_names must be nonempty strings")
    if len(set(instance_names)) != size:
        raise ValueError("instance_names must be unique")
    target = _vector(labels, "labels", size)
    gender = _vector(genders, "genders", size)
    if not np.isin(target, [0, 1]).all():
        raise ValueError("labels must be explicitly binarized as 0 or 1")
    if not np.isin(gender, [0, 1]).all():
        raise ValueError("genders must use codes 0=M or 1=F")

    frame = pd.DataFrame(matrix, columns=feature_names, index=instance_names)
    frame[protected] = 1.0 - gender
    frame["liked"] = target
    dataset = StandardDataset(
        df=frame,
        label_name="liked",
        favorable_classes=[1],
        protected_attribute_names=[protected],
        privileged_classes=[[1]],
        metadata={"protected_attribute_maps": [{1.0: "M", 0.0: "F"}]},
    )
    if (
        _validate_rows(dataset, "dataset") != size
        or list(dataset.instance_names) != list(instance_names)
    ):
        raise ValueError("AIF360 unexpectedly dropped or reordered input rows")
    if not np.array_equal(dataset.labels.ravel(), target):
        raise ValueError("AIF360 unexpectedly remapped the binary labels")
    if not np.array_equal(dataset.protected_attributes.ravel(), 1.0 - gender):
        raise ValueError("AIF360 unexpectedly remapped the protected values")
    return dataset


def prediction_dataset(
    truth: BinaryLabelDataset,
    ratings: np.ndarray,
    threshold: float = 3.0,
) -> BinaryLabelDataset:
    """Copy truth and set binary predictions from scores on the 1–5 scale.

    AIF360 ``scores`` are the linear transform ``(rating - 1) / 4``. They are
    *not calibrated relevance probabilities*. All 18 reported metrics use hard
    labels; the rating threshold is applied before that linear transform.
    """
    size = _validate_rows(truth, "truth")
    if truth.favorable_label != 1 or truth.unfavorable_label != 0:
        raise ValueError("truth must use favorable_label=1 and unfavorable_label=0")
    values = _vector(ratings, "ratings", size)
    if ((values < 1) | (values > 5)).any():
        raise ValueError("ratings must be on the 1–5 scale")
    if not np.isfinite(threshold) or not 1 <= threshold <= 5:
        raise ValueError("threshold must be finite and on the 1–5 scale")
    prediction = truth.copy(deepcopy=True)
    prediction.labels = (values >= threshold).astype(np.float64).reshape(-1, 1)
    prediction.scores = ((values - 1.0) / 4.0).reshape(-1, 1)
    return prediction


def compute_metrics(
    dataset_true: BinaryLabelDataset,
    dataset_pred: BinaryLabelDataset,
    unprivileged_groups: list[dict[str, float]],
    privileged_groups: list[dict[str, float]],
    disp: bool = False,
) -> OrderedDict[str, float]:
    """Return the guide's exact keys; represent undefined metrics with NaN.

    Finite values come from AIF360. Its zero-denominator precision/FDR/FOR
    fallbacks are explicitly changed from zero to NaN, and infinities become
    NaN. Callers must serialize these as null with an explanation rather than
    treating missing denominators as perfect or zero fairness.
    """
    size = _validate_rows(dataset_true, "dataset_true")
    if _validate_rows(dataset_pred, "dataset_pred") != size:
        raise ValueError("truth and prediction must have the same number of rows")
    if list(dataset_true.instance_names) != list(dataset_pred.instance_names):
        raise ValueError("truth and prediction instance_names must be in the same order")
    metric = ClassificationMetric(
        dataset_true,
        dataset_pred,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        values = OrderedDict(
            [
                ("true_negative_rate", metric.specificity()),
                ("true_positive_rate", metric.sensitivity()),
                ("false_positive_rate", metric.false_positive_rate()),
                ("false_negative_rate", metric.false_negative_rate()),
                ("precision", metric.precision()),
                ("false_discovery_rate", metric.false_discovery_rate()),
                ("false_omission_rate", metric.false_omission_rate()),
                ("accuracy", metric.accuracy()),
                (
                    "Balanced accuracy",
                    0.5 * (metric.true_positive_rate() + metric.true_negative_rate()),
                ),
                ("error_rate", metric.error_rate()),
                ("average_odds_difference", metric.average_odds_difference()),
                ("average_abs_odds_difference", metric.average_abs_odds_difference()),
                ("selection_rate", metric.selection_rate()),
                ("disparate_impact", metric.disparate_impact()),
                ("statistical_parity_difference", metric.statistical_parity_difference()),
                ("generalized_entropy_index", metric.generalized_entropy_index()),
                ("theil_index", metric.theil_index()),
                ("equal_opportunity_difference", metric.equal_opportunity_difference()),
            ]
        )
    counts = metric.binary_confusion_matrix()
    if counts["TP"] + counts["FP"] == 0:
        values["precision"] = values["false_discovery_rate"] = float("nan")
    if counts["TN"] + counts["FN"] == 0:
        values["false_omission_rate"] = float("nan")
    result = OrderedDict(
        (name, float(value) if np.isfinite(value) else float("nan"))
        for name, value in values.items()
    )
    if disp:
        for name, value in result.items():
            print(f"{name} = {value:.4f}")
    return result
