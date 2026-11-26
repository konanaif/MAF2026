"""MAF adapter for DRAF. Default comparison: identical MLP, lambda=0 vs 1."""
from pathlib import Path
import warnings
import numpy as np
from aif360.datasets import BinaryLabelDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from MAF.algorithms.preprocessing.optim_preproc_helpers.data_prepro_function import (
    load_preproc_data_adult, load_preproc_data_german,
)
from MAF.metric import common_utils
from MAF.utils.common import fix_seed
from MAF.algorithms.inprocessing.draf_core.model import DRAFClassifier, select_threshold
from MAF.algorithms.inprocessing.draf_core.subgroups import validate_sensitive
from MAF.algorithms.inprocessing.draf_core.adult_sensitive import extend_adult_sensitive


class DRAF:
    def __init__(self, dataset_name="adult", protected="sex", *, dataset=None,
                 sensitive_attributes=None, seed=1, epochs=200, lr=1e-3,
                 lambda_fair=1.0, n_low_frac=0.01, af_max_order=(1, 2),
                 max_groups=2048, base_model="mlp", baseline="mlp", device="cpu",
                 x_sensitive="concat", calibrate=True, adv_steps=3, verbose=False,
                 threshold_metric="balanced_accuracy", **kwargs):
        if dataset_name not in {"adult", "german"}:
            raise ValueError("Supported datasets: adult, german.")
        # Accept MAF's shared constructor kwargs; DRAF uses the explicit options below.
        self.options = dict(lambda_fair=lambda_fair, epochs=epochs, lr=lr, seed=seed,
                            n_low_frac=n_low_frac, af_max_order=af_max_order,
                            max_groups=max_groups, base_model=base_model, device=device,
                            calibrate=calibrate, adv_steps=adv_steps, verbose=verbose,
                            threshold_metric=threshold_metric)
        self.model_ = DRAFClassifier(**self.options)
        if baseline not in {"mlp", "logistic"}:
            raise ValueError("baseline must be mlp (same backbone) or logistic (additional comparison).")
        if x_sensitive not in {"concat", "drop"}:
            raise ValueError("x_sensitive must be concat or drop.")
        self.dataset_name, self.protected, self.seed = dataset_name, protected, int(seed)
        self.baseline, self.x_sensitive = baseline, x_sensitive
        fix_seed(self.seed)
        loaded_builtin = dataset is None
        if dataset is None:
            loaders = {"adult": load_preproc_data_adult, "german": load_preproc_data_german}
            # MAF preprocessing, with all its protected attributes retained for DRAF.
            dataset = loaders[dataset_name]()
            if dataset_name == "adult":
                dataset = extend_adult_sensitive(dataset, Path(__file__).resolve().parents[2] / "data/adult")
        if not isinstance(dataset, BinaryLabelDataset):
            raise TypeError("dataset must be an AIF360 BinaryLabelDataset.")
        self.dataset_orig = dataset.copy(deepcopy=True)
        self.label_policy_ = "dataset favorable_label metadata"
        self.data_warnings_ = []
        self.sensitive_attributes = list(dataset.protected_attribute_names if sensitive_attributes is None
                                         else sensitive_attributes)
        if (not self.sensitive_attributes or len(set(self.sensitive_attributes)) != len(self.sensitive_attributes)
                or set(self.sensitive_attributes) - set(dataset.protected_attribute_names)
                or protected not in self.sensitive_attributes):
            raise ValueError("Sensitive attributes must be distinct dataset attributes and include protected.")
        weights = np.asarray(dataset.instance_weights)
        if not np.isfinite(weights).all() or (weights <= 0).any() or not np.allclose(weights, weights[0]):
            raise ValueError("Upstream DRAF is unweighted; non-uniform instance_weights are unsupported.")
        # Exact split calls used by MAF2025 GerryFairClassifier. No stratification.
        self.dataset_orig_train, remainder = self.dataset_orig.split([0.7], shuffle=True, seed=self.seed)
        self.dataset_orig_valid, self.dataset_orig_test = remainder.split([0.5], shuffle=True, seed=self.seed)
        position = dataset.protected_attribute_names.index(protected)
        privileged = set(dataset.privileged_protected_attributes[position].tolist())
        unprivileged = set(dataset.unprivileged_protected_attributes[position].tolist())
        if (not privileged or not unprivileged or privileged & unprivileged
                or privileged | unprivileged != {0, 1}):
            raise ValueError("Protected metadata must partition binary values 0 and 1.")
        self.privileged_groups = [{protected: value} for value in sorted(privileged)]
        self.unprivileged_groups = [{protected: value} for value in sorted(unprivileged)]
        self.feature_indices_ = [i for i, name in enumerate(dataset.feature_names)
                                 if x_sensitive != "drop" or name not in self.sensitive_attributes]
        self.feature_names_ = [dataset.feature_names[i] for i in self.feature_indices_]
        if not self.feature_indices_:
            raise ValueError("No model features remain.")
        self.scaler_ = StandardScaler()
        self.X_train = self.scaler_.fit_transform(self.dataset_orig_train.features[:, self.feature_indices_]).astype(np.float32)
        self.X_valid = self._features(self.dataset_orig_valid)
        self.X_test = self._features(self.dataset_orig_test)
        self.y_train, self.y_valid, self.y_test = [self._labels(split) for split in self.splits]
        positions = [dataset.protected_attribute_names.index(name) for name in self.sensitive_attributes]
        self.S_train, self.S_valid, self.S_test = [
            validate_sensitive(split.protected_attributes[:, positions]) for split in self.splits]

    @property
    def splits(self):
        return self.dataset_orig_train, self.dataset_orig_valid, self.dataset_orig_test

    @staticmethod
    def _labels(dataset):
        return (dataset.labels.ravel() == dataset.favorable_label).astype(int)

    def _features(self, dataset):
        return self.scaler_.transform(dataset.features[:, self.feature_indices_]).astype(np.float32)

    @staticmethod
    def _prediction(dataset, scores, labels):
        result = dataset.copy(deepcopy=True)
        result.scores = np.asarray(scores).reshape(-1, 1)
        result.labels = np.where(labels, dataset.favorable_label, dataset.unfavorable_label).reshape(-1, 1)
        result.validate_dataset()
        return result

    def baseline_fit(self):
        if self.baseline == "logistic":
            self.baseline_model_ = LogisticRegression(max_iter=1000, random_state=self.seed)
            self.baseline_model_.fit(self.X_train, self.y_train)
            self.baseline_threshold_ = select_threshold(
                self.baseline_model_.predict_proba(self.X_valid)[:, 1], self.y_valid,
                self.options["threshold_metric"]) if self.options["calibrate"] else 0.5
        else:
            self.baseline_model_ = DRAFClassifier(**dict(self.options, lambda_fair=0.0)).fit(
                self.X_train, self.y_train, self.S_train, self.X_valid, self.y_valid)
            self.baseline_threshold_ = self.baseline_model_.threshold_
        scores = self.baseline_model_.predict_proba(self.X_test)[:, 1]
        self.baseline_prediction_ = self._prediction(self.dataset_orig_test, scores, scores >= self.baseline_threshold_)
        return self.baseline_prediction_

    def fit(self):
        self.model_.fit(self.X_train, self.y_train, self.S_train, self.X_valid, self.y_valid)
        return self

    def draf_fit(self):
        self.fit()
        self.draf_prediction_ = self.predict(self.dataset_orig_test)
        return self.draf_prediction_

    def predict(self, dataset):
        if not isinstance(dataset, BinaryLabelDataset):
            raise TypeError("dataset must be an AIF360 BinaryLabelDataset.")
        if (dataset.label_names != self.dataset_orig.label_names
                or dataset.favorable_label != self.dataset_orig.favorable_label
                or dataset.unfavorable_label != self.dataset_orig.unfavorable_label):
            raise ValueError("Prediction label definitions must match training.")
        if dataset.feature_names != self.dataset_orig.feature_names:
            raise ValueError("Feature names and order must match training.")
        scores = self.model_.predict_proba(self._features(dataset))[:, 1]
        return self._prediction(dataset, scores, scores >= self.model_.threshold_)

    def compute_metrics(self, prediction):
        return common_utils.compute_metrics(self.dataset_orig_test, prediction,
                                            self.unprivileged_groups, self.privileged_groups)

    def run(self):
        # Exactly two fits by default; the lambda=0 MLP is the displayed baseline.
        self.metrics_orig_ = self.metrics_transf_ = None
        before, after = self.baseline_fit(), self.draf_fit()
        self.diagnostics_ = {}
        self.warnings_ = list(self.data_warnings_)
        for name, prediction in (("baseline", before), ("draf", after)):
            labels, scores = self._labels(prediction), prediction.scores.ravel()
            self.diagnostics_[name] = dict(
                positive_rate=float(labels.mean()),
                roc_auc=float(roc_auc_score(self.y_test, scores)) if np.unique(self.y_test).size == 2 else None,
                degenerate_predictions=np.unique(labels).size < 2,
                confusion_matrix=confusion_matrix(self.y_test, labels, labels=[0, 1]).tolist())
            if np.unique(labels).size < 2:
                self.warnings_.append(dict(code="CONSTANT_PREDICTIONS", model=name,
                    message="Only one class was predicted. Zero fairness gaps do not establish fairness improvement."))
        original, transformed = self.compute_metrics(before), self.compute_metrics(after)
        self.metrics_orig_, self.metrics_transf_ = original, transformed
        for name, metrics in (("baseline", original), ("draf", transformed)):
            undefined = [key for key, value in metrics.items() if not np.isfinite(value)]
            if undefined:
                self.warnings_.append(dict(code="UNDEFINED_METRICS", model=name, metrics=undefined,
                                           message="Some MAF metrics are undefined for this evaluation."))
        self.evaluation_status_ = "warning" if self.warnings_ else "ok"
        for warning in self.warnings_:
            warnings.warn(warning["message"], RuntimeWarning, stacklevel=2)
        return original, transformed


if __name__ == "__main__":
    import torch
    torch.set_num_threads(4)
    original, transformed = DRAF().run()
    print("Metrics for original data:")
    print(original)
    print("Metrics for transformed data:")
    print(transformed)
