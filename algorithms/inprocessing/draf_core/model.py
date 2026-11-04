"""DRAF's original MLP, adversarial regression penalty, and training loop.

The classification and adversary updates follow upstream fair_losses.py and
methods/__init__.py at upstream 79a9eea.
"""
import warnings
from numbers import Integral

import numpy as np
import torch
from torch import nn

from .subgroups import SubgroupFamily, validate_sensitive


def regression_penalty(yv, gz, eps=1e-6):
    """Upstream log(1 + |Cov(Cv, g(f)) / Var(Cv)|), with a zero-variance guard."""
    yc, gc = yv - yv.mean(), gz - gz.mean()
    slope = (yc * gc).mean() / yc.square().mean().clamp_min(eps)
    return torch.log1p(slope.abs())


class Predictor(nn.Module):
    def __init__(self, n_features, base_model):
        super().__init__()
        self.net = (nn.Sequential(nn.Linear(n_features, 100), nn.ReLU(), nn.Linear(100, 1))
                    if base_model == "mlp" else nn.Linear(n_features, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DRAdversary:
    def __init__(self, masks, lr, adv_steps):
        self.masks = masks
        self.v = nn.Parameter(torch.randn(masks.shape[1], device=masks.device))
        with torch.no_grad():
            self.v.div_(self.v.norm() + 1e-12)
        self.g = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid()).to(masks.device)
        self.opt_g = torch.optim.Adam(self.g.parameters(), lr=1.5 * lr)
        self.opt_v = torch.optim.Adam([self.v], lr=4.0 * lr)
        self.adv_steps = adv_steps

    def penalty(self, logits, detached_direction=True):
        direction = self.v / (self.v.norm() + 1e-12)
        yv = self.masks @ direction
        if detached_direction:
            yv = yv.detach()
        gz = self.g(logits.sigmoid().unsqueeze(1)).squeeze(-1)
        return regression_penalty(yv, gz)

    def update(self, model, features):
        with torch.no_grad():
            logits = model(features)
        for _ in range(self.adv_steps):
            loss = -self.penalty(logits, detached_direction=False)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite DRAF adversarial loss.")
            self.opt_g.zero_grad()
            self.opt_v.zero_grad()
            loss.backward()
            self.opt_g.step()
            self.opt_v.step()


class DRAFClassifier:
    def __init__(self, *, lambda_fair, epochs, lr, seed,
                 n_low_frac, af_max_order, max_groups,
                 base_model, device, adv_steps,
                 calibrate, verbose, threshold_metric, warmup_epochs=10):
        for name, value, minimum in (("epochs", epochs, 1), ("adv_steps", adv_steps, 1),
                                     ("warmup_epochs", warmup_epochs, 0), ("seed", seed, 0)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if lr <= 0 or not np.isfinite(lr):
            raise ValueError("lr must be finite and positive.")
        if lambda_fair < 0 or not np.isfinite(lambda_fair):
            raise ValueError("lambda_fair must be finite and nonnegative.")
        if base_model not in {"mlp", "linear"}:
            raise ValueError("base_model must be mlp or linear.")
        if threshold_metric not in {"balanced_accuracy", "accuracy"}:
            raise ValueError("threshold_metric must be balanced_accuracy or accuracy.")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu, cuda, or auto.")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested, but no CUDA device is available.")
        self.lambda_fair, self.epochs, self.lr = float(lambda_fair), int(epochs), float(lr)
        self.seed, self.device = int(seed), device
        self.base_model, self.adv_steps = base_model, int(adv_steps)
        self.warmup_epochs, self.calibrate, self.verbose = int(warmup_epochs), calibrate, verbose
        self.threshold_metric = threshold_metric
        self.family_ = SubgroupFamily(n_low_frac, af_max_order, max_groups)

    @staticmethod
    def _xy(x, y):
        x, y = np.asarray(x, dtype=np.float32), np.asarray(y).ravel()
        if x.ndim != 2 or len(x) != len(y) or not len(x) or not x.shape[1]:
            raise ValueError("X and y must have compatible, nonempty shapes.")
        if not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
            raise ValueError("Features must be finite and labels binary (0/1).")
        return x, y.astype(np.float32)

    def fit(self, X, y, sensitive, X_val, y_val):
        X, y = self._xy(X, y)
        X_val, y_val = self._xy(X_val, y_val)
        sensitive = validate_sensitive(sensitive)
        if len(sensitive) != len(X) or X_val.shape[1] != X.shape[1]:
            raise ValueError("Training, validation, and sensitive shapes do not match.")
        if np.unique(y).size != 2 or np.unique(y_val).size != 2:
            raise ValueError("Both labels must occur in training and validation sets.")
        self.n_features_in_ = X.shape[1]
        masks = self.family_.fit_transform(sensitive)
        self.n_groups_ = masks.shape[1]
        if self.lambda_fair > 0 and not self.n_groups_:
            raise ValueError("No eligible subgroups: lower n_low_frac or supply varying sensitive attributes.")
        if self.family_.truncated_:
            warnings.warn("Subgroup family reached max_groups; increase it for fuller coverage.", RuntimeWarning)
        # Keep the library call from reseeding the caller's random generators.
        devices = list(range(torch.cuda.device_count())) if self.device == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            if self.device == "cpu":
                # torch.manual_seed also reseeds CUDA even during a CPU fit.
                torch.default_generator.manual_seed(self.seed)
            else:
                torch.manual_seed(self.seed)
            self._fit(X, y, masks, X_val, y_val)
        return self

    def _fit(self, X, y, masks, X_val, y_val):
        as_tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=self.device)
        x, target, xv, yv = map(as_tensor, (X, y, X_val, y_val))
        self.model_ = Predictor(X.shape[1], self.base_model).to(self.device)
        adversary = DRAdversary(as_tensor(masks), self.lr, self.adv_steps) if self.lambda_fair > 0 else None
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=1e-6)
        bce = nn.BCEWithLogitsLoss()
        self.history_, best_state, best_score = [], None, -np.inf
        selection_start = min(self.epochs - 1, int(0.75 * self.epochs) + 1)
        decay_start = int(0.85 * self.epochs)
        for epoch in range(self.epochs):
            self.model_.train()
            logits = self.model_(x)
            cls_loss = bce(logits, target)
            weight = self.lambda_fair
            if epoch < self.warmup_epochs:
                weight *= epoch / max(1, self.warmup_epochs)
            elif epoch >= decay_start:
                weight *= 1 - (epoch - decay_start) / max(1, self.epochs - decay_start)
            if adversary:
                for parameter in adversary.g.parameters():
                    parameter.requires_grad_(False)
                penalty = adversary.penalty(logits)
            else:
                penalty = logits.new_zeros(())
            total = cls_loss + weight * penalty
            if not torch.isfinite(total):
                raise FloatingPointError(f"Non-finite DRAF training loss at epoch {epoch + 1}.")
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            if adversary:
                for parameter in adversary.g.parameters():
                    parameter.requires_grad_(True)
                adversary.update(self.model_, x)
            self.model_.eval()
            with torch.no_grad():
                val_logits = self.model_(xv)
                val_accuracy = ((val_logits.sigmoid() >= 0.5) == yv).float().mean().item()
            if epoch >= selection_start and val_accuracy > best_score + 1e-8:
                best_score, self.best_epoch_ = val_accuracy, epoch + 1
                best_state = {key: value.detach().cpu().clone() for key, value in self.model_.state_dict().items()}
            self.history_.append(dict(epoch=epoch + 1, bce=float(cls_loss.detach().cpu()),
                                      penalty=float(penalty.detach().cpu()), lambda_effective=weight,
                                      total=float(total.detach().cpu()), val_accuracy=val_accuracy))
            if self.verbose and (epoch == 0 or (epoch + 1) % 25 == 0 or epoch == self.epochs - 1):
                print(f"epoch={epoch + 1}/{self.epochs} bce={cls_loss.item():.5f} "
                      f"penalty={penalty.item():.5f} val_accuracy={val_accuracy:.4f}", flush=True)
        self.model_.load_state_dict(best_state)
        self.best_val_accuracy_ = best_score
        tail = self.history_[-min(20, len(self.history_)):]
        self.training_diagnostics_ = dict(
            epochs_completed=len(self.history_), selected_epoch=self.best_epoch_,
            selected_last_epoch=self.best_epoch_ == self.epochs,
            tail_window=len(tail),
            tail_val_accuracy_change=tail[-1]["val_accuracy"] - tail[0]["val_accuracy"],
            tail_training_bce_change=tail[-1]["bce"] - tail[0]["bce"],
            final_training_bce=tail[-1]["bce"])
        self.temperature_, self.threshold_ = 1.0, 0.5
        if self.calibrate:
            self._calibrate(xv, yv)

    def _calibrate(self, X_val, y_val):
        with torch.no_grad():
            logits = self.model_(X_val)
        log_temperature = torch.zeros((), device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([log_temperature], lr=0.05)
        for _ in range(200):
            loss = nn.functional.binary_cross_entropy_with_logits(logits / (log_temperature.exp() + 1e-6), y_val)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.temperature_ = float(np.clip(log_temperature.detach().exp().cpu().item(), 0.25, 4.0))
        with torch.no_grad():
            proba = (logits / self.temperature_).sigmoid().cpu().numpy()
        target = y_val.cpu().numpy()
        self.threshold_ = select_threshold(proba, target, self.threshold_metric)

    def predict_proba(self, X):
        if not hasattr(self, "model_"):
            raise RuntimeError("Fit the classifier before prediction.")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_ or not np.isfinite(X).all():
            raise ValueError("Prediction features must be finite and match training columns.")
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.as_tensor(X, device=self.device))
            proba = (logits / self.temperature_).sigmoid().cpu().numpy()
        return np.column_stack([1 - proba, proba])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold_).astype(int)


def select_threshold(proba, target, metric):
    """Validation only; break BA ties near 0.5, report collapse via the adapter.

    accuracy is the explicit upstream reproduction policy (first maximum).
    """
    proba, target = np.asarray(proba).ravel(), np.asarray(target).ravel()
    if (len(proba) != len(target) or not np.isfinite(proba).all()
            or not np.isin(target, [0, 1]).all() or np.unique(target).size != 2):
        raise ValueError("Threshold selection needs finite scores and both binary labels.")
    if metric == "accuracy":
        selected, best = 0.5, -1.0
        for threshold in np.linspace(0.05, 0.95, 37):
            accuracy = float(((proba >= threshold) == target).mean())
            if accuracy > best:
                selected, best = float(threshold), accuracy
        return selected
    if metric != "balanced_accuracy":
        raise ValueError("Unknown threshold metric.")
    candidates = []
    for threshold in np.linspace(0.05, 0.95, 37):
        labels = proba >= threshold
        score = 0.5 * (labels[target == 1].mean() + (~labels[target == 0]).mean())
        candidates.append((float(score), -abs(float(threshold) - 0.5), -float(threshold)))
    return -max(candidates)[2]
