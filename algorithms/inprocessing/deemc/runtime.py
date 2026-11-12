"""Portable inference helpers; no training is performed in this module."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from . import _glocal as original



def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def require_finite(value, name):
    ok = bool(torch.isfinite(value).all()) if isinstance(value, torch.Tensor) else bool(np.isfinite(value).all())
    if not ok:
        raise ValueError(f"NaN/Inf in {name}")


def load_inputs(config, data_dir):
    """Reconstruct the original split from saved source-row indices, without RNG."""
    data_dir = Path(data_dir).resolve()
    manifest = read_json(data_dir / config["split_manifest"])
    for name, expected in manifest["data_sha256"].items():
        if sha256(data_dir / name) != expected:
            raise ValueError(f"Data checksum mismatch: {name}")
    rows = np.genfromtxt(data_dir / "movielens_1m_dataset.dat", delimiter="::", dtype=np.int32)
    split_path = data_dir / config["split_artifact"]
    if sha256(split_path) != manifest["artifact_sha256"]:
        raise ValueError("Saved split artifact checksum mismatch")
    with np.load(split_path, allow_pickle=False) as saved:
        user_ids = saved["user_ids"].copy()
        movie_ids = saved["movie_ids"].copy()
        test_rows = saved["test_row_indices"].copy()
    if not np.array_equal(user_ids, np.unique(rows[:, 0])) or not np.array_equal(movie_ids, np.unique(rows[:, 1])):
        raise ValueError("User/movie mapping mismatch")
    n_m, n_u = len(movie_ids), len(user_ids)
    mi = np.searchsorted(movie_ids, rows[:, 1])
    ui = np.searchsorted(user_ids, rows[:, 0])
    if np.unique(mi.astype(np.int64) * n_u + ui).size != len(rows):
        raise ValueError("Duplicate rating pairs require original shuffled assignment order")
    if len(test_rows) != int(config["test_fraction"] * len(rows)):
        raise ValueError("Wrong test split size")
    if np.unique(test_rows).size != len(test_rows) or test_rows.min() < 0 or test_rows.max() >= len(rows):
        raise ValueError("Invalid test source-row indices")
    is_test = np.zeros(len(rows), dtype=bool)
    is_test[test_rows] = True
    train_r = np.zeros((n_m, n_u), dtype=np.float32)
    test_r = np.zeros_like(train_r)
    train_r[mi[~is_test], ui[~is_test]] = rows[~is_test, 2]
    test_r[mi[is_test], ui[is_test]] = rows[is_test, 2]
    train_m = (train_r > 1e-12).astype(np.float32)
    test_m = (test_r > 1e-12).astype(np.float32)
    for name, array in {"train_r": train_r, "test_r": test_r, "train_m": train_m, "test_m": test_m}.items():
        if array_hash(array) != manifest["matrix_sha256"][name]:
            raise ValueError(f"Original split matrix mismatch: {name}")
    groups = original.load_ml1m_dee_metadata(str(data_dir) + "/", user_ids, movie_ids)
    return train_r, train_m, test_r, test_m, user_ids, movie_ids, groups


# 체크포인트에 저장된 학습 환경 경로·장치. 결과와 무관하므로 config.json 에 두지 않는다
ENVIRONMENT_KEYS = {"data_path", "run_dir", "device"}


def lambda0_training_config(training, lambda_dee):
    """ablation 체크포인트 검사용: config.json training 에서 lambda_dee 만 바꾼 사본."""
    expected = json.loads(json.dumps(training))
    expected["lambda_dee"] = lambda_dee
    return expected


def check_training_config(saved, expected):
    """체크포인트의 학습 설정이 config.json 의 training 과 같은지(환경 키 제외) 확인."""
    saved = {k: v for k, v in saved.items() if k not in ENVIRONMENT_KEYS}
    if saved != expected:
        raise ValueError("Checkpoint config differs from original experiment config")


def load_glocal(path, training_config, n_m, n_u, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    check_training_config(checkpoint["config"], training_config)
    hp = training_config["hyperparameters"]
    model = original.GLocalK(n_m, n_u, hp)
    state = checkpoint["model_state_dict"]
    for name, tensor in state.items():
        require_finite(tensor, f"{path.name}:{name}")
    keys = model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model, {"path": path.name, "sha256": sha256(path),
                   "strict_load": True, "missing_keys": list(keys.missing_keys),
                   "unexpected_keys": list(keys.unexpected_keys),
                   "state_tensors": len(state), "all_weights_finite": True}


@torch.inference_mode()
def glocal_predict(model, train_r, train_m):
    # Exact original fine-tuning prediction path. DEE affects training only.
    pred = model.finetune_prediction_and_loss(train_r, train_r, train_m)[0]
    require_finite(pred, "GLocal-K prediction")
    return pred.cpu().numpy().astype(np.float32)


@torch.inference_mode()
def mf_predict(path, user_ids, movie_ids, config, device):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["training_config"] != config["mf_training"]:
        raise ValueError("MF training config mismatch")
    if not np.array_equal(saved["user_ids"].numpy(), user_ids) or not np.array_equal(saved["movie_ids"].numpy(), movie_ids):
        raise ValueError("MF index mapping mismatch")
    arrays = {}
    shapes = {"item_factors": (len(movie_ids), config["mf_training"]["k"]),
              "user_factors": (len(user_ids), config["mf_training"]["k"]),
              "item_bias": (len(movie_ids),), "user_bias": (len(user_ids),)}
    for name, shape in shapes.items():
        tensor = saved[name]
        if tuple(tensor.shape) != shape:
            raise ValueError(f"Wrong MF tensor shape: {name}")
        require_finite(tensor, f"MF:{name}")
        arrays[name] = tensor.to(device)
    require_finite(saved["global_mean"], "MF global mean")
    full = (saved["global_mean"] + arrays["item_bias"][:, None] + arrays["user_bias"][None, :]
            + arrays["item_factors"] @ arrays["user_factors"].t())
    require_finite(full, "MF prediction")
    return full.cpu().numpy().astype(np.float32), saved["origin"]


def blend(predictions, mf, weight):
    # Preserve the original float64 accumulation and float32 cast boundaries.
    ensemble = ((predictions[0].astype(np.float64) + predictions[1].astype(np.float64)) / 2).astype(np.float32)
    result = ((1 - weight) * ensemble.astype(np.float64) + weight * mf.astype(np.float64)).astype(np.float32)
    require_finite(result, "blended prediction")
    return result


def metrics(prediction, inputs, threshold=3.0):
    train_r, train_m, test_r, test_m, _, _, (user_groups, item_groups) = inputs
    original.set_rating_bounds("ML-1M")
    result = original.evaluate_predictions(prediction, train_r, train_m, test_r, test_m, map_threshold=4.0)
    dee, details = original.dee_metric(original.clip_predictions(prediction), user_groups, item_groups, threshold=threshold)
    result["dee"] = float(dee)
    return result, {str(k): v for k, v in details.items()}
