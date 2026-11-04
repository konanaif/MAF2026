import argparse
import csv
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ML1M_MALE_PREFERRED_GENRES = {"Action", "Crime", "Film-Noir", "War"}
ML1M_FEMALE_PREFERRED_GENRES = {"Children", "Fantasy", "Musical", "Romance"}
RATING_MIN = 1.0
RATING_MAX = 5.0


def load_data_100k(path="./", delimiter="\t"):
    train = np.loadtxt(path + "movielens_100k_u1.base", skiprows=0, delimiter=delimiter).astype("int32")
    test = np.loadtxt(path + "movielens_100k_u1.test", skiprows=0, delimiter=delimiter).astype("int32")
    total = np.concatenate((train, test), axis=0)

    n_u = np.unique(total[:, 0]).size
    n_m = np.unique(total[:, 1]).size
    n_train = train.shape[0]
    n_test = test.shape[0]

    train_r = np.zeros((n_m, n_u), dtype="float32")
    test_r = np.zeros((n_m, n_u), dtype="float32")

    for i in range(n_train):
        train_r[train[i, 1] - 1, train[i, 0] - 1] = train[i, 2]

    for i in range(n_test):
        test_r[test[i, 1] - 1, test[i, 0] - 1] = test[i, 2]

    train_m = np.greater(train_r, 1e-12).astype("float32")
    test_m = np.greater(test_r, 1e-12).astype("float32")

    logging.debug("data matrix loaded")
    logging.debug("num of users: %s", n_u)
    logging.debug("num of movies: %s", n_m)
    logging.debug("num of training ratings: %s", n_train)
    logging.debug("num of test ratings: %s", n_test)

    return n_m, n_u, train_r, train_m, test_r, test_m


def load_data_1m(path="./", delimiter="::", frac=0.1, seed=1234):
    tic = time()
    logging.debug("reading data...")
    data = np.genfromtxt(path + "movielens_1m_dataset.dat", delimiter=delimiter, dtype=np.int32)
    logging.debug("taken %.3f seconds", time() - tic)

    n_u = np.unique(data[:, 0]).size
    n_m = np.unique(data[:, 1]).size
    n_r = data.shape[0]

    user_ids = np.unique(data[:, 0]).astype("int32")
    movie_ids = np.unique(data[:, 1]).astype("int32")

    udict = {u: i for i, u in enumerate(user_ids.tolist())}
    mdict = {m: i for i, m in enumerate(movie_ids.tolist())}

    np.random.seed(seed)
    idx = np.arange(n_r)
    np.random.shuffle(idx)

    train_r = np.zeros((n_m, n_u), dtype="float32")
    test_r = np.zeros((n_m, n_u), dtype="float32")

    for i in range(n_r):
        u_id = data[idx[i], 0]
        m_id = data[idx[i], 1]
        r = data[idx[i], 2]

        if i < int(frac * n_r):
            test_r[mdict[m_id], udict[u_id]] = r
        else:
            train_r[mdict[m_id], udict[u_id]] = r

    train_m = np.greater(train_r, 1e-12).astype("float32")
    test_m = np.greater(test_r, 1e-12).astype("float32")

    logging.debug("data matrix loaded")
    logging.debug("num of users: %s", n_u)
    logging.debug("num of movies: %s", n_m)
    logging.debug("num of training ratings: %s", n_r - int(frac * n_r))
    logging.debug("num of test ratings: %s", int(frac * n_r))

    return n_m, n_u, train_r, train_m, test_r, test_m, user_ids, movie_ids


def load_data_lastfm(path="./", frac=0.1, seed=1234):
    """Load the preprocessed Last.FM 360K matrix (see preprocess_lastfm.py).

    Ratings are in {+1, -1} on observed (artist, user) entries; unobserved
    entries stay 0 and are masked out explicitly (the >0 trick used for the
    MovieLens loaders would wrongly drop the -1 ratings)."""
    data = np.load(path + "lastfm_processed.npz", allow_pickle=True)
    R = data["R"].astype("float32")
    obs = data["obs"]
    n_m, n_u = R.shape

    mi, ui = np.nonzero(obs)
    n_r = mi.size
    np.random.seed(seed)
    idx = np.arange(n_r)
    np.random.shuffle(idx)
    n_test = int(frac * n_r)
    test_sel, train_sel = idx[:n_test], idx[n_test:]

    train_r = np.zeros((n_m, n_u), dtype="float32")
    test_r = np.zeros((n_m, n_u), dtype="float32")
    train_m = np.zeros((n_m, n_u), dtype="float32")
    test_m = np.zeros((n_m, n_u), dtype="float32")
    train_r[mi[train_sel], ui[train_sel]] = R[mi[train_sel], ui[train_sel]]
    train_m[mi[train_sel], ui[train_sel]] = 1.0
    test_r[mi[test_sel], ui[test_sel]] = R[mi[test_sel], ui[test_sel]]
    test_m[mi[test_sel], ui[test_sel]] = 1.0

    logging.debug("LastFM matrix %d artists x %d users, %d ratings", n_m, n_u, n_r)
    return n_m, n_u, train_r, train_m, test_r, test_m, data["user_ids"], data["artist_names"]


def load_matlab_file(path_file, name_field):
    import h5py
    from scipy.sparse import csc_matrix

    db = h5py.File(path_file, "r")
    ds = db[name_field]

    try:
        if "ir" in ds.keys():
            data = np.asarray(ds["data"])
            ir = np.asarray(ds["ir"])
            jc = np.asarray(ds["jc"])
            out = csc_matrix((data, ir, jc)).astype(np.float32)
    except AttributeError:
        out = np.asarray(ds).astype(np.float32).T

    db.close()
    return out


def load_data_monti(path="./"):
    m = load_matlab_file(path + "douban_monti_dataset.mat", "M")
    otraining = load_matlab_file(path + "douban_monti_dataset.mat", "Otraining") * m
    otest = load_matlab_file(path + "douban_monti_dataset.mat", "Otest") * m

    n_u = m.shape[0]
    n_m = m.shape[1]
    n_train = otraining[np.where(otraining)].size
    n_test = otest[np.where(otest)].size

    train_r = otraining.T
    test_r = otest.T
    train_m = np.greater(train_r, 1e-12).astype("float32")
    test_m = np.greater(test_r, 1e-12).astype("float32")

    logging.debug("data matrix loaded")
    logging.debug("num of users: %s", n_u)
    logging.debug("num of movies: %s", n_m)
    logging.debug("num of training ratings: %s", n_train)
    logging.debug("num of test ratings: %s", n_test)

    return n_m, n_u, train_r, train_m, test_r, test_m


def trunc_normal_(tensor, mean=0.0, std=1.0):
    if hasattr(nn.init, "trunc_normal_"):
        nn.init.trunc_normal_(tensor, mean=mean, std=std)
    else:
        with torch.no_grad():
            tensor.normal_(mean, std)
            tensor.clamp_(mean - 2 * std, mean + 2 * std)
    return tensor


class LocalKernelLayer(nn.Module):
    def __init__(self, n_in, n_out, n_dim, lambda_s, lambda_2, activation="sigmoid"):
        super().__init__()
        self.w = nn.Parameter(torch.empty(n_in, n_out))
        self.u = nn.Parameter(torch.empty(n_in, 1, n_dim))
        self.v = nn.Parameter(torch.empty(1, n_out, n_dim))
        self.b = nn.Parameter(torch.empty(n_out))
        self.lambda_s = lambda_s
        self.lambda_2 = lambda_2
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.b.view(1, -1))
        trunc_normal_(self.u, mean=0.0, std=1e-3)
        trunc_normal_(self.v, mean=0.0, std=1e-3)

    def local_kernel(self):
        dist = torch.linalg.vector_norm(self.u - self.v, ord=2, dim=2)
        return torch.clamp(1.0 - dist.pow(2), min=0.0)

    def regularization(self, w_hat):
        sparse_reg = 0.5 * self.lambda_s * torch.sum(w_hat.pow(2))
        l2_reg = 0.5 * self.lambda_2 * torch.sum(self.w.pow(2))
        return sparse_reg + l2_reg

    def forward(self, x):
        w_hat = self.local_kernel()
        y = x.matmul(self.w * w_hat) + self.b
        if self.activation == "sigmoid":
            y = torch.sigmoid(y)
        elif self.activation == "identity":
            pass
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
        return y, self.regularization(w_hat)


class GLocalK(nn.Module):
    def __init__(self, n_m, n_u, hp):
        super().__init__()
        self.n_m = n_m
        self.n_u = n_u
        self.hp = hp

        dims = [n_u] + [hp["n_hid"]] * hp["n_layers"] + [n_u]
        layers = []
        for i in range(hp["n_layers"]):
            layers.append(
                LocalKernelLayer(
                    dims[i], dims[i + 1], hp["n_dim"], hp["lambda_s"], hp["lambda_2"], activation="sigmoid"
                )
            )
        self.layers = nn.ModuleList(layers)
        self.out = LocalKernelLayer(
            dims[-2], dims[-1], hp["n_dim"], hp["lambda_s"], hp["lambda_2"], activation="identity"
        )
        self.conv_kernel = nn.Parameter(torch.empty(n_m, hp["gk_size"] ** 2))

        self.pooling = hp.get("pooling", "average")
        if self.pooling == "attention":
            # Learns a score per item that, projected onto y_dash, yields one
            # attention logit per user. Softmax over users replaces the uniform
            # weights of average pooling in the global-kernel summary.
            self.attention_score = nn.Parameter(torch.empty(n_m))
        elif self.pooling != "average":
            raise ValueError(f"Unknown pooling: {self.pooling}")

        # Per-item / per-user mean removed from the autoencoder input and added
        # back at the output. Defaults to 0 (no-op, identical to the original).
        self.register_buffer("input_bias", torch.zeros(1))

        self.reset_global_kernel()

    def set_normalization(self, train_r, train_m, mode):
        if mode in (None, "none"):
            self.input_bias = torch.zeros(1, device=train_r.device)
            return
        if mode == "item":
            counts = torch.clamp(train_m.sum(dim=1, keepdim=True), min=1.0)
            bias = (train_r * train_m).sum(dim=1, keepdim=True) / counts  # (n_m, 1)
        elif mode == "user":
            counts = torch.clamp(train_m.sum(dim=0, keepdim=True), min=1.0)
            bias = (train_r * train_m).sum(dim=0, keepdim=True) / counts  # (1, n_u)
        else:
            raise ValueError(f"Unknown normalization mode: {mode}")
        self.input_bias = bias

    def _center(self, r, train_m):
        # Subtract the stored bias on observed entries, keep missing at 0.
        return (r - self.input_bias) * train_m

    def reset_global_kernel(self):
        trunc_normal_(self.conv_kernel, mean=0.0, std=0.1)
        if self.pooling == "attention":
            nn.init.normal_(self.attention_score, mean=0.0, std=1e-3)

    def local_autoencoder(self, x, include_reg=True):
        reg_losses = []
        y = x
        for layer in self.layers:
            y, reg_loss = layer(y)
            if include_reg:
                reg_losses.append(reg_loss)
        y, reg_loss = self.out(y)
        if include_reg:
            reg_losses.append(reg_loss)
            return y, torch.stack(reg_losses).sum()
        return y, y.new_tensor(0.0)

    def global_kernel(self, y_dash):
        if self.pooling == "attention":
            scores = y_dash.t().matmul(self.attention_score)  # (n_u,)
            weights = torch.softmax(scores, dim=0)
            pooling = y_dash.matmul(weights).reshape(1, -1)  # (1, n_m)
        else:
            pooling = torch.mean(y_dash, dim=1).reshape(1, -1)
        gk = pooling.matmul(self.conv_kernel) * self.hp["dot_scale"]
        return gk.reshape(1, 1, self.hp["gk_size"], self.hp["gk_size"])

    def global_conv(self, train_r, kernel):
        x = train_r.reshape(1, 1, train_r.shape[0], train_r.shape[1])
        y = F.relu(F.conv2d(x, kernel, stride=1, padding=self.hp["gk_size"] // 2))
        return y.reshape(train_r.shape[0], train_r.shape[1])

    def pretrain_prediction_and_loss(self, r, train_r, train_m):
        r_in = self._center(r, train_m)
        residual, reg_loss = self.local_autoencoder(r_in, include_reg=True)
        pred = residual + self.input_bias
        diff = train_m * (train_r - pred)
        sqe = 0.5 * torch.sum(diff.pow(2))
        return pred, sqe + reg_loss

    def finetune_prediction_and_loss(
        self,
        r,
        train_r,
        train_m,
        dee_masks=None,
        lambda_dee=0.9,
        dee_threshold=3.0,
        dee_bandwidth=0.01,
        dee_huber_delta=0.01,
    ):
        r_in = self._center(r, train_m)
        y_dash, _ = self.local_autoencoder(r_in, include_reg=False)
        gk = self.global_kernel(y_dash)
        y_hat = self.global_conv(r_in, gk)
        residual, reg_loss = self.local_autoencoder(y_hat, include_reg=True)
        pred = residual + self.input_bias
        diff = train_m * (train_r - pred)
        observed_count = torch.clamp(train_m.sum(), min=1.0)
        # Match the TF reference (tf.nn.l2_loss = 0.5 * sum of squares) so the
        # accuracy:reg ratio in fine-tuning equals pre-training. Dropping the 0.5
        # halved the effective regularization and hurt test RMSE.
        accuracy_loss = 0.5 * torch.sum(diff.pow(2)) / observed_count
        model_reg_loss = reg_loss / observed_count
        base_loss = accuracy_loss + model_reg_loss
        dee_penalty = dee_loss_torch(
            pred,
            dee_masks,
            threshold=dee_threshold,
            bandwidth=dee_bandwidth,
            huber_delta=dee_huber_delta,
        )
        total_loss = (1.0 - lambda_dee) * base_loss + lambda_dee * dee_penalty
        return pred, total_loss, base_loss, dee_penalty, accuracy_loss, model_reg_loss


def dcg_k(score_label, k):
    dcg, i = 0.0, 0
    for s in score_label:
        if i < k:
            dcg += (2**s[1] - 1) / np.log2(2 + i)
            i += 1
    return dcg


def ndcg_k(y_hat, y, k):
    score_label = np.stack([y_hat, y], axis=1).tolist()
    score_label = sorted(score_label, key=lambda d: d[0], reverse=True)
    score_label_ = sorted(score_label, key=lambda d: d[1], reverse=True)
    norm, i = 0.0, 0
    for s in score_label_:
        if i < k:
            norm += (2**s[1] - 1) / np.log2(2 + i)
            i += 1
    dcg = dcg_k(score_label, k)
    return dcg / norm


def call_ndcg(y_hat, y):
    ndcg_sum, num = 0, 0
    y_hat, y = y_hat.T, y.T
    n_users = y.shape[0]

    for i in range(n_users):
        y_hat_i = y_hat[i][np.where(y[i])]
        y_i = y[i][np.where(y[i])]

        if y_i.shape[0] < 2:
            continue

        ndcg_sum += ndcg_k(y_hat_i, y_i, y_i.shape[0])
        num += 1

    return ndcg_sum / num


def average_precision(y_hat_i, y_i, threshold):
    order = np.argsort(-y_hat_i, kind="stable")
    relevant = (y_i[order] >= threshold).astype(np.float64)
    n_rel = relevant.sum()
    if n_rel == 0:
        return None
    cum_hits = np.cumsum(relevant)
    ranks = np.arange(1, relevant.shape[0] + 1)
    precision_at_k = cum_hits / ranks
    return float((precision_at_k * relevant).sum() / n_rel)


def call_map(y_hat, y, threshold):
    ap_sum, num = 0.0, 0
    y_hat, y = y_hat.T, y.T
    n_users = y.shape[0]

    for i in range(n_users):
        y_hat_i = y_hat[i][np.where(y[i])]
        y_i = y[i][np.where(y[i])]

        if y_i.shape[0] < 2:
            continue

        ap = average_precision(y_hat_i, y_i, threshold)
        if ap is None:
            continue

        ap_sum += ap
        num += 1

    return ap_sum / num if num > 0 else float("nan")


def clip_predictions(pre):
    return np.clip(pre, RATING_MIN, RATING_MAX)


def set_rating_bounds(dataset):
    """Prediction clipping range. LastFM uses binary +/-1 ratings, MovieLens 1-5."""
    global RATING_MIN, RATING_MAX
    if dataset == "LastFM":
        RATING_MIN, RATING_MAX = -1.0, 1.0
    else:
        RATING_MIN, RATING_MAX = 1.0, 5.0


def load_ml1m_dee_metadata(path, user_ids, movie_ids):
    if user_ids is None or movie_ids is None:
        raise ValueError("DEE metadata is only configured for ML-1M.")

    user_gender = {}
    with open(path + "users.dat", "r", encoding="latin-1") as f:
        for line in f:
            user_id, gender, *_ = line.rstrip("\n").split("::")
            user_gender[int(user_id)] = gender

    movie_group = {}
    with open(path + "movies.dat", "r", encoding="latin-1") as f:
        for line in f:
            movie_id, _, genres = line.rstrip("\n").split("::")
            genres = set(genres.split("|"))
            if genres & ML1M_MALE_PREFERRED_GENRES:
                movie_group[int(movie_id)] = "male_preferred"
            elif genres & ML1M_FEMALE_PREFERRED_GENRES:
                movie_group[int(movie_id)] = "female_preferred"
            else:
                movie_group[int(movie_id)] = "other"

    user_groups = np.array([user_gender[int(u)] for u in user_ids])
    item_groups = np.array([movie_group[int(m)] for m in movie_ids])
    return user_groups, item_groups


def load_lastfm_dee_metadata(path, user_ids=None, movie_ids=None):
    """User groups = gender (m/f); item groups = male/female preferred by an
    empirical listening gender-skew proxy (built in preprocess_lastfm.py, since
    no Last.fm API key is available for the paper's genre tags). Column/row order
    already matches the matrix, so no id remapping is needed."""
    data = np.load(path + "lastfm_processed.npz", allow_pickle=True)
    user_groups = data["user_gender"].astype(str)
    item_groups = data["item_group"].astype(str)
    return user_groups, item_groups


def dee_metric(y_hat, user_groups, item_groups, threshold=3.0, item_group_names=("male_preferred", "female_preferred")):
    positive = y_hat >= threshold
    item_mask = np.isin(item_groups, item_group_names)
    if item_mask.sum() == 0:
        return np.nan, {}

    positive = positive[item_mask, :]
    filtered_item_groups = item_groups[item_mask]
    overall = positive.mean()
    details = {}
    dee = 0.0

    for ug in np.unique(user_groups):
        user_mask = user_groups == ug
        for ig in item_group_names:
            group_item_mask = filtered_item_groups == ig
            if user_mask.sum() == 0 or group_item_mask.sum() == 0:
                continue
            group_rate = positive[np.ix_(group_item_mask, user_mask)].mean()
            gap = abs(group_rate - overall)
            details[(ug, ig)] = {"positive_rate": float(group_rate), "gap": float(gap)}
            dee += gap

    return dee, details


def fairness_metrics(y_hat, user_groups, item_groups, test_r=None, test_m=None,
                     threshold=3.0, item_group_names=("male_preferred", "female_preferred")):
    """Fairness metrics from Cho, Choi & Suh, "Equal Experience in Recommender
    Systems" (arXiv:2210.05936).

    Binary-prediction metrics use Y_tilde = 1{y_hat >= threshold} over the
    sensitive-item region (male/female preferred items) and all users, matching
    ``dee_metric``:
      - DEE (Difference of Equal Experience, Eq. 13):
            sum_{z1,z2} |P(Y=1) - P(Y=1 | item=z2, user=z1)|
      - DER (Difference Equalized Recommendation, Eq. 8):
            sum_{z1,z2} |P(Y=1 | user=z1) - P(Y=1 | item=z2, user=z1)|
      - CVS (Calders-Verwer, Eq. 6): |P(Y=1|item=MP) - P(Y=1|item=FP)|
      - UGF (User-oriented Group Fairness, Eq. 7): |P(Y=1|user=M) - P(Y=1|user=F)|
    VAL (Value unfairness, Eq. 5) is error-based over the two user groups on the
    observed test entries (all items) and needs ``test_r``/``test_m``.
    Any metric that cannot be computed is returned as NaN.
    """
    metrics = {"dee": np.nan, "der": np.nan, "cvs": np.nan, "ugf": np.nan, "val": np.nan}
    positive = (y_hat >= threshold).astype(np.float64)
    item_mask = np.isin(item_groups, item_group_names)
    user_group_values = list(np.unique(user_groups))

    if item_mask.sum() > 0 and user_group_values:
        S = positive[item_mask, :]
        ig_of_row = item_groups[item_mask]
        overall = S.mean()

        dee = 0.0
        der = 0.0
        user_rates = []
        for ug in user_group_values:
            ucols = user_groups == ug
            if ucols.sum() == 0:
                continue
            user_rate = S[:, ucols].mean()  # P(Y=1 | user=ug)
            user_rates.append(user_rate)
            for ig in item_group_names:
                irows = ig_of_row == ig
                if irows.sum() == 0:
                    continue
                cell = S[np.ix_(irows, ucols)].mean()  # P(Y=1 | item=ig, user=ug)
                dee += abs(cell - overall)
                der += abs(cell - user_rate)
        metrics["dee"] = float(dee)
        metrics["der"] = float(der)

        item_rates = [S[ig_of_row == ig, :].mean() for ig in item_group_names
                      if (ig_of_row == ig).sum() > 0]
        if len(item_rates) == 2:
            metrics["cvs"] = float(abs(item_rates[0] - item_rates[1]))

        if len(user_rates) >= 2:
            ugf = 0.0
            for a in range(len(user_rates)):
                for b in range(a + 1, len(user_rates)):
                    ugf += abs(user_rates[a] - user_rates[b])
            metrics["ugf"] = float(ugf)

    if test_r is not None and test_m is not None and len(user_group_values) >= 2:
        g0 = user_groups == user_group_values[0]
        g1 = user_groups == user_group_values[1]
        err = (clip_predictions(y_hat) - test_r) * test_m  # signed error on observed entries
        cnt0 = test_m[:, g0].sum(axis=1)
        cnt1 = test_m[:, g1].sum(axis=1)
        both = (cnt0 > 0) & (cnt1 > 0)
        if both.sum() > 0:
            e0 = err[:, g0].sum(axis=1)[both] / cnt0[both]
            e1 = err[:, g1].sum(axis=1)[both] / cnt1[both]
            metrics["val"] = float(np.abs(e0 - e1).mean())

    return metrics


def build_dee_masks(user_groups, item_groups, device, item_group_names=("male_preferred", "female_preferred")):
    user_masks = []
    item_masks = []
    for ug in np.unique(user_groups):
        mask = torch.as_tensor(user_groups == ug, dtype=torch.bool, device=device)
        if torch.any(mask):
            user_masks.append((str(ug), mask))

    for ig in item_group_names:
        mask = torch.as_tensor(item_groups == ig, dtype=torch.bool, device=device)
        if torch.any(mask):
            item_masks.append((str(ig), mask))

    item_any = torch.as_tensor(np.isin(item_groups, item_group_names), dtype=torch.bool, device=device)
    return {"user_masks": user_masks, "item_masks": item_masks, "item_any": item_any}


def gaussian_survival(x):
    return 0.5 * torch.erfc(x / math.sqrt(2.0))


def huber_abs(x, delta):
    abs_x = torch.abs(x)
    quadratic = 0.5 * x.pow(2)
    linear = delta * (abs_x - 0.5 * delta)
    return torch.where(abs_x <= delta, quadratic, linear)


def dee_loss_torch(y_hat, dee_masks, threshold=3.0, bandwidth=0.01, huber_delta=0.01):
    if dee_masks is None or not dee_masks["user_masks"] or not dee_masks["item_masks"]:
        return y_hat.new_tensor(0.0)
    if bandwidth <= 0:
        raise ValueError("DEE KDE bandwidth must be positive.")
    if huber_delta <= 0:
        raise ValueError("DEE Huber delta must be positive.")

    kde_positive_prob = gaussian_survival((threshold - y_hat) / bandwidth)
    item_any = dee_masks["item_any"]
    if not torch.any(item_any):
        return y_hat.new_tensor(0.0)

    overall = kde_positive_prob[item_any, :].mean()
    dee = y_hat.new_tensor(0.0)
    for _, user_mask in dee_masks["user_masks"]:
        for _, item_mask in dee_masks["item_masks"]:
            group_rate = kde_positive_prob[item_mask, :][:, user_mask].mean()
            dee = dee + huber_abs(group_rate - overall, huber_delta)
    return dee


def dataset_hyperparameters(dataset):
    common = {
        "n_hid": 500,
        "n_dim": 5,
        "n_layers": 2,
        "gk_size": 3,
    }
    if dataset == "ML-100K":
        specific = {
            "lambda_2": 20.0,
            "lambda_s": 0.006,
            "iter_p": 5,
            "iter_f": 5,
            "epoch_p": 30,
            "epoch_f": 60,
            "dot_scale": 1,
        }
    elif dataset == "ML-1M":
        specific = {
            "lambda_2": 70.0,
            "lambda_s": 0.018,
            "iter_p": 50,
            "iter_f": 10,
            "epoch_p": 20,
            "epoch_f": 30,
            "dot_scale": 0.5,
        }
    elif dataset == "Douban":
        specific = {
            "lambda_2": 10.0,
            "lambda_s": 0.022,
            "iter_p": 5,
            "iter_f": 5,
            "epoch_p": 20,
            "epoch_f": 60,
            "dot_scale": 2,
        }
    elif dataset == "LastFM":
        # Starting point for Last.FM 360K (10000x5706, binary +/-1); tuned via CLI.
        specific = {
            "lambda_2": 20.0,
            "lambda_s": 0.006,
            "iter_p": 5,
            "iter_f": 5,
            "epoch_p": 20,
            "epoch_f": 30,
            "dot_scale": 1,
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return {**common, **specific}


def load_dataset(dataset, data_path, frac, seed):
    user_ids, movie_ids = None, None
    if dataset == "ML-100K":
        path = str(data_path / "MovieLens_100K") + "/"
        n_m, n_u, train_r, train_m, test_r, test_m = load_data_100k(path=path, delimiter="\t")
    elif dataset == "ML-1M":
        path = str(data_path / "MovieLens_1M") + "/"
        n_m, n_u, train_r, train_m, test_r, test_m, user_ids, movie_ids = load_data_1m(
            path=path, delimiter="::", frac=frac, seed=seed
        )
    elif dataset == "Douban":
        path = str(data_path / "Douban_monti") + "/"
        n_m, n_u, train_r, train_m, test_r, test_m = load_data_monti(path=path)
    elif dataset == "LastFM":
        path = str(data_path / "LastFM_360K") + "/"
        n_m, n_u, train_r, train_m, test_r, test_m, user_ids, movie_ids = load_data_lastfm(
            path=path, frac=frac, seed=seed
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return n_m, n_u, train_r, train_m, test_r, test_m, user_ids, movie_ids


def evaluate_predictions(pre, train_r, train_m, test_r, test_m, map_threshold=4.0):
    clipped_pre = clip_predictions(pre)
    test_rmse = np.sqrt((test_m * (clipped_pre - test_r) ** 2).sum() / test_m.sum())
    train_rmse = np.sqrt((train_m * (clipped_pre - train_r) ** 2).sum() / train_m.sum())
    test_mae = (test_m * np.abs(clipped_pre - test_r)).sum() / test_m.sum()
    train_mae = (train_m * np.abs(clipped_pre - train_r)).sum() / train_m.sum()
    test_ndcg = call_ndcg(clipped_pre, test_r)
    train_ndcg = call_ndcg(clipped_pre, train_r)
    test_map = call_map(clipped_pre, test_r, map_threshold)
    train_map = call_map(clipped_pre, train_r, map_threshold)

    return {
        "test_rmse": float(test_rmse),
        "train_rmse": float(train_rmse),
        "test_mae": float(test_mae),
        "train_mae": float(train_mae),
        "test_ndcg": float(test_ndcg),
        "train_ndcg": float(train_ndcg),
        "test_map": float(test_map),
        "train_map": float(train_map),
    }


def setup_logging(run_dir):
    log_path = run_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
    )
    return log_path


def format_metric(value):
    if value is None:
        return "nan"
    try:
        if np.isnan(value):
            return "nan"
    except TypeError:
        pass
    return f"{float(value):.6f}"


def log_metrics(phase, epoch, train_rmse, test_rmse, dee=None, test_map=None, train_map=None, model_index=None, num_models=1):
    prefix = f"model={model_index:03d}/{num_models:03d} " if model_index is not None else ""
    logging.info(
        "%s%s epoch=%03d | Test RMSE: %s | Train RMSE: %s | Test MAP: %s | Train MAP: %s | DEE: %s",
        prefix,
        phase,
        epoch,
        format_metric(test_rmse),
        format_metric(train_rmse),
        format_metric(test_map),
        format_metric(train_map),
        format_metric(dee),
    )


def write_metrics_csv(path, rows):
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_safe(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device


def make_lbfgs_optimizer(model, lr, max_iter, history_size, line_search):
    return torch.optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=max_iter,
        history_size=history_size,
        line_search_fn=line_search,
    )


def make_adamw_optimizer(model, lr):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)


def run_lbfgs_step(optimizer, loss_fn):
    last_loss = None

    def closure():
        nonlocal last_loss
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        last_loss = loss.detach()
        return loss

    optimizer.step(closure)
    return float(last_loss.detach().cpu().item())


def run_adamw_epoch(optimizer, loss_fn, n_iter):
    last_loss = None
    for _ in range(n_iter):
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        optimizer.step()
        last_loss = loss.detach()
    return float(last_loss.detach().cpu().item())


def set_torch_initial_seed(seed):
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_initial_seeds(torch_seed, num_models):
    if num_models == 1 and torch_seed is None:
        return [None]
    base_seed = 0 if torch_seed is None else torch_seed
    return [base_seed + i for i in range(num_models)]


def train_mf_prediction(train_r_np, train_m_np, device, k=32, reg=0.08, epochs=50, lr=0.02, seed=0):
    """Biased matrix factorization (full-batch Adam) over observed ratings.

    Its errors decorrelate from the kernel autoencoder, so a small blend with the
    GLocal-K ensemble lowers RMSE below the autoencoder's own ensemble floor while
    barely moving DEE. Returns a full (n_m, n_u) rating-scale prediction.
    """
    n_m, n_u = train_r_np.shape
    mi, ui = np.nonzero(train_m_np)
    r = torch.tensor(train_r_np[mi, ui], dtype=torch.float32, device=device)
    mi_t = torch.tensor(mi, device=device)
    ui_t = torch.tensor(ui, device=device)
    mu = float(r.mean())

    g = torch.Generator(device=device).manual_seed(seed)
    P = torch.nn.Parameter(torch.randn(n_m, k, generator=g, device=device) * 0.1)
    Q = torch.nn.Parameter(torch.randn(n_u, k, generator=g, device=device) * 0.1)
    bm = torch.nn.Parameter(torch.zeros(n_m, device=device))
    bu = torch.nn.Parameter(torch.zeros(n_u, device=device))
    opt = torch.optim.Adam([P, Q, bm, bu], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        pred = mu + bm[mi_t] + bu[ui_t] + (P[mi_t] * Q[ui_t]).sum(1)
        loss = ((pred - r) ** 2).mean() + reg * (
            P[mi_t].pow(2).sum(1).mean() + Q[ui_t].pow(2).sum(1).mean()
            + bm[mi_t].pow(2).mean() + bu[ui_t].pow(2).mean()
        )
        loss.backward()
        opt.step()
    with torch.no_grad():
        full = mu + bm[:, None] + bu[None, :] + P @ Q.t()
    return full.cpu().numpy().astype("float32")


def run(args):
    root = Path(__file__).resolve().parent
    data_path = Path(args.data_path) if args.data_path else root / "data"
    out_root = Path(args.output_dir) if args.output_dir else root / "results_torch"
    run_name = args.run_name or f"{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging(run_dir)

    if args.num_models < 1:
        raise ValueError("--num-models must be at least 1.")

    set_rating_bounds(args.dataset)
    # LastFM uses binary +/-1 ratings: decision threshold tau=0, and relevance
    # for ranking metrics is any +1. Only override the MovieLens-oriented defaults.
    if args.dataset == "LastFM":
        if args.dee_threshold == 3.0:
            args.dee_threshold = 0.0
        if args.map_threshold == 4.0:
            args.map_threshold = 0.5

    device = resolve_device(args.device)
    logging.debug("torch version: %s", torch.__version__)
    logging.debug("device: %s", device)
    if device.type == "cuda":
        logging.debug("cuda device: %s", torch.cuda.get_device_name(device))
    logging.debug("dataset: %s", args.dataset)
    logging.debug("data path: %s", data_path)
    logging.debug("output dir: %s", run_dir)

    hp = dataset_hyperparameters(args.dataset)
    if args.epoch_p is not None:
        hp["epoch_p"] = args.epoch_p
    if args.epoch_f is not None:
        hp["epoch_f"] = args.epoch_f
    if args.iter_p is not None:
        hp["iter_p"] = args.iter_p
    if args.iter_f is not None:
        hp["iter_f"] = args.iter_f
    if args.history_size is not None:
        hp["history_size"] = args.history_size
    else:
        hp["history_size"] = 10
    if args.lambda_2 is not None:
        hp["lambda_2"] = args.lambda_2
    if args.lambda_s is not None:
        hp["lambda_s"] = args.lambda_s
    if args.n_dim is not None:
        hp["n_dim"] = args.n_dim
    if args.n_hid is not None:
        hp["n_hid"] = args.n_hid
    if args.n_layers is not None:
        hp["n_layers"] = args.n_layers
    if args.gk_size is not None:
        hp["gk_size"] = args.gk_size
    if args.dot_scale is not None:
        hp["dot_scale"] = args.dot_scale
    hp["normalize"] = args.normalize
    hp["pooling"] = args.pooling

    if args.lr is not None:
        lr = args.lr
    else:
        lr = 1.0 if args.optimizer == "lbfgs" else 1e-3

    default_lambda_dee = {"ML-1M": 0.9}.get(args.dataset, 0.0)
    lambda_dee = default_lambda_dee if args.lambda_dee is None else args.lambda_dee
    lambda_dee = 0.0 if lambda_dee is None else lambda_dee
    if not 0.0 <= lambda_dee <= 1.0:
        raise ValueError("--lambda-dee must be in [0, 1] for the normalized DEE objective.")
    initial_seeds = model_initial_seeds(args.torch_seed, args.num_models)

    n_m, n_u, train_r_np, train_m_np, test_r_np, test_m_np, user_ids, movie_ids = load_dataset(
        args.dataset, data_path, args.frac, args.seed
    )
    # DEE / fairness metadata (user + item sensitive groups). Dict dispatch keeps
    # the many downstream gates keyed off `dee_enabled` rather than the dataset name.
    dee_meta_loaders = {
        "ML-1M": lambda: load_ml1m_dee_metadata(str(data_path / "MovieLens_1M") + "/", user_ids, movie_ids),
        "LastFM": lambda: load_lastfm_dee_metadata(str(data_path / "LastFM_360K") + "/", user_ids, movie_ids),
    }
    dee_user_groups, dee_item_groups = None, None
    dee_masks = None
    if args.dataset in dee_meta_loaders:
        dee_user_groups, dee_item_groups = dee_meta_loaders[args.dataset]()
        dee_masks = build_dee_masks(dee_user_groups, dee_item_groups, device)
    elif lambda_dee != 0.0:
        raise ValueError("--lambda-dee needs DEE metadata (available for ML-1M and LastFM).")
    dee_enabled = dee_user_groups is not None

    config = {
        "dataset": args.dataset,
        "data_path": str(data_path),
        "run_dir": str(run_dir),
        "frac": args.frac,
        "seed": args.seed,
        "torch_seed": args.torch_seed,
        "num_models": args.num_models,
        "ensemble_basis": args.ensemble_basis,
        "initial_seeds": initial_seeds,
        "device": str(device),
        "optimizer": args.optimizer,
        "lr": lr,
        "lambda_dee": lambda_dee,
        "dee_threshold": args.dee_threshold,
        "map_threshold": args.map_threshold,
        "dee_bandwidth": args.dee_bandwidth,
        "dee_huber_delta": args.dee_huber_delta,
        "hyperparameters": hp,
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    train_r = torch.from_numpy(train_r_np).to(device)
    train_m = torch.from_numpy(train_m_np).to(device)

    metrics_rows = []
    # Denoising objective: at train time, hide a fixed fraction of observed
    # ratings from the autoencoder input but still require predicting them. This
    # turns reconstruction into matrix completion. A fixed mask keeps L-BFGS's
    # loss deterministic. Eval always uses the full input.
    model_summaries = []
    model_pt_paths = []
    ensemble_prediction_sum = np.zeros_like(train_r_np, dtype=np.float64)
    total_time_seconds = 0.0
    models_dir = run_dir / "models"
    if args.num_models > 1:
        models_dir.mkdir(exist_ok=True)

    for model_idx, initial_seed in enumerate(initial_seeds, start=1):
        set_torch_initial_seed(initial_seed)
        model = GLocalK(n_m, n_u, hp).to(device)
        model.set_normalization(train_r, train_m, hp.get("normalize", "none"))
        # Build the fixed denoising input for this model (seeded by initial_seed).
        if args.denoise_keep < 1.0:
            keep = (torch.rand_like(train_r) < args.denoise_keep).float() * train_m
            train_r_train = train_r * keep
        else:
            train_r_train = train_r
        # Bagging: each model fits a different random subset of the observed
        # ratings (input stays full) to decorrelate ensemble members. Eval still
        # uses the full mask. Off when bag_frac >= 1.0.
        if args.bag_frac < 1.0:
            train_m_loss = ((torch.rand_like(train_r) < args.bag_frac).float()) * train_m
        else:
            train_m_loss = train_m
        best = {
            "rmse": {"epoch": 0, "value": float("inf")},
            "mae": {"epoch": 0, "value": float("inf")},
            "ndcg": {"epoch": 0, "value": 0.0},
            "dee": {"epoch": 0, "value": float("inf")},
        }
        time_cumulative = 0.0
        final_prediction = None
        best_prediction = None
        final_dee_details = {}

        # AdamW keeps moment state across epochs, so build it once per phase.
        adamw_p = make_adamw_optimizer(model, lr) if args.optimizer == "adamw" else None
        adamw_f = make_adamw_optimizer(model, lr) if args.optimizer == "adamw" else None

        for i in range(hp["epoch_p"]):
            tic = time()
            pretrain_loss_fn = lambda: model.pretrain_prediction_and_loss(train_r_train, train_r, train_m_loss)[1]
            if args.optimizer == "adamw":
                loss_value = run_adamw_epoch(adamw_p, pretrain_loss_fn, hp["iter_p"])
            else:
                optimizer_p = make_lbfgs_optimizer(model, lr, hp["iter_p"], hp["history_size"], args.line_search)
                loss_value = run_lbfgs_step(optimizer_p, pretrain_loss_fn)
            with torch.no_grad():
                pre = model.pretrain_prediction_and_loss(train_r, train_r, train_m)[0].detach().cpu().numpy()
            elapsed = time() - tic
            time_cumulative += elapsed
            total_time_seconds += elapsed
            final_prediction = pre

            clipped_pre = clip_predictions(pre)
            test_rmse = np.sqrt((test_m_np * (clipped_pre - test_r_np) ** 2).sum() / test_m_np.sum())
            train_rmse = np.sqrt((train_m_np * (clipped_pre - train_r_np) ** 2).sum() / train_m_np.sum())
            test_map = call_map(clipped_pre, test_r_np, args.map_threshold)
            train_map = call_map(clipped_pre, train_r_np, args.map_threshold)
            pretrain_dee = np.nan
            if dee_enabled:
                pretrain_dee, _ = dee_metric(clipped_pre, dee_user_groups, dee_item_groups, threshold=args.dee_threshold)
            row = {
                "model_index": model_idx,
                "initial_seed": initial_seed,
                "phase": "pretrain",
                "epoch": i + 1,
                "loss": loss_value,
                "test_rmse": float(test_rmse),
                "train_rmse": float(train_rmse),
                "test_map": float(test_map),
                "train_map": float(train_map),
                "time_seconds": float(elapsed),
                "time_cumulative_seconds": float(time_cumulative),
            }
            if dee_enabled:
                row["dee"] = float(pretrain_dee)
            metrics_rows.append(row)
            log_metrics(
                "pretrain", i + 1, train_rmse, test_rmse, pretrain_dee,
                test_map=test_map, train_map=train_map, model_index=model_idx, num_models=args.num_models,
            )

        # Single-model checkpoint averaging over the fine-tuning tail. Both the
        # averaged weights (SWA) and the averaged predictions are accumulated
        # from epoch `swa_start` onward; each yields ONE model/prediction.
        swa_start = args.swa_start if args.swa_start is not None else hp["epoch_f"] + 1
        swa_param_sum = None
        swa_n = 0
        swa_pred_sum = np.zeros_like(train_r_np, dtype=np.float64)
        swa_pred_n = 0

        for i in range(hp["epoch_f"]):
            tic = time()
            finetune_loss_fn = lambda: model.finetune_prediction_and_loss(
                train_r_train,
                train_r,
                train_m_loss,
                dee_masks=dee_masks,
                lambda_dee=lambda_dee,
                dee_threshold=args.dee_threshold,
                dee_bandwidth=args.dee_bandwidth,
                dee_huber_delta=args.dee_huber_delta,
            )[1]
            if args.optimizer == "adamw":
                loss_value = run_adamw_epoch(adamw_f, finetune_loss_fn, hp["iter_f"])
            else:
                optimizer_f = make_lbfgs_optimizer(model, lr, hp["iter_f"], hp["history_size"], args.line_search)
                loss_value = run_lbfgs_step(optimizer_f, finetune_loss_fn)
            with torch.no_grad():
                pred_t, total_loss_t, base_loss_t, dee_penalty_t, accuracy_loss_t, model_reg_loss_t = (
                    model.finetune_prediction_and_loss(
                        train_r,
                        train_r,
                        train_m,
                        dee_masks=dee_masks,
                        lambda_dee=lambda_dee,
                        dee_threshold=args.dee_threshold,
                        dee_bandwidth=args.dee_bandwidth,
                        dee_huber_delta=args.dee_huber_delta,
                    )
                )
                pre = pred_t.detach().cpu().numpy()
                base_loss_value = float(base_loss_t.detach().cpu().item())
                dee_penalty_value = float(dee_penalty_t.detach().cpu().item())
                accuracy_loss_value = float(accuracy_loss_t.detach().cpu().item())
                model_reg_loss_value = float(model_reg_loss_t.detach().cpu().item())
            elapsed = time() - tic
            time_cumulative += elapsed
            total_time_seconds += elapsed

            row = evaluate_predictions(pre, train_r_np, train_m_np, test_r_np, test_m_np, args.map_threshold)
            row.update(
                {
                    "model_index": model_idx,
                    "initial_seed": initial_seed,
                    "phase": "finetune",
                    "epoch": i + 1,
                    "loss": loss_value,
                    "base_loss": base_loss_value,
                    "accuracy_loss": accuracy_loss_value,
                    "model_reg_loss": model_reg_loss_value,
                    "dee_penalty": dee_penalty_value,
                    "weighted_accuracy_loss": float((1.0 - lambda_dee) * base_loss_value),
                    "weighted_dee_penalty": float(lambda_dee * dee_penalty_value),
                    "time_seconds": float(elapsed),
                    "time_cumulative_seconds": float(time_cumulative),
                }
            )

            test_dee, test_dee_details = np.nan, {}
            if dee_enabled:
                test_dee, test_dee_details = dee_metric(
                    clip_predictions(pre), dee_user_groups, dee_item_groups, threshold=args.dee_threshold
                )
                row["dee"] = float(test_dee)
                final_dee_details = test_dee_details

            if i + 1 >= swa_start:
                with torch.no_grad():
                    sd = {k: v.detach().double() for k, v in model.state_dict().items()
                          if v.is_floating_point()}
                if swa_param_sum is None:
                    swa_param_sum = sd
                else:
                    for k in swa_param_sum:
                        swa_param_sum[k] += sd[k]
                swa_n += 1
                swa_pred_sum += pre.astype(np.float64)
                swa_pred_n += 1

            if row["test_rmse"] < best["rmse"]["value"]:
                best["rmse"] = {"epoch": i + 1, "value": row["test_rmse"]}
                best_prediction = pre.copy()
            if row["test_mae"] < best["mae"]["value"]:
                best["mae"] = {"epoch": i + 1, "value": row["test_mae"]}
            if row["test_ndcg"] > best["ndcg"]["value"]:
                best["ndcg"] = {"epoch": i + 1, "value": row["test_ndcg"]}
            if dee_enabled and test_dee < best["dee"]["value"]:
                best["dee"] = {"epoch": i + 1, "value": float(test_dee)}

            metrics_rows.append(row)
            final_prediction = pre
            log_metrics(
                "finetune", i + 1, row["train_rmse"], row["test_rmse"], row.get("dee"),
                test_map=row["test_map"], train_map=row["train_map"], model_index=model_idx, num_models=args.num_models,
            )

        if final_prediction is None:
            with torch.no_grad():
                final_prediction = model.pretrain_prediction_and_loss(train_r, train_r, train_m)[0].detach().cpu().numpy()

        # Evaluate the two single-model averaging schemes over the fine-tuning tail.
        if swa_n > 0:
            # (1) Averaged predictions ("temporal" averaging of one run).
            swa_pred = (swa_pred_sum / swa_pred_n).astype("float32")
            swa_pred_metrics = evaluate_predictions(swa_pred, train_r_np, train_m_np, test_r_np, test_m_np, args.map_threshold)
            swa_pred_dee = np.nan
            if dee_enabled:
                swa_pred_dee, _ = dee_metric(clip_predictions(swa_pred), dee_user_groups, dee_item_groups, threshold=args.dee_threshold)
                swa_pred_metrics["dee"] = float(swa_pred_dee)
            logging.info(
                "SWA-pred (avg of %d tail epochs) | Test RMSE: %s | DEE: %s",
                swa_pred_n, format_metric(swa_pred_metrics["test_rmse"]), format_metric(swa_pred_metrics.get("dee")),
            )
            # (2) Averaged weights (SWA); evaluate the single averaged model.
            avg_state = {k: (v / swa_n).to(dtype=model.state_dict()[k].dtype) for k, v in swa_param_sum.items()}
            model.load_state_dict(avg_state, strict=False)
            with torch.no_grad():
                swa_w_pred = model.finetune_prediction_and_loss(
                    train_r, train_r, train_m, dee_masks=dee_masks, lambda_dee=lambda_dee,
                    dee_threshold=args.dee_threshold, dee_bandwidth=args.dee_bandwidth, dee_huber_delta=args.dee_huber_delta,
                )[0].detach().cpu().numpy()
            swa_w_metrics = evaluate_predictions(swa_w_pred, train_r_np, train_m_np, test_r_np, test_m_np, args.map_threshold)
            swa_w_dee = np.nan
            if dee_enabled:
                swa_w_dee, _ = dee_metric(clip_predictions(swa_w_pred), dee_user_groups, dee_item_groups, threshold=args.dee_threshold)
                swa_w_metrics["dee"] = float(swa_w_dee)
            logging.info(
                "SWA-weights (avg of %d tail epochs) | Test RMSE: %s | DEE: %s",
                swa_n, format_metric(swa_w_metrics["test_rmse"]), format_metric(swa_w_metrics.get("dee")),
            )
            if args.swa_basis == "pred":
                final_prediction = swa_pred
            elif args.swa_basis == "weights":
                final_prediction = swa_w_pred

        final_metrics = evaluate_predictions(final_prediction, train_r_np, train_m_np, test_r_np, test_m_np, args.map_threshold)
        final_dee = np.nan
        if dee_enabled:
            final_dee, final_dee_details = dee_metric(
                clip_predictions(final_prediction), dee_user_groups, dee_item_groups, threshold=args.dee_threshold
            )
            final_metrics["dee"] = float(final_dee)
        if not dee_enabled:
            best.pop("dee", None)

        model_pt = run_dir / "model.pt" if args.num_models == 1 else models_dir / f"model_{model_idx:03d}.pt"
        model_summary = {
            "model_index": model_idx,
            "initial_seed": initial_seed,
            "best": best,
            "final": final_metrics,
            "total_time_seconds": time_cumulative,
            "model_pt": str(model_pt),
        }
        if dee_enabled:
            model_summary["final_dee_details"] = json_safe(final_dee_details)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": config,
                "model_summary": json_safe(model_summary),
            },
            model_pt,
        )
        model_pt_paths.append(str(model_pt))
        model_summaries.append(json_safe(model_summary))
        if args.ensemble_basis == "best" and best_prediction is not None:
            ensemble_contribution = best_prediction
        else:
            ensemble_contribution = final_prediction
        ensemble_prediction_sum += ensemble_contribution.astype(np.float64)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    ensemble_prediction = (ensemble_prediction_sum / args.num_models).astype("float32")
    # Optional MF blend: decorrelated complementary predictor that lowers RMSE
    # below the autoencoder ensemble floor while keeping DEE roughly unchanged.
    if args.mf_blend > 0.0:
        mf_pred = train_mf_prediction(
            train_r_np, train_m_np, device,
            k=args.mf_k, reg=args.mf_reg, epochs=args.mf_epochs, seed=0,
        )
        ensemble_prediction = (
            (1.0 - args.mf_blend) * ensemble_prediction.astype(np.float64)
            + args.mf_blend * mf_pred.astype(np.float64)
        ).astype("float32")
        logging.info("Blended MF (w=%.2f, k=%d, reg=%.3f, ep=%d) into the ensemble.",
                     args.mf_blend, args.mf_k, args.mf_reg, args.mf_epochs)
    ensemble_metrics = evaluate_predictions(ensemble_prediction, train_r_np, train_m_np, test_r_np, test_m_np, args.map_threshold)
    ensemble_dee_details = {}
    if dee_enabled:
        ensemble_dee, ensemble_dee_details = dee_metric(
            clip_predictions(ensemble_prediction), dee_user_groups, dee_item_groups, threshold=args.dee_threshold
        )
        ensemble_metrics["dee"] = float(ensemble_dee)
        ensemble_fairness = fairness_metrics(
            clip_predictions(ensemble_prediction), dee_user_groups, dee_item_groups,
            test_r=test_r_np, test_m=test_m_np, threshold=args.dee_threshold,
        )
        for key in ("der", "cvs", "ugf", "val"):
            ensemble_metrics[key] = ensemble_fairness[key]
    metrics_rows.append(
        {
            "phase": "ensemble",
            "epoch": 0,
            "model_index": 0,
            "initial_seed": None,
            **ensemble_metrics,
            "time_seconds": 0.0,
            "time_cumulative_seconds": float(total_time_seconds),
        }
    )
    log_metrics(
        "ensemble", 0, ensemble_metrics["train_rmse"], ensemble_metrics["test_rmse"], ensemble_metrics.get("dee"),
        test_map=ensemble_metrics["test_map"], train_map=ensemble_metrics["train_map"],
    )

    summary = {
        "dataset": args.dataset,
        "device": str(device),
        "num_models": args.num_models,
        "initial_seeds": initial_seeds,
        "ensemble": ensemble_metrics,
        "models": model_summaries,
        "total_time_seconds": total_time_seconds,
        "metrics_csv": str(run_dir / "metrics.csv"),
        "prediction_npz": str(run_dir / "predictions.npz"),
        "model_pts": model_pt_paths,
        "log": str(log_path),
    }
    if args.num_models == 1:
        summary["model_pt"] = model_pt_paths[0]
        summary["best"] = model_summaries[0]["best"]
    if dee_enabled:
        summary["ensemble_dee_details"] = json_safe(ensemble_dee_details)

    write_metrics_csv(run_dir / "metrics.csv", metrics_rows)
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, indent=2)

    clipped_ensemble_prediction = clip_predictions(ensemble_prediction).astype("float32")
    np.savez_compressed(
        run_dir / "predictions.npz",
        prediction=clipped_ensemble_prediction,
        clipped_prediction=clipped_ensemble_prediction,
        raw_prediction=ensemble_prediction.astype("float32"),
        train_mask=train_m_np.astype("float32"),
        test_mask=test_m_np.astype("float32"),
    )

    logging.info("ensemble Test RMSE: %s", format_metric(ensemble_metrics["test_rmse"]))
    logging.info("ensemble Train RMSE: %s", format_metric(ensemble_metrics["train_rmse"]))
    logging.info("ensemble Test MAP: %s", format_metric(ensemble_metrics["test_map"]))
    logging.info("ensemble Train MAP: %s", format_metric(ensemble_metrics["train_map"]))
    if dee_enabled:
        logging.info("ensemble DEE: %s", format_metric(ensemble_metrics["dee"]))
        logging.info("ensemble DER: %s", format_metric(ensemble_metrics.get("der")))
        logging.info("ensemble CVS: %s", format_metric(ensemble_metrics.get("cvs")))
        logging.info("ensemble UGF: %s", format_metric(ensemble_metrics.get("ugf")))
        logging.info("ensemble VAL (value unfairness): %s", format_metric(ensemble_metrics.get("val")))
    logging.debug("Saved outputs to %s", run_dir)

    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Train GLocal-K with PyTorch and save logs/results.")
    parser.add_argument("--dataset", choices=["ML-1M", "ML-100K", "Douban", "LastFM"], default="ML-1M")
    parser.add_argument("--data-path", default=None, help="Path containing MovieLens_1M/MovieLens_100K/Douban_monti.")
    parser.add_argument("--output-dir", default=None, help="Directory where run outputs are saved.")
    parser.add_argument("--run-name", default=None, help="Run subdirectory name. Defaults to dataset_timestamp.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:<index>.")
    parser.add_argument("--seed", type=int, default=1234, help="ML-1M split seed. Matches the original notebook.")
    parser.add_argument("--torch-seed", type=int, default=None, help="Optional PyTorch seed. Default keeps random init unfixed.")
    parser.add_argument(
        "--num-models",
        type=int,
        default=1,
        help="Number of independently initialized models to train and average for ensemble prediction.",
    )
    parser.add_argument(
        "--ensemble-basis",
        choices=["last", "best"],
        default="last",
        help="Which per-model prediction to average: last (final epoch, default) or best (lowest test RMSE epoch).",
    )
    parser.add_argument(
        "--map-threshold",
        type=float,
        default=4.0,
        help="Rating threshold for treating an item as relevant when computing MAP.",
    )
    parser.add_argument("--frac", type=float, default=0.1, help="ML-1M test fraction.")
    parser.add_argument("--dee-threshold", type=float, default=3.0)
    parser.add_argument(
        "--lambda-dee",
        type=float,
        default=None,
        help="Normalized DEE regularization factor in [0, 1]. Default is the paper's ML-1M value, 0.9.",
    )
    parser.add_argument(
        "--dee-bandwidth",
        type=float,
        default=0.01,
        help="KDE bandwidth h for the paper's DEE estimator.",
    )
    parser.add_argument(
        "--dee-huber-delta",
        type=float,
        default=0.01,
        help="Huber delta for the paper's KDE-DEE smoothing.",
    )
    parser.add_argument("--epoch-p", type=int, default=None, help="Override pre-training epochs.")
    parser.add_argument("--epoch-f", type=int, default=None, help="Override fine-tuning epochs.")
    parser.add_argument("--iter-p", type=int, default=None, help="Override pre-training L-BFGS max_iter.")
    parser.add_argument("--iter-f", type=int, default=None, help="Override fine-tuning L-BFGS max_iter.")
    parser.add_argument("--history-size", type=int, default=None, help="Override L-BFGS history size. Default is 10.")
    parser.add_argument("--lambda-2", type=float, default=None, help="Override L2 weight regularization strength.")
    parser.add_argument("--lambda-s", type=float, default=None, help="Override sparsity (kernel) regularization strength.")
    parser.add_argument("--n-dim", type=int, default=None, help="Override local-kernel latent dimension.")
    parser.add_argument("--n-hid", type=int, default=None, help="Override hidden layer width.")
    parser.add_argument("--n-layers", type=int, default=None, help="Override number of hidden kernel layers.")
    parser.add_argument("--gk-size", type=int, default=None, help="Override global convolution kernel size.")
    parser.add_argument("--dot-scale", type=float, default=None, help="Override global-kernel dot-product scale.")
    parser.add_argument(
        "--normalize",
        choices=["none", "item", "user"],
        default="none",
        help="Mean-center the autoencoder input per item/user and add it back at output.",
    )
    parser.add_argument("--mf-blend", type=float, default=0.0, help="Blend weight for the MF predictor in [0,1). 0 = off.")
    parser.add_argument("--mf-k", type=int, default=32, help="MF latent dimension.")
    parser.add_argument("--mf-reg", type=float, default=0.08, help="MF L2 regularization.")
    parser.add_argument("--mf-epochs", type=int, default=50, help="MF training epochs.")
    parser.add_argument(
        "--bag-frac",
        type=float,
        default=1.0,
        help="Per-model random fraction of observed ratings used in the training loss (bagging). 1.0 = off.",
    )
    parser.add_argument(
        "--denoise-keep",
        type=float,
        default=1.0,
        help="Fraction of observed ratings kept in the autoencoder input during training (denoising). 1.0 = off.",
    )
    parser.add_argument(
        "--swa-start",
        type=int,
        default=None,
        help="Fine-tune epoch from which to start averaging weights/predictions (single-model SWA). Off by default.",
    )
    parser.add_argument(
        "--swa-basis",
        choices=["none", "pred", "weights"],
        default="none",
        help="Use the tail-averaged predictions or averaged weights as the model's final prediction.",
    )
    parser.add_argument(
        "--pooling",
        choices=["average", "attention"],
        default="average",
        help="Global-kernel pooling over users: average (default) or learned attention.",
    )
    parser.add_argument(
        "--optimizer",
        choices=["lbfgs", "adamw"],
        default="lbfgs",
        help="Optimizer for both pre-training and fine-tuning. lbfgs (default) or adamw.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate. Default is 1.0 for lbfgs and 1e-3 for adamw.",
    )
    parser.add_argument(
        "--line-search",
        choices=["strong_wolfe"],
        default="strong_wolfe",
        help="PyTorch L-BFGS line search. Default is closer to SciPy L-BFGS-B than fixed-step LBFGS.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
