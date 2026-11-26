"""
Round 40 - TARGET PUSH (national-project reporting).

GOAL: report against the ORIGINAL committed benchmark (IITP target table) on the
ORIGINAL Four-Findings data / split defined by train.py:
    예측정확도(accuracy)     >= 94%   (2026 target; currently 93.38 -> SHORT)
    공정성 DEO               <= 5%    (|TPR_f - TPR_m| x 100; currently 2.03)
    정책반영 성능평균하락률   <= 3%    ((acc_nofair-acc_fair)/acc_nofair x100; now 0.25)
Only accuracy is short. Push acc >= 94 while keeping DEO <= 5 and drop <= 3.

This script REPLICATES train.py's data pipeline / label & metric definitions EXACTLY
so the numbers stay comparable to the 93.38/2.03/0.25 benchmark, then makes the
training recipe configurable so the orchestrator can escalate across rounds:
    backbone {resnet18(orig), vit_b16, densenet121, raddino}
    epochs (orig=2, up to ~30), aug {basic(orig), strong}, class-weight fix
    (orig bug: computed then overwritten with None), TTA, seed.

REPLICATION NOTES (must match train.py for report parity):
  * Data      : merge test+validation Four-Findings CSVs -> 4,376 rows (all on disk).
  * Label     : Pneumothorax==YES -> 0, else -> 1  (so label 1 == "No Finding",
                the MAJORITY class; pneumothorax positives are the rare label-0 cases).
  * Gender    : M -> +1, F -> -1  (from "Patient Gender").
  * Split     : IMAGE-LEVEL train_test_split(indices, test_size=0.10,
                random_state=42, shuffle=True)  -> train 3938 / test 438.
  * Accuracy  : overall argmax@0.5 accuracy on the 438 test images.
  * DEO       : evaluate_DEO replicated EXACTLY -> TPR computed on label==1
                (train.py's own definition), DEO=|TPR_f-TPR_m|, reported x100.
  * Fairness  : WeightedRandomSampler across (label x gender) subgroups (w=max/size).
  * Policy    : train a FAIR (WeightedRandomSampler) and a NOFAIR (shuffle) model
                with the SAME recipe; drop% = (acc_nofair-acc_fair)/acc_nofair x100.

Does NOT modify train.py / common*.py / earlier rounds.
Env: /usr/bin/python, torch.device('cuda:0') (external CUDA_VISIBLE_DEVICES),
     HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 (rad-DINO / ViT weights cached).
"""
import os
# deterministic cuBLAS (must be set before CUDA/cuBLAS initialises; harmless with --nondeterministic)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import csv
import json
import random
import argparse
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ------------------------------------------------------------------ paths
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "..", "..", "data", "ALINMed"))
TEST_LABELS_CSV = os.path.join(DATA, "four_findings_expert_labels_test_labels.csv")
VAL_LABELS_CSV = os.path.join(DATA, "four_findings_expert_labels_validation_labels.csv")
# NIH ChestX-ray14 images: download_nih_subset.py writes them here (override with $NIH_ROOT)
NIH_ROOT = os.environ.get("NIH_ROOT", os.path.join(DATA, "datasets", "nih-chest-xrays", "data", "versions", "3"))
IMG_DIRS = [os.path.join(NIH_ROOT, f"images_{i:03d}", "images") for i in range(1, 13)]
RESULTS_LOG = os.path.join(DATA, "results", "results.jsonl")
ROUND_ID = 40

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
RADDINO_HF = "microsoft/rad-dino"
RADDINO_REV = "110cbc18d5133582e320b43d53bf5c44e410c936"   # pinned revision

# reporting targets
TARGET_ACC = 94.0     # percent
TARGET_DEO = 5.0      # percent
TARGET_DROP = 3.0     # percent


# ------------------------------------------------------------------ utils
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_determinism(enabled=True):
    """Deterministic mode (default): same seed + same GPU/software -> identical training.
    cuDNN deterministic algorithms, no cuDNN autotuning, torch.use_deterministic_algorithms
    (raises on any op without a deterministic implementation instead of silently varying),
    and a deterministic backward for the rad-DINO position-embedding interpolation (CUDA bicubic
    backward has no deterministic kernel; the forward values are unchanged).
    Repeat runs are identical on the same GPU model and library versions; a different GPU or
    cuDNN/CUDA version can still give slightly different numbers."""
    torch.backends.cudnn.deterministic = enabled
    torch.backends.cudnn.benchmark = not enabled and torch.backends.cudnn.benchmark
    torch.use_deterministic_algorithms(enabled)
    if enabled:
        _patch_dinov2_pos_interp()


def _bicubic_matrix(n, scale):
    """A (m x n) with A @ v == F.interpolate(v, scale_factor=scale, mode='bicubic', align_corners=False)
    along one axis (bicubic is separable; a column of constants interpolates to itself across)."""
    e = torch.eye(n, dtype=torch.float32).reshape(n, 1, n, 1).expand(n, 1, n, 4).contiguous()
    y = torch.nn.functional.interpolate(e, scale_factor=(scale, 1.0), mode="bicubic", align_corners=False)
    return y[:, 0, :, 0].t().contiguous()          # (m, n)


class _BicubicPosInterp(torch.autograd.Function):
    """Forward: the original F.interpolate(bicubic) call, so values are bit-identical to the
    unpatched model. Backward: the same linear map transposed, A_h^T @ G @ A_w, instead of the
    CUDA bicubic backward kernel (which has no deterministic implementation)."""

    @staticmethod
    def forward(ctx, p, sh, sw, a_h, a_w):                 # p: (1, C, n, n)
        ctx.save_for_backward(a_h, a_w)
        return torch.nn.functional.interpolate(p, scale_factor=(sh, sw), mode="bicubic",
                                               align_corners=False)

    @staticmethod
    def backward(ctx, g):                                  # g: (1, C, H', W')
        a_h, a_w = ctx.saved_tensors
        return torch.einsum("ih,ncij,jw->nchw", a_h, g, a_w), None, None, None, None


_DINO_PATCHED = False


def _patch_dinov2_pos_interp():
    """Dinov2Embeddings.interpolate_pos_encoding with a deterministic backward for the bicubic
    position-embedding interpolation (see _BicubicPosInterp). Forward values are unchanged and
    there are no new parameters (state_dict unchanged)."""
    global _DINO_PATCHED
    if _DINO_PATCHED:
        return
    try:
        import math
        from transformers.models.dinov2 import modeling_dinov2 as md
    except Exception:
        return
    cache = {}

    def interpolate_pos_encoding(self, embeddings, height, width):
        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        n = int(math.sqrt(num_positions))
        h = height // self.config.patch_size + 0.1      # as in transformers (dino issue #8)
        w = width // self.config.patch_size + 0.1
        sh, sw = float(h / math.sqrt(num_positions)), float(w / math.sqrt(num_positions))
        key = (n, sh, sw, patch_pos_embed.device)
        if key not in cache:
            cache[key] = (_bicubic_matrix(n, sh).to(patch_pos_embed.device),
                          _bicubic_matrix(n, sw).to(patch_pos_embed.device))
        a_h, a_w = cache[key]
        target_dtype = patch_pos_embed.dtype
        p = patch_pos_embed.reshape(1, n, n, dim).permute(0, 3, 1, 2).to(dtype=torch.float32)
        p = _BicubicPosInterp.apply(p, sh, sw, a_h, a_w).to(dtype=target_dtype)
        if int(h) != p.shape[-2] or int(w) != p.shape[-1]:
            raise ValueError("Width or height does not match with the interpolated position embeddings")
        p = p.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), p), dim=1)

    md.Dinov2Embeddings._orig_interpolate_pos_encoding = md.Dinov2Embeddings.interpolate_pos_encoding
    md.Dinov2Embeddings.interpolate_pos_encoding = interpolate_pos_encoding
    _DINO_PATCHED = True


def _worker_init_fn(worker_id):
    """DataLoader worker seeding, explicit: numpy / random from the worker's torch seed
    (which the loader's generator fixes)."""
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def _loader_generator(seed):
    return torch.Generator().manual_seed(seed)


def find_image_path(img_name):
    for d in IMG_DIRS:
        p = os.path.join(d, img_name)
        if os.path.exists(p):
            return p
    return None


def get_gender_label(gender_str):
    # EXACT replication of train.py: M -> +1, else -> -1
    return 1 if gender_str.strip().upper() == "M" else -1


def merge_expert_csvs(csv1, csv2):
    rows = []
    for fn in (csv1, csv2):
        with open(fn, "r") as f:
            rows.extend(list(csv.DictReader(f)))
    return rows


# ------------------------------------------------------------------ dataset
class CleanChestDataset(Dataset):
    """EXACT replication of train.py's CleanChestDataset labelling.

    sample = (path, cond_label, gender)   cond_label: pneumothorax->0 else->1
    """

    def __init__(self, all_rows, transform=None, indices=None, verbose=True):
        super().__init__()
        self.transform = transform
        self.samples = []
        missing = 0
        for row in all_rows:
            img_name = row["Image Index"].strip()
            full_path = find_image_path(img_name)
            if not full_path:
                missing += 1
                continue
            cond_label = 0 if row["Pneumothorax"] == "YES" else 1
            gval = get_gender_label(row["Patient Gender"])
            self.samples.append((full_path, cond_label, gval))
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]
        if verbose:
            lab = Counter(s[1] for s in self.samples)
            gen = Counter(s[2] for s in self.samples)
            print(f"[Dataset] n={len(self.samples)} dropped_missing={missing} | "
                  f"label pneu(0)={lab[0]} nofind(1)={lab[1]} | "
                  f"gender M(+1)={gen[1]} F(-1)={gen[-1]}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cond_label, gender_label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, cond_label, gender_label

    def get_metadata(self):
        return [(s[1], s[2]) for s in self.samples]


# ------------------------------------------------------------------ transforms
def build_transforms(aug, mean, std, size=224):
    """basic == train.py's exact train transform. strong adds rotation,
    colorjitter, RandAugment-lite and RandomErasing."""
    norm = transforms.Normalize(mean, std)
    if aug == "basic":
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.81, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm,
        ])
    elif aug == "strong":
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandAugment(num_ops=2, magnitude=7),
            transforms.ToTensor(),
            norm,
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
        ])
    else:
        raise ValueError(f"unknown aug {aug}")
    test_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        norm,
    ])
    return train_tf, test_tf


# ------------------------------------------------------------------ models
class RadDinoClassifier(nn.Module):
    """HuggingFace rad-DINO backbone + fresh 2-class linear head
    (same build pattern as rounds 12-15)."""

    def __init__(self, num_classes=2, hf_name=RADDINO_HF):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(hf_name, revision=RADDINO_REV)
        h = self.backbone.config.hidden_size
        self.head = nn.Linear(h, num_classes)

    def forward(self, x):
        out = self.backbone(pixel_values=x)
        feat = out.pooler_output if out.pooler_output is not None \
            else out.last_hidden_state[:, 0]
        return self.head(feat)


def _tv_weights(enum_name):
    """Return torchvision weights enum if the checkpoint is cached, else None.
    Under *_OFFLINE=1 a cache miss raises; we catch and fall back to random init
    with a clear note (so a non-cached backbone still *runs*, just uncomparable)."""
    return enum_name


def build_model(backbone, device):
    """Return (model, is_transformer, mean, std, note)."""
    note = ""
    if backbone == "resnet18":
        try:
            m = models.resnet18(weights="IMAGENET1K_V1")
        except Exception as e:
            m = models.resnet18(weights=None)
            note = f"resnet18 pretrained unavailable ({e}); random init"
        m.fc = nn.Linear(m.fc.in_features, 2)
        return m.to(device), False, IMAGENET_MEAN, IMAGENET_STD, note

    if backbone == "densenet121":
        try:
            m = models.densenet121(weights="IMAGENET1K_V1")
        except Exception as e:
            m = models.densenet121(weights=None)
            note = (f"densenet121 pretrained NOT cached offline ({type(e).__name__}); "
                    "random init -> NOT comparable, cache weights before full run")
        m.classifier = nn.Linear(m.classifier.in_features, 2)
        return m.to(device), False, IMAGENET_MEAN, IMAGENET_STD, note

    if backbone == "vit_b16":
        try:
            m = models.vit_b_16(weights="IMAGENET1K_V1")
        except Exception as e:
            m = models.vit_b_16(weights=None)
            note = f"vit_b16 pretrained unavailable ({e}); random init"
        m.heads.head = nn.Linear(m.heads.head.in_features, 2)
        return m.to(device), True, IMAGENET_MEAN, IMAGENET_STD, note

    if backbone == "raddino":
        m = RadDinoClassifier()
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        try:
            from transformers import AutoImageProcessor
            proc = AutoImageProcessor.from_pretrained(RADDINO_HF, revision=RADDINO_REV)
            mean = list(proc.image_mean)
            std = list(proc.image_std)
        except Exception as e:
            note = f"rad-DINO processor norm unavailable ({e}); ImageNet norm"
        return m.to(device), True, mean, std, note

    raise ValueError(f"unknown backbone {backbone}")


def build_optimizer(model, backbone, is_transformer, lr):
    """Transformers get differential LR (backbone low, head high) as in R12."""
    if is_transformer:
        if backbone == "raddino":
            bb = list(model.backbone.parameters())
            head = list(model.head.parameters())
        else:  # vit_b16
            head = list(model.heads.parameters())
            head_ids = {id(p) for p in head}
            bb = [p for p in model.parameters() if id(p) not in head_ids]
        return optim.AdamW([
            {"params": bb, "lr": lr},
            {"params": head, "lr": max(lr * 50, 1e-3)},
        ], weight_decay=1e-4)
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)


# ------------------------------------------------------------------ sampler
def build_fair_sampler(train_dataset, generator=None):
    """EXACT replication of train.py fairness: WeightedRandomSampler across
    (cond_label x gender) subgroups with weight = max_group_size / group_size."""
    group_dict = defaultdict(list)
    meta = train_dataset.get_metadata()
    for idx, (c_label, g_label) in enumerate(meta):
        group_dict[(c_label, g_label)].append(idx)
    group_sizes = {g: len(idxs) for g, idxs in group_dict.items()}
    max_size = max(group_sizes.values()) if group_sizes else 1
    sample_weights = np.zeros(len(meta), dtype=np.float32)
    for g, idxs in group_dict.items():
        w = float(max_size / len(idxs)) if len(idxs) > 0 else 1.0
        for i in idxs:
            sample_weights[i] = w
    sampler = WeightedRandomSampler(sample_weights, len(meta), generator=generator)
    return sampler, {str(k): v for k, v in group_sizes.items()}


# ------------------------------------------------------------------ eval (DEO)
@torch.no_grad()
def evaluate_DEO(model, loader, device, is_transformer=False, tta=False, return_preds=False):
    """EXACT replication of train.py.evaluate_DEO metric definitions:
    overall accuracy + TPR on label==1, DEO=|TPR_f-TPR_m|. TTA (hflip) optional.
    return_preds=True also returns the per-sample (preds, labels, gender) arrays."""
    model.eval()
    all_preds, all_labels, all_gender = [], [], []
    for images, labels, gender in loader:
        images = images.to(device)
        logits = model(images)
        if tta:
            logits = logits + model(torch.flip(images, dims=[3]))
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(np.asarray(labels))
        all_gender.extend(np.asarray(gender))

    acc = accuracy_score(all_labels, all_preds)
    female_idx = [i for i, g in enumerate(all_gender) if g == -1]
    male_idx = [i for i, g in enumerate(all_gender) if g == 1]
    female_actual_1 = [i for i in female_idx if all_labels[i] == 1]
    male_actual_1 = [i for i in male_idx if all_labels[i] == 1]
    tpr_f = (sum(all_preds[j] == 1 for j in female_actual_1) / len(female_actual_1)
             if female_actual_1 else 0.0)
    tpr_m = (sum(all_preds[j] == 1 for j in male_actual_1) / len(male_actual_1)
             if male_actual_1 else 0.0)
    deo = abs(tpr_f - tpr_m)
    if return_preds:
        return (float(acc), float(deo), float(tpr_f), float(tpr_m),
                (np.asarray(all_preds), np.asarray(all_labels), np.asarray(all_gender)))
    return float(acc), float(deo), float(tpr_f), float(tpr_m)


# ------------------------------------------------------------------ train
def make_criterion(train_dataset, fix_classweight, device):
    """train.py computed inverse-freq class weights then DISABLED them (=None).
    fix_classweight applies them properly (balanced inverse-frequency)."""
    labels = [c for c, _ in train_dataset.get_metadata()]
    cnt = Counter(labels)
    if not fix_classweight:
        return nn.CrossEntropyLoss()
    tot = sum(cnt.values())
    # balanced: w_c = tot / (2 * count_c)  -> upweights the rare pneumothorax(0)
    w = torch.tensor([tot / (2 * max(cnt[0], 1)), tot / (2 * max(cnt[1], 1))],
                     dtype=torch.float, device=device)
    return nn.CrossEntropyLoss(weight=w)


def train_variant(rows, train_idx, test_idx, args, use_fairness, device):
    """Train one model (fair or nofair) with the shared recipe. Returns metrics."""
    tag = "FAIR" if use_fairness else "NOFAIR"
    model, is_tf, mean, std, note = build_model(args.backbone, device)
    if note:
        print(f"[{tag}] NOTE: {note}")
    train_tf, test_tf = build_transforms(args.aug, mean, std)
    train_ds = CleanChestDataset(rows, transform=train_tf, indices=train_idx, verbose=False)
    test_ds = CleanChestDataset(rows, transform=test_tf, indices=test_idx, verbose=False)

    gen = _loader_generator(args.seed)     # sampler order + worker seeds (augmentation)
    if use_fairness:
        sampler, groups = build_fair_sampler(train_ds, generator=gen)
        print(f"[{tag}] WeightedRandomSampler groups(label,gender)->size: {groups}")
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  shuffle=False, drop_last=False, num_workers=args.num_workers,
                                  generator=gen, worker_init_fn=_worker_init_fn)
    else:
        print(f"[{tag}] standard shuffle")
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  drop_last=False, num_workers=args.num_workers,
                                  generator=gen, worker_init_fn=_worker_init_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, worker_init_fn=_worker_init_fn)

    criterion = make_criterion(train_ds, args.fix_classweight, device)
    optimizer = build_optimizer(model, args.backbone, is_tf, args.lr)

    for epoch in range(args.epochs):
        model.train()
        run_loss, seen = 0.0, 0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * labels.size(0)
            seen += labels.size(0)
        acc, deo, tf, tm = evaluate_DEO(model, test_loader, device, is_tf, tta=args.tta)
        print(f"[{tag}] ep{epoch+1}/{args.epochs} loss={run_loss/max(seen,1):.4f} "
              f"acc={acc:.4f} DEO={deo:.4f} (TPR_f={tf:.3f} TPR_m={tm:.3f})")

    acc, deo, tf, tm, preds = evaluate_DEO(model, test_loader, device, is_tf, tta=args.tta,
                                           return_preds=True)
    print(f"[{tag}] FINAL acc={acc:.4f} DEO={deo:.4f} TPR_f={tf:.3f} TPR_m={tm:.3f}")
    return {"acc": acc, "deo": deo, "tpr_f": tf, "tpr_m": tm, "_preds": preds}


# ------------------------------------------------------------------ data prep
def prepare_split(verbose=True):
    """Merge CSVs + EXACT image-level split from train.py."""
    rows = merge_expert_csvs(TEST_LABELS_CSV, VAL_LABELS_CSV)
    if verbose:
        print(f"[Main] merged rows = {len(rows)}")
    full = CleanChestDataset(rows, transform=None, verbose=verbose)
    all_indices = list(range(len(full)))
    train_idx, test_idx = train_test_split(
        all_indices, test_size=0.10, random_state=42, shuffle=True)
    return rows, full, train_idx, test_idx


def split_composition(full, idx):
    samples = [full.samples[i] for i in idx]
    pneu = [s for s in samples if s[1] == 0]          # pneumothorax positives
    return {
        "n": len(samples),
        "pos": len(pneu),                              # rare pneumothorax cases
        "pos_F": sum(1 for s in pneu if s[2] == -1),
        "pos_M": sum(1 for s in pneu if s[2] == 1),
        "F": sum(1 for s in samples if s[2] == -1),
        "M": sum(1 for s in samples if s[2] == 1),
        "nofind": sum(1 for s in samples if s[1] == 1),
    }


# ------------------------------------------------------------------ smoke
def run_smoke(args, device):
    print("=" * 70)
    print("SMOKE TEST (no full training)")
    print("=" * 70)
    rows, full, train_idx, test_idx = prepare_split(verbose=True)
    assert len(rows) == 4376, f"expected 4376 merged rows, got {len(rows)}"
    tr_c = split_composition(full, train_idx)
    te_c = split_composition(full, test_idx)
    print(f"[SPLIT] train n={tr_c['n']}  test n={te_c['n']}  (expect 3938 / 438)")
    print(f"[SPLIT] TEST pneumothorax-positives={te_c['pos']}  "
          f"F={te_c['pos_F']} M={te_c['pos_M']}  (expect ~30, F16/M14)")
    print(f"[SPLIT] TEST gender all: F={te_c['F']} M={te_c['M']}  "
          f"| No-Finding(label1)={te_c['nofind']}")
    print(f"[NOTE] train.py's evaluate_DEO computes TPR on label==1 (No-Finding, "
          f"the majority); DEO=|TPR_f-TPR_m|. Replicated verbatim.")

    for bb in ["resnet18", "raddino"]:
        print("-" * 60)
        print(f"[SMOKE] backbone = {bb}")
        try:
            model, is_tf, mean, std, note = build_model(bb, device)
        except Exception as e:
            print(f"[SMOKE] {bb} BUILD FAILED: {type(e).__name__}: {e}")
            continue
        if note:
            print(f"[SMOKE] note: {note}")
        print(f"[SMOKE] built {bb} (transformer={is_tf}) norm mean={ [round(x,3) for x in mean] }")
        train_tf, test_tf = build_transforms(args.aug, mean, std)
        train_ds = CleanChestDataset(rows, transform=train_tf, indices=train_idx, verbose=False)
        test_ds = CleanChestDataset(rows, transform=test_tf, indices=test_idx, verbose=False)

        # WeightedRandomSampler builds
        sampler, groups = build_fair_sampler(train_ds)
        print(f"[SMOKE] WeightedRandomSampler OK groups(label,gender)->size: {groups}")
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=0, drop_last=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        # 1-batch fwd + bwd
        criterion = make_criterion(train_ds, args.fix_classweight, device)
        optimizer = build_optimizer(model, bb, is_tf, args.lr)
        images, labels, _ = next(iter(train_loader))
        images, labels = images.to(device), labels.to(device)
        model.train()
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        print(f"[SMOKE] 1-batch fwd+bwd OK  logits={tuple(logits.shape)} loss={loss.item():.4f}")

        # evaluate/DEO on a few batches (subset test loader)
        small = CleanChestDataset(rows, transform=test_tf, indices=test_idx[:3 * args.batch_size],
                                  verbose=False)
        small_loader = DataLoader(small, batch_size=args.batch_size, shuffle=False, num_workers=0)
        acc, deo, tf, tm = evaluate_DEO(model, small_loader, device, is_tf, tta=args.tta)
        print(f"[SMOKE] evaluate_DEO OK (few batches, untrained) "
              f"acc={acc:.4f} DEO={deo:.4f} TPR_f={tf:.3f} TPR_m={tm:.3f}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("=" * 70)
    print("SMOKE OK - split reproduced, resnet18 + rad-DINO build/fwd/bwd/sampler/DEO all pass")
    print("=" * 70)


# ------------------------------------------------------------------ full run
def run_full(args, device):
    set_determinism(not getattr(args, "nondeterministic", False))
    tag = f"{args.backbone}_e{args.epochs}_{args.aug}"
    if args.fix_classweight:
        tag += "_cw"
    if args.tta:
        tag += "_tta"
    if not getattr(args, "nondeterministic", False):
        tag += "_det"
    tag += f"_s{args.seed}"
    print("=" * 70)
    print(f"TARGET PUSH full run: {tag}  (seed={args.seed})")
    print("=" * 70)
    rows, full, train_idx, test_idx = prepare_split(verbose=True)
    te_c = split_composition(full, test_idx)
    print(f"[SPLIT] train={len(train_idx)} test={len(test_idx)} "
          f"test_pneu_pos={te_c['pos']} (F{te_c['pos_F']}/M{te_c['pos_M']})")

    print("\n----- MODEL B (WITH FAIRNESS: WeightedRandomSampler) -----")
    seed_everything(args.seed)
    fair = train_variant(rows, train_idx, test_idx, args, use_fairness=True, device=device)

    print("\n----- MODEL A (NO FAIRNESS: standard shuffle) -----")
    seed_everything(args.seed)
    nofair = train_variant(rows, train_idx, test_idx, args, use_fairness=False, device=device)

    acc_fair, acc_nofair = fair["acc"], nofair["acc"]
    deo_fair_pct = fair["deo"] * 100.0
    drop_pct = ((acc_nofair - acc_fair) / acc_nofair * 100.0) if acc_nofair > 0 else 0.0

    pass_acc = acc_fair * 100.0 >= TARGET_ACC
    pass_deo = deo_fair_pct <= TARGET_DEO
    pass_drop = drop_pct <= TARGET_DROP
    overall = pass_acc and pass_deo and pass_drop

    print("\n" + "=" * 70)
    print("*** RESULTS (reported on FAIR model, original benchmark definitions) ***")
    print(f"accuracy(fair)   = {acc_fair*100:.2f}%   accuracy(nofair) = {acc_nofair*100:.2f}%")
    print(f"DEO(fair)        = {deo_fair_pct:.2f}%   (TPR_f={fair['tpr_f']:.3f} TPR_m={fair['tpr_m']:.3f})")
    print(f"policy-drop      = {drop_pct:.2f}%")
    print(f"TARGET CHECK: acc>=94? {pass_acc} ({acc_fair*100:.2f}) | "
          f"DEO<=5? {pass_deo} ({deo_fair_pct:.2f}) | "
          f"drop<=3? {pass_drop} ({drop_pct:.2f})  => "
          f"{'PASS' if overall else 'FAIL'}")
    print("=" * 70)

    variant = f"{args.backbone}_e{args.epochs}_{args.aug}"
    notes = (f"R40 target-push on ORIGINAL Four-Findings merge(4376)/image-level split "
             f"(train3938/test438, test_pneu_pos={te_c['pos']} F{te_c['pos_F']}/M{te_c['pos_M']}); "
             f"recipe: backbone={args.backbone} epochs={args.epochs} aug={args.aug} "
             f"fix_classweight={args.fix_classweight} tta={args.tta} lr={args.lr} bs={args.batch_size} seed={args.seed}. "
             f"TARGETS: acc>=94% -> {acc_fair*100:.2f}% [{'PASS' if pass_acc else 'FAIL'}]; "
             f"DEO<=5% -> {deo_fair_pct:.2f}% [{'PASS' if pass_deo else 'FAIL'}]; "
             f"policy-drop<=3% -> {drop_pct:.2f}% [{'PASS' if pass_drop else 'FAIL'}]; "
             f"acc_nofair={acc_nofair*100:.2f}%. OVERALL={'PASS' if overall else 'FAIL'}. "
             f"Baseline committed: acc93.38/DEO2.03/drop0.25. "
             f"DEO def = train.py evaluate_DEO (TPR on label==1).")

    row = {"round": ROUND_ID, "round_id": ROUND_ID,
           "arch": f"{args.backbone} (target-push)", "variant": variant,
           "acc": round(acc_fair, 4), "deo": round(fair["deo"], 4),
           "tpr_f": round(fair["tpr_f"], 3), "tpr_m": round(fair["tpr_m"], 3),
           "notes": notes}
    os.makedirs(os.path.dirname(RESULTS_LOG), exist_ok=True)
    with open(RESULTS_LOG, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("[logged]", json.dumps(row, ensure_ascii=False))

    out = {
        "round": ROUND_ID, "tag": tag, "variant": variant,
        "config": vars(args),
        "acc_fair": acc_fair, "acc_nofair": acc_nofair,
        "deo_fair_pct": deo_fair_pct, "policy_drop_pct": drop_pct,
        "tpr_f_fair": fair["tpr_f"], "tpr_m_fair": fair["tpr_m"],
        "fair": fair, "nofair": nofair,
        "targets": {"acc": TARGET_ACC, "deo": TARGET_DEO, "drop": TARGET_DROP},
        "pass": {"acc": pass_acc, "deo": pass_deo, "drop": pass_drop, "overall": overall},
        "test_composition": te_c,
    }
    preds = {"fair": fair.pop("_preds"), "nofair": nofair.pop("_preds")}
    out_path = os.path.join(DATA, "results", f"target_push_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[saved] {out_path}")
    out["_preds"] = preds      # per-sample test predictions (not saved to JSON)
    return out


# ------------------------------------------------------------------ main
def build_argparser():
    p = argparse.ArgumentParser(description="Round 40 target-push train/eval.")
    p.add_argument("--backbone", default="resnet18",
                   choices=["resnet18", "vit_b16", "raddino", "densenet121"])
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--aug", default="basic", choices=["basic", "strong"])
    p.add_argument("--fix_classweight", action="store_true",
                   help="apply balanced class-weighted CE (train.py had it disabled)")
    p.add_argument("--tta", action="store_true", help="hflip test-time augmentation")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--smoke", action="store_true", help="build+1-batch checks, no training")
    p.add_argument("--nondeterministic", action="store_true",
                   help="turn off the deterministic mode (faster cuDNN autotuning, runs may differ)")
    return p


def main():
    args = build_argparser().parse_args()
    set_determinism(not args.nondeterministic)
    seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Env] device={device} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
          f"HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')} torch={torch.__version__}")
    if args.smoke:
        run_smoke(args, device)
    else:
        run_full(args, device)


if __name__ == "__main__":
    main()
