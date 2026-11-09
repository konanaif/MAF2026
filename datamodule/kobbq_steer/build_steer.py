"""KoBBQ-Steer 스티어링 행렬 생성

실행: python -m MAF.datamodule.kobbq_steer.build_steer [--data-dir DIR] [--model-dir DIR] [--output DIR] [--device cuda:0]
산출: <model_dir>/acts/<run>/ (활성화, 약 26 GB), <model_dir>/steer/<run>/{directions.pt, alphasteer/*.pt}. GPU 필요
진행은 <output>/progress.log 에 기록한다.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from MAF.algorithms.inprocessing.kobbq_steer.alphasteer import (
    benign_cov,
    null_basis,
    solve_tilde_delta,
)
from MAF.algorithms.inprocessing.kobbq_steer.common import (
    get_logger,
    load_model,
    run_dir,
    save_json,
    seed_all,
)
from MAF.algorithms.inprocessing.kobbq_steer.directions import (
    add_pred_role,
    balanced_pos_neg,
    diff_in_means,
    layer_cos,
    probe_curve,
)
from MAF.algorithms.inprocessing.kobbq_steer.eval_kmmlu import shots_by_subject
from MAF.algorithms.inprocessing.kobbq_steer.extract import (
    extract,
    load_layer,
    load_layers,
    load_meta,
)
from MAF.algorithms.inprocessing.kobbq_steer.plotting import SERIES, plt, save
from MAF.algorithms.inprocessing.kobbq_steer.prompts import (
    LETTERS,
    benign_prompt,
    kmmlu_prompt,
    kobbq_prompt,
)
from MAF.datamodule.kobbq_steer.prepare_data import make_config, write_config
from sklearn.metrics import roc_auc_score

EXTRACT_SETS = ("kobbq_trainval", "caa_unknown", "caa_stereo", "benign", "kmmlu_val")


def progress(output, msg):
    path = Path(output) / "progress.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")


def extract_acts(cfg, model, tok, sets=EXTRACT_SETS, log=print):
    """02: last-prompt-token / CAA answer-token activations for every set (resumable per set)."""
    P = cfg["paths"]
    proc = Path(P["processed"])
    L = model.config.num_hidden_layers
    layers = list(range(L))
    A = run_dir(cfg, "acts")
    E = cfg["extract"]

    kb = pd.read_parquet(proc / "kobbq.parquet")
    assert "test" not in set(kb[kb.split != "test"].split)
    for name in sets:
        if (A / name / "DONE").exists():
            log(f"skip {name} (DONE)")
            continue
        if name == "kobbq_trainval":
            meta = kb[kb.split.isin(["train", "val"])].reset_index(drop=True)
            texts = [kobbq_prompt(tok, r, cfg) for r in meta.to_dict("records")]
            letters = "ABC"
        elif name in ("caa_unknown", "caa_stereo"):
            meta = kb[(kb.split == "train") & (kb.context_type == "ambig")].reset_index(
                drop=True
            )
            role = name.split("_")[1]
            texts = [
                kobbq_prompt(
                    tok, r, cfg, answer_letter=LETTERS[r[f"{role}_option_idx"]]
                )
                for r in meta.to_dict("records")
            ]
            letters = "ABC"
        elif name == "benign":
            meta = pd.read_parquet(proc / "benign.parquet")
            texts = [benign_prompt(tok, t, cfg) for t in meta.text]
            letters = "ABC"  # scoring irrelevant here
        elif name == "kmmlu_val":
            meta = pd.read_parquet(proc / "kmmlu_val.parquet")
            shots = shots_by_subject(
                pd.read_parquet(proc / "kmmlu_dev.parquet"), cfg["kmmlu"]["n_shot"]
            )
            texts = [
                kmmlu_prompt(tok, r, shots[r["subject"]], cfg)
                for r in meta.to_dict("records")
            ]
            letters = "ABCD"
        else:
            raise ValueError(name)
        if name.startswith("caa"):
            # sanity: the captured position is the appended letter token
            ids = tok(texts[0], add_special_tokens=False).input_ids
            assert tok.decode(ids[-1]) in "ABC", tok.decode(ids[-3:])
        log(f"{name}: example prompt\n{texts[0]}")
        extract(
            model,
            tok,
            texts,
            meta,
            A / name,
            letters,
            layers,
            E["shard_size"],
            E["max_tokens_per_batch"],
            log,
        )
    return A


def build_directions(cfg, log=print):
    """04: r_amb (behavior, CAA), r_disamb, layer-wise probe, benign residual norms."""
    seed = cfg["seed"]
    seed_all(seed)
    A, S, R = run_dir(cfg, "acts"), run_dir(cfg, "steer"), run_dir(cfg, "results")

    meta = add_pred_role(load_meta(A / "kobbq_trainval"))
    L = int(
        torch.load(min((A / "kobbq_trainval").glob("shard_*.pt")), mmap=True)[
            "acts"
        ].shape[0]
    )
    layers = list(range(L))
    acts = load_layers(A / "kobbq_trainval", layers, torch.bfloat16)  # [L, n, d]
    log(f"acts {tuple(acts.shape)}")

    tr_amb = meta[(meta.split == "train") & (meta.context_type == "ambig")]
    log(
        "train-ambig pred roles: "
        + str(tr_amb.pred_role.value_counts().to_dict())
        + " (anti errors are excluded from D_pos/D_neg)"
    )
    pos_m = meta.pred_role.eq("unknown") & meta.index.isin(tr_amb.index)
    neg_m = meta.pred_role.eq("stereo") & meta.index.isin(tr_amb.index)
    pos, neg, tab = balanced_pos_neg(meta, pos_m, neg_m, seed=seed, log=log)
    tab.to_csv(R / "balance_r_amb.csv", index=False)
    log(
        f"r_amb behavior: D_pos={len(pos)} D_neg={len(neg)} (balanced per category x polarity)"
    )
    for nm, ix in (("pos", pos), ("neg", neg)):
        log(
            f"  letter of chosen option ({nm}): {meta.loc[ix, 'pred_idx'].value_counts().sort_index().to_dict()}"
        )
    r_beh = diff_in_means(acts, pos, neg)

    # CAA: one (unknown, stereo) pair per train-ambig prompt -> balanced by construction
    cu = load_layers(A / "caa_unknown", layers, torch.bfloat16)
    cs = load_layers(A / "caa_stereo", layers, torch.bfloat16)
    r_caa = (cu.double().mean(1) - cs.double().mean(1)).float()
    del cu, cs

    # r_disamb: disamb correct vs. stereo errors in counter-biased contexts (train)
    tr_dis = meta[(meta.split == "train") & (meta.context_type == "disambig")]
    pos_d = meta.index.isin(tr_dis.index) & (meta.pred_idx == meta.gold_idx)
    neg_d = (
        meta.index.isin(tr_dis.index)
        & meta.context_bias.eq("counter_biased")
        & meta.pred_role.eq("stereo")
    )
    pd_, nd_, tab_d = balanced_pos_neg(
        meta,
        pd.Series(pos_d, meta.index),
        pd.Series(neg_d, meta.index),
        seed=seed,
        log=log,
    )
    tab_d.to_csv(R / "balance_r_disamb.csv", index=False)
    log(f"r_disamb: D_pos={len(pd_)} D_neg={len(nd_)}")
    r_dis = diff_in_means(acts, pd_, nd_)

    # mean residual norm per layer on benign-train (CAA scaling)
    bmeta = load_meta(A / "benign")
    btr = np.where(bmeta.split.values == "train")[0]
    bacts = load_layers(A / "benign", layers, torch.bfloat16)
    mean_norm = torch.stack(
        [bacts[l, torch.as_tensor(btr)].float().norm(dim=-1).mean() for l in layers]
    )
    del bacts

    cos_bc = layer_cos(r_beh, r_caa)
    cos_ad = layer_cos(r_beh, r_dis)
    torch.save(
        {
            "r_behavior": r_beh,
            "r_caa": r_caa,
            "r_disamb": r_dis,
            "mean_norm": mean_norm,
            "n_pos": len(pos),
            "n_neg": len(neg),
        },
        S / "directions.pt",
    )

    # probe: train on balanced train-ambig (unknown=1 vs stereo=0), eval on val-ambig templates
    va = meta[
        (meta.split == "val")
        & (meta.context_type == "ambig")
        & meta.pred_role.isin(["unknown", "stereo"])
    ]
    tr_idx = np.concatenate([pos, neg])
    y_tr = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    y_va = (va.pred_role == "unknown").to_numpy().astype(int)
    probe = probe_curve(
        acts[:, torch.as_tensor(tr_idx)].float(),
        y_tr,
        acts[:, torch.as_tensor(va.index.to_numpy())].float(),
        y_va,
    )
    probe["r_norm_behavior"] = r_beh.norm(dim=-1).numpy()
    probe["r_norm_caa"] = r_caa.norm(dim=-1).numpy()
    probe["mean_norm_benign"] = mean_norm.numpy()
    probe["cos_behavior_caa"] = cos_bc
    probe["cos_amb_disamb"] = cos_ad
    probe.to_csv(R / "directions_by_layer.csv", index=False)
    log(
        f"val-ambig probe set: n={len(va)} unknown={y_va.sum()}\n{probe.round(3).to_string(index=False)}"
    )

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(layers, probe.auroc, color=SERIES[0], label="AUROC")
    ax[0].plot(layers, probe.acc, color=SERIES[1], label="accuracy")
    ax[0].set(
        title="Probe: unknown vs stereo (train→val templates)",
        xlabel="layer",
        ylim=(0.4, 1.0),
    )
    ax[0].legend()
    ax[1].plot(layers, cos_bc, color=SERIES[0], label="cos(r_behavior, r_CAA)")
    ax[1].plot(layers, cos_ad, color=SERIES[1], label="cos(r_amb, r_disamb)")
    ax[1].axhline(0, color="#a3a29d", lw=0.8)
    ax[1].set(title="Direction cosine by layer", xlabel="layer", ylim=(-1, 1))
    ax[1].legend()
    save(fig, R / "directions_probe_cos.png")
    save_json(
        {
            "n_pos": len(pos),
            "n_neg": len(neg),
            "n_pos_dis": len(pd_),
            "n_neg_dis": len(nd_),
            "train_amb_roles": tr_amb.pred_role.value_counts().to_dict(),
        },
        R / "directions_summary.json",
    )
    return S / "directions.pt"


def build_alphasteer(cfg, device, log=print, layers="all", hb_add_disamb=False):
    """05: null-space projection P and steering matrix Delta for every layer / p / alpha / r."""
    seed_all(cfg["seed"])
    A, S, R = run_dir(cfg, "acts"), run_dir(cfg, "steer"), run_dir(cfg, "results")
    tag = "alphasteer_hbdis" if hb_add_disamb else "alphasteer"
    out = S / tag
    out.mkdir(exist_ok=True)
    dev = torch.device(device)

    D = torch.load(S / "directions.pt")
    rk = {"behavior": D["r_behavior"], "caa": D["r_caa"]}
    L = D["r_behavior"].shape[0]
    layers = range(L) if layers == "all" else [int(x) for x in layers.split(",")]
    ac = cfg["alphasteer"]

    km = load_meta(A / "kobbq_trainval")
    bm = load_meta(A / "benign")
    kmmlu = A / "kmmlu_val"

    def ix(m):
        return torch.as_tensor(np.where(m)[0])

    i_tr_amb = ix((km.split == "train") & (km.context_type == "ambig"))
    i_va_amb = ix((km.split == "val") & (km.context_type == "ambig"))
    i_va_dis = ix((km.split == "val") & (km.context_type == "disambig"))
    i_tr_dis = ix((km.split == "train") & (km.context_type == "disambig"))
    i_b_tr, i_b_va = ix(bm.split == "train"), ix(bm.split == "val")
    log(
        f"H_b benign-train={len(i_b_tr)}{' + train disamb=' + str(len(i_tr_dis)) if hb_add_disamb else ''} "
        f"| H_m train-ambig={len(i_tr_amb)}"
    )

    rows = []
    for l in layers:
        t0 = time.time()
        kact = load_layer(A / "kobbq_trainval", l)
        bact = load_layer(A / "benign", l)
        Hb = [bact[i_b_tr]] + ([kact[i_tr_dis]] if hb_add_disamb else [])
        C, nb = benign_cov((x for h in Hb for x in h.split(2048)), bact.shape[1], dev)
        Hm = kact[i_tr_amb].to(dev, torch.float64)
        probes = {
            "benign_val": bact[i_b_va],
            "kobbq_val_amb": kact[i_va_amb],
            "kobbq_val_dis": kact[i_va_dis],
            "kmmlu_val": load_layer(kmmlu, l),
        }
        probes = {k: v.to(dev, torch.float64) for k, v in probes.items()}
        b_tr = bact[i_b_tr].to(dev, torch.float64)
        for p in ac["sweep_null_ratio"]:
            U, _evals = null_basis(C, p)

            def leak(h, U=U):
                return (h @ U).norm(dim=1) / h.norm(dim=1)

            leak_tr, leak_va = leak(b_tr), leak(probes["benign_val"])
            for a in ac["sweep_ridge"]:
                for rname, rv in rk.items():
                    r = rv[l].to(dev, torch.float64)
                    w, rr, info = solve_tilde_delta(Hm, U, r, a)
                    torch.save(
                        {
                            "w": w.float().cpu(),
                            "r": rr.float().cpu(),
                            "layer": l,
                            "p": p,
                            "alpha": a,
                            "r_kind": rname,
                            "n_b": nb,
                        },
                        out / f"L{l}_p{p}_a{a}_{rname}.pt",
                    )
                    g = {
                        k: (v @ w).abs() * r.norm() for k, v in probes.items()
                    }  # ||Delta h||
                    dm = (Hm @ w)[:, None] * r[None, :]
                    rows.append(
                        {
                            "layer": l,
                            "p": p,
                            "alpha": a,
                            "r_kind": rname,
                            **info,
                            "leak_benign_train_mean": float(leak_tr.mean()),
                            "leak_benign_train_max": float(leak_tr.max()),
                            "leak_benign_val_mean": float(leak_va.mean()),
                            "leak_benign_val_max": float(leak_va.max()),
                            "cos_mean_Dh_m_r": float(
                                torch.nn.functional.cosine_similarity(dm.mean(0), r, 0)
                            ),
                            "gain_train_amb_mean": float(
                                (Hm @ w).mean()
                            ),  # Delta h_m = gain * r; target gain = 1
                            "r_norm": float(r.norm()),
                            **{
                                f"dh_{k}_median": float(v.median())
                                for k, v in g.items()
                            },
                            "auroc_dh_amb_vs_benign": float(
                                roc_auc_score(
                                    np.r_[
                                        np.ones(len(g["kobbq_val_amb"])),
                                        np.zeros(len(g["benign_val"])),
                                    ],
                                    torch.cat([g["kobbq_val_amb"], g["benign_val"]])
                                    .cpu()
                                    .numpy(),
                                )
                            ),
                            "auroc_dh_amb_vs_kmmlu": float(
                                roc_auc_score(
                                    np.r_[
                                        np.ones(len(g["kobbq_val_amb"])),
                                        np.zeros(len(g["kmmlu_val"])),
                                    ],
                                    torch.cat([g["kobbq_val_amb"], g["kmmlu_val"]])
                                    .cpu()
                                    .numpy(),
                                )
                            ),
                        }
                    )
        log(
            f"layer {l} done in {time.time() - t0:.0f}s | "
            + str({k: round(v, 4) for k, v in rows[-1].items() if isinstance(v, float)})
        )
        pd.DataFrame(rows).to_csv(R / f"{tag}_diag.csv", index=False)
    return out


def build_steer(cfg, device="cuda:0"):
    if not torch.cuda.is_available():
        raise RuntimeError("build_steer needs a CUDA device")
    output = cfg["paths"]["results"]
    log = get_logger("build_steer", Path(output) / "logs/build_steer.log")
    seed_all(cfg["seed"])
    sets = ",".join(EXTRACT_SETS)
    progress(output, f"build_steer 1/3 extract activations (sets: {sets})")
    model, tok = load_model(cfg, device)
    extract_acts(cfg, model, tok, log=log.info)
    del model
    torch.cuda.empty_cache()
    progress(output, "build_steer 2/3 directions")
    directions = build_directions(cfg, log=log.info)
    progress(output, "build_steer 3/3 alphasteer")
    alphasteer = build_alphasteer(cfg, device, log=log.info)
    progress(output, "build_steer done")
    return {"directions": directions, "alphasteer": alphasteer}


def main(argv=None):
    ap = argparse.ArgumentParser(description="KoBBQ-Steer steering matrices")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)
    cfg = make_config(args.data_dir, args.model_dir, args.output)
    write_config(cfg, cfg["paths"]["results"])
    try:
        out = build_steer(cfg, args.device)
    except (RuntimeError, FileNotFoundError) as e:
        print(f"build_steer: {e}", file=sys.stderr)
        return 1
    print({k: str(v) for k, v in out.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
