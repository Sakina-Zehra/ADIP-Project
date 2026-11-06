"""
step4_severity.py  --  Step 4: severity SCORE (size + depth + roughness + stones + water), fitted and
                        validated against your merged human labels.

What this does, in order:
  1. Loads your merged labels and the Step-3 feature files.
  2. Builds 5 "pillars" (size, depth, roughness, stones, water) by rank-normalising raw features against
     TRAIN only (so val/test never leak into the normalisation).
  3. Fits pillar WEIGHTS on TRAIN, two ways:
       - "fixed"  : equal weights (0.2 each) -- the simple, proposal-style baseline, no learning.
       - "tuned"  : weights chosen to maximise agreement with your TRAIN labels (Spearman rank correlation).
     Class thresholds (continuous score -> 1/2/3) are fit on TRAIN to maximise weighted kappa.
  4. Evaluates BOTH weight sets on VAL (or TEST with --test), TWICE each:
       - using features from the GROUND-TRUTH masks  (upper bound: "if segmentation were perfect")
       - using features from the PREDICTED masks     (the real, fully-automatic pipeline)
     The same weights/thresholds (fit on GT-mask TRAIN features only) are applied to both, so the GAP
     between the two tells you how much segmentation error is costing you.
  5. Ablation: refits "tuned" with each pillar removed, so you can see which pillars your TRAIN labels
     actually reward (e.g. does removing the water pillar hurt agreement, as your rubric intends?).

================================================================================================
REQUIRED INPUT: a merged labels file  (CSV or XLSX), default path  severity_labels.csv
  - one row per image_id
  - one column per labeller holding their 1/2/3 score. Any column whose name starts with "score"
    (case-insensitive) is treated as one labeller's score, e.g.  score_a, score_b  (any number >=1).
  - optional: columns starting with "conf" hold that labeller's confidence (1-3). If present, a row
    where EVERY labeller's confidence is 1 is excluded from TRAIN fitting (still scored on val/test).
  - You must build this file yourself from your two labelling sheets, e.g.:
        import pandas as pd
        a = pd.read_excel("labels_A.xlsx")[["image_id", "score", "confidence"]].rename(
                columns={"score": "score_a", "confidence": "conf_a"})
        b = pd.read_excel("labels_B.xlsx")[["image_id", "score", "confidence"]].rename(
                columns={"score": "score_b", "confidence": "conf_b"})
        a.merge(b, on="image_id").to_csv("severity_labels.csv", index=False)
    (adjust the column names on the right of the rename() calls to match your actual sheet headers)

USAGE (after step3_features.py has produced features_gt_train / features_gt_val / features_pred_val):
    python step4_severity.py                     # fit on train, evaluate on val
    python step4_severity.py --test               # FINAL: fit on train+val, evaluate on test ONCE
    python step4_severity.py --labels my_labels.csv

OUTPUTS (folder 'outputs'):
    severity_config.json          the fitted pillar weights + class thresholds (both "fixed" and "tuned")
    severity_results_<split>.csv  the main table: agreement for fixed/tuned x gt-mask/pred-mask
    severity_ablation_<split>.csv which pillar matters how much, per your labels
    severity_scores_<split>.csv   per-image predicted continuous score + class, both mask types
    severity_confusion_<split>.png
================================================================================================
"""
import argparse, os, sys, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr, kendalltau
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score   # moved from inside functions to avoid repeated import overhead
import pothole_common as pc

PILLARS = {                      # pillar name -> (raw feature column, direction: +1 = higher raw = more severe)
    # FIX: use diam_cm_or_px (added in step3) so size pillar works when depth is unreliable
    # (diam_cm is NaN when depth fails -> pillar silently defaulted to 0.5; diam_cm_or_px uses pixel fallback)
    "size":  ("diam_cm_or_px", +1),
    "depth": ("depth_p90", +1),
    "rough": ("rough_lap_in", +1),
    "stone": ("stone_dens_in", +1),
    "water": ("water_cue", +1),
}
PNAMES = list(PILLARS.keys())
N_CLASSES = 3                    # labels are 1, 2, 3


# ====================================================================== labels
def load_labels(path):
    """Two supported formats:
    (a) PRE-MERGED (preferred): a file produced by merge_labels.py, already containing target_cont,
        target_class and high_disagreement -- used as-is, no re-aggregation here.
    (b) RAW (fallback, for quick testing without running merge_labels.py first): a file with one
        score_* column per labeller, averaged/medianed here with no confidence weighting."""
    if not os.path.exists(path):
        sys.exit(f"{path} not found. Run merge_labels.py first (see its docstring), or see this file's "
                 "docstring for the simpler fallback format.")
    df = pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv(path)
    if "image_id" not in df.columns:
        sys.exit(f"{path} has no 'image_id' column (columns found: {list(df.columns)})")
    if {"target_cont", "target_class"}.issubset(df.columns):
        print(f"labels: using pre-merged target_cont/target_class from {path} (from merge_labels.py)")
        df["target_class"] = df["target_class"].astype(int)
        df["low_conf_only"] = df["high_disagreement"].astype(bool) if "high_disagreement" in df.columns else False
        if "high_disagreement" in df.columns:
            print(f"{int(df.high_disagreement.sum())} images flagged high_disagreement -> excluded from weight FITTING only")
        return df[["image_id", "target_cont", "target_class", "low_conf_only"]]
    score_cols = [c for c in df.columns if c.lower().startswith("score")]
    conf_cols = [c for c in df.columns if c.lower().startswith("conf")]
    if not score_cols:
        sys.exit(f"{path} has no target_cont/target_class (merge_labels.py output) and no column starting "
                 "with 'score' either (columns found: {list(df.columns)}).")
    print(f"labels: no target_cont/target_class found -> using the RAW fallback (simple mean/median, "
         "NOT confidence-weighted). Run merge_labels.py for the proper confidence-weighted aggregation.")
    for c in score_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        bad = df[c].notna() & ~df[c].isin([1, 2, 3])
        if bad.any():
            print(f"!! {bad.sum()} rows in column '{c}' are not 1/2/3 and will be ignored for that column "
                  f"(examples: {df.loc[bad, c].unique()[:5]})")
            df.loc[bad, c] = np.nan
    df["n_labellers"] = df[score_cols].notna().sum(axis=1)
    df = df[df.n_labellers > 0].copy()
    df["target_cont"] = df[score_cols].mean(axis=1)
    df["target_class"] = df[score_cols].median(axis=1).round().clip(1, 3).astype(int)
    df["low_conf_only"] = False
    if conf_cols:
        for c in conf_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["min_conf"] = df[conf_cols].min(axis=1, skipna=True)
        df["low_conf_only"] = df["min_conf"] <= 1
    if len(score_cols) >= 2:
        agree = df[score_cols].apply(lambda r: r.dropna().nunique() <= 1, axis=1)
        print(f"labeller agreement: identical score in {100 * agree.mean():.0f}% of {len(df)} rows "
              f"across columns {score_cols}")
    return df[["image_id", "target_cont", "target_class", "low_conf_only"]]


# ====================================================================== pillars
def build_pillars(feat_df, ref=None):
    """ref=None: this IS the reference (train) -> returns (pillar_df, ref). ref given: map onto the train
    distribution via percentile lookup, so val/test never influence the normalisation."""
    out = pd.DataFrame(index=feat_df.index)
    ref_out = {} if ref is None else None
    for name, (col, sign) in PILLARS.items():
        if col not in feat_df.columns:
            sys.exit(f"feature file has no column '{col}' needed for pillar '{name}'. Was it produced by the "
                     "current step3_features.py?")
        v = feat_df[col].astype(float) * sign
        if ref is None:
            finite = np.sort(v[np.isfinite(v)].values)
            ref_out[name] = finite if len(finite) >= 5 else None
            base = finite
        else:
            base = ref[name]
        if base is None or len(base) < 5:
            out[name] = 0.5
            continue
        pr = np.interp(v.values, base, np.linspace(0, 1, len(base)), left=0.0, right=1.0)
        pr = np.where(np.isfinite(v.values), pr, np.nan)
        fill = 0.5 if ref is None else float(np.nanmedian(out[name]) if name in out else 0.5)
        out[name] = np.where(np.isnan(pr), 0.5, pr)          # missing feature (e.g. depth unreliable) -> neutral 0.5
    return (out, ref_out) if ref is None else out


# ====================================================================== weights + thresholds
def softmax_w(z, mask=None):
    z = np.asarray(z, float)
    if mask is not None:
        z = np.where(mask, z, -1e9)
    e = np.exp(z - z.max())
    return e / e.sum()


def fit_weights(P, y, active=None, restarts=8, seed=0):
    """Maximise Spearman(score, y) over the simplex of pillar weights (restricted to `active` pillars)."""
    idx = [i for i, n in enumerate(PNAMES) if active is None or n in active]
    mask = np.array([n in (active or PNAMES) for n in PNAMES])
    X = P[PNAMES].values
    rng = np.random.default_rng(seed)

    def neg_rho(z):
        w = softmax_w(z, mask)
        s = X @ w
        if np.std(s) < 1e-9:
            return 0.0
        return -spearmanr(s, y)[0]

    best = None
    for r in range(restarts):
        z0 = rng.normal(0, 1.5, len(PNAMES)) if r else np.zeros(len(PNAMES))
        res = minimize(neg_rho, z0, method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-4, maxiter=400))
        if best is None or res.fun < best.fun:
            best = res
    w = softmax_w(best.x, mask)
    return {n: float(w[i]) for i, n in enumerate(PNAMES)}, -float(best.fun)


def fit_thresholds(score, y_class, n_classes=N_CLASSES):
    """Two cut points on `score` that maximise quadratic weighted kappa against y_class (1..n_classes)."""
    qs = np.percentile(score, np.arange(5, 96, 3))
    cands = sorted(set(np.round(qs, 4)))
    best = (-1, (np.median(score) - 0.1, np.median(score) + 0.1))
    for i, t1 in enumerate(cands):
        for t2 in cands[i + 1:]:
            pred = np.digitize(score, [t1, t2]) + 1
            k = cohen_kappa_score(y_class, pred, weights="quadratic")
            if k > best[0]:
                best = (k, (t1, t2))
    return list(best[1]), best[0]


def apply_formula(P, weights, thresholds):
    score = (P[PNAMES].values * np.array([weights[n] for n in PNAMES])).sum(1)
    cls = np.digitize(score, thresholds) + 1
    return score, cls


# ====================================================================== evaluation
def agreement(score, cls, target_cont, target_class):
    rho = spearmanr(score, target_cont)[0]
    tau = kendalltau(score, target_cont)[0]
    kap = cohen_kappa_score(target_class, cls, weights="quadratic")
    acc = float((cls == target_class).mean())
    acc1 = float((np.abs(cls - target_class) <= 1).mean())
    return dict(n=len(score), spearman=round(float(rho), 3), kendall_tau=round(float(tau), 3),
               weighted_kappa=round(float(kap), 3), exact_acc=round(acc, 3), within1_acc=round(acc1, 3))


def confusion_png(cls, target_class, path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(target_class, cls, labels=[1, 2, 3])
        fig, ax = plt.subplots(figsize=(3.5, 3.2))
        im = ax.imshow(cm, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels([1, 2, 3]); ax.set_yticks([0, 1, 2]); ax.set_yticklabels([1, 2, 3])
        ax.set_xlabel("predicted"); ax.set_ylabel("labelled"); ax.set_title(title, fontsize=9)
        fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    except Exception as ex:
        print("(confusion plot skipped:", ex, ")")


# ====================================================================== run
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="severity_labels.csv")
    ap.add_argument("--test", action="store_true", help="final evaluation on the test split (use once)")
    args = ap.parse_args()
    os.makedirs(pc.OUT, exist_ok=True)
    eval_name = "test" if args.test else "val"
    fit_name = "train+val" if args.test else "train"
    fit_splits = ["train", "val"] if args.test else ["train"]

    lab = load_labels(args.labels)
    print(f"labels: {len(lab)} images, class counts {dict(lab.target_class.value_counts().sort_index())}, "
         f"{int(lab.low_conf_only.sum())} excluded from fitting as low-confidence")

    def load_feat(kind, split):
        p = os.path.join(pc.OUT, f"features_{kind}_{split}.csv")
        if not os.path.exists(p):
            sys.exit(f"{p} not found. Run step3_features.py --split {split}"
                     f"{'' if kind == 'gt' else ' (after step2_segment.py run' + (' --test' if split == 'test' else '') + ')'} first.")
        return pd.read_csv(p)

    gt_fit = pd.concat([load_feat("gt", s) for s in fit_splits], ignore_index=True)
    fit_df = gt_fit.merge(lab, on="image_id").query("not low_conf_only")
    if len(fit_df) < 20:
        sys.exit(f"only {len(fit_df)} labelled images with ground-truth features available to fit on -- check that "
                 "image_ids in your labels file match mapping.csv, and that step3_features.py has been run for "
                 f"{fit_splits}.")
    print(f"fitting on {fit_name}: {len(fit_df)} labelled images with ground-truth-mask features")
    P_fit, ref = build_pillars(fit_df)

    w_fixed = {n: 1.0 / len(PNAMES) for n in PNAMES}
    s_fixed_fit, _ = apply_formula(P_fit, w_fixed, [0, 0])           # placeholder thresholds, refit below
    t_fixed, k_fixed = fit_thresholds(s_fixed_fit, fit_df.target_class.values)
    w_tuned, rho_fit = fit_weights(P_fit, fit_df.target_cont.values)
    s_tuned_fit, _ = apply_formula(P_fit, w_tuned, [0, 0])
    t_tuned, k_tuned = fit_thresholds(s_tuned_fit, fit_df.target_class.values)
    print(f"\nFIXED (equal) weights: {dict((k, round(v, 3)) for k, v in w_fixed.items())}")
    print(f"TUNED weights        : {dict((k, round(v, 3)) for k, v in w_tuned.items())}  (train Spearman {rho_fit:.3f})")
    json.dump(dict(fixed=dict(weights=w_fixed, thresholds=t_fixed), tuned=dict(weights=w_tuned, thresholds=t_tuned),
                   fit_split=fit_name, pillars={k: v[0] for k, v in PILLARS.items()}),
             open(os.path.join(pc.OUT, "severity_config.json"), "w"), indent=2)

    # ---------------- evaluate on GT-mask features (upper bound) and predicted-mask features (real pipeline)
    gt_ev = load_feat("gt", eval_name).merge(lab, on="image_id")
    pred_ev = load_feat("pred", eval_name).merge(lab, on="image_id")
    both = gt_ev.merge(pred_ev[["image_id"]], on="image_id")        # only images present in both, for a fair gap
    gt_ev = gt_ev[gt_ev.image_id.isin(both.image_id)].reset_index(drop=True)
    pred_ev = pred_ev[pred_ev.image_id.isin(both.image_id)].reset_index(drop=True)
    print(f"\nevaluating on {eval_name}: {len(gt_ev)} labelled images present in both mask types")

    rows, scores_out = [], []
    for wname, w, t in (("fixed", w_fixed, t_fixed), ("tuned", w_tuned, t_tuned)):
        for mkind, mdf in (("ground_truth_mask", gt_ev), ("predicted_mask", pred_ev)):
            Pe = build_pillars(mdf, ref)
            score, cls = apply_formula(Pe, w, t)
            r = agreement(score, cls, mdf.target_cont.values, mdf.target_class.values)
            r.update(weights=wname, mask=mkind)
            rows.append(r)
            if wname == "tuned":
                scores_out.append(pd.DataFrame(dict(image_id=mdf.image_id, mask=mkind, score=score, pred_class=cls,
                                                     target_class=mdf.target_class, target_cont=mdf.target_cont)))
                confusion_png(cls, mdf.target_class.values, os.path.join(pc.OUT, f"severity_confusion_{eval_name}_{mkind}.png"),
                             f"{eval_name} / {mkind} / tuned")
    R = pd.DataFrame(rows)[["weights", "mask", "n", "spearman", "kendall_tau", "weighted_kappa", "exact_acc", "within1_acc"]]
    pd.set_option("display.width", 160)
    print("\n=== SEVERITY AGREEMENT WITH YOUR LABELS ===")
    print(R.to_string(index=False))
    R.round(3).to_csv(os.path.join(pc.OUT, f"severity_results_{eval_name}.csv"), index=False)
    gap = R[(R.weights == "tuned")].set_index("mask").spearman
    if "ground_truth_mask" in gap and "predicted_mask" in gap:
        print(f"\nsegmentation cost (tuned weights): Spearman drops by {gap['ground_truth_mask'] - gap['predicted_mask']:.3f} "
             "going from ground-truth masks to our automatic predicted masks")
    pd.concat(scores_out, ignore_index=True).round(4).to_csv(os.path.join(pc.OUT, f"severity_scores_{eval_name}.csv"), index=False)

    # ---------------- ablation: does each pillar earn its place, per YOUR labels?
    print(f"\n=== ABLATION (tuned weights, fit on {fit_name} GT-mask features, evaluated on {eval_name} GT-mask features) ===")
    abl = []
    w_full, rho_full = fit_weights(P_fit, fit_df.target_cont.values)
    s_full, _ = apply_formula(P_fit, w_full, [0, 0])
    t_full, _ = fit_thresholds(s_full, fit_df.target_class.values)
    Pe_gt = build_pillars(gt_ev, ref)
    se_full, ce_full = apply_formula(Pe_gt, w_full, t_full)
    r_full = agreement(se_full, ce_full, gt_ev.target_cont.values, gt_ev.target_class.values)
    abl.append(dict(removed="(none, full model)", train_spearman=round(rho_full, 3), **{f"val_{k}": v for k, v in r_full.items()}))
    for drop in PNAMES:
        active = [n for n in PNAMES if n != drop]
        w_d, rho_d = fit_weights(P_fit, fit_df.target_cont.values, active=active)
        s_d, _ = apply_formula(P_fit, w_d, [0, 0])
        t_d, _ = fit_thresholds(s_d, fit_df.target_class.values)
        se_d, ce_d = apply_formula(Pe_gt, w_d, t_d)
        r_d = agreement(se_d, ce_d, gt_ev.target_cont.values, gt_ev.target_class.values)
        abl.append(dict(removed=drop, train_spearman=round(rho_d, 3), **{f"val_{k}": v for k, v in r_d.items()}))
    A = pd.DataFrame(abl)[["removed", "train_spearman", "val_spearman", "val_weighted_kappa"]]
    print(A.to_string(index=False))
    print("(a pillar whose removal barely changes val_spearman is not earning its weight in your labels; "
         "a large drop means your labellers DO reward that cue)")
    A.round(3).to_csv(os.path.join(pc.OUT, f"severity_ablation_{eval_name}.csv"), index=False)


if __name__ == "__main__":
    main()
