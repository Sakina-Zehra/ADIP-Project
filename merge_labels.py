# """
# merge_labels.py  --  merge 4 labellers' severity sheets into one file for step4_severity.py.

# Strategy (confidence-weighted aggregation; see chat for the literature rationale):
#   1. Per image, the continuous severity target is a CONFIDENCE-WEIGHTED MEAN of the 4 raw scores
#      (default weight = the confidence value itself: confidence 1/2/3 -> weight 1/2/3, so a confident
#      rating counts for more than an unsure one, but no rating is dropped to zero).
#   2. The class target (1/2/3) is that weighted mean, rounded.
#   3. Images where the 4 raw scores disagree by MORE than --disagree-threshold (default 1 point) are
#      flagged high_disagreement=True. step4_severity.py excludes these from WEIGHT FITTING (so one
#      ambiguous image can't tilt the formula) but still scores them on val/test, so you can see how
#      the model handles genuinely ambiguous cases rather than hiding them.
#   4. Pairwise weighted (quadratic) kappa between every pair of the 4 raters is computed and reported --
#      this is your reliability ceiling: no automatic score should be expected to beat how well your own
#      labellers agree with each other.

# REQUIRED INPUT: 4 sheets (.xlsx or .csv), each with an image_id column, a severity-score column and a
# confidence column. If your column headers differ from "image_id" / "severity" / "confidence", pass
# --score-col / --conf-col / --id-col.

# USAGE:
#     python merge_labels.py sheet1.xlsx sheet2.xlsx sheet3.xlsx sheet4.xlsx
#     python merge_labels.py s1.xlsx s2.xlsx s3.xlsx s4.xlsx --score-col "Severity Score" --conf-col "Confidence"
#     python merge_labels.py s1.xlsx s2.xlsx s3.xlsx s4.xlsx --out severity_labels.csv --disagree-threshold 1

# OUTPUT: severity_labels.csv (by default) -- feed this straight to step4_severity.py with --labels.
# Also writes outputs/label_merge_summary.txt and outputs/label_rater_kappa.csv.
# """
# import argparse, os, sys, itertools
# import numpy as np, pandas as pd
# from sklearn.metrics import cohen_kappa_score
# import pothole_common as pc


# def load_sheet(path, id_col, score_col, conf_col):
#     df = pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv(path)
#     for c in (id_col, score_col, conf_col):
#         if c not in df.columns:
#             sys.exit(f"{path}: column '{c}' not found (columns present: {list(df.columns)}). "
#                      "Pass --id-col / --score-col / --conf-col to match your sheet's headers.")
#     df = df[[id_col, score_col, conf_col]].rename(columns={id_col: "image_id", score_col: "score", conf_col: "conf"})
#     df["image_id"] = df["image_id"].astype(str).str.strip()
#     df["score"] = pd.to_numeric(df["score"], errors="coerce")
#     df["conf"] = pd.to_numeric(df["conf"], errors="coerce")
#     bad_s = df.score.notna() & ~df.score.isin([1, 2, 3])
#     bad_c = df.conf.notna() & ~df.conf.isin([1, 2, 3])
#     if bad_s.any():
#         print(f"!! {path}: {bad_s.sum()} score values are not 1/2/3, set to NaN (examples: {df.loc[bad_s, 'score'].unique()[:5]})")
#         df.loc[bad_s, "score"] = np.nan
#     if bad_c.any():
#         print(f"!! {path}: {bad_c.sum()} confidence values are not 1/2/3, set to NaN (examples: {df.loc[bad_c, 'conf'].unique()[:5]})")
#         df.loc[bad_c, "conf"] = np.nan
#     dup = df.image_id.duplicated()
#     if dup.any():
#         print(f"!! {path}: {dup.sum()} duplicate image_id rows, keeping the first occurrence of each")
#         df = df[~dup]
#     return df.dropna(subset=["image_id"]).set_index("image_id")


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("sheets", nargs=4, help="exactly 4 sheet paths (.xlsx or .csv)")
#     ap.add_argument("--id-col", default="image_id")
#     ap.add_argument("--score-col", default="severity")
#     ap.add_argument("--conf-col", default="confidence")
#     ap.add_argument("--out", default="severity_labels.csv")
#     ap.add_argument("--disagree-threshold", type=float, default=1.0,
#                     help="raw-score spread above this -> flagged high_disagreement (excluded from weight FITTING only)")
#     args = ap.parse_args()
#     os.makedirs(pc.OUT, exist_ok=True)

#     sheets = [load_sheet(p, args.id_col, args.score_col, args.conf_col) for p in args.sheets]
#     all_ids = set(pc.load_index().image_id)
#     for p, s in zip(args.sheets, sheets):
#         missing_from_mapping = set(s.index) - all_ids
#         if missing_from_mapping:
#             print(f"!! {p}: {len(missing_from_mapping)} image_id values not found in mapping.csv, e.g. {list(missing_from_mapping)[:5]} "
#                  "-- check --id-col / that this sheet uses the same image_id as labelling.py")
#         if len(s) != len(all_ids):
#             print(f"!! {p}: {len(s)} rows, but mapping.csv has {len(all_ids)} images -- this sheet looks incomplete")

#     common = set.intersection(*(set(s.index) for s in sheets))
#     print(f"{len(common)} images present in ALL 4 sheets (out of {len(all_ids)} total images)")
#     common = sorted(common)
#     S = np.array([sheets[k].loc[common, "score"].values for k in range(4)])     # 4 x n
#     Cw = np.array([sheets[k].loc[common, "conf"].values for k in range(4)])     # 4 x n, used as the weight directly

#     valid = ~np.isnan(S) & ~np.isnan(Cw)
#     w = np.where(valid, Cw, 0.0)
#     wsum = w.sum(0)
#     if (wsum == 0).any():
#         bad = [common[i] for i in np.flatnonzero(wsum == 0)]
#         sys.exit(f"{len(bad)} images have no valid (score, confidence) pair across all 4 sheets, e.g. {bad[:5]} -- fix those rows first")
#     target_cont = np.where(valid, S, 0).sum(0) / wsum                            # confidence-weighted mean
#     target_class = np.clip(np.round(target_cont), 1, 3).astype(int)
#     spread = np.nanmax(S, axis=0) - np.nanmin(S, axis=0)
#     high_disagree = spread > args.disagree_threshold

#     out = pd.DataFrame({"image_id": common, "target_cont": target_cont, "target_class": target_class,
#                         "disagreement_spread": spread, "high_disagreement": high_disagree})
#     for k in range(4):
#         out[f"score_{k+1}"] = S[k]
#         out[f"conf_{k+1}"] = Cw[k]
#     out.to_csv(args.out, index=False)
#     print(f"[ok] wrote {args.out}: {len(out)} images, class counts {dict(pd.Series(target_class).value_counts().sort_index())}, "
#          f"{int(high_disagree.sum())} flagged high_disagreement (spread > {args.disagree_threshold})")

#     # ---------------- pairwise weighted kappa = your reliability ceiling
#     rows = []
#     for i, j in itertools.combinations(range(4), 2):
#         ok = ~np.isnan(S[i]) & ~np.isnan(S[j])
#         k = cohen_kappa_score(S[i, ok], S[j, ok], weights="quadratic") if ok.sum() > 5 else float("nan")
#         rows.append(dict(rater_a=i + 1, rater_b=j + 1, n=int(ok.sum()), weighted_kappa=round(float(k), 3)))
#     K = pd.DataFrame(rows)
#     K.to_csv(os.path.join(pc.OUT, "label_rater_kappa.csv"), index=False)
#     print("\npairwise weighted kappa (reliability ceiling for the severity model):")
#     print(K.to_string(index=False))
#     print(f"average pairwise kappa: {K.weighted_kappa.mean():.3f}")
#     with open(os.path.join(pc.OUT, "label_merge_summary.txt"), "w", encoding="utf-8") as f:
#         f.write(f"{len(out)} images merged from 4 sheets\nclass counts: {dict(pd.Series(target_class).value_counts().sort_index())}\n"
#                f"high_disagreement (spread>{args.disagree_threshold}): {int(high_disagree.sum())}\n\n"
#                f"pairwise weighted kappa:\n{K.to_string(index=False)}\naverage: {K.weighted_kappa.mean():.3f}\n")


# if __name__ == "__main__":
#     main()
"""
merge_labels.py  --  merge several labellers' severity sheets into one file for step4_severity.py.

STRATEGY (rank-normalised, confidence-weighted aggregation):
Investigating this project's own 4 labellers showed their disagreement is mostly a SCALE OFFSET, not
random noise: one rater used "severe" liberally, another almost never used it, even though both ranked
individual potholes against each other fairly similarly (positive Spearman correlation between every
pair). Averaging raw 1/2/3 scores lets the strict rater's near-constant "1" drown out the others. The
fix used here:
  1. Within EACH rater's own sheet, convert their raw 1/2/3 score to a percentile RANK (0-1) relative to
     that rater's own 1000 scores (tied scores share the average rank). This puts every rater on the same
     "where does this fall in MY OWN usage of the scale" footing, regardless of how liberally they used
     "3". A rater who only ever gives "3" to their most severe 3% of images still has those images land
     near the top of the 0-1 scale -- their rare "3" is no longer outvoted just because they used it less.
  2. Average the 4 raters' normalised ranks, weighted by that rater's stated CONFIDENCE for that image
     (confidence 1/2/3 used directly as the weight, so a confident rating counts for more).
  3. Rescale the combined 0-1 score back to a 1-3 continuous target (for readability).
  4. Convert the continuous target to a 1/2/3 class using thresholds chosen so the overall class
     PROPORTIONS match the average of what the 4 raters actually used (not a forced even 33/33/33 split).
  5. Images where the 4 raters' normalised ranks disagree by more than --disagree-threshold (default 0.5,
     on the 0-1 scale) are flagged high_disagreement. step4_severity.py excludes these from weight
     FITTING only -- they are still scored on val/test.
  6. Reports TWO reliability numbers, because they answer different questions:
       - pairwise weighted kappa on RAW scores: the classic reliability figure, but it is pulled down by
         the scale offset above (a rater who is consistently one notch stricter looks "unreliable" to
         kappa even if their ordering of images is sound).
       - pairwise Spearman correlation on RAW scores: ignores the scale offset, so it reflects whether
         raters agree on RELATIVE severity, which is what the rank-normalisation above actually uses.

REQUIRED INPUT: one sheet (.xlsx or .csv) per labeller, each with an image_id column, a severity-score
column and a confidence column (headers default to "image_id" / "severity" / "confidence"; override with
--id-col / --score-col / --conf-col if yours differ). Works with any number of labellers (2 or more), not
just 4.

USAGE:
    python merge_labels.py sheet1.xlsx sheet2.xlsx sheet3.xlsx sheet4.xlsx
    python merge_labels.py s1.xlsx s2.xlsx s3.xlsx s4.xlsx --out severity_labels.csv --disagree-threshold 0.5

OUTPUT: severity_labels.csv (by default) -- feed this straight to step4_severity.py with --labels.
Also writes outputs/label_merge_summary.txt, outputs/label_rater_kappa.csv, outputs/label_rater_spearman.csv.
"""
import argparse, os, sys, itertools
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import cohen_kappa_score
import pothole_common as pc


def load_sheet(path, id_col, score_col, conf_col):
    df = pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv(path)
    for c in (id_col, score_col, conf_col):
        if c not in df.columns:
            sys.exit(f"{path}: column '{c}' not found (columns present: {list(df.columns)}). "
                     "Pass --id-col / --score-col / --conf-col to match your sheet's headers.")
    df = df[[id_col, score_col, conf_col]].rename(columns={id_col: "image_id", score_col: "score", conf_col: "conf"})
    df["image_id"] = df["image_id"].astype(str).str.strip()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["conf"] = pd.to_numeric(df["conf"], errors="coerce")
    bad_s = df.score.notna() & ~df.score.isin([1, 2, 3])
    bad_c = df.conf.notna() & ~df.conf.isin([1, 2, 3])
    if bad_s.any():
        print(f"!! {path}: {bad_s.sum()} score values are not 1/2/3, set to NaN (examples: {df.loc[bad_s, 'score'].unique()[:5]})")
        df.loc[bad_s, "score"] = np.nan
    if bad_c.any():
        print(f"!! {path}: {bad_c.sum()} confidence values are not 1/2/3, set to NaN (examples: {df.loc[bad_c, 'conf'].unique()[:5]})")
        df.loc[bad_c, "conf"] = np.nan
    dup = df.image_id.duplicated()
    if dup.any():
        print(f"!! {path}: {dup.sum()} duplicate image_id rows, keeping the first occurrence of each")
        df = df[~dup]
    n_na = df.score.isna().sum()
    if n_na:
        print(f"!! {path}: {n_na} rows have no usable score -> this rater will be excluded for those images")
    return df.dropna(subset=["image_id"]).set_index("image_id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sheets", nargs="+", help="2 or more sheet paths (.xlsx or .csv), one per labeller")
    ap.add_argument("--id-col", default="image_id")
    ap.add_argument("--score-col", default="severity")
    ap.add_argument("--conf-col", default="confidence")
    ap.add_argument("--out", default="severity_labels.csv")
    ap.add_argument("--disagree-threshold", type=float, default=0.5,
                    help="spread of the NORMALISED 0-1 ranks above this -> flagged high_disagreement "
                        "(excluded from weight FITTING only)")
    args = ap.parse_args()
    if len(args.sheets) < 2:
        sys.exit("need at least 2 sheets")
    os.makedirs(pc.OUT, exist_ok=True)
    R = len(args.sheets)

    sheets = [load_sheet(p, args.id_col, args.score_col, args.conf_col) for p in args.sheets]
    all_ids = set(pc.load_index().image_id)
    for p, s in zip(args.sheets, sheets):
        missing_from_mapping = set(s.index) - all_ids
        if missing_from_mapping:
            print(f"!! {p}: {len(missing_from_mapping)} image_id values not found in mapping.csv, e.g. {list(missing_from_mapping)[:5]} "
                 "-- check --id-col / that this sheet uses the same image_id as labelling.py")
        if len(s) != len(all_ids):
            print(f"!! {p}: {len(s)} rows, but mapping.csv has {len(all_ids)} images -- this sheet looks incomplete")

    common = sorted(set.intersection(*(set(s.index) for s in sheets)))
    print(f"{len(common)} images present in ALL {R} sheets (out of {len(all_ids)} total images)")
    if len(common) < 10:
        sys.exit("fewer than 10 images in common across all sheets -- check image_id matches between sheets")

    S = np.array([sheets[k].loc[common, "score"].values for k in range(R)], dtype=float)   # R x n, raw 1/2/3
    Cw = np.array([sheets[k].loc[common, "conf"].values for k in range(R)], dtype=float)   # R x n, confidence (weight)
    n = len(common)

    # ---------------- step 1: rank-normalise each rater's scores to their OWN 0-1 usage distribution
    Rk = np.full_like(S, np.nan)
    for k in range(R):
        v = S[k]
        ok = ~np.isnan(v)
        if ok.sum() >= 2:
            Rk[k, ok] = (rankdata(v[ok], method="average") - 0.5) / ok.sum()    # in (0, 1)
    assert np.nanmin(Rk) >= 0 and np.nanmax(Rk) <= 1, "normalised ranks escaped [0,1] -- bug, stop"

    # ---------------- step 2: confidence-weighted average of the normalised ranks
    valid = ~np.isnan(Rk) & ~np.isnan(Cw)
    w = np.where(valid, Cw, 0.0)
    wsum = w.sum(0)
    if (wsum == 0).any():
        bad = [common[i] for i in np.flatnonzero(wsum == 0)]
        sys.exit(f"{len(bad)} images have no valid (score, confidence) pair across all sheets, e.g. {bad[:5]} -- fix those rows first")
    combined01 = np.where(valid, Rk * w, 0).sum(0) / wsum                       # THE weighted sum, divided by weight sum
    assert combined01.min() >= -1e-9 and combined01.max() <= 1 + 1e-9, "combined score escaped [0,1] -- bug, stop"

    # ---------------- step 3: rescale 0-1 -> 1-3 for a readable continuous target
    target_cont = 1.0 + 2.0 * combined01

    # ---------------- step 4: class thresholds chosen to match the AVERAGE of the raters' own class proportions
    props = np.array([[(S[k] == c).sum() / np.isfinite(S[k]).sum() for c in (1, 2, 3)] for k in range(R)])
    avg_props = props.mean(0)                                                   # [p(class1), p(class2), p(class3)]
    avg_props = avg_props / avg_props.sum()
    cut1, cut2 = np.cumsum(avg_props)[:2]
    t1, t2 = np.quantile(combined01, [cut1, cut2])
    if t1 >= t2:                                                                # degenerate (e.g. one class ~0%) -> fall back
        t1, t2 = np.quantile(combined01, [1 / 3, 2 / 3])
    target_class = np.digitize(combined01, [t1, t2]) + 1

    # ---------------- step 5: disagreement flag, on the NORMALISED ranks (not raw scores, which conflate
    # scale-offset with real disagreement -- see the docstring)
    spread = np.nanmax(Rk, axis=0) - np.nanmin(Rk, axis=0)
    high_disagree = spread > args.disagree_threshold

    out = pd.DataFrame({"image_id": common, "target_cont": target_cont, "target_class": target_class,
                        "normalised_rank_spread": spread, "high_disagreement": high_disagree})
    for k in range(R):
        out[f"score_{k+1}"] = S[k]
        out[f"conf_{k+1}"] = Cw[k]
        out[f"rank_{k+1}"] = Rk[k]
    out.to_csv(args.out, index=False)
    print(f"\n[ok] wrote {args.out}: {len(out)} images")
    print(f"average rater class proportions used for thresholds: class1={avg_props[0]:.2f} class2={avg_props[1]:.2f} class3={avg_props[2]:.2f}")
    print(f"resulting merged class counts: {dict(pd.Series(target_class).value_counts().sort_index())}")
    print(f"{int(high_disagree.sum())} images flagged high_disagreement (normalised-rank spread > {args.disagree_threshold})")

    # ---------------- reliability: raw-score kappa (scale-sensitive) AND raw-score Spearman (scale-robust)
    krows, srows = [], []
    for i, j in itertools.combinations(range(R), 2):
        ok = ~np.isnan(S[i]) & ~np.isnan(S[j])
        k = cohen_kappa_score(S[i, ok], S[j, ok], weights="quadratic") if ok.sum() > 5 else float("nan")
        rho = spearmanr(S[i, ok], S[j, ok])[0] if ok.sum() > 5 else float("nan")
        krows.append(dict(rater_a=i + 1, rater_b=j + 1, n=int(ok.sum()), weighted_kappa=round(float(k), 3)))
        srows.append(dict(rater_a=i + 1, rater_b=j + 1, n=int(ok.sum()), spearman=round(float(rho), 3)))
    K, SP = pd.DataFrame(krows), pd.DataFrame(srows)
    K.to_csv(os.path.join(pc.OUT, "label_rater_kappa.csv"), index=False)
    SP.to_csv(os.path.join(pc.OUT, "label_rater_spearman.csv"), index=False)
    print("\npairwise weighted kappa on RAW scores (scale-sensitive -- penalised by any rater using the scale "
         "more strictly/leniently than the others, even if their ordering of images agrees):")
    print(K.to_string(index=False))
    print(f"average: {K.weighted_kappa.mean():.3f}")
    print("\npairwise Spearman correlation on RAW scores (scale-robust -- this is what the rank-normalised "
         "merge above actually relies on):")
    print(SP.to_string(index=False))
    print(f"average: {SP.spearman.mean():.3f}")
    with open(os.path.join(pc.OUT, "label_merge_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"{len(out)} images merged from {R} sheets (rank-normalised, confidence-weighted)\n"
               f"average rater class proportions: {avg_props.round(3).tolist()}\n"
               f"merged class counts: {dict(pd.Series(target_class).value_counts().sort_index())}\n"
               f"high_disagreement (normalised spread>{args.disagree_threshold}): {int(high_disagree.sum())}\n\n"
               f"pairwise weighted kappa (raw scores):\n{K.to_string(index=False)}\naverage: {K.weighted_kappa.mean():.3f}\n\n"
               f"pairwise Spearman (raw scores):\n{SP.to_string(index=False)}\naverage: {SP.spearman.mean():.3f}\n")


if __name__ == "__main__":
    main()