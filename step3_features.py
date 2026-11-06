"""
step3_features.py  --  Step 3: severity FEATURES (size, depth, roughness/stones, water cues).

Computes the same features from (a) the ground-truth masks and (b) our predicted masks, then measures how
well the predicted-mask features match the ground-truth ones. No severity formula yet.

Usage (after step2_segment.py run, which saves outputs/pred_masks_val.npz):
    python step3_features.py --split val                    # GT + predicted features and the fidelity table
    python step3_features.py --split all --gt-only          # GT features for all images (for the severity step)
    add  --workers 4  to use several CPU cores

Outputs (folder 'outputs'):
    features_gt_<split>.csv, features_pred_<split>.csv, feature_fidelity_<split>.csv
    water_cue_top_<split>.jpg, stone_top_<split>.jpg   -> LOOK AT THEM: do the top-ranked images really show water / stones?
"""
import argparse, os, sys
import numpy as np, cv2, pandas as pd
from scipy.stats import spearmanr
from concurrent.futures import ProcessPoolExecutor
import pothole_common as pc

MAX_STONE_CM = 15.0      # largest stone/rock the blob detector looks for (diameter)
STONE_MIN_CM2 = 3.0      # blobs smaller than this are noise
STONE_MAX_CM2 = 400.0    # blobs larger than this are not single stones
BIG_STONE_CM = 8.0       # "big rock" = blob with equivalent diameter at least this
MIN_PX_DEPTH = 200       # valid depth pixels needed for depth roughness

GROUP_ORDER = ["area", "n_blobs", "length", "width", "diam", "z_cm", "depth", "rough", "stone", "big_stone",
               "smooth", "chroma", "dark", "d_a", "d_b", "spec"]


def ell(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(k), int(k)))


def _design(xs, ys):
    x = xs / pc.W - 0.5
    y = ys / pc.H - 0.5
    return np.c_[x * x, x * y, y * y, x, y, np.ones_like(x)]


def compute_features(rgb, d, m):
    """rgb: BGR 480x640, d: depth cm (NaN invalid) 480x640, m: mask {0,1} 480x640 -> dict of floats."""
    m = (m > 0).astype(np.uint8)
    area = int(m.sum())
    f = dict(area_px=float(area), area_frac=area / (pc.W * pc.H), n_blobs=float(cv2.connectedComponents(m)[0] - 1))
    if area < 50:
        return f
    inner = cv2.dilate(m, np.ones((21, 21), np.uint8))
    outer = cv2.dilate(m, np.ones((91, 91), np.uint8))
    ring = (outer == 1) & (inner == 0)
    zr = d[ring & ~np.isnan(d)]
    z = float(np.median(zr)) if zr.size >= 200 else float("nan")
    cpp = pc.cm_per_pixel(z) if np.isfinite(z) else float("nan")     # cm per pixel at road level
    f["z_cm"] = z
    # ---------------- size
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rw, rh = cv2.minAreaRect(max(cs, key=cv2.contourArea))[1]
    f["area_cm2"] = area * cpp ** 2
    f["length_cm"] = max(rw, rh) * cpp
    f["width_cm"] = min(rw, rh) * cpp
    f["diam_cm"] = 2 * np.sqrt(f["area_cm2"] / np.pi)
    # FIX: pixel-based size fallback — used by step4 severity when depth is unreliable (diam_cm is NaN).
    # Not physically calibrated but correctly rank-orders potholes by size.
    diam_px = 2 * np.sqrt(area / np.pi)
    f["diam_px_norm"] = diam_px / pc.W                           # normalised by image width (0..1)
    # Combined feature: use cm if depth is reliable, otherwise fall back to pixel-normalised diameter
    f["diam_cm_or_px"] = f["diam_cm"] if np.isfinite(f["diam_cm"]) else f["diam_px_norm"] * 30.0  # ~scale to cm range
    # ---------------- depth
    rel, ok, rms, reason = pc.rel_depth(d, m)
    st = pc.depth_stats(rel, m)
    f.update(depth_ok=float(ok), depth_p50=st["p50"], depth_p90=st["p90"], depth_max=st["max"], depth_valid_frac=st["valid_frac"])
    core = cv2.erode(m, ell(5)) > 0
    if core.sum() < 0.3 * area:
        core = m > 0
    # deep_core: a second, size-SCALED erosion (not a fixed 5px) used only for the stone/water appearance
    # cues below. A predicted mask that is e.g. 50% too large still has ~5px of slack near its edge, so the
    # fixed "core" above still includes a lot of road right at the boundary; a radius-proportional erosion
    # pulls further in and keeps roughness/depth (which already tested "fair"/"OK") on the gentler "core".
    r_eq = float(np.sqrt(area / np.pi))
    dk = int(np.clip(0.30 * r_eq, 5, 40))
    deep_core = cv2.erode(m, ell(2 * dk + 1)) > 0
    if deep_core.sum() < 0.15 * area:          # mask too small/thin to erode that far -> fall back
        deep_core = core
    ys, xs = np.nonzero(core & ~np.isnan(rel))
    if len(xs) >= MIN_PX_DEPTH:
        zz = rel[ys, xs].astype(np.float64)
        A = _design(xs.astype(np.float64), ys.astype(np.float64))
        res = zz - A @ np.linalg.lstsq(A, zz, rcond=None)[0]
        f["depth_rough"] = float(res.std())                      # bumpiness of the pothole floor (ridges, stones)
        f["depth_range"] = float(np.percentile(zz, 90) - np.percentile(zz, 10))
    # ---------------- appearance: roughness, stones, water cues (inside vs road ring)
    ringb = ring
    if ringb.sum() < 200 or core.sum() < 50:
        return f
    lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)
    L, A_, B_ = cv2.split(lab)
    Lg = cv2.GaussianBlur(L, (0, 0), 1.5)
    lap = np.abs(cv2.Laplacian(Lg, cv2.CV_32F))
    f["rough_lap_in"] = float(lap[core].mean())
    f["rough_lap_ratio"] = f["rough_lap_in"] / (float(lap[ringb].mean()) + 0.5)
    Lf = L.astype(np.float32)
    mu = cv2.blur(Lf, (9, 9))
    sd = np.sqrt(np.maximum(cv2.blur(Lf * Lf, (9, 9)) - mu * mu, 0))
    f["smooth_ratio"] = float(sd[deep_core].mean() / (sd[ringb].mean() + 0.5))   # low = smoother than the road (water, mud)
    a, b = A_.astype(np.float32), B_.astype(np.float32)
    f["chroma_std_in"] = float(np.sqrt(a[deep_core].var() + b[deep_core].var()))  # low = uniform colour
    f["dark_in"] = float((L[ringb].mean() - L[deep_core].mean()) / 255.0)
    f["pale_cue"] = float(np.clip(1.0 - abs(f["dark_in"]) * 4.0, 0.0, 1.0))   # 1 = looks about the same
                                                                                 # brightness as the road
                                                                                 # (pale/low-contrast); 0 = clearly
                                                                                 # darker or lighter than the road
    f["d_a"] = float((a[deep_core].mean() - a[ringb].mean()) / 255.0)
    f["d_b"] = float((b[deep_core].mean() - b[ringb].mean()) / 255.0)            # negative = bluer (sky reflection)
    f["spec_frac"] = float((L[deep_core] > np.percentile(L[ringb], 98)).mean())  # bright reflection patches
    if np.isfinite(cpp):
        r_px = int(np.clip(MAX_STONE_CM / 2 / cpp, 4, 40))
        k = ell(2 * r_px + 1)
        tops = [cv2.morphologyEx(Lg, cv2.MORPH_TOPHAT, k), cv2.morphologyEx(Lg, cv2.MORPH_BLACKHAT, k)]   # bright / dark blobs

        def thr(t):
            v = t[ringb].astype(np.float32)
            return max(8.0, float(np.median(v) + 3 * 1.4826 * np.median(np.abs(v - np.median(v)))))
        blob = (tops[0] > thr(tops[0])) | (tops[1] > thr(tops[1]))

        def stones(region):
            n, lab_, st_, _ = cv2.connectedComponentsWithStats((blob & region).astype(np.uint8), connectivity=8)
            a_cm2 = st_[1:, cv2.CC_STAT_AREA] * cpp ** 2
            keep = (a_cm2 >= STONE_MIN_CM2) & (a_cm2 <= STONE_MAX_CM2)
            reg_cm2 = max(float(region.sum()) * cpp ** 2, 1e-6)
            diam = 2 * np.sqrt(a_cm2[keep] / np.pi)
            return int(keep.sum()), 100.0 * keep.sum() / reg_cm2, float(a_cm2[keep].sum() / reg_cm2), int((diam >= BIG_STONE_CM).sum())
        n_in, dens_in, frac_in, big_in = stones(deep_core)        # stones counted in the shrunken, more-trustworthy region
        _, dens_r, _, _ = stones(ringb)
        f.update(stone_n_in=float(n_in), stone_dens_in=dens_in, stone_frac_in=frac_in, big_stone_n_in=float(big_in),
                 stone_dens_ratio=dens_in / (dens_r + 0.05))
    return f


def job(args):
    iid, rgb_path, depth_path, mask_src, kind = args
    cv2.setNumThreads(1)
    try:
        rgb, _ = pc.load_rgb(rgb_path)
        d = pc.load_depth_cm(depth_path)
        if kind == "gt":
            m = pc.gt_mask(mask_src)[0]
        else:
            m = np.unpackbits(mask_src)[:pc.H * pc.W].reshape(pc.H, pc.W)
        out = compute_features(rgb, d, m)
        out["image_id"] = iid
        return kind, out
    except Exception as ex:
        return kind, dict(image_id=iid, error=repr(ex))


def add_water_cue(F):
    """Rank-average of two cues that actually correlate with labelled severity in this dataset:
    low smooth_ratio (smoother than the road) and HIGH chroma_std_in (more colour variation inside).
    chroma_std_in's sign is flipped from the original design: univariate testing against the real
    labels showed MORE colour variation -> MORE severe, the opposite of the "uniform = water" assumption.
    spec_frac (bright reflections) is dropped: it showed ~zero correlation with severity (Spearman -0.01),
    so it was adding noise, not signal."""
    parts = []
    for col, sign in (("smooth_ratio", -1), ("chroma_std_in", +1)):
        if col in F:
            parts.append((F[col] * sign).rank(pct=True))
    F["water_cue"] = pd.concat(parts, axis=1).mean(axis=1) if parts else np.nan
    return F


def fidelity(G, P):
    rows = []
    feats = [c for c in G.columns if c not in ("image_id", "error") and c in P.columns]
    for c in feats:
        g, p = G[c].astype(float), P[c].astype(float)
        ok = g.notna() & p.notna()
        if c.startswith("depth_") and c not in ("depth_ok", "depth_valid_frac"):
            ok &= (G["depth_ok"] > 0) & (P["depth_ok"] > 0)
        if ok.sum() < 5 or g[ok].nunique() < 2:
            continue
        diff = (p - g)[ok]
        rel = (diff.abs() / (g[ok].abs() + 1e-6))
        rho = spearmanr(g[ok], p[ok])[0]
        rows.append(dict(feature=c, n=int(ok.sum()), spearman=round(float(rho), 3), median_rel_err=round(float(rel.median()), 3),
                         within25pct=round(float((rel <= 0.25).mean()), 2), bias=round(float(diff.median()), 4),
                         verdict="OK" if rho >= 0.9 else ("fair" if rho >= 0.75 else "WEAK")))
    T = pd.DataFrame(rows)
    T["o"] = [next((i for i, g_ in enumerate(GROUP_ORDER) if r.startswith(g_)), 99) for r in T.feature]
    return T.sort_values(["o", "feature"]).drop(columns="o")


def montage(ids, paths, masks, title, out):
    tiles = []
    for i, txt in ids:
        rgb, _ = pc.load_rgb(paths.at[i, "rgb_path"])
        tiles.append(pc.overlay(rgb, gt=masks[i], text=f"{i} {txt}", scale=0.5))
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    pc.imwrite(out, np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val", "test", "all"])
    ap.add_argument("--gt-only", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    df, sp = pc.load_index(), pd.read_csv(pc.SPLITS)
    paths = df.set_index("image_id")
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    gt_only = args.gt_only or args.split in ("train", "all")
    if gt_only and not args.gt_only:
        print("(predicted masks exist only for val/test -> computing ground-truth features only)")
    ids = list(sp.image_id[sp.split.isin(splits)])
    os.makedirs(pc.OUT, exist_ok=True)
    jobs = [(i, paths.at[i, "rgb_path"], paths.at[i, "depth_path"], paths.at[i, "label_path"], "gt") for i in ids]
    if not gt_only:
        f = os.path.join(pc.OUT, f"pred_masks_{args.split}.npz")
        if not os.path.exists(f):
            sys.exit(f"{f} not found. Run  python step2_segment.py run{' --test' if args.split == 'test' else ''}  first.")
        z = np.load(f)
        for i, pk in zip(z["ids"], z["packed"]):
            jobs.append((str(i), paths.at[str(i), "rgb_path"], paths.at[str(i), "depth_path"], pk, "pred"))
    res = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for n, r in enumerate(ex.map(job, jobs, chunksize=4), 1):
                res.append(r)
                if n % 100 == 0:
                    print(f"  {n}/{len(jobs)}")
    else:
        for n, j in enumerate(jobs, 1):
            res.append(job(j))
            if n % 50 == 0:
                print(f"  {n}/{len(jobs)}")
    errs = [r for k, r in res if "error" in r]
    if errs:
        print(f"!! {len(errs)} images failed:", [(e['image_id'], e['error']) for e in errs[:3]])
    G = pd.DataFrame([r for k, r in res if k == "gt" and "error" not in r])
    G = add_water_cue(G)
    G.round(4).to_csv(os.path.join(pc.OUT, f"features_gt_{args.split}.csv"), index=False)
    print(f"[ok] ground-truth features: {len(G)} images -> features_gt_{args.split}.csv")
    gt_masks = {i: pc.gt_mask(paths.at[i, "label_path"])[0] for i in G.image_id}
    if "water_cue" in G:
        top = G.sort_values("water_cue", ascending=False).head(12)
        montage([(r.image_id, f"water {r.water_cue:.2f}") for r in top.itertuples()], paths, gt_masks, "water", os.path.join(pc.OUT, f"water_cue_top_{args.split}.jpg"))
    if "stone_dens_in" in G:
        top = G.sort_values("stone_dens_in", ascending=False).head(12)
        montage([(r.image_id, f"stones {r.stone_dens_in:.2f}/100cm2") for r in top.itertuples()], paths, gt_masks, "stones", os.path.join(pc.OUT, f"stone_top_{args.split}.jpg"))
    print("[ok] wrote water_cue_top / sto-ne_top pictures (12 highest-ranked images with the true outline)")
    if "pale_cue" in G:
        top = G.sort_values("pale_cue", ascending=False).head(12)
        montage([(r.image_id, f"pale {r.pale_cue:.2f}") for r in top.itertuples()], paths, gt_masks, "pale", os.path.join(pc.OUT, f"pale_top_{args.split}.jpg"))
    if gt_only:
        return
    P = pd.DataFrame([r for k, r in res if k == "pred" and "error" not in r])
    P = add_water_cue(P)
    P.round(4).to_csv(os.path.join(pc.OUT, f"features_pred_{args.split}.csv"), index=False)
    M = G.merge(P, on="image_id", suffixes=("_g", "_p"))
    Gm = M[["image_id"] + [c + "_g" for c in G.columns if c != "image_id"]].rename(columns=lambda c: c[:-2] if c.endswith("_g") else c)
    Pm = M[["image_id"] + [c + "_p" for c in P.columns if c != "image_id"]].rename(columns=lambda c: c[:-2] if c.endswith("_p") else c)
    T = fidelity(Gm, Pm)
    T.to_csv(os.path.join(pc.OUT, f"feature_fidelity_{args.split}.csv"), index=False)
    pd.set_option("display.width", 200)
    print(f"\n=== FEATURES FROM PREDICTED MASKS vs FROM GROUND-TRUTH MASKS ({len(Gm)} images, {args.split}) ===")
    print("spearman >= 0.90 = OK (ranks images like the true masks do), 0.75-0.90 = fair, below = WEAK")
    print(T.to_string(index=False))
    key = T[T.feature.isin(["area_cm2", "length_cm", "depth_p90", "rough_lap_ratio", "stone_dens_in", "smooth_ratio"])]
    print(f"\nheadline features OK: {int((key.verdict == 'OK').sum())} of {len(key)}")


if __name__ == "__main__":
    main()