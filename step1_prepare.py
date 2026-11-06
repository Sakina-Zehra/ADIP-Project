"""
step1_prepare.py  --  Step 1: check the data, verify the depth module, create the frozen train/val/test split.

Run once (from the folder that contains labelling.py, mapping.csv and 'PUBLIC POTHOLE DATASET'):
    python step1_prepare.py
Re-running does NOT overwrite splits.csv (it is frozen); use --force only if you really want a new split.

Outputs (folder 'outputs'):
    dataset_report.csv   one row per image (sizes, polygon counts, mask area, depth validity, depth reliability ...)
    dataset_report.txt   the summary printed on screen
    qc_gt_overlay.jpg    12 random images with the ground-truth outline -> LOOK AT IT
    splits.csv           (in the project folder) image_id, group, split
"""
import argparse, os, sys, collections
import numpy as np, cv2, pandas as pd
import pothole_common as pc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite an existing splits.csv")
    ap.add_argument("--seed", type=int, default=pc.SEED)
    args = ap.parse_args()
    os.makedirs(pc.OUT, exist_ok=True)
    log = []

    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        log.append(s)

    pc.selftest()
    say("[ok] metric self-test passed")
    df = pc.load_index()
    say(f"[ok] mapping.csv: {len(df)} images, all rgb/depth/label files exist")

    # ---------------------------------------------------------------- per-image checks
    rows = []
    for k, r in df.iterrows():
        rgb, (w0, h0) = pc.load_rgb(r.rgb_path)
        d = pc.load_depth_cm(r.depth_path)
        m, npoly, bad = pc.gt_mask(r.label_path)
        area = int(m.sum())
        ys, xs = np.nonzero(m)
        touch = bool(area and (xs.min() == 0 or ys.min() == 0 or xs.max() == pc.W - 1 or ys.max() == pc.H - 1))
        ncomp = int(cv2.connectedComponents(m)[0] - 1) if area else 0
        rel, ok, rms, reason = pc.rel_depth(d, m)
        st = pc.depth_stats(rel, m)
        _, spike = pc.clean_depth(d)
        rows.append(dict(
            image_id=r.image_id, rgb_w=w0, rgb_h=h0, n_polygons=npoly, bad_label_lines=bad,
            gt_area_frac=area / (pc.W * pc.H), gt_components=ncomp, gt_touches_border=touch,
            depth_invalid_frac=float(np.isnan(d).mean()), depth_spike_frac=spike,
            depth_ok=ok, depth_reason=reason, road_fit_rms=rms, depth_p90_gt=st["p90"],
            depth_valid_in_mask=st["valid_frac"], map_ok=r.get("depth_reliable", ""), map_rms=r.get("road_fit_rms_cm", "")))
        if (k + 1) % 200 == 0:
            print(f"  checked {k + 1}/{len(df)}")
    rep = pd.DataFrame(rows)
    rep.to_csv(os.path.join(pc.OUT, "dataset_report.csv"), index=False)

    # ---------------------------------------------------------------- summary + red flags
    say("\n=== DATASET ===")
    sizes = collections.Counter(zip(rep.rgb_w, rep.rgb_h))
    say("RGB original sizes (w,h):", dict(sizes), "-> all are resized to 640x480; polygons are normalised so they stay aligned")
    say(f"images with NO polygon (empty label): {(rep.n_polygons == 0).sum()}   <- these have no ground truth and are excluded from Dice/IoU")
    say(f"images with >1 polygon: {(rep.n_polygons > 1).sum()} ; with >1 separate blob in the mask: {(rep.gt_components > 1).sum()}")
    say(f"label lines skipped (too short / odd): {int(rep.bad_label_lines.sum())}  (should be 0)")
    a = rep.gt_area_frac[rep.n_polygons > 0]
    say("GT mask area as % of image, percentiles 1/10/50/90/99:",
        " / ".join(f"{100 * np.percentile(a, p):.1f}" for p in (1, 10, 50, 90, 99)))
    say(f"GT touching the image border: {int(rep.gt_touches_border.sum())}")
    say(f"tiny GT (<0.3% of image): {(a < 0.003).sum()}   huge GT (>45% of image): {(a > 0.45).sum()}")
    say(f"depth invalid pixels per image, median / 90th pct: {100 * rep.depth_invalid_frac.median():.1f}% / {100 * rep.depth_invalid_frac.quantile(.9):.1f}%")
    say(f"depth spike pixels removed by clean_depth, median: {100 * rep.depth_spike_frac.median():.2f}%")

    say("\n=== DEPTH MODULE vs labelling.py (this must agree) ===")
    nflag = int((~rep.depth_ok).sum())
    say(f"depth flagged unreliable now: {nflag} of {len(rep)}", dict(collections.Counter(rep.depth_reason[~rep.depth_ok])))
    if (rep.map_ok != "").all():
        mine = np.where(rep.depth_ok, "yes", "no")
        disagree = rep.image_id[mine != rep.map_ok].tolist()
        rms_bad = []
        for i, r in rep.iterrows():
            try:
                if abs(float(r.map_rms) - float(r.road_fit_rms)) > 0.011:
                    rms_bad.append(r.image_id)
            except (ValueError, TypeError):
                pass
        say(f"reliable-flag disagrees with mapping.csv on {len(disagree)} images {disagree[:10]}")
        say(f"road-fit rms differs by >0.01 cm on {len(rms_bad)} images {rms_bad[:10]}")
        if disagree or rms_bad:
            say("  !! The depth module does not reproduce labelling.py for these images. Most likely cause: "
                "MAX_AREA_FRAC in pothole_common.py differs from the value labelling.py was last run with. "
                "Fix that BEFORE trusting any depth number.")
        else:
            say("  [ok] depth module reproduces labelling.py exactly")
    dc = rep.depth_p90_gt[rep.depth_ok]
    say(f"depth below road inside GT (90th pct), reliable images, percentiles 10/50/90: "
        f"{np.nanpercentile(dc, 10):.1f} / {np.nanpercentile(dc, 50):.1f} / {np.nanpercentile(dc, 90):.1f} cm")
    weak = rep[(rep.depth_ok) & (rep.depth_p90_gt < 0.5)].image_id.tolist()
    say(f"reliable depth but GT region NOT deeper than road (p90 < 0.5 cm): {len(weak)} images {weak[:10]}"
        "  <- either genuinely flat potholes or an RGB/depth/label mismatch; look at a few")

    # ---------------------------------------------------------------- split (frozen)
    say("\n=== SPLIT ===")
    if os.path.exists(pc.SPLITS) and not args.force:
        sp = pd.read_csv(pc.SPLITS)
        say(f"[ok] splits.csv already exists -> kept unchanged ({len(sp)} rows). Use --force to rebuild.")
        if set(sp.image_id) != set(df.image_id):
            # FIX: this was a silent warning before — it must be a hard exit.
            # Continuing with a stale split silently trains/evaluates on wrong images.
            import sys
            sys.exit("ERROR: splits.csv image_ids do not match mapping.csv. "
                     "Dataset has changed since the split was created. "
                     "Re-run with --force to rebuild the split.")
    else:
        sp, gap = pc.make_splits(df, seed=args.seed)
        sp.to_csv(pc.SPLITS, index=False)
        say(f"[new] splits.csv written (group = images taken within {gap}s of each other stay together)")
    g = sp.groupby("group").size()
    say(f"groups: {len(g)}, largest group: {g.max()} images")
    mm = sp.merge(rep, on="image_id")
    for s in ("train", "val", "test"):
        x = mm[mm.split == s]
        say(f"  {s:5s}: {len(x):4d} images | median GT area {100 * x.gt_area_frac.median():.1f}% | "
            f"depth unreliable {100 * (~x.depth_ok).mean():.0f}% | multi-blob GT {int((x.gt_components > 1).sum())}")
    # duplicate-timestamp pairs must stay together
    key = df.rgb.str.split("_").str[:2].str.join("_")
    both = sp.merge(df[["image_id"]].assign(key=key), on="image_id")
    leak = both.groupby("key").split.nunique()
    say(f"duplicate-timestamp pairs split across sets: {int((leak > 1).sum())}  (must be 0)")

    # ---------------------------------------------------------------- QC picture
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(df), size=min(12, len(df)), replace=False)
    tiles = []
    for k in pick:
        r = df.iloc[k]
        rgb, _ = pc.load_rgb(r.rgb_path)
        m, _, _ = pc.gt_mask(r.label_path)
        tiles.append(pc.overlay(rgb, gt=m, text=r.image_id, scale=0.5))
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    pc.imwrite(os.path.join(pc.OUT, "qc_gt_overlay.jpg"), grid)
    say(f"\n[ok] wrote {os.path.join(pc.OUT, 'qc_gt_overlay.jpg')}  -> open it: every green outline must sit on a pothole.")
    with open(os.path.join(pc.OUT, "dataset_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log))


if __name__ == "__main__":
    main()
