"""
pothole_common.py  --  shared foundation for the pothole project (Step 1).

Everything that must behave IDENTICALLY in every later script lives here:
  * paths + index (reads the mapping.csv written by labelling.py, so image_id
    means the same image for the labellers, the splits and the models)
  * label/mask reading (YOLOv8 polygons -> binary mask)
  * depth loading, cleaning and the road-surface fit (copied from labelling.py)
  * grouped train/val/test split
  * Dice / IoU metrics

Do not edit constants marked "MUST MATCH labelling.py" unless you also change
labelling.py and regenerate the labelling images.
"""
import os, re, json
import numpy as np
import cv2
import pandas as pd

# ------------------------------------------------------------------ paths
BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "PUBLIC POTHOLE DATASET")
MAPPING = os.path.join(BASE, "mapping.csv")          # written by labelling.py
SPLITS = os.path.join(BASE, "splits.csv")            # written once by step1_prepare.py
OUT = os.path.join(BASE, "outputs")
CACHE = os.path.join(BASE, "cache")

W, H = 640, 480                                      # every image/mask/depth is handled at this size
SEED = 42

# ------------------------------------------------------------------ depth constants
# MUST MATCH labelling.py (so the depth module reproduces what the labellers saw)
RMS_MAX = 1.0            # cm, road-fit error above this = depth flagged unreliable
MIN_RING = 3000          # min road pixels around the pothole needed for the fit
MAX_AREA_FRAC = 0.45     # outline covering more of the image than this = depth unreliable
MIN_COVER = 0.7          # share of the road ring that must lie inside the image with valid depth
HFOV_DEG = 65.0          # assumed horizontal field of view (D415 spec) -> only used for cm conversions


# ------------------------------------------------------------------ safe image io (works with any Windows path)
def imread(path, flags=cv2.IMREAD_COLOR):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if img is None:
        raise IOError(f"could not read image: {path}")
    return img


def imwrite(path, img, quality=92):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise IOError(f"could not encode {path}")
    buf.tofile(path)


# ------------------------------------------------------------------ index
def load_index():
    """One row per image_id with absolute paths. mapping.csv is the single source of truth."""
    if not os.path.exists(MAPPING):
        raise FileNotFoundError(f"{MAPPING} not found. Put these scripts in the same folder as labelling.py "
                                "(the folder that contains mapping.csv).")
    df = pd.read_csv(MAPPING, dtype=str).fillna("")
    for c in ("image_id", "rgb", "depth", "label"):
        if c not in df.columns:
            raise ValueError(f"mapping.csv has no column '{c}' (columns: {list(df.columns)})")
    df["rgb_path"] = [os.path.join(ROOT, "images", x) for x in df["rgb"]]
    df["depth_path"] = [os.path.join(ROOT, "depths", x) for x in df["depth"]]
    df["label_path"] = [os.path.join(ROOT, "labels", x) for x in df["label"]]
    missing = [p for c in ("rgb_path", "depth_path", "label_path") for p in df[c] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} files listed in mapping.csv are missing, e.g. {missing[:3]}")
    if df["image_id"].duplicated().any():
        raise ValueError("duplicate image_id in mapping.csv")
    df["t"] = [parse_time(x) for x in df["rgb"]]
    return df.reset_index(drop=True)


def parse_time(name):
    """'20250227_135438_color_png.rf....jpg' -> seconds (float) or NaN."""
    m = re.match(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})", os.path.basename(name))
    if not m:
        return float("nan")
    y, mo, d, hh, mm, ss = map(int, m.groups())
    try:
        return float(pd.Timestamp(y, mo, d, hh, mm, ss).timestamp())
    except Exception:
        return float("nan")


# ------------------------------------------------------------------ labels / masks
def read_polygons(label_path, w=W, h=H):
    """YOLOv8-seg label file -> (list of int32 polygons in pixels, number of skipped lines).
    Same rules as labelling.py: a line needs class id + at least 3 points, otherwise it is skipped."""
    polys, bad = [], 0
    with open(label_path, "r") as f:
        lines = f.read().strip().splitlines()
    for line in lines:
        p = line.split()
        if not p:
            continue
        if len(p) < 7:
            bad += 1
            continue
        try:
            vals = np.array(p[1:], dtype=float)
        except ValueError:
            bad += 1
            continue
        if len(vals) % 2:                     # odd number of coordinates -> drop the stray one
            vals = vals[:-1]
            bad += 1
        xy = vals.reshape(-1, 2) * [w, h]
        polys.append(xy.astype(np.int32))     # truncation, identical to labelling.py
    return polys, bad


def gt_mask(label_path, w=W, h=H):
    """Union of all pothole polygons -> uint8 {0,1} mask. Returns (mask, n_polygons, n_bad_lines)."""
    polys, bad = read_polygons(label_path, w, h)
    m = np.zeros((h, w), np.uint8)
    for p in polys:
        cv2.fillPoly(m, [p], 1)
    return m, len(polys), bad


def load_rgb(path):
    """BGR image resized to 640x480 (as in labelling.py). Returns (img, original (w,h))."""
    img = imread(path)
    h0, w0 = img.shape[:2]
    if (w0, h0) != (W, H):
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    return img, (w0, h0)


# ------------------------------------------------------------------ depth
def load_depth_cm(path):
    """uint16 millimetres -> float32 cm, invalid (<=0) = NaN, always 480x640."""
    d = np.load(path).astype(np.float32) * 0.1
    d[d <= 0] = np.nan
    if d.shape != (H, W):
        d = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)
    return d


def clean_depth(d, spike_cm=3.0):
    """Proposal step 'clean up the depth data': invalid pixels are already NaN; additionally remove
    isolated spikes that differ from their 5x5 median by more than spike_cm. Returns (cleaned, spike_fraction).
    NOTE: the labellers saw RAW depth, so rel_depth() below defaults to raw."""
    if not np.isfinite(d).any():
        return d.copy(), 0.0
    filled = np.where(np.isnan(d), np.nanmedian(d), d).astype(np.float32)
    med = cv2.medianBlur(filled, 5)
    spikes = np.abs(filled - med) > spike_cm
    spikes &= ~np.isnan(d)
    out = d.copy()
    out[spikes] = np.nan
    return out, float(spikes.sum() / max(np.isfinite(d).sum(), 1))


def _design(xs, ys, w, h):
    x = xs / w - 0.5
    y = ys / h - 0.5
    return np.c_[x * x, x * y, y * y, x, y, np.ones_like(x)]


def _fit_surface(d, ring):
    """Quadratic road surface fitted to the ring of road around the pothole (robust to stones)."""
    ys, xs = np.nonzero(ring & ~np.isnan(d))
    if len(xs) < MIN_RING:
        return None
    if len(xs) > 20000:
        sel = np.random.default_rng(0).choice(len(xs), 20000, replace=False)
        ys, xs = ys[sel], xs[sel]
    h, w = d.shape
    A = _design(xs.astype(np.float64), ys.astype(np.float64), w, h)
    z = d[ys, xs].astype(np.float64)
    c = np.linalg.lstsq(A, z, rcond=None)[0]
    for _ in range(3):
        r = z - A @ c
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
        keep = np.abs(r) < 2.5 * s
        if keep.sum() < 500:
            break
        A, z = A[keep], z[keep]
        c = np.linalg.lstsq(A, z, rcond=None)[0]
    rms = float(np.sqrt(np.mean((z - A @ c) ** 2)))
    return c, rms


def rel_depth(d, m):
    """Depth relative to the road surface fitted AROUND mask m (works for GT masks and predicted masks).
    d: float cm (NaN = invalid), m: uint8 {0,1} mask, both 480x640.
    Returns (rel, ok, rms, reason). rel > 0 means deeper than the road. Logic = labelling.py."""
    h, w = d.shape
    m = (m > 0).astype(np.uint8)
    if m.sum() == 0:
        return np.full_like(d, np.nan), False, float("nan"), "empty mask"
    inner = cv2.dilate(m, np.ones((21, 21), np.uint8))
    outer = cv2.dilate(m, np.ones((91, 91), np.uint8))
    ring = (outer == 1) & (inner == 0)
    fit = _fit_surface(d, ring)
    if fit is None:
        med = np.nanmedian(d) if np.isfinite(d).any() else np.nan
        return d - med, False, float("nan"), "too little road around outline"
    c, rms = fit
    yy, xx = np.mgrid[0:h, 0:w]
    base = (_design(xx.ravel().astype(np.float64), yy.ravel().astype(np.float64), w, h) @ c).reshape(h, w)
    rel = (d - base).astype(np.float32)
    reason = ""
    if rms >= RMS_MAX:
        reason = "road surface fit poor"
    elif m.sum() > MAX_AREA_FRAC * h * w:
        reason = "outline too large"
    else:
        mp = cv2.copyMakeBorder(m, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=0)
        expected = int(((cv2.dilate(mp, np.ones((91, 91), np.uint8)) == 1) &
                        (cv2.dilate(mp, np.ones((21, 21), np.uint8)) == 0)).sum())
        coverage = int((ring & ~np.isnan(d)).sum()) / max(expected, 1)
        if coverage < MIN_COVER:
            reason = "road missing on several sides"
    return rel, reason == "", rms, reason


def depth_stats(rel, m):
    """Depth-below-road summary inside mask m (cm). Used later for severity features."""
    v = rel[(m > 0) & ~np.isnan(rel)]
    if v.size < 20:
        return dict(p50=np.nan, p90=np.nan, max=np.nan, valid_frac=0.0)
    return dict(p50=float(np.percentile(v, 50)), p90=float(np.percentile(v, 90)),
                max=float(v.max()), valid_frac=float(v.size / max((m > 0).sum(), 1)))


def cm_per_pixel(z_cm):
    """Approximate ground size of one pixel at camera distance z_cm (camera assumed to look straight down)."""
    return 2.0 * z_cm * np.tan(np.radians(HFOV_DEG / 2)) / W


# ------------------------------------------------------------------ split
def make_splits(df, seed=SEED, fracs=(0.70, 0.15, 0.15), gaps=(120, 30, 5)):
    """Grouped split: images captured within `gap` seconds of each other (and duplicate-timestamp
    pairs) stay together, so near-identical frames never sit in both train and test.
    Returns (df with columns group, split) and the gap that was used."""
    names = ("train", "val", "test")
    d = df[["image_id", "t"]].copy()
    n = len(d)
    used = None
    for gap in gaps:
        order = d.sort_values(["t", "image_id"], na_position="last").index.to_numpy()
        grp = np.zeros(n, int)
        g, prev = 0, None
        gid = {}
        for i in order:
            t = d.at[i, "t"]
            if prev is None or np.isnan(t) or np.isnan(prev) or (t - prev) > gap:
                g += 1
            gid[i] = g
            prev = t
        d["group"] = pd.Series(gid)
        biggest = d["group"].value_counts().max()
        used = gap
        if biggest <= max(10, 0.05 * n):
            break
    sizes = d["group"].value_counts()
    rng = np.random.default_rng(seed)
    gids = np.array(sizes.index)
    rng.shuffle(gids)
    gids = sorted(gids, key=lambda g_: -sizes[g_])        # stable: big groups first, random among equals
    targets = {s: f * n for s, f in zip(names, fracs)}
    counts = {s: 0 for s in names}
    assign = {}
    for g_ in gids:
        s = max(names, key=lambda s_: (targets[s_] - counts[s_]) / targets[s_])
        assign[g_] = s
        counts[s] += int(sizes[g_])
    d["split"] = d["group"].map(assign)
    return d[["image_id", "group", "split"]], used


# ------------------------------------------------------------------ metrics
def dice_iou(pred, gt):
    p = np.asarray(pred).astype(bool)
    g = np.asarray(gt).astype(bool)
    inter = float((p & g).sum())
    ps, gs = float(p.sum()), float(g.sum())
    if ps == 0 and gs == 0:
        return 1.0, 1.0
    return 2 * inter / (ps + gs), inter / (ps + gs - inter)


def summarize(dice, iou, tp=None, fp=None, fn=None):
    dice, iou = np.asarray(dice, float), np.asarray(iou, float)
    out = dict(n=len(dice), dice_mean=dice.mean(), dice_median=np.median(dice), iou_mean=iou.mean(),
               iou_median=np.median(iou), hit50=float((iou >= 0.5).mean()), hit70=float((iou >= 0.7).mean()))
    if tp is not None:
        tp, fp, fn = float(np.sum(tp)), float(np.sum(fp)), float(np.sum(fn))
        out["pooled_iou"] = tp / max(tp + fp + fn, 1)
        out["precision"] = tp / max(tp + fp, 1)
        out["recall"] = tp / max(tp + fn, 1)
    return out


def boot_ci(x, n=1000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def selftest():
    """Metric sanity checks. Raises AssertionError if the harness itself is broken."""
    a = np.zeros((10, 10), np.uint8); a[2:6, 2:6] = 1          # 16 px
    b = np.zeros((10, 10), np.uint8); b[4:8, 4:8] = 1          # 16 px, overlap 2x2 = 4
    assert dice_iou(a, a) == (1.0, 1.0)
    assert dice_iou(a, np.zeros_like(a)) == (0.0, 0.0)
    d, i = dice_iou(a, b)
    assert abs(d - 2 * 4 / 32) < 1e-9 and abs(i - 4 / 28) < 1e-9
    assert dice_iou(np.zeros_like(a), np.zeros_like(a)) == (1.0, 1.0)
    return True


def overlay(bgr, gt=None, pred=None, text=None, scale=1.0):
    """GT outline green, prediction outline red."""
    img = bgr.copy()
    for m, col in ((gt, (0, 255, 0)), (pred, (0, 0, 255))):
        if m is None:
            continue
        cs, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cs, -1, col, 2)
    if text:
        cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img
