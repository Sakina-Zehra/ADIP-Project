# # """
# # step2_segment.py  --  Step 2 (version 2): classical pothole segmentation, threshold-hybrid pipeline.

# # Pipeline (RGB only, no pretrained deep models):
# #   1. Preprocess   CLAHE (contrast), bilateral denoise + unsharp mask ("blur removal"); Lab colour smoothing.
# #   2. Candidates   (a) histogram thresholding (Otsu and triangle, both polarities) of lightness, two colour
# #                       channels and local texture energy            [Koch & Brilakis 2011; Jakstys 2016]
# #                   (b) k-means colour clustering and local adaptive thresholding
# #                   (c) filled outlines of closed Canny edge groups   [proposal]
# #                   Every mask is cleaned with morphology; lobed blobs are split by watershed on the distance
# #                   transform [Chung & Khan 2019]; near-duplicates are removed (NMS).
# #   3. Features     shape + edge support, and texture/colour of the INSIDE compared with a ring of ROAD around it.
# #   4. Classifier   Random Forest / SVM scores every candidate; every candidate that scores high enough and does
# #                   not overlap a better one is kept (several potholes per image are allowed).
# #   5. Boundary     optional marker-controlled watershed that snaps the outline to Canny edges (kept only if it
# #                   helps on validation).

# # Usage (after step1_prepare.py):
# #     python step2_segment.py build --workers 4      # slow part, run once, cached in 'cache'
# #     python step2_segment.py run                    # train on TRAIN, tune/evaluate on VAL, write all reports
# #     python step2_segment.py run --test             # FINAL: refit on train+val, evaluate on TEST (do this once)
# # Quick smoke test on 40 images:  build --limit 40  then  run --smoke
# # """
# # import argparse, os, sys, json, pickle, time, hashlib, warnings
# # import numpy as np, cv2, pandas as pd
# # from concurrent.futures import ProcessPoolExecutor
# # from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.svm import SVC
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.pipeline import make_pipeline
# # import pothole_common as pc

# # warnings.filterwarnings("ignore", category=RuntimeWarning)

# # # ====================================================================== configuration
# # C = dict(
# #     work_w=320, work_h=240,            # candidates + features are computed at half resolution (speed, less noise)
# #     clahe_clip=3.0, clahe_tile=8,      # contrast adjustment (3.0 > 2.0: helps dark/bright road images)
# #     bilateral=(5, 25, 5),              # edge-preserving denoise (d, sigmaColor, sigmaSpace)
# #     unsharp_sigma=2.0, unsharp_amount=0.6,   # mild "blur removal"
# #     thr_sigma=2.0,                     # smoothing of L/a/b before thresholding (removes single stones)
# #     tex_win=9,                         # window of the local-texture (std) channel
# #     kmeans_ks=(2, 3),                  # colour clusters
# #     kmeans_attempts=5,                 # k-means restarts (was 2 implicitly; 5 gives more stable cluster centres)
# #     adapt_blocks=(51, 121), adapt_c=6, # local adaptive threshold: block size (odd) and offset
# #     blur_sigmas=(1.5, 3.0), canny_pcts=(88, 94), ref_pct=90,   # Canny: two smoothing scales, strongest 12% / 6% of gradients
# #     close_ks=(5, 11),                  # closing for Canny outlines
# #     open_k=5,                          # opening that removes thin tails from filled regions
# #     mask_open=3, mask_close=7,         # clean-up of threshold masks (close joins stones inside a pothole)
# #     max_comps_per_mask=12,             # largest blobs kept per binary mask
# #     split_solidity=0.85, split_frac=0.5,   # watershed split of lobed blobs: only if solidity < 0.85; markers at 50% of max distance
# #     min_area_frac=0.003, max_area_frac=0.90,
# #     nms_iou=0.85, max_cands=120,
# #     ring_scale=0.35, ring_min=5, ring_max=30, ring_gap=2,   # road ring around a candidate (pixels at work resolution)
# #     pos_iou=0.5, neg_iou=0.3,          # labels for training (see make_labels)
# #     rel_pos=0.10, rel_neg=0.20,
# #     neg_pos_ratio=3,                   # FIX: subsample negatives to this multiple of positives in build_xy
# #     lbp_P=16, lbp_R=2,                # FIX: radius 2 (was 1) captures pothole texture at a meaningful scale
# #     max_regions=15, overlap_skip=0.3,  # several potholes per image; skip a region that overlaps a kept one by >30% of the smaller
# #     snap_frac=0.25,                    # boundary snap band = 25% of the region's equivalent radius
# #     svm_max_samples=20000,
# # )
# # PIPELINE_VERSION = 3                   # bumped: LBP params changed -> cache must be rebuilt
# # WORK = (C["work_w"], C["work_h"])
# # LBP_BINS = C["lbp_P"] + 2
# # GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")
# # SRC_NAMES = ("canny", "otsu", "triangle", "kmeans", "adaptive", "split", "depth")


# # def ell(k):
# #     return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(k), int(k)))


# # def cfg_hash():
# #     s = json.dumps(dict(C, v=PIPELINE_VERSION), sort_keys=True, default=str)
# #     return hashlib.md5(s.encode()).hexdigest()[:8]


# # # ====================================================================== feature names and groups
# # FEATURES = (["area_frac", "solidity", "circ", "elong", "extent", "ell_res", "cx", "cy", "border_frac",
# #              "edge_support", "grad_b"] + [f"src_{s}" for s in SRC_NAMES])
# # N_SHAPE = len(FEATURES)
# # FEATURES += ["mean_in", "std_in", "dmean", "std_ratio", "da", "db", "lap_in", "lap_ratio", "grad_in", "grad_ratio",
# #              "ed_in", "ed_ring", "ed_diff", "entropy_in"]
# # FEATURES += [f"lbp_in_{i}" for i in range(LBP_BINS)] + [f"lbp_d_{i}" for i in range(LBP_BINS)] + ["lbp_chi2", "lbp_chi2_fine"]
# # FEATURES += [f"glcm_in_{p}" for p in GLCM_PROPS] + [f"glcm_r_{p}" for p in GLCM_PROPS] + [f"glcm_d_{p}" for p in GLCM_PROPS]
# # FEATURES += ["ring_ok"]
# # GROUPS = {"shape_edge": list(range(N_SHAPE)), "texture": list(range(N_SHAPE, len(FEATURES))),
# #           "all": list(range(len(FEATURES)))}
# # IDX = {n: i for i, n in enumerate(FEATURES)}


# # # ====================================================================== 1. preprocessing
# # def make_ctx(bgr):
# #     """Everything computed once per image at working resolution."""
# #     h, w = bgr.shape[:2]
# #     lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
# #     L, A, B = cv2.split(lab)
# #     # --- for EDGES: contrast adjustment + denoise + mild deblur (proposal: "contrast adjustment, blur removal")
# #     clahe = cv2.createCLAHE(clipLimit=C["clahe_clip"], tileGridSize=(C["clahe_tile"],) * 2)
# #     Lc = clahe.apply(L)
# #     den = cv2.bilateralFilter(Lc, *C["bilateral"])
# #     blur = cv2.GaussianBlur(den, (0, 0), C["unsharp_sigma"])
# #     enh = cv2.addWeighted(den, 1 + C["unsharp_amount"], blur, -C["unsharp_amount"], 0)

# #     def canny_at(sigma, pct):
# #         g = cv2.GaussianBlur(enh, (0, 0), sigma)
# #         gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
# #         gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
# #         hi = float(np.percentile(np.hypot(gx, gy), pct))      # image-adaptive threshold (handles lighting changes)
# #         return cv2.Canny(g, 0.5 * hi, hi, L2gradient=True)

# #     edge_maps = [canny_at(s, p) for s in C["blur_sigmas"] for p in C["canny_pcts"]]
# #     E_ref = canny_at(C["blur_sigmas"][0], C["ref_pct"])
# #     edge_near = cv2.dilate(E_ref, np.ones((5, 5), np.uint8)) > 0
# #     edge_snap = cv2.dilate(canny_at(C["blur_sigmas"][0], C["canny_pcts"][-1]), np.ones((3, 3), np.uint8)) > 0
# #     # --- for THRESHOLDING: smoothed raw Lab channels + local texture energy
# #     sg = C["thr_sigma"]
# #     Ls, As, Bs = (cv2.GaussianBlur(x, (0, 0), sg) for x in (L, A, B))
# #     Lf = L.astype(np.float32)
# #     mu = cv2.blur(Lf, (C["tex_win"],) * 2)
# #     mu2 = cv2.blur(Lf * Lf, (C["tex_win"],) * 2)
# #     T = cv2.GaussianBlur(np.sqrt(np.maximum(mu2 - mu * mu, 0)), (0, 0), sg)
# #     lo, hi = np.percentile(T, 1), np.percentile(T, 99)
# #     T8 = np.clip((T - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
# #     bgr_s = cv2.bilateralFilter(bgr, 5, 25, 5)               # smoothed colour image for the watershed steps
# #     # --- for FEATURES use the RAW lightness (CLAHE would equalise the very inside/outside contrast we want to measure)
# #     Lg = cv2.GaussianBlur(L, (0, 0), 1.0)
# #     gx = cv2.Sobel(Lg, cv2.CV_32F, 1, 0, ksize=3)
# #     gy = cv2.Sobel(Lg, cv2.CV_32F, 0, 1, ksize=3)
# #     mag = np.hypot(gx, gy)
# #     lap = np.abs(cv2.Laplacian(Lg, cv2.CV_32F))
# #     # Multi-scale LBP: fine (R=1) + coarse (R=2) concatenated in full_feats for richer texture descriptor
# #     lbp = local_binary_pattern(L, C["lbp_P"], C["lbp_R"], method="uniform").astype(np.uint8)
# #     lbp_fine = local_binary_pattern(L, 8, 1, method="uniform").astype(np.uint8)   # always R=1 for fine scale
# #     return dict(h=h, w=w, L=L, A=A, B=B, edge_maps=edge_maps, E_ref=E_ref, edge_near=edge_near, edge_snap=edge_snap,
# #                 chan=dict(L=Ls, A=As, B=Bs, T=T8), bgr_s=bgr_s,
# #                 mag=mag, mag_mean=float(mag.mean()) + 1e-6, lap=lap, lbp=lbp, lbp_fine=lbp_fine)


# # # ====================================================================== 2. candidate generation
# # def fill_outer(m_u8):
# #     cs, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #     out = np.zeros_like(m_u8)
# #     cv2.drawContours(out, cs, -1, 1, -1)
# #     return out


# # def clean_region(m_u8):
# #     """Shape clean-up: remove thin tails, keep the largest blob, fill holes."""
# #     m = cv2.morphologyEx(m_u8, cv2.MORPH_OPEN, ell(C["open_k"]))
# #     n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
# #     if n <= 1:
# #         return None
# #     k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
# #     return fill_outer((lab == k).astype(np.uint8))


# # def split_lobes(comp, ctx, amin):
# #     """Watershed on the distance transform: separates potholes that were merged into one blob."""
# #     cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #     area = float(comp.sum())
# #     harea = max(cv2.contourArea(cv2.convexHull(max(cs, key=cv2.contourArea))), 1.0)
# #     if area / harea >= C["split_solidity"]:
# #         return []
# #     D = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
# #     sure_fg = (D > C["split_frac"] * D.max()).astype(np.uint8)
# #     n, mk = cv2.connectedComponents(sure_fg)
# #     if n - 1 < 2:
# #         return []
# #     markers = np.zeros(comp.shape, np.int32)
# #     markers[comp == 0] = 1
# #     markers[mk > 0] = mk[mk > 0] + 1
# #     # FIX: pass a cropped, correctly-sized colour image instead of the full bgr_s.
# #     # Using the full image caused surrounding colours to bleed into the split boundary.
# #     h_c, w_c = comp.shape
# #     bgr_crop = ctx["bgr_s"][:h_c, :w_c].copy()   # comp is always at work resolution so shapes match
# #     cv2.watershed(bgr_crop, markers)
# #     parts = []
# #     for k in range(2, n + 1):
# #         part = ((markers == k) & (comp > 0)).astype(np.uint8)
# #         if part.sum() >= amin:
# #             parts.append(part)
# #     return parts


# # def binary_masks(ctx):
# #     """All thresholding-based binary masks, tagged with their source name."""
# #     h, w = ctx["h"], ctx["w"]
# #     out = []
# #     for ch in ctx["chan"].values():                     # (a) histogram thresholding, both polarities
# #         for src, flag in (("otsu", cv2.THRESH_OTSU), ("triangle", cv2.THRESH_TRIANGLE)):
# #             t, _ = cv2.threshold(ch, 0, 255, cv2.THRESH_BINARY + flag)
# #             out.append(((ch < t).astype(np.uint8), src))
# #             out.append(((ch > t).astype(np.uint8), src))
# #     feats = np.dstack([ctx["chan"]["L"], ctx["chan"]["A"], ctx["chan"]["B"]]).reshape(-1, 3).astype(np.float32)
# #     crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
# #     for k in C["kmeans_ks"]:                            # (b) colour clustering
# #         cv2.setRNGSeed(0)
# #         _, labels, _ = cv2.kmeans(feats, k, None, crit, C["kmeans_attempts"], cv2.KMEANS_PP_CENTERS)
# #         labels = labels.reshape(h, w)
# #         for j in range(k):
# #             out.append(((labels == j).astype(np.uint8), "kmeans"))
# #     Ls = ctx["chan"]["L"]
# #     for b in C["adapt_blocks"]:                         # (b) local adaptive thresholding (small potholes on uniform road)
# #         out.append((cv2.adaptiveThreshold(Ls, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, b, C["adapt_c"]), "adaptive"))
# #         out.append((cv2.adaptiveThreshold(Ls, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, b, -C["adapt_c"]), "adaptive"))
# #     return out


# # def raw_candidates(ctx):
# #     h, w = ctx["h"], ctx["w"]
# #     amin, amax = C["min_area_frac"] * h * w, C["max_area_frac"] * h * w
# #     out, seen = [], set()

# #     def add(m, src):
# #         if m is None:
# #             return
# #         a = int(m.sum())
# #         if not (amin <= a <= amax):
# #             return
# #         key = (cv2.resize(m.astype(np.uint8), (64, 48), interpolation=cv2.INTER_AREA) > 0).tobytes()
# #         if key in seen:                                 # exact duplicates (same blob from different thresholds)
# #             return
# #         seen.add(key)
# #         out.append((m.astype(bool), SRC_NAMES.index(src)))

# #     for m, src in binary_masks(ctx):
# #         m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ell(C["mask_open"]))
# #         m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ell(C["mask_close"]))
# #         n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
# #         order = np.argsort(-st[1:, cv2.CC_STAT_AREA])[:C["max_comps_per_mask"]] + 1
# #         for j in order:
# #             if st[j, cv2.CC_STAT_AREA] < amin:
# #                 continue
# #             comp = fill_outer((lab == j).astype(np.uint8))
# #             add(comp, src)
# #             for part in split_lobes(comp, ctx, amin):
# #                 add(clean_region(part), "split")
# #     for E in ctx["edge_maps"]:                          # (c) filled outlines of closed edge groups (proposal)
# #         for k in C["close_ks"]:
# #             closed = cv2.morphologyEx(E, cv2.MORPH_CLOSE, ell(k))
# #             cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #             for c in cnts:
# #                 if cv2.contourArea(c) < 0.5 * amin:
# #                     continue
# #                 f = np.zeros((h, w), np.uint8)
# #                 cv2.drawContours(f, [c], -1, 1, -1)
# #                 add(clean_region(f), "canny")
# #     # (d) depth-anomaly candidate: regions that are noticeably deeper than their surroundings.
# #     # This is a direct physical cue — potholes are literally holes in the road.
# #     if ctx.get("depth_anomaly") is not None:
# #         d_map = ctx["depth_anomaly"]
# #         d_closed = cv2.morphologyEx(d_map, cv2.MORPH_CLOSE, ell(11))
# #         n_d, lab_d, st_d, _ = cv2.connectedComponentsWithStats(d_closed, connectivity=8)
# #         order_d = np.argsort(-st_d[1:, cv2.CC_STAT_AREA])[:C["max_comps_per_mask"]] + 1
# #         for j in order_d:
# #             if st_d[j, cv2.CC_STAT_AREA] < amin:
# #                 continue
# #             comp = fill_outer((lab_d == j).astype(np.uint8))
# #             add(clean_region(comp), "depth")
# #             for part in split_lobes(comp, ctx, amin):
# #                 add(clean_region(part), "depth")
# #     return out


# # def base_feats(mask, ctx):
# #     """Shape + edge-support features (cheap; also used to rank candidates before the classifier exists)."""
# #     h, w = mask.shape
# #     ys, xs = np.nonzero(mask)
# #     area = len(xs)
# #     x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
# #     cm = np.pad(mask[y0:y1 + 1, x0:x1 + 1].astype(np.uint8), 1)
# #     cnts, _ = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# #     c = max(cnts, key=cv2.contourArea)
# #     per = max(cv2.arcLength(c, True), 1.0)
# #     harea = max(cv2.contourArea(cv2.convexHull(c)), 1.0)
# #     rw, rh = cv2.minAreaRect(c)[1]
# #     ell_res = 1.0
# #     if len(c) >= 5:
# #         try:
# #             e = cv2.fitEllipse(c)
# #             em = np.zeros_like(cm)
# #             cv2.ellipse(em, e, 1, -1)
# #             ell_res = float((em.astype(bool) ^ cm.astype(bool)).sum()) / area
# #         except cv2.error:
# #             pass
# #     frame = int(mask[0, :].sum() + mask[-1, :].sum() + mask[:, 0].sum() + mask[:, -1].sum())
# #     bnd = cm.astype(bool) & ~(cv2.erode(cm, np.ones((3, 3), np.uint8)) > 0)
# #     en = np.pad(ctx["edge_near"][y0:y1 + 1, x0:x1 + 1], 1)
# #     mg = np.pad(ctx["mag"][y0:y1 + 1, x0:x1 + 1], 1)
# #     return dict(area_frac=area / (h * w), solidity=min(area / harea, 1.0), circ=min(4 * np.pi * area / per ** 2, 1.5),
# #                 elong=min(max(rw, rh) / max(min(rw, rh), 1.0), 20.0), extent=area / ((x1 - x0 + 1) * (y1 - y0 + 1)),
# #                 ell_res=min(ell_res, 5.0), cx=float(xs.mean()) / w, cy=float(ys.mean()) / h,
# #                 border_frac=min(frame / per, 1.0),
# #                 edge_support=float(en[bnd].mean()) if bnd.any() else 0.0,
# #                 grad_b=float(mg[bnd].mean() / ctx["mag_mean"]) if bnd.any() else 0.0)


# # def heuristic_score(b, prior):
# #     """Technique 1 on its own (no learning): closed, edge-supported, compact, plausible-size, not cut by the border."""
# #     ap = max(np.exp(-0.5 * ((np.log(max(b["area_frac"], 1e-6)) - prior[0]) / prior[1]) ** 2), 0.05)
# #     return float(b["edge_support"] * (0.5 + 0.5 * b["solidity"]) * ap * (1 - 0.5 * b["border_frac"]))


# # def nms_and_cap(cands, prior):
# #     """Score, drop near-duplicates (IoU > nms_iou on 1/4-res masks), keep the best max_cands."""
# #     if not cands:
# #         return []
# #     scored = [(heuristic_score(b, prior), m, src, b) for m, src, b in cands]
# #     scored.sort(key=lambda t: -t[0])
# #     small = np.stack([cv2.resize(m.astype(np.uint8), (C["work_w"] // 4, C["work_h"] // 4),
# #                                  interpolation=cv2.INTER_AREA).ravel() > 0 for _, m, _, _ in scored]).astype(np.float32)
# #     inter = small @ small.T
# #     ar = small.sum(1)
# #     iou = inter / np.maximum(ar[:, None] + ar[None, :] - inter, 1.0)
# #     keep = []
# #     for i in range(len(scored)):
# #         if all(iou[i, j] <= C["nms_iou"] for j in keep):
# #             keep.append(i)
# #         if len(keep) >= C["max_cands"]:
# #             break
# #     return [scored[i] for i in keep]


# # def full_feats(mask, ctx, base, src):
# #     """Features: base shape/edge + inside-vs-road-ring texture comparison."""
# #     h, w = mask.shape
# #     ys, xs = np.nonzero(mask)
# #     area = len(xs)
# #     x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
# #     rw = int(np.clip(C["ring_scale"] * np.sqrt(area), C["ring_min"], C["ring_max"]))
# #     mg = rw + C["ring_gap"] + 2
# #     cx0, cy0, cx1, cy1 = max(0, x0 - mg), max(0, y0 - mg), min(w, x1 + mg + 1), min(h, y1 + mg + 1)
# #     m = mask[cy0:cy1, cx0:cx1].astype(np.uint8)
# #     ring = (cv2.dilate(m, ell(2 * rw + 1)) > 0) & ~(cv2.dilate(m, ell(2 * C["ring_gap"] + 1)) > 0)
# #     core = cv2.erode(m, ell(5)) > 0               # inside, away from the blurry boundary
# #     if core.sum() < 0.3 * m.sum():
# #         core = m > 0
# #     ring_ok = bool(ring.sum() >= 50)
# #     L = ctx["L"][cy0:cy1, cx0:cx1]
# #     Af = ctx["A"][cy0:cy1, cx0:cx1].astype(np.float32)
# #     Bf = ctx["B"][cy0:cy1, cx0:cx1].astype(np.float32)
# #     lap = ctx["lap"][cy0:cy1, cx0:cx1]
# #     mag = ctx["mag"][cy0:cy1, cx0:cx1]
# #     Er = (ctx["E_ref"][cy0:cy1, cx0:cx1] > 0)
# #     lb = ctx["lbp"][cy0:cy1, cx0:cx1]
# #     lb_fine = ctx["lbp_fine"][cy0:cy1, cx0:cx1]   # R=1 fine-scale LBP for cross-scale chi2
# #     nan = np.nan
# #     Li = L[core].astype(np.float32)
# #     f = {k: base[k] for k in FEATURES[:11]}
# #     for n, s in enumerate(SRC_NAMES):
# #         f[f"src_{s}"] = 1.0 if src == n else 0.0
# #     f["mean_in"] = Li.mean() / 255.0
# #     f["std_in"] = Li.std() / 255.0
# #     f["lap_in"] = lap[core].mean()
# #     f["grad_in"] = mag[core].mean() / ctx["mag_mean"]
# #     f["ed_in"] = Er[core].mean()
# #     hist = np.bincount((Li.astype(np.int32) >> 4), minlength=16)[:16] / max(len(Li), 1)
# #     f["entropy_in"] = float(-(hist[hist > 0] * np.log(hist[hist > 0])).sum())
# #     hin = np.bincount(lb[core], minlength=LBP_BINS)[:LBP_BINS] / max(core.sum(), 1)
# #     for i in range(LBP_BINS):
# #         f[f"lbp_in_{i}"] = hin[i]
# #     # Fine-scale (R=1) LBP histogram inside — used for the cross-scale chi2 below
# #     FINE_BINS = 10   # 8+2 for P=8,R=1
# #     hin_fine = np.bincount(lb_fine[core], minlength=FINE_BINS)[:FINE_BINS] / max(core.sum(), 1)

# #     def glcm_vec(sel):
# #         q = (L >> 4).astype(np.uint8)
# #         q = np.where(sel, q, 16).astype(np.uint8)          # level 16 = "ignore" (outside the region)
# #         g = graycomatrix(q, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=17, symmetric=True, normed=False)
# #         P = g[:16, :16].astype(np.float64)
# #         if P.sum() < 30:
# #             return [nan] * len(GLCM_PROPS)
# #         return [float(graycoprops(P, p).mean()) for p in GLCM_PROPS]

# #     gin = glcm_vec(core)
# #     for p, v in zip(GLCM_PROPS, gin):
# #         f[f"glcm_in_{p}"] = v
# #     if ring_ok:
# #         Lr = L[ring].astype(np.float32)
# #         f["dmean"] = (Li.mean() - Lr.mean()) / 255.0
# #         f["std_ratio"] = min(Li.std() / (Lr.std() + 1.0), 20.0)
# #         f["da"] = (Af[core].mean() - Af[ring].mean()) / 255.0
# #         f["db"] = (Bf[core].mean() - Bf[ring].mean()) / 255.0
# #         lr = lap[ring].mean()
# #         f["lap_ratio"] = min(f["lap_in"] / (lr + 0.5), 50.0)
# #         f["grad_ratio"] = min(f["grad_in"] / (mag[ring].mean() / ctx["mag_mean"] + 0.05), 50.0)
# #         f["ed_ring"] = Er[ring].mean()
# #         f["ed_diff"] = f["ed_in"] - f["ed_ring"]
# #         hr = np.bincount(lb[ring], minlength=LBP_BINS)[:LBP_BINS] / max(ring.sum(), 1)
# #         for i in range(LBP_BINS):
# #             f[f"lbp_d_{i}"] = hin[i] - hr[i]
# #         f["lbp_chi2"] = float(0.5 * np.sum((hin - hr) ** 2 / (hin + hr + 1e-9)))
# #         # Cross-scale chi2: compare fine-scale (R=1) inner vs ring — captures texture at a different granularity
# #         FINE_BINS = 10
# #         hr_fine = np.bincount(lb_fine[ring], minlength=FINE_BINS)[:FINE_BINS] / max(ring.sum(), 1)
# #         f["lbp_chi2_fine"] = float(0.5 * np.sum((hin_fine - hr_fine) ** 2 / (hin_fine + hr_fine + 1e-9)))
# #         gr = glcm_vec(ring)
# #         for p, a, b in zip(GLCM_PROPS, gin, gr):
# #             f[f"glcm_r_{p}"] = b
# #             f[f"glcm_d_{p}"] = a - b
# #     f["ring_ok"] = 1.0 if ring_ok else 0.0
# #     v = np.array([f.get(n, nan) for n in FEATURES], np.float64)
# #     return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# # def rle_encode(m):
# #     f = m.ravel().astype(np.int8)
# #     d = np.diff(np.concatenate([[0], f, [0]]))
# #     return np.flatnonzero(d == 1).astype(np.int32), np.flatnonzero(d == -1).astype(np.int32)


# # def rle_decode(rle, shape):
# #     n = shape[0] * shape[1]
# #     d = np.zeros(n + 1, np.int8)
# #     d[rle[0]] += 1
# #     d[rle[1]] -= 1
# #     return (np.cumsum(d[:n]) > 0).reshape(shape)


# # def load_work(rgb_path):
# #     bgr, _ = pc.load_rgb(rgb_path)
# #     return cv2.resize(bgr, WORK, interpolation=cv2.INTER_AREA)


# # def instance_masks(label_path):
# #     """One work-resolution mask per ground-truth polygon (a pothole each)."""
# #     polys, _ = pc.read_polygons(label_path)
# #     out = []
# #     for p in polys:
# #         m = np.zeros((pc.H, pc.W), np.uint8)
# #         cv2.fillPoly(m, [p], 1)
# #         mw = cv2.resize(m, WORK, interpolation=cv2.INTER_NEAREST).astype(bool)
# #         if mw.any():
# #             out.append(mw)
# #     return out


# # def process_image(args):
# #     """One image -> candidates (RLE), feature matrix, IoU with every ground-truth pothole."""
# #     iid, rgb_path, depth_path, label_path, prior = args
# #     cv2.setNumThreads(1)                          # avoid CPU oversubscription when several workers run
# #     cv2.setRNGSeed(0)
# #     try:
# #         ctx = make_ctx(load_work(rgb_path))
# #         # Add depth-anomaly candidate map to ctx if depth data is available
# #         if depth_path and os.path.exists(depth_path):
# #             try:
# #                 d_full = pc.load_depth_cm(depth_path)
# #                 d_small = cv2.resize(
# #                     np.nan_to_num(d_full, nan=np.nanmedian(d_full)).astype(np.float32),
# #                     WORK, interpolation=cv2.INTER_NEAREST)
# #                 local_med = cv2.medianBlur(d_small, 15)
# #                 anomaly = ((d_small - local_med) > 2.0) & np.isfinite(
# #                     cv2.resize(d_full.astype(np.float32), WORK, interpolation=cv2.INTER_NEAREST))
# #                 ctx["depth_anomaly"] = (anomaly.astype(np.uint8) * 255)
# #             except Exception:
# #                 ctx["depth_anomaly"] = None
# #         else:
# #             ctx["depth_anomaly"] = None
# #         inst = instance_masks(label_path)
# #         raw = raw_candidates(ctx)
# #         pre = [(m, s, base_feats(m, ctx)) for m, s in raw]
# #         kept = nms_and_cap(pre, prior)
# #         feats, rles, srcs, sa, masks = [], [], [], [], []
# #         for score, m, src, base in kept:
# #             feats.append(full_feats(m, ctx, base, src))
# #             rles.append(rle_encode(m))
# #             srcs.append(src)
# #             sa.append(score)
# #             masks.append(m)
# #         n = len(masks)
# #         if n and inst:
# #             Cm = np.stack([m.ravel() for m in masks]).astype(np.float32)
# #             Im = np.stack([g.ravel() for g in inst]).astype(np.float32)
# #             inter = Cm @ Im.T
# #             iou_mat = inter / np.maximum(Cm.sum(1)[:, None] + Im.sum(1)[None, :] - inter, 1.0)
# #             U = np.any(np.stack(inst), axis=0).ravel().astype(np.float32)
# #             iu = (Cm @ U) / np.maximum(Cm.sum(1) + U.sum() - Cm @ U, 1.0)
# #         else:
# #             iou_mat = np.zeros((n, len(inst)), np.float32)
# #             iu = np.zeros(n, np.float32)
# #         iou = iou_mat.max(1) if iou_mat.shape[1] else np.zeros(n, np.float32)
# #         ref = (iou_mat.max(0)[iou_mat.argmax(1)] if iou_mat.shape[1] and n else np.zeros(n, np.float32))
# #         return dict(id=iid, ok=True, has_gt=bool(inst), n_inst=len(inst),
# #                     feats=np.array(feats, np.float32).reshape(n, len(FEATURES)),
# #                     iou=iou.astype(np.float32), ref=ref.astype(np.float32), iou_u=iu.astype(np.float32),
# #                     iou_mat=iou_mat.astype(np.float16), rle=rles, src=np.array(srcs, np.int8),
# #                     score_a=np.array(sa, np.float32), n_raw=len(raw))
# #     except Exception as ex:                       # never let one bad image kill a 1000-image run
# #         return dict(id=iid, ok=False, error=repr(ex))


# # # ====================================================================== build (cache)
# # def cache_path(smoke=False):
# #     return os.path.join(pc.CACHE, f"features_{cfg_hash()}{'_smoke' if smoke else ''}.pkl")


# # def train_prior(df, sp):
# #     """Area prior (log-normal) over individual TRAIN ground-truth potholes."""
# #     ids = set(sp.image_id[sp.split == "train"])
# #     la = []
# #     for _, r in df[df.image_id.isin(ids)].iterrows():
# #         polys, _ = pc.read_polygons(r.label_path)
# #         for p in polys:
# #             m = np.zeros((pc.H, pc.W), np.uint8)
# #             cv2.fillPoly(m, [p], 1)
# #             if m.sum() > 0:
# #                 la.append(np.log(m.sum() / (pc.W * pc.H)))
# #     return float(np.mean(la)), float(max(np.std(la), 0.7))


# # def cmd_build(args):
# #     if not os.path.exists(pc.SPLITS):
# #         sys.exit("splits.csv not found. Run step1_prepare.py first.")
# #     os.makedirs(pc.CACHE, exist_ok=True)
# #     df, sp = pc.load_index(), pd.read_csv(pc.SPLITS)
# #     prior = train_prior(df, sp)
# #     print(f"area prior from TRAIN GT: mean log-area {prior[0]:.2f} (~{100 * np.exp(prior[0]):.1f}% of image), sd {prior[1]:.2f}")
# #     sub = df if not args.limit else df.sample(n=min(args.limit, len(df)), random_state=0)
# #     jobs = [(r.image_id, r.rgb_path, getattr(r, "depth_path", None), r.label_path, prior) for r in sub.itertuples()]
# #     t0 = time.time()
# #     results = []
# #     if args.workers > 1:
# #         with ProcessPoolExecutor(max_workers=args.workers) as ex:
# #             for i, res in enumerate(ex.map(process_image, jobs, chunksize=4), 1):
# #                 results.append(res)
# #                 if i % 50 == 0:
# #                     print(f"  {i}/{len(jobs)}  ({time.time() - t0:.0f}s)")
# #     else:
# #         for i, j in enumerate(jobs, 1):
# #             results.append(process_image(j))
# #             if i % 25 == 0:
# #                 print(f"  {i}/{len(jobs)}  ({time.time() - t0:.0f}s)")
# #     bad = [r for r in results if not r["ok"]]
# #     good = {r["id"]: r for r in results if r["ok"]}
# #     if bad:
# #         print(f"!! {len(bad)} images FAILED and are excluded:", [(b['id'], b['error']) for b in bad[:5]])
# #     nc = np.array([len(r["iou"]) for r in good.values()])
# #     print(f"candidates per image: median {int(np.median(nc))}, min {nc.min()}, max {nc.max()}; images with 0 candidates: {(nc == 0).sum()}")
# #     with open(cache_path(args.limit > 0), "wb") as f:
# #         pickle.dump(dict(meta=dict(cfg=C, features=FEATURES, prior=prior), entries=good, failed=bad), f)
# #     print(f"[ok] cached {len(good)} images in {time.time() - t0:.0f}s -> {cache_path(args.limit > 0)}")


# # # ====================================================================== model helpers
# # def make_model(kind, p):
# #     if kind == "rf":
# #         return RandomForestClassifier(n_estimators=200, min_samples_leaf=3, max_features="sqrt",
# #                                       class_weight="balanced_subsample", n_jobs=-1, random_state=0)
# #     return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=p.get("C", 1.0), gamma="scale",
# #                                               class_weight="balanced", probability=True, random_state=0))


# # def make_labels(iou, ref):
# #     """1 = good outline of a ground-truth pothole (IoU>=0.5 and close to the best candidate for THAT pothole),
# #     0 = not a pothole outline (IoU<0.3, or clearly worse than the best candidate for its pothole),
# #     -1 = ambiguous, left out of training."""
# #     y = np.full(len(iou), -1, int)
# #     y[iou >= np.maximum(C["pos_iou"], ref - C["rel_pos"])] = 1
# #     neg = (iou < C["neg_iou"]) | ((ref >= C["pos_iou"]) & (iou < ref - C["rel_neg"]))
# #     y[neg & (y != 1)] = 0
# #     return y


# # def build_xy(entries, ids, cols, shuffle_labels=False):
# #     X, y = [], []
# #     for i in ids:
# #         e = entries[i]
# #         if not e["has_gt"] or len(e["iou"]) == 0:
# #             continue
# #         lab = make_labels(e["iou"], e["ref"])
# #         keep = lab >= 0
# #         X.append(e["feats"][keep][:, cols])
# #         y.append(lab[keep])
# #     X, y = np.vstack(X), np.concatenate(y)
# #     if y.sum() == 0 or y.sum() == len(y):
# #         raise RuntimeError("training set has only one class; candidate generation found no good candidates (see 'build' output)")
# #     # FIX: subsample negatives to neg_pos_ratio:1 to combat extreme class imbalance.
# #     # With ~120 candidates per image and ~1 positive, the raw ratio is ~119:1 which biases
# #     # the classifier to always predict "not pothole". Subsampling to 3:1 gives a cleaner boundary.
# #     if not shuffle_labels:
# #         pos_idx = np.flatnonzero(y == 1)
# #         neg_idx = np.flatnonzero(y == 0)
# #         max_neg = C["neg_pos_ratio"] * len(pos_idx)
# #         if len(neg_idx) > max_neg:
# #             rng = np.random.default_rng(42)
# #             neg_sub = rng.choice(neg_idx, max_neg, replace=False)
# #             sel = np.concatenate([pos_idx, neg_sub])
# #             rng.shuffle(sel)
# #             X, y = X[sel], y[sel]
# #     if shuffle_labels:
# #         y = np.random.default_rng(0).permutation(y)
# #     return X, y


# # def fit_model(kind, p, entries, ids, cols, shuffle_labels=False):
# #     X, y = build_xy(entries, ids, cols, shuffle_labels)
# #     if kind == "svm" and len(y) > C["svm_max_samples"]:
# #         rng = np.random.default_rng(0)
# #         pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
# #         n_neg = max(C["svm_max_samples"] - len(pos), 1000)
# #         sel = np.concatenate([pos, rng.choice(neg, min(n_neg, len(neg)), replace=False)])
# #         X, y = X[sel], y[sel]
# #     m = make_model(kind, p)
# #     m.fit(X, y)
# #     return m


# # def proba(model, e, cols):
# #     if len(e["iou"]) == 0:
# #         return np.zeros(0)
# #     return model.predict_proba(e["feats"][:, cols])[:, 1]


# # # ====================================================================== prediction + evaluation
# # def fill_holes(m):
# #     return fill_outer(m.astype(np.uint8))


# # def to_full(mask_work):
# #     """work-res bool mask -> clean 640x480 uint8 mask."""
# #     f = cv2.resize(mask_work.astype(np.float32), (pc.W, pc.H), interpolation=cv2.INTER_LINEAR) > 0.5
# #     return fill_holes(cv2.morphologyEx(f.astype(np.uint8), cv2.MORPH_OPEN, ell(3)))


# # def choose(e, scores, t_add):
# #     """Best candidate always; further candidates with score>=t_add that do not overlap a kept one by more than
# #     overlap_skip of the smaller region (several potholes per image), at most max_regions."""
# #     if len(scores) == 0:
# #         return []
# #     order = np.argsort(-scores)
# #     sel = [int(order[0])]
# #     if t_add is not None:
# #         shape = (WORK[1], WORK[0])
# #         masks = {sel[0]: rle_decode(e["rle"][sel[0]], shape)}
# #         for j in order[1:]:
# #             if len(sel) >= C["max_regions"] or scores[j] < t_add:
# #                 break
# #             mj = rle_decode(e["rle"][int(j)], shape)
# #             aj = max(int(mj.sum()), 1)
# #             if all((mj & masks[k]).sum() / max(min(aj, int(masks[k].sum())), 1) <= C["overlap_skip"] for k in sel):
# #                 sel.append(int(j))
# #                 masks[int(j)] = mj
# #     return sel


# # def snap_region(region, ctx):
# #     """Marker-controlled watershed: the outline moves to the strongest nearby edge inside a band around the region.
# #     Canny edges are drawn as bright ridges that the flood cannot cross cheaply."""
# #     reg = region.astype(np.uint8)
# #     area = int(reg.sum())
# #     # FIX: if the region touches the image border, watershed will snap the boundary to the image edge
# #     # instead of a real Canny edge, producing a cut-off mask. Skip snap for border-touching regions.
# #     if (reg[0, :].any() or reg[-1, :].any() or reg[:, 0].any() or reg[:, -1].any()):
# #         return region
# #     r = np.sqrt(area / np.pi)
# #     k = int(max(3, round(C["snap_frac"] * r)))
# #     inner = cv2.erode(reg, ell(2 * k + 1))
# #     outer = cv2.dilate(reg, ell(2 * k + 1))
# #     if inner.sum() < 20:
# #         return region
# #     img = ctx["bgr_s"].copy()
# #     img[ctx["edge_snap"]] = 255
# #     markers = np.zeros(reg.shape, np.int32)
# #     markers[outer == 0] = 1
# #     markers[inner > 0] = 2
# #     cv2.watershed(img, markers)
# #     out = markers == 2
# #     if out.sum() < 0.5 * area or out.sum() > 1.5 * area:
# #         return region
# #     return out


# # def pred_work_mask(e, sel, snap_ctx=None):
# #     shape = (WORK[1], WORK[0])
# #     m = np.zeros(shape, bool)
# #     for j in sel:
# #         r = rle_decode(e["rle"][j], shape)
# #         m |= snap_region(r, snap_ctx) if snap_ctx is not None else r
# #     return m


# # def score_row(pred_full, gt_full):
# #     d, i = pc.dice_iou(pred_full, gt_full)
# #     p, g = pred_full.astype(bool), gt_full.astype(bool)
# #     tp = float((p & g).sum())
# #     return dict(dice=d, iou=i, tp=tp, fp=float(p.sum() - tp), fn=float(g.sum() - tp), pred_frac=p.mean())


# # def oracle_multi_sel(e):
# #     """Best candidate for every ground-truth pothole (upper bound for several potholes per image)."""
# #     m = e["iou_mat"].astype(np.float32)
# #     sel = []
# #     for j in range(m.shape[1] if m.ndim == 2 else 0):
# #         if m.shape[0] == 0:
# #             break
# #         k = int(np.argmax(m[:, j]))
# #         if m[k, j] >= 0.3 and k not in sel:
# #             sel.append(k)
# #     return sel


# # def eval_variant(ids, entries, gts, scorer, t_add=None, mode="score", ctx_of=None):
# #     """scorer(entry)->scores. mode: score | oracle | oracle_multi | random.
# #     ctx_of(id) -> image context enables the boundary snap."""
# #     rows, preds = {}, {}
# #     rng = np.random.default_rng(0)
# #     for i in ids:
# #         e = entries[i]
# #         if len(e["iou"]) == 0:
# #             sel = []
# #         elif mode == "oracle":
# #             sel = [int(np.argmax(e["iou_u"]))]
# #         elif mode == "oracle_multi":
# #             sel = oracle_multi_sel(e) or [int(np.argmax(e["iou_u"]))]
# #         elif mode == "random":
# #             sel = [int(rng.integers(len(e["iou"])))]
# #         else:
# #             sel = choose(e, scorer(e), t_add)
# #         if sel:
# #             pred = to_full(pred_work_mask(e, sel, ctx_of(i) if ctx_of else None))
# #         else:
# #             pred = np.zeros((pc.H, pc.W), np.uint8)
# #         r = score_row(pred, gts[i])
# #         r["n_regions"] = len(sel)
# #         rows[i], preds[i] = r, pred
# #     return rows, preds


# # def table_row(name, rows):
# #     df = pd.DataFrame(rows.values())
# #     s = pc.summarize(df.dice, df.iou, df.tp, df.fp, df.fn)
# #     lo, hi = pc.boot_ci(df.dice.values)
# #     s.update(variant=name, dice_ci=f"{lo:.3f}-{hi:.3f}")
# #     return s


# # # ====================================================================== run
# # def cmd_run(args):
# #     path = cache_path(args.smoke)
# #     if not os.path.exists(path):
# #         sys.exit(f"cache not found ({path}). Run:  python step2_segment.py build")
# #     with open(path, "rb") as f:
# #         pk = pickle.load(f)
# #     entries = pk["entries"]
# #     df, sp = pc.load_index(), pd.read_csv(pc.SPLITS)
# #     paths = df.set_index("image_id")
# #     split_ids = {s: [i for i in sp.image_id[sp.split == s] if i in entries and entries[i]["has_gt"]]
# #                  for s in ("train", "val", "test")}
# #     for s, v in split_ids.items():
# #         print(f"{s}: {len(v)} usable images")
# #     n_empty = sum(1 for e in entries.values() if not e["has_gt"])
# #     if n_empty:
# #         print(f"({n_empty} images without ground truth are excluded)")
# #     os.makedirs(pc.OUT, exist_ok=True)
# #     cfg_path = os.path.join(pc.OUT, "best_config.json")
# #     log = []

# #     def say(*a):
# #         s = " ".join(str(x) for x in a)
# #         print(s)
# #         log.append(s)

# #     if args.test:
# #         if not os.path.exists(cfg_path):
# #             sys.exit("best_config.json missing. Run `run` (validation) first, then `run --test`.")
# #         best = json.load(open(cfg_path))
# #         fit_ids, eval_name = split_ids["train"] + split_ids["val"], "test"
# #         say(f"FINAL TEST RUN: refitting '{best['kind']}' on train+val, evaluating on test ONCE")
# #     else:
# #         best, fit_ids, eval_name = None, split_ids["train"], "val"
# #     ev_ids = split_ids[eval_name]
# #     gts = {i: pc.gt_mask(paths.at[i, "label_path"])[0] for i in ev_ids}
# #     cols_all, cols_se, cols_tx = GROUPS["all"], GROUPS["shape_edge"], GROUPS["texture"]
# #     # FIX: bounded cache - holding all 1000 image contexts simultaneously exhausts RAM.
# #     # Keep only the 60 most recently used contexts (each ~320x240x~10 arrays ~few MB).
# #     _ctx_cache = {}
# #     _ctx_order = []
# #     _CTX_MAXSIZE = 60

# #     def ctx_of(i):
# #         if i not in _ctx_cache:
# #             if len(_ctx_order) >= _CTX_MAXSIZE:
# #                 evict = _ctx_order.pop(0)
# #                 _ctx_cache.pop(evict, None)
# #             _ctx_cache[i] = make_ctx(load_work(paths.at[i, "rgb_path"]))
# #             _ctx_order.append(i)
# #         return _ctx_cache[i]

# #     # ---------------- model selection on VAL (only in validation mode)
# #     if best is None:
# #         kinds = [("rf", {})] if args.models == "rf" else [("rf", {}), ("svm", {"C": 1.0}), ("svm", {"C": 10.0})]
# #         sel_rows = []
# #         for kind, p in kinds:
# #             t0 = time.time()
# #             m = fit_model(kind, p, entries, fit_ids, cols_all)
# #             rows, _ = eval_variant(ev_ids, entries, gts, lambda e, m=m: proba(m, e, cols_all))
# #             s = table_row(f"{kind} {p}", rows)
# #             say(f"model {kind} {p}: val Dice {s['dice_mean']:.3f} (top-1 only)  [{time.time() - t0:.0f}s]")
# #             sel_rows.append((s["dice_mean"], kind, p, m))
# #         sel_rows.sort(key=lambda t: -t[0])
# #         _, kind, p, model_all = sel_rows[0]
# #         best_t, best_d = None, -1
# #         for t in (None, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):          # probability needed to keep an extra region
# #             rows, _ = eval_variant(ev_ids, entries, gts, lambda e: proba(model_all, e, cols_all), t_add=t)
# #             d = float(np.mean([r["dice"] for r in rows.values()]))
# #             say(f"  extra-region threshold {t}: val Dice {d:.3f}")
# #             if d > best_d + 1e-4:
# #                 best_t, best_d = t, d
# #         rows, _ = eval_variant(ev_ids, entries, gts, lambda e: proba(model_all, e, cols_all), t_add=best_t, ctx_of=ctx_of)
# #         d_snap = float(np.mean([r["dice"] for r in rows.values()]))
# #         refine = bool(d_snap > best_d + 0.003)                   # keep the boundary snap only if it clearly helps
# #         say(f"  boundary snap: val Dice {d_snap:.3f} vs {best_d:.3f} without -> {'KEPT' if refine else 'not used'}")
# #         best = dict(kind=kind, params=p, t_add=best_t, refine=refine)
# #         json.dump(best, open(cfg_path, "w"), indent=1)
# #         say(f"chosen: {best}")
# #     kind, p, t_add, refine = best["kind"], best["params"], best["t_add"], best.get("refine", False)

# #     # ---------------- variants, oracles and baselines
# #     say(f"\n=== {eval_name.upper()} RESULTS (full resolution 640x480) ===")
# #     m_all = fit_model(kind, p, entries, fit_ids, cols_all)
# #     m_se = fit_model(kind, p, entries, fit_ids, cols_se)
# #     m_tx = fit_model(kind, p, entries, fit_ids, cols_tx)
# #     chi_med = float(np.median(np.concatenate([entries[i]["feats"][:, IDX["lbp_chi2"]] for i in fit_ids if len(entries[i]["iou"])])))
# #     f_all = lambda e: proba(m_all, e, cols_all)
# #     full_name = "E+ FULL (classifier, several potholes" + (", boundary snap)" if refine else ")")
# #     variants = {
# #         "A edge/shape heuristic (no learning)": (lambda e: e["score_a"], None, "score", None),
# #         "B A + texture rule": (lambda e: e["score_a"] * e["feats"][:, IDX["lbp_chi2"]] / (e["feats"][:, IDX["lbp_chi2"]] + chi_med + 1e-9), None, "score", None),
# #         "C classifier on shape/edge only": (lambda e: proba(m_se, e, cols_se), None, "score", None),
# #         "D classifier on texture only": (lambda e: proba(m_tx, e, cols_tx), None, "score", None),
# #         "E FULL: classifier on all features (1 region)": (f_all, None, "score", None),
# #         full_name: (f_all, t_add, "score", ctx_of if refine else None),
# #         "oracle single (best one candidate)": (None, None, "oracle", None),
# #         "oracle multi (best candidate per pothole)": (None, None, "oracle_multi", None),
# #         "random candidate (lower bound)": (None, None, "random", None),
# #     }
# #     results, preds_all, table = {}, {}, []
# #     for name, (fn, t, mode, cf) in variants.items():
# #         rows, preds = eval_variant(ev_ids, entries, gts, fn, t_add=t, mode=mode, ctx_of=cf)
# #         results[name], preds_all[name] = rows, preds
# #         table.append(table_row(name, rows))
# #     acc = np.zeros((pc.H, pc.W), np.float32)
# #     tr_gts = []
# #     for i in split_ids["train"]:
# #         g = pc.gt_mask(paths.at[i, "label_path"])[0]
# #         acc += g
# #         tr_gts.append(g)
# #     acc /= max(len(tr_gts), 1)
# #     cands_mm = [(acc >= f * acc.max()).astype(np.uint8) for f in (0.3, 0.5, 0.7, 0.9)]
# #     mm = max(cands_mm, key=lambda mk: np.mean([pc.dice_iou(mk, g)[0] for g in tr_gts]))   # best on TRAIN
# #     rows = {i: dict(score_row(mm, gts[i]), n_regions=1) for i in ev_ids}
# #     results["mean-mask baseline (fixed centred blob)"] = rows
# #     table.append(table_row("mean-mask baseline (fixed centred blob)", rows))
# #     m_sh = fit_model(kind, p, entries, fit_ids, cols_all, shuffle_labels=True)
# #     rows, _ = eval_variant(ev_ids, entries, gts, lambda e: proba(m_sh, e, cols_all))
# #     table.append(table_row("SANITY: trained on shuffled labels", rows))

# #     T = pd.DataFrame(table).set_index("variant")
# #     cols = ["n", "dice_mean", "dice_ci", "dice_median", "iou_mean", "pooled_iou", "precision", "recall", "hit50", "hit70"]
# #     pd.set_option("display.width", 220)
# #     say(T[cols].round(3).to_string())
# #     T[cols].round(4).to_csv(os.path.join(pc.OUT, f"variants_{eval_name}.csv"))

# #     # candidate coverage (where does the pipeline lose?)
# #     best_u = np.array([e["iou_u"].max() if len(e["iou_u"]) else 0 for e in (entries[i] for i in ev_ids)])
# #     inst_best = np.concatenate([e["iou_mat"].astype(np.float32).max(0) if e["iou_mat"].ndim == 2 and e["iou_mat"].shape[0] else
# #                                 np.zeros(e["n_inst"]) for e in (entries[i] for i in ev_ids)])
# #     say(f"\ncandidate coverage (work res): best single candidate vs whole GT has IoU>=0.5 in {100 * (best_u >= 0.5).mean():.0f}% of images, "
# #         f">=0.7 in {100 * (best_u >= 0.7).mean():.0f}%, >=0.85 in {100 * (best_u >= 0.85).mean():.0f}%")
# #     say(f"ground-truth potholes ({len(inst_best)} in {len(ev_ids)} images) that have a candidate with IoU>=0.5: {100 * (inst_best >= 0.5).mean():.0f}%, >=0.7: {100 * (inst_best >= 0.7).mean():.0f}%")
# #     srcs = [SRC_NAMES[int(entries[i]["src"][int(np.argmax(entries[i]["iou_u"]))])] for i in ev_ids
# #             if len(entries[i]["iou_u"]) and entries[i]["iou_u"].max() >= 0.5]
# #     if srcs:
# #         vc = pd.Series(srcs).value_counts()
# #         say("source of the best candidate (IoU>=0.5):", {k: int(v) for k, v in vc.items()}, "<- sources that never win can be removed")
# #     rank1 = []
# #     for i in ev_ids[:60]:
# #         e = entries[i]
# #         ctx = ctx_of(i)
# #         g = cv2.resize(gts[i], WORK, interpolation=cv2.INTER_NEAREST)
# #         n, lab, st, _ = cv2.connectedComponentsWithStats(g, connectivity=8)
# #         if n <= 1:
# #             continue
# #         g = lab == (1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))
# #         fv = full_feats(g, ctx, base_feats(g, ctx), 0)[None]
# #         s_gt = m_all.predict_proba(fv[:, cols_all])[0, 1]
# #         rank1.append((s_gt, s_gt - (proba(m_all, e, cols_all).max() if len(e["iou"]) else 0)))
# #     if rank1:
# #         rk = np.array(rank1)
# #         say(f"SANITY GT-as-candidate: the TRUE outline is accepted (probability>=0.5) in {100 * (rk[:, 0] >= 0.5).mean():.0f}% of {len(rk)} images "
# #             f"and scores within 0.1 of the best generated candidate in {100 * (rk[:, 1] >= -0.1).mean():.0f}% "
# #             "(should be high; if low, the features cannot recognise even a perfect region)")

# #     # ---------------- per-image table, failure taxonomy, depth diagnostic
# #     R = pd.DataFrame({i: results[full_name][i] for i in ev_ids}).T.astype(float)
# #     R.index.name = "image_id"
# #     R["dice_E"] = R["dice"]
# #     for key, nm in (("A", "A edge/shape heuristic (no learning)"), ("C", "C classifier on shape/edge only"),
# #                     ("D", "D classifier on texture only"), ("oracle", "oracle multi (best candidate per pothole)")):
# #         R[f"dice_{key}"] = [results[nm][i]["dice"] for i in ev_ids]
# #     R["oracle_iou"] = [results["oracle multi (best candidate per pothole)"][i]["iou"] for i in ev_ids]
# #     R["gt_frac"] = [gts[i].mean() for i in ev_ids]
# #     R["gt_blobs"] = [int(cv2.connectedComponents(gts[i])[0] - 1) for i in ev_ids]

# #     def ftype(r):
# #         if r.iou >= 0.75:
# #             return "good"
# #         if r.oracle_iou < 0.5:
# #             return "candidate_generation_failed"
# #         if r.iou < 0.5:
# #             return "wrong_candidate_selected"
# #         return "boundary_imprecise"
# #     R["failure_type"] = R.apply(ftype, axis=1)
# #     R["size_error"] = np.where(R.pred_frac > 1.5 * R.gt_frac, "over", np.where(R.pred_frac < R.gt_frac / 1.5, "under", "ok"))
# #     dg, dp, okg, okp, mL = [], [], [], [], []
# #     for i in ev_ids:
# #         d = pc.load_depth_cm(paths.at[i, "depth_path"])
# #         mL.append(float(cv2.cvtColor(pc.load_rgb(paths.at[i, "rgb_path"])[0], cv2.COLOR_BGR2LAB)[..., 0].mean()))
# #         pred = preds_all[full_name][i]
# #         rg, ogk, _, _ = pc.rel_depth(d, gts[i])
# #         rp, opk, _, _ = pc.rel_depth(d, pred)
# #         dg.append(pc.depth_stats(rg, gts[i])["p90"]); dp.append(pc.depth_stats(rp, pred)["p90"])
# #         okg.append(ogk); okp.append(opk)
# #     R["depth_p90_gt"], R["depth_p90_pred"], R["depth_ok_gt"], R["depth_ok_pred"], R["mean_L"] = dg, dp, okg, okp, mL
# #     R.round(4).to_csv(os.path.join(pc.OUT, f"per_image_{eval_name}.csv"))
# #     say("\nfailure types (final pipeline):", {k: int(v) for k, v in R.failure_type.value_counts().items()})
# #     say("size errors:", {k: int(v) for k, v in R.size_error.value_counts().items()}, "| images with several GT blobs:", int((R.gt_blobs > 1).sum()))
# #     ratio = (R.pred_frac / R.gt_frac).replace([np.inf, -np.inf], np.nan).dropna()
# #     say(f"predicted/true area ratio, percentiles 10/50/90: {' / '.join(f'{np.percentile(ratio, q):.2f}' for q in (10, 50, 90))}; "
# #         f"within +-25% of true area: {100 * ((ratio > 0.75) & (ratio < 1.25)).mean():.0f}%")
# #     bd = []
# #     for fac, nice in (("gt_frac", "pothole size"), ("mean_L", "image brightness"), ("depth_p90_gt", "depth (cm below road)")):
# #         v = R[fac].astype(float)
# #         if v.notna().sum() < 9 or v.nunique() < 3:
# #             continue
# #         for b, g in R.groupby(pd.qcut(v, 3, duplicates="drop"), observed=True):
# #             bd.append(dict(factor=nice, bin=str(b), n=len(g), A_heuristic=g.dice_A.mean(), C_shape_edge=g.dice_C.mean(),
# #                            D_texture=g.dice_D.mean(), E_full=g.dice_E.mean(), oracle=g.dice_oracle.mean()))
# #     if bd:
# #         BD = pd.DataFrame(bd)
# #         BD.round(3).to_csv(os.path.join(pc.OUT, f"breakdown_{eval_name}.csv"), index=False)
# #         say("\nmean Dice by condition (terciles):")
# #         say(BD.round(3).to_string(index=False))
# #     both = R[R.depth_ok_gt & R.depth_ok_pred].dropna(subset=["depth_p90_gt", "depth_p90_pred"])
# #     if len(both) > 3:
# #         from scipy.stats import spearmanr
# #         rho = spearmanr(both.depth_p90_gt, both.depth_p90_pred)[0]
# #         say(f"depth preview: on {len(both)} images with reliable depth, depth-below-road from the PREDICTED mask vs the GT mask: "
# #             f"Spearman {rho:.2f}, mean abs difference {np.mean(np.abs(both.depth_p90_gt - both.depth_p90_pred)):.2f} cm")
# #     if kind == "rf":
# #         imp = pd.Series(m_all.feature_importances_, index=FEATURES).sort_values(ascending=False)
# #         imp.round(4).to_csv(os.path.join(pc.OUT, "feature_importance.csv"), header=["importance"])
# #         say("top-10 features:", ", ".join(f"{k} {v:.3f}" for k, v in imp.head(10).items()))
# #         # Feature pruning suggestion: flag bottom-20% features that contribute < 0.1% each.
# #         # Removing noise features often improves generalisation on small datasets.
# #         threshold = max(imp.max() * 0.001, 1e-5)
# #         weak = imp[imp < threshold]
# #         if len(weak):
# #             say(f"  [{len(weak)} low-importance features each contribute <{threshold*100:.3f}% — "
# #                 f"consider removing: {', '.join(weak.index[:8].tolist())}{'...' if len(weak) > 8 else ''}]")

# #     # ---------------- predicted masks for step3_features.py
# #     ids_out = list(ev_ids)
# #     packed = np.stack([np.packbits(preds_all[full_name][i].astype(np.uint8).ravel()) for i in ids_out])
# #     np.savez_compressed(os.path.join(pc.OUT, f"pred_masks_{eval_name}.npz"), ids=np.array(ids_out), packed=packed)

# #     # ---------------- overlays: worst and random cases
# #     odir = os.path.join(pc.OUT, f"overlays_{eval_name}")
# #     os.makedirs(odir, exist_ok=True)
# #     for f_ in os.listdir(odir):
# #         os.remove(os.path.join(odir, f_))
# #     rng = np.random.default_rng(0)
# #     worst = R.sort_values("dice").index[:16].tolist()
# #     rest = [i for i in R.index if i not in worst]
# #     rand = list(rng.choice(rest, size=min(8, len(rest)), replace=False)) if rest else []
# #     for tag, lst in (("worst", worst), ("random", rand)):
# #         for n, i in enumerate(lst):
# #             rgb, _ = pc.load_rgb(paths.at[i, "rgb_path"])
# #             ctx = ctx_of(i)
# #             left = pc.overlay(rgb, gt=gts[i], pred=preds_all[full_name][i],
# #                               text=f"{i} dice {R.at[i, 'dice']:.2f} {R.at[i, 'failure_type']}")
# #             right = cv2.cvtColor(cv2.resize(ctx["E_ref"], (pc.W, pc.H), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
# #             right = pc.overlay(right, gt=gts[i], pred=preds_all[full_name][i], text="edge map + GT(green)/pred(red)")
# #             pc.imwrite(os.path.join(odir, f"{tag}_{n:02d}_{i}.jpg"), np.hstack([left, right]))
# #     say(f"\n[ok] overlays in {odir}  (worst 16 + 8 random)")
# #     try:
# #         import matplotlib
# #         matplotlib.use("Agg")
# #         import matplotlib.pyplot as plt
# #         fig, ax = plt.subplots(figsize=(7, 4))
# #         for nm, lab_ in (("A edge/shape heuristic (no learning)", "A"), (full_name, "final"), ("oracle multi (best candidate per pothole)", "oracle")):
# #             ax.hist([results[nm][i]["dice"] for i in ev_ids], bins=20, range=(0, 1), alpha=0.5, label=lab_)
# #         ax.set_xlabel("Dice per image"); ax.set_ylabel("images"); ax.legend()
# #         fig.tight_layout(); fig.savefig(os.path.join(pc.OUT, f"dice_hist_{eval_name}.png"), dpi=120)
# #     except Exception as ex:
# #         say("(matplotlib plot skipped:", ex, ")")
# #     with open(os.path.join(pc.OUT, f"summary_{eval_name}.txt"), "w", encoding="utf-8") as f:
# #         f.write("\n".join(log))


# # def main():
# #     ap = argparse.ArgumentParser()
# #     sub = ap.add_subparsers(dest="cmd", required=True)
# #     b = sub.add_parser("build")
# #     b.add_argument("--workers", type=int, default=1)
# #     b.add_argument("--limit", type=int, default=0, help="smoke test on a random subset")
# #     r = sub.add_parser("run")
# #     r.add_argument("--test", action="store_true", help="final evaluation on the test split (use once)")
# #     r.add_argument("--smoke", action="store_true")
# #     r.add_argument("--models", default="rf,svm", choices=["rf", "rf,svm"])
# #     args = ap.parse_args()
# #     (cmd_build if args.cmd == "build" else cmd_run)(args)


# # if __name__ == "__main__":
# #     main()

# """
# step2_segment.py  --  Step 2 (version 2): classical pothole segmentation, threshold-hybrid pipeline.

# Pipeline (RGB only, no pretrained deep models):
#   1. Preprocess   CLAHE (contrast), bilateral denoise + unsharp mask ("blur removal"); Lab colour smoothing.
#   2. Candidates   (a) histogram thresholding (Otsu and triangle, both polarities) of lightness, two colour
#                       channels and local texture energy            [Koch & Brilakis 2011; Jakstys 2016]
#                   (b) k-means colour clustering and local adaptive thresholding
#                   (c) filled outlines of closed Canny edge groups   [proposal]
#                   Every mask is cleaned with morphology; lobed blobs are split by watershed on the distance
#                   transform [Chung & Khan 2019]; near-duplicates are removed (NMS).
#   3. Features     shape + edge support, and texture/colour of the INSIDE compared with a ring of ROAD around it.
#   4. Classifier   Random Forest / SVM scores every candidate; every candidate that scores high enough and does
#                   not overlap a better one is kept (several potholes per image are allowed).
#   5. Boundary     optional marker-controlled watershed that snaps the outline to Canny edges (kept only if it
#                   helps on validation).

# Usage (after step1_prepare.py):
#     python step2_segment.py build --workers 4      # slow part, run once, cached in 'cache'
#     python step2_segment.py run                    # train on TRAIN, tune/evaluate on VAL, write all reports
#     python step2_segment.py run --test             # FINAL: refit on train+val, evaluate on TEST (do this once)
# Quick smoke test on 40 images:  build --limit 40  then  run --smoke
# """
# import argparse, os, sys, json, pickle, time, hashlib, warnings
# import numpy as np, cv2, pandas as pd
# from concurrent.futures import ProcessPoolExecutor
# from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import make_pipeline
# import pothole_common as pc

# warnings.filterwarnings("ignore", category=RuntimeWarning)

# # ====================================================================== configuration
# C = dict(
#     work_w=320, work_h=240,            # candidates + features are computed at half resolution (speed, less noise)
#     clahe_clip=2.0, clahe_tile=8,      # contrast adjustment
#     bilateral=(5, 25, 5),              # edge-preserving denoise (d, sigmaColor, sigmaSpace)
#     unsharp_sigma=2.0, unsharp_amount=0.6,   # mild "blur removal"
#     thr_sigma=2.0,                     # smoothing of L/a/b before thresholding (removes single stones)
#     tex_win=9,                         # window of the local-texture (std) channel
#     kmeans_ks=(2, 3),                  # colour clusters
#     adapt_blocks=(51, 121), adapt_c=6, # local adaptive threshold: block size (odd) and offset
#     blur_sigmas=(1.5, 3.0), canny_pcts=(88, 94), ref_pct=90,   # Canny: two smoothing scales, strongest 12% / 6% of gradients
#     close_ks=(5, 11),                  # closing for Canny outlines
#     open_k=5,                          # opening that removes thin tails from filled regions
#     mask_open=3, mask_close=7,         # clean-up of threshold masks (close joins stones inside a pothole)
#     max_comps_per_mask=12,             # largest blobs kept per binary mask
#     split_solidity=0.85, split_frac=0.5,   # watershed split of lobed blobs: only if solidity < 0.85; markers at 50% of max distance
#     min_area_frac=0.003, max_area_frac=0.90,
#     nms_iou=0.85, max_cands=120,
#     ring_scale=0.35, ring_min=5, ring_max=30, ring_gap=2,   # road ring around a candidate (pixels at work resolution)
#     pos_iou=0.5, neg_iou=0.3,          # labels for training (see make_labels)
#     rel_pos=0.10, rel_neg=0.20,
#     lbp_P=8, lbp_R=1,
#     max_regions=15, overlap_skip=0.3,  # several potholes per image; skip a region that overlaps a kept one by >30% of the smaller
#     snap_frac=0.25,                    # boundary snap band = 25% of the region's equivalent radius
#     svm_max_samples=20000,
# )
# PIPELINE_VERSION = 3          # bumped: ellipse-fit candidate source added
# WORK = (C["work_w"], C["work_h"])
# LBP_BINS = C["lbp_P"] + 2
# GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")
# SRC_NAMES = ("canny", "otsu", "triangle", "kmeans", "adaptive", "split", "ellipse")


# def ell(k):
#     return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(k), int(k)))


# def cfg_hash():
#     s = json.dumps(dict(C, v=PIPELINE_VERSION), sort_keys=True, default=str)
#     return hashlib.md5(s.encode()).hexdigest()[:8]


# # ====================================================================== feature names and groups
# FEATURES = (["area_frac", "solidity", "circ", "elong", "extent", "ell_res", "cx", "cy", "border_frac",
#              "edge_support", "grad_b"] + [f"src_{s}" for s in SRC_NAMES])
# N_SHAPE = len(FEATURES)
# FEATURES += ["mean_in", "std_in", "dmean", "std_ratio", "da", "db", "lap_in", "lap_ratio", "grad_in", "grad_ratio",
#              "ed_in", "ed_ring", "ed_diff", "entropy_in"]
# FEATURES += [f"lbp_in_{i}" for i in range(LBP_BINS)] + [f"lbp_d_{i}" for i in range(LBP_BINS)] + ["lbp_chi2"]
# FEATURES += [f"glcm_in_{p}" for p in GLCM_PROPS] + [f"glcm_r_{p}" for p in GLCM_PROPS] + [f"glcm_d_{p}" for p in GLCM_PROPS]
# FEATURES += ["ring_ok"]
# GROUPS = {"shape_edge": list(range(N_SHAPE)), "texture": list(range(N_SHAPE, len(FEATURES))),
#           "all": list(range(len(FEATURES)))}
# IDX = {n: i for i, n in enumerate(FEATURES)}


# # ====================================================================== 1. preprocessing
# def make_ctx(bgr):
#     """Everything computed once per image at working resolution."""
#     h, w = bgr.shape[:2]
#     lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
#     L, A, B = cv2.split(lab)
#     # --- for EDGES: contrast adjustment + denoise + mild deblur (proposal: "contrast adjustment, blur removal")
#     clahe = cv2.createCLAHE(clipLimit=C["clahe_clip"], tileGridSize=(C["clahe_tile"],) * 2)
#     Lc = clahe.apply(L)
#     den = cv2.bilateralFilter(Lc, *C["bilateral"])
#     blur = cv2.GaussianBlur(den, (0, 0), C["unsharp_sigma"])
#     enh = cv2.addWeighted(den, 1 + C["unsharp_amount"], blur, -C["unsharp_amount"], 0)

#     def canny_at(sigma, pct):
#         g = cv2.GaussianBlur(enh, (0, 0), sigma)
#         gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
#         gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
#         hi = float(np.percentile(np.hypot(gx, gy), pct))      # image-adaptive threshold (handles lighting changes)
#         return cv2.Canny(g, 0.5 * hi, hi, L2gradient=True)

#     edge_maps = [canny_at(s, p) for s in C["blur_sigmas"] for p in C["canny_pcts"]]
#     E_ref = canny_at(C["blur_sigmas"][0], C["ref_pct"])
#     edge_near = cv2.dilate(E_ref, np.ones((5, 5), np.uint8)) > 0
#     edge_snap = cv2.dilate(canny_at(C["blur_sigmas"][0], C["canny_pcts"][-1]), np.ones((3, 3), np.uint8)) > 0
#     # --- for THRESHOLDING: smoothed raw Lab channels + local texture energy
#     sg = C["thr_sigma"]
#     Ls, As, Bs = (cv2.GaussianBlur(x, (0, 0), sg) for x in (L, A, B))
#     Lf = L.astype(np.float32)
#     mu = cv2.blur(Lf, (C["tex_win"],) * 2)
#     mu2 = cv2.blur(Lf * Lf, (C["tex_win"],) * 2)
#     T = cv2.GaussianBlur(np.sqrt(np.maximum(mu2 - mu * mu, 0)), (0, 0), sg)
#     lo, hi = np.percentile(T, 1), np.percentile(T, 99)
#     T8 = np.clip((T - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
#     bgr_s = cv2.bilateralFilter(bgr, 5, 25, 5)               # smoothed colour image for the watershed steps
#     # --- for FEATURES use the RAW lightness (CLAHE would equalise the very inside/outside contrast we want to measure)
#     Lg = cv2.GaussianBlur(L, (0, 0), 1.0)
#     gx = cv2.Sobel(Lg, cv2.CV_32F, 1, 0, ksize=3)
#     gy = cv2.Sobel(Lg, cv2.CV_32F, 0, 1, ksize=3)
#     mag = np.hypot(gx, gy)
#     lap = np.abs(cv2.Laplacian(Lg, cv2.CV_32F))
#     lbp = local_binary_pattern(L, C["lbp_P"], C["lbp_R"], method="uniform").astype(np.uint8)
#     return dict(h=h, w=w, L=L, A=A, B=B, edge_maps=edge_maps, E_ref=E_ref, edge_near=edge_near, edge_snap=edge_snap,
#                 chan=dict(L=Ls, A=As, B=Bs, T=T8), bgr_s=bgr_s,
#                 mag=mag, mag_mean=float(mag.mean()) + 1e-6, lap=lap, lbp=lbp)


# # ====================================================================== 2. candidate generation
# def fill_outer(m_u8):
#     cs, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     out = np.zeros_like(m_u8)
#     cv2.drawContours(out, cs, -1, 1, -1)
#     return out


# def clean_region(m_u8):
#     """Shape clean-up: remove thin tails, keep the largest blob, fill holes."""
#     m = cv2.morphologyEx(m_u8, cv2.MORPH_OPEN, ell(C["open_k"]))
#     n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
#     if n <= 1:
#         return None
#     k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
#     return fill_outer((lab == k).astype(np.uint8))


# def ellipse_candidate(comp):
#     """Fit an ellipse to the candidate's largest contour and return it as a FILLED mask -- a smooth,
#     regularised version of the same region. Added as an EXTRA candidate alongside the original comp
#     (never replacing it): a free-form thresholded blob can keep following a stray edge indefinitely
#     (this is our main over-size failure mode -- predicted/true area ratio has a long heavy tail), while
#     a fitted ellipse physically cannot. [literature: the candidate RoI is commonly modelled as an ellipse
#     before texture confirmation, e.g. Koch & Brilakis-style pipelines]. Returns None if the contour has
#     too few points to fit (cv2.fitEllipse needs >=5)."""
#     cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not cs:
#         return None
#     c = max(cs, key=cv2.contourArea)
#     if len(c) < 5:
#         return None
#     try:
#         e = cv2.fitEllipse(c)
#     except cv2.error:
#         return None
#     em = np.zeros(comp.shape, np.uint8)
#     cv2.ellipse(em, e, 1, -1)
#     return em if em.sum() > 0 else None


# def split_lobes(comp, ctx, amin):
#     """Watershed on the distance transform: separates potholes that were merged into one blob."""
#     cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     area = float(comp.sum())
#     harea = max(cv2.contourArea(cv2.convexHull(max(cs, key=cv2.contourArea))), 1.0)
#     if area / harea >= C["split_solidity"]:
#         return []
#     D = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
#     sure_fg = (D > C["split_frac"] * D.max()).astype(np.uint8)
#     n, mk = cv2.connectedComponents(sure_fg)
#     if n - 1 < 2:
#         return []
#     markers = np.zeros(comp.shape, np.int32)
#     markers[comp == 0] = 1
#     markers[mk > 0] = mk[mk > 0] + 1
#     cv2.watershed(ctx["bgr_s"], markers)
#     parts = []
#     for k in range(2, n + 1):
#         part = ((markers == k) & (comp > 0)).astype(np.uint8)
#         if part.sum() >= amin:
#             parts.append(part)
#     return parts


# def binary_masks(ctx):
#     """All thresholding-based binary masks, tagged with their source name."""
#     h, w = ctx["h"], ctx["w"]
#     out = []
#     for ch in ctx["chan"].values():                     # (a) histogram thresholding, both polarities
#         for src, flag in (("otsu", cv2.THRESH_OTSU), ("triangle", cv2.THRESH_TRIANGLE)):
#             t, _ = cv2.threshold(ch, 0, 255, cv2.THRESH_BINARY + flag)
#             out.append(((ch < t).astype(np.uint8), src))
#             out.append(((ch > t).astype(np.uint8), src))
#     feats = np.dstack([ctx["chan"]["L"], ctx["chan"]["A"], ctx["chan"]["B"]]).reshape(-1, 3).astype(np.float32)
#     crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
#     for k in C["kmeans_ks"]:                            # (b) colour clustering
#         cv2.setRNGSeed(0)
#         _, labels, _ = cv2.kmeans(feats, k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
#         labels = labels.reshape(h, w)
#         for j in range(k):
#             out.append(((labels == j).astype(np.uint8), "kmeans"))
#     Ls = ctx["chan"]["L"]
#     for b in C["adapt_blocks"]:                         # (b) local adaptive thresholding (small potholes on uniform road)
#         out.append((cv2.adaptiveThreshold(Ls, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, b, C["adapt_c"]), "adaptive"))
#         out.append((cv2.adaptiveThreshold(Ls, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, b, -C["adapt_c"]), "adaptive"))
#     return out


# def raw_candidates(ctx):
#     h, w = ctx["h"], ctx["w"]
#     amin, amax = C["min_area_frac"] * h * w, C["max_area_frac"] * h * w
#     out, seen = [], set()

#     def add(m, src):
#         if m is None:
#             return
#         a = int(m.sum())
#         if not (amin <= a <= amax):
#             return
#         key = (cv2.resize(m.astype(np.uint8), (64, 48), interpolation=cv2.INTER_AREA) > 0).tobytes()
#         if key in seen:                                 # exact duplicates (same blob from different thresholds)
#             return
#         seen.add(key)
#         out.append((m.astype(bool), SRC_NAMES.index(src)))

#     for m, src in binary_masks(ctx):
#         m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ell(C["mask_open"]))
#         m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ell(C["mask_close"]))
#         n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
#         order = np.argsort(-st[1:, cv2.CC_STAT_AREA])[:C["max_comps_per_mask"]] + 1
#         for j in order:
#             if st[j, cv2.CC_STAT_AREA] < amin:
#                 continue
#             comp = fill_outer((lab == j).astype(np.uint8))
#             add(comp, src)
#             add(ellipse_candidate(comp), "ellipse")
#             for part in split_lobes(comp, ctx, amin):
#                 part = clean_region(part)
#                 add(part, "split")
#                 if part is not None:
#                     add(ellipse_candidate(part), "ellipse")
#     for E in ctx["edge_maps"]:                          # (c) filled outlines of closed edge groups (proposal)
#         for k in C["close_ks"]:
#             closed = cv2.morphologyEx(E, cv2.MORPH_CLOSE, ell(k))
#             cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             for c in cnts:
#                 if cv2.contourArea(c) < 0.5 * amin:
#                     continue
#                 f = np.zeros((h, w), np.uint8)
#                 cv2.drawContours(f, [c], -1, 1, -1)
#                 f = clean_region(f)
#                 add(f, "canny")
#                 if f is not None:
#                     add(ellipse_candidate(f), "ellipse")
#     return out


# def base_feats(mask, ctx):
#     """Shape + edge-support features (cheap; also used to rank candidates before the classifier exists)."""
#     h, w = mask.shape
#     ys, xs = np.nonzero(mask)
#     area = len(xs)
#     x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
#     cm = np.pad(mask[y0:y1 + 1, x0:x1 + 1].astype(np.uint8), 1)
#     cnts, _ = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     c = max(cnts, key=cv2.contourArea)
#     per = max(cv2.arcLength(c, True), 1.0)
#     harea = max(cv2.contourArea(cv2.convexHull(c)), 1.0)
#     rw, rh = cv2.minAreaRect(c)[1]
#     ell_res = 1.0
#     if len(c) >= 5:
#         try:
#             e = cv2.fitEllipse(c)
#             em = np.zeros_like(cm)
#             cv2.ellipse(em, e, 1, -1)
#             ell_res = float((em.astype(bool) ^ cm.astype(bool)).sum()) / area
#         except cv2.error:
#             pass
#     frame = int(mask[0, :].sum() + mask[-1, :].sum() + mask[:, 0].sum() + mask[:, -1].sum())
#     bnd = cm.astype(bool) & ~(cv2.erode(cm, np.ones((3, 3), np.uint8)) > 0)
#     en = np.pad(ctx["edge_near"][y0:y1 + 1, x0:x1 + 1], 1)
#     mg = np.pad(ctx["mag"][y0:y1 + 1, x0:x1 + 1], 1)
#     return dict(area_frac=area / (h * w), solidity=min(area / harea, 1.0), circ=min(4 * np.pi * area / per ** 2, 1.5),
#                 elong=min(max(rw, rh) / max(min(rw, rh), 1.0), 20.0), extent=area / ((x1 - x0 + 1) * (y1 - y0 + 1)),
#                 ell_res=min(ell_res, 5.0), cx=float(xs.mean()) / w, cy=float(ys.mean()) / h,
#                 border_frac=min(frame / per, 1.0),
#                 edge_support=float(en[bnd].mean()) if bnd.any() else 0.0,
#                 grad_b=float(mg[bnd].mean() / ctx["mag_mean"]) if bnd.any() else 0.0)


# def heuristic_score(b, prior):
#     """Technique 1 on its own (no learning): closed, edge-supported, compact, plausible-size, not cut by the border."""
#     ap = max(np.exp(-0.5 * ((np.log(max(b["area_frac"], 1e-6)) - prior[0]) / prior[1]) ** 2), 0.05)
#     return float(b["edge_support"] * (0.5 + 0.5 * b["solidity"]) * ap * (1 - 0.5 * b["border_frac"]))


# def nms_and_cap(cands, prior):
#     """Score, drop near-duplicates (IoU > nms_iou on 1/4-res masks), keep the best max_cands."""
#     if not cands:
#         return []
#     scored = [(heuristic_score(b, prior), m, src, b) for m, src, b in cands]
#     scored.sort(key=lambda t: -t[0])
#     small = np.stack([cv2.resize(m.astype(np.uint8), (C["work_w"] // 4, C["work_h"] // 4),
#                                  interpolation=cv2.INTER_AREA).ravel() > 0 for _, m, _, _ in scored]).astype(np.float32)
#     inter = small @ small.T
#     ar = small.sum(1)
#     iou = inter / np.maximum(ar[:, None] + ar[None, :] - inter, 1.0)
#     keep = []
#     for i in range(len(scored)):
#         if all(iou[i, j] <= C["nms_iou"] for j in keep):
#             keep.append(i)
#         if len(keep) >= C["max_cands"]:
#             break
#     return [scored[i] for i in keep]


# def full_feats(mask, ctx, base, src):
#     """Features: base shape/edge + inside-vs-road-ring texture comparison."""
#     h, w = mask.shape
#     ys, xs = np.nonzero(mask)
#     area = len(xs)
#     x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
#     rw = int(np.clip(C["ring_scale"] * np.sqrt(area), C["ring_min"], C["ring_max"]))
#     mg = rw + C["ring_gap"] + 2
#     cx0, cy0, cx1, cy1 = max(0, x0 - mg), max(0, y0 - mg), min(w, x1 + mg + 1), min(h, y1 + mg + 1)
#     m = mask[cy0:cy1, cx0:cx1].astype(np.uint8)
#     ring = (cv2.dilate(m, ell(2 * rw + 1)) > 0) & ~(cv2.dilate(m, ell(2 * C["ring_gap"] + 1)) > 0)
#     core = cv2.erode(m, ell(5)) > 0               # inside, away from the blurry boundary
#     if core.sum() < 0.3 * m.sum():
#         core = m > 0
#     ring_ok = bool(ring.sum() >= 50)
#     L = ctx["L"][cy0:cy1, cx0:cx1]
#     Af = ctx["A"][cy0:cy1, cx0:cx1].astype(np.float32)
#     Bf = ctx["B"][cy0:cy1, cx0:cx1].astype(np.float32)
#     lap = ctx["lap"][cy0:cy1, cx0:cx1]
#     mag = ctx["mag"][cy0:cy1, cx0:cx1]
#     Er = (ctx["E_ref"][cy0:cy1, cx0:cx1] > 0)
#     lb = ctx["lbp"][cy0:cy1, cx0:cx1]
#     nan = np.nan
#     Li = L[core].astype(np.float32)
#     f = {k: base[k] for k in FEATURES[:11]}
#     for n, s in enumerate(SRC_NAMES):
#         f[f"src_{s}"] = 1.0 if src == n else 0.0
#     f["mean_in"] = Li.mean() / 255.0
#     f["std_in"] = Li.std() / 255.0
#     f["lap_in"] = lap[core].mean()
#     f["grad_in"] = mag[core].mean() / ctx["mag_mean"]
#     f["ed_in"] = Er[core].mean()
#     hist = np.bincount((Li.astype(np.int32) >> 4), minlength=16)[:16] / max(len(Li), 1)
#     f["entropy_in"] = float(-(hist[hist > 0] * np.log(hist[hist > 0])).sum())
#     hin = np.bincount(lb[core], minlength=LBP_BINS)[:LBP_BINS] / max(core.sum(), 1)
#     for i in range(LBP_BINS):
#         f[f"lbp_in_{i}"] = hin[i]

#     def glcm_vec(sel):
#         q = (L >> 4).astype(np.uint8)
#         q = np.where(sel, q, 16).astype(np.uint8)          # level 16 = "ignore" (outside the region)
#         g = graycomatrix(q, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=17, symmetric=True, normed=False)
#         P = g[:16, :16].astype(np.float64)
#         if P.sum() < 30:
#             return [nan] * len(GLCM_PROPS)
#         return [float(graycoprops(P, p).mean()) for p in GLCM_PROPS]

#     gin = glcm_vec(core)
#     for p, v in zip(GLCM_PROPS, gin):
#         f[f"glcm_in_{p}"] = v
#     if ring_ok:
#         Lr = L[ring].astype(np.float32)
#         f["dmean"] = (Li.mean() - Lr.mean()) / 255.0
#         f["std_ratio"] = min(Li.std() / (Lr.std() + 1.0), 20.0)
#         f["da"] = (Af[core].mean() - Af[ring].mean()) / 255.0
#         f["db"] = (Bf[core].mean() - Bf[ring].mean()) / 255.0
#         lr = lap[ring].mean()
#         f["lap_ratio"] = min(f["lap_in"] / (lr + 0.5), 50.0)
#         f["grad_ratio"] = min(f["grad_in"] / (mag[ring].mean() / ctx["mag_mean"] + 0.05), 50.0)
#         f["ed_ring"] = Er[ring].mean()
#         f["ed_diff"] = f["ed_in"] - f["ed_ring"]
#         hr = np.bincount(lb[ring], minlength=LBP_BINS)[:LBP_BINS] / max(ring.sum(), 1)
#         for i in range(LBP_BINS):
#             f[f"lbp_d_{i}"] = hin[i] - hr[i]
#         f["lbp_chi2"] = float(0.5 * np.sum((hin - hr) ** 2 / (hin + hr + 1e-9)))
#         gr = glcm_vec(ring)
#         for p, a, b in zip(GLCM_PROPS, gin, gr):
#             f[f"glcm_r_{p}"] = b
#             f[f"glcm_d_{p}"] = a - b
#     f["ring_ok"] = 1.0 if ring_ok else 0.0
#     v = np.array([f.get(n, nan) for n in FEATURES], np.float64)
#     return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# def rle_encode(m):
#     f = m.ravel().astype(np.int8)
#     d = np.diff(np.concatenate([[0], f, [0]]))
#     return np.flatnonzero(d == 1).astype(np.int32), np.flatnonzero(d == -1).astype(np.int32)


# def rle_decode(rle, shape):
#     n = shape[0] * shape[1]
#     d = np.zeros(n + 1, np.int8)
#     d[rle[0]] += 1
#     d[rle[1]] -= 1
#     return (np.cumsum(d[:n]) > 0).reshape(shape)


# def load_work(rgb_path):
#     bgr, _ = pc.load_rgb(rgb_path)
#     return cv2.resize(bgr, WORK, interpolation=cv2.INTER_AREA)


# def instance_masks(label_path):
#     """One work-resolution mask per ground-truth polygon (a pothole each)."""
#     polys, _ = pc.read_polygons(label_path)
#     out = []
#     for p in polys:
#         m = np.zeros((pc.H, pc.W), np.uint8)
#         cv2.fillPoly(m, [p], 1)
#         mw = cv2.resize(m, WORK, interpolation=cv2.INTER_NEAREST).astype(bool)
#         if mw.any():
#             out.append(mw)
#     return out


# def process_image(args):
#     """One image -> candidates (RLE), feature matrix, IoU with every ground-truth pothole."""
#     iid, rgb_path, label_path, prior = args
#     cv2.setNumThreads(1)                          # avoid CPU oversubscription when several workers run
#     cv2.setRNGSeed(0)
#     try:
#         ctx = make_ctx(load_work(rgb_path))
#         inst = instance_masks(label_path)
#         raw = raw_candidates(ctx)
#         pre = [(m, s, base_feats(m, ctx)) for m, s in raw]
#         kept = nms_and_cap(pre, prior)
#         feats, rles, srcs, sa, masks = [], [], [], [], []
#         for score, m, src, base in kept:
#             feats.append(full_feats(m, ctx, base, src))
#             rles.append(rle_encode(m))
#             srcs.append(src)
#             sa.append(score)
#             masks.append(m)
#         n = len(masks)
#         if n and inst:
#             Cm = np.stack([m.ravel() for m in masks]).astype(np.float32)
#             Im = np.stack([g.ravel() for g in inst]).astype(np.float32)
#             inter = Cm @ Im.T
#             iou_mat = inter / np.maximum(Cm.sum(1)[:, None] + Im.sum(1)[None, :] - inter, 1.0)
#             U = np.any(np.stack(inst), axis=0).ravel().astype(np.float32)
#             iu = (Cm @ U) / np.maximum(Cm.sum(1) + U.sum() - Cm @ U, 1.0)
#         else:
#             iou_mat = np.zeros((n, len(inst)), np.float32)
#             iu = np.zeros(n, np.float32)
#         iou = iou_mat.max(1) if iou_mat.shape[1] else np.zeros(n, np.float32)
#         ref = (iou_mat.max(0)[iou_mat.argmax(1)] if iou_mat.shape[1] and n else np.zeros(n, np.float32))
#         return dict(id=iid, ok=True, has_gt=bool(inst), n_inst=len(inst),
#                     feats=np.array(feats, np.float32).reshape(n, len(FEATURES)),
#                     iou=iou.astype(np.float32), ref=ref.astype(np.float32), iou_u=iu.astype(np.float32),
#                     iou_mat=iou_mat.astype(np.float16), rle=rles, src=np.array(srcs, np.int8),
#                     score_a=np.array(sa, np.float32), n_raw=len(raw))
#     except Exception as ex:                       # never let one bad image kill a 1000-image run
#         return dict(id=iid, ok=False, error=repr(ex))


# # ====================================================================== build (cache)
# def cache_path(smoke=False):
#     return os.path.join(pc.CACHE, f"features_{cfg_hash()}{'_smoke' if smoke else ''}.pkl")


# def train_prior(df, sp):
#     """Area prior (log-normal) over individual TRAIN ground-truth potholes."""
#     ids = set(sp.image_id[sp.split == "train"])
#     la = []
#     for _, r in df[df.image_id.isin(ids)].iterrows():
#         polys, _ = pc.read_polygons(r.label_path)
#         for p in polys:
#             m = np.zeros((pc.H, pc.W), np.uint8)
#             cv2.fillPoly(m, [p], 1)
#             if m.sum() > 0:
#                 la.append(np.log(m.sum() / (pc.W * pc.H)))
#     return float(np.mean(la)), float(max(np.std(la), 0.7))


# def cmd_build(args):
#     if not os.path.exists(pc.SPLITS):
#         sys.exit("splits.csv not found. Run step1_prepare.py first.")
#     os.makedirs(pc.CACHE, exist_ok=True)
#     df, sp = pc.load_index(), pd.read_csv(pc.SPLITS)
#     prior = train_prior(df, sp)
#     print(f"area prior from TRAIN GT: mean log-area {prior[0]:.2f} (~{100 * np.exp(prior[0]):.1f}% of image), sd {prior[1]:.2f}")
#     sub = df if not args.limit else df.sample(n=min(args.limit, len(df)), random_state=0)
#     jobs = [(r.image_id, r.rgb_path, r.label_path, prior) for r in sub.itertuples()]
#     t0 = time.time()
#     results = []
#     if args.workers > 1:
#         with ProcessPoolExecutor(max_workers=args.workers) as ex:
#             for i, res in enumerate(ex.map(process_image, jobs, chunksize=4), 1):
#                 results.append(res)
#                 if i % 50 == 0:
#                     print(f"  {i}/{len(jobs)}  ({time.time() - t0:.0f}s)")
#     else:
#         for i, j in enumerate(jobs, 1):
#             results.append(process_image(j))
#             if i % 25 == 0:
#                 print(f"  {i}/{len(jobs)}  ({time.time() - t0:.0f}s)")
#     bad = [r for r in results if not r["ok"]]
#     good = {r["id"]: r for r in results if r["ok"]}
#     if bad:
#         print(f"!! {len(bad)} images FAILED and are excluded:", [(b['id'], b['error']) for b in bad[:5]])
#     nc = np.array([len(r["iou"]) for r in good.values()])
#     print(f"candidates per image: median {int(np.median(nc))}, min {nc.min()}, max {nc.max()}; images with 0 candidates: {(nc == 0).sum()}")
#     with open(cache_path(args.limit > 0), "wb") as f:
#         pickle.dump(dict(meta=dict(cfg=C, features=FEATURES, prior=prior), entries=good, failed=bad), f)
#     print(f"[ok] cached {len(good)} images in {time.time() - t0:.0f}s -> {cache_path(args.limit > 0)}")


# # ====================================================================== model helpers
# def make_model(kind, p):
#     if kind == "rf":
#         return RandomForestClassifier(n_estimators=200, min_samples_leaf=3, max_features="sqrt",
#                                       class_weight="balanced_subsample", n_jobs=-1, random_state=0)
#     return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=p.get("C", 1.0), gamma="scale",
#                                               class_weight="balanced", probability=True, random_state=0))


# def make_labels(iou, ref):
#     """1 = good outline of a ground-truth pothole (IoU>=0.5 and close to the best candidate for THAT pothole),
#     0 = not a pothole outline (IoU<0.3, or clearly worse than the best candidate for its pothole),
#     -1 = ambiguous, left out of training."""
#     y = np.full(len(iou), -1, int)
#     y[iou >= np.maximum(C["pos_iou"], ref - C["rel_pos"])] = 1
#     neg = (iou < C["neg_iou"]) | ((ref >= C["pos_iou"]) & (iou < ref - C["rel_neg"]))
#     y[neg & (y != 1)] = 0
#     return y


# def build_xy(entries, ids, cols, shuffle_labels=False):
#     X, y = [], []
#     for i in ids:
#         e = entries[i]
#         if not e["has_gt"] or len(e["iou"]) == 0:
#             continue
#         lab = make_labels(e["iou"], e["ref"])
#         keep = lab >= 0
#         X.append(e["feats"][keep][:, cols])
#         y.append(lab[keep])
#     X, y = np.vstack(X), np.concatenate(y)
#     if y.sum() == 0 or y.sum() == len(y):
#         raise RuntimeError("training set has only one class; candidate generation found no good candidates (see 'build' output)")
#     if shuffle_labels:
#         y = np.random.default_rng(0).permutation(y)
#     return X, y


# def fit_model(kind, p, entries, ids, cols, shuffle_labels=False):
#     X, y = build_xy(entries, ids, cols, shuffle_labels)
#     if kind == "svm" and len(y) > C["svm_max_samples"]:
#         rng = np.random.default_rng(0)
#         pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
#         n_neg = max(C["svm_max_samples"] - len(pos), 1000)
#         sel = np.concatenate([pos, rng.choice(neg, min(n_neg, len(neg)), replace=False)])
#         X, y = X[sel], y[sel]
#     m = make_model(kind, p)
#     m.fit(X, y)
#     return m


# def proba(model, e, cols):
#     if len(e["iou"]) == 0:
#         return np.zeros(0)
#     return model.predict_proba(e["feats"][:, cols])[:, 1]


# # ====================================================================== prediction + evaluation
# def fill_holes(m):
#     return fill_outer(m.astype(np.uint8))


# def to_full(mask_work):
#     """work-res bool mask -> clean 640x480 uint8 mask."""
#     f = cv2.resize(mask_work.astype(np.float32), (pc.W, pc.H), interpolation=cv2.INTER_LINEAR) > 0.5
#     return fill_holes(cv2.morphologyEx(f.astype(np.uint8), cv2.MORPH_OPEN, ell(3)))


# def choose(e, scores, t_add):
#     """Best candidate always; further candidates with score>=t_add that do not overlap a kept one by more than
#     overlap_skip of the smaller region (several potholes per image), at most max_regions."""
#     if len(scores) == 0:
#         return []
#     order = np.argsort(-scores)
#     sel = [int(order[0])]
#     if t_add is not None:
#         shape = (WORK[1], WORK[0])
#         masks = {sel[0]: rle_decode(e["rle"][sel[0]], shape)}
#         for j in order[1:]:
#             if len(sel) >= C["max_regions"] or scores[j] < t_add:
#                 break
#             mj = rle_decode(e["rle"][int(j)], shape)
#             aj = max(int(mj.sum()), 1)
#             if all((mj & masks[k]).sum() / max(min(aj, int(masks[k].sum())), 1) <= C["overlap_skip"] for k in sel):
#                 sel.append(int(j))
#                 masks[int(j)] = mj
#     return sel


# def snap_region(region, ctx):
#     """Marker-controlled watershed: the outline moves to the strongest nearby edge inside a band around the region.
#     Canny edges are drawn as bright ridges that the flood cannot cross cheaply."""
#     reg = region.astype(np.uint8)
#     area = int(reg.sum())
#     r = np.sqrt(area / np.pi)
#     k = int(max(3, round(C["snap_frac"] * r)))
#     inner = cv2.erode(reg, ell(2 * k + 1))
#     outer = cv2.dilate(reg, ell(2 * k + 1))
#     if inner.sum() < 20:
#         return region
#     img = ctx["bgr_s"].copy()
#     img[ctx["edge_snap"]] = 255
#     markers = np.zeros(reg.shape, np.int32)
#     markers[outer == 0] = 1
#     markers[inner > 0] = 2
#     cv2.watershed(img, markers)
#     out = markers == 2
#     if out.sum() < 0.5 * area or out.sum() > 1.5 * area:
#         return region
#     return out


# def pred_work_mask(e, sel, snap_ctx=None):
#     shape = (WORK[1], WORK[0])
#     m = np.zeros(shape, bool)
#     for j in sel:
#         r = rle_decode(e["rle"][j], shape)
#         m |= snap_region(r, snap_ctx) if snap_ctx is not None else r
#     return m


# def score_row(pred_full, gt_full):
#     d, i = pc.dice_iou(pred_full, gt_full)
#     p, g = pred_full.astype(bool), gt_full.astype(bool)
#     tp = float((p & g).sum())
#     return dict(dice=d, iou=i, tp=tp, fp=float(p.sum() - tp), fn=float(g.sum() - tp), pred_frac=p.mean())


# def oracle_multi_sel(e):
#     """Best candidate for every ground-truth pothole (upper bound for several potholes per image)."""
#     m = e["iou_mat"].astype(np.float32)
#     sel = []
#     for j in range(m.shape[1] if m.ndim == 2 else 0):
#         if m.shape[0] == 0:
#             break
#         k = int(np.argmax(m[:, j]))
#         if m[k, j] >= 0.3 and k not in sel:
#             sel.append(k)
#     return sel


# def eval_variant(ids, entries, gts, scorer, t_add=None, mode="score", ctx_of=None):
#     """scorer(entry)->scores. mode: score | oracle | oracle_multi | random.
#     ctx_of(id) -> image context enables the boundary snap."""
#     rows, preds = {}, {}
#     rng = np.random.default_rng(0)
#     for i in ids:
#         e = entries[i]
#         if len(e["iou"]) == 0:
#             sel = []
#         elif mode == "oracle":
#             sel = [int(np.argmax(e["iou_u"]))]
#         elif mode == "oracle_multi":
#             sel = oracle_multi_sel(e) or [int(np.argmax(e["iou_u"]))]
#         elif mode == "random":
#             sel = [int(rng.integers(len(e["iou"])))]
#         else:
#             sel = choose(e, scorer(e), t_add)
#         if sel:
#             pred = to_full(pred_work_mask(e, sel, ctx_of(i) if ctx_of else None))
#         else:
#             pred = np.zeros((pc.H, pc.W), np.uint8)
#         r = score_row(pred, gts[i])
#         r["n_regions"] = len(sel)
#         rows[i], preds[i] = r, pred
#     return rows, preds


# def table_row(name, rows):
#     df = pd.DataFrame(rows.values())
#     s = pc.summarize(df.dice, df.iou, df.tp, df.fp, df.fn)
#     lo, hi = pc.boot_ci(df.dice.values)
#     s.update(variant=name, dice_ci=f"{lo:.3f}-{hi:.3f}")
#     return s


# # ====================================================================== run
# def cmd_run(args):
#     path = cache_path(args.smoke)
#     if not os.path.exists(path):
#         sys.exit(f"cache not found ({path}). Run:  python step2_segment.py build")
#     with open(path, "rb") as f:
#         pk = pickle.load(f)
#     entries = pk["entries"]
#     df, sp = pc.load_index(), pd.read_csv(pc.SPLITS)
#     paths = df.set_index("image_id")
#     split_ids = {s: [i for i in sp.image_id[sp.split == s] if i in entries and entries[i]["has_gt"]]
#                  for s in ("train", "val", "test")}
#     for s, v in split_ids.items():
#         print(f"{s}: {len(v)} usable images")
#     n_empty = sum(1 for e in entries.values() if not e["has_gt"])
#     if n_empty:
#         print(f"({n_empty} images without ground truth are excluded)")
#     os.makedirs(pc.OUT, exist_ok=True)
#     cfg_path = os.path.join(pc.OUT, "best_config.json")
#     log = []

#     def say(*a):
#         s = " ".join(str(x) for x in a)
#         print(s)
#         log.append(s)

#     if args.test:
#         if not os.path.exists(cfg_path):
#             sys.exit("best_config.json missing. Run `run` (validation) first, then `run --test`.")
#         best = json.load(open(cfg_path))
#         fit_ids, eval_name = split_ids["train"] + split_ids["val"], "test"
#         say(f"FINAL TEST RUN: refitting '{best['kind']}' on train+val, evaluating on test ONCE")
#     else:
#         best, fit_ids, eval_name = None, split_ids["train"], "val"
#     ev_ids = split_ids[eval_name]
#     gts = {i: pc.gt_mask(paths.at[i, "label_path"])[0] for i in ev_ids}
#     cols_all, cols_se, cols_tx = GROUPS["all"], GROUPS["shape_edge"], GROUPS["texture"]
#     ctx_cache = {}

#     def ctx_of(i):
#         if i not in ctx_cache:
#             ctx_cache[i] = make_ctx(load_work(paths.at[i, "rgb_path"]))
#         return ctx_cache[i]

#     # ---------------- model selection on VAL (only in validation mode)
#     if best is None:
#         kinds = [("rf", {})] if args.models == "rf" else [("rf", {}), ("svm", {"C": 1.0}), ("svm", {"C": 10.0})]
#         sel_rows = []
#         for kind, p in kinds:
#             t0 = time.time()
#             m = fit_model(kind, p, entries, fit_ids, cols_all)
#             rows, _ = eval_variant(ev_ids, entries, gts, lambda e, m=m: proba(m, e, cols_all))
#             s = table_row(f"{kind} {p}", rows)
#             say(f"model {kind} {p}: val Dice {s['dice_mean']:.3f} (top-1 only)  [{time.time() - t0:.0f}s]")
#             sel_rows.append((s["dice_mean"], kind, p, m))
#         sel_rows.sort(key=lambda t: -t[0])
#         _, kind, p, model_all = sel_rows[0]
#         best_t, best_d = None, -1
#         for t in (None, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):          # probability needed to keep an extra region
#             rows, _ = eval_variant(ev_ids, entries, gts, lambda e: proba(model_all, e, cols_all), t_add=t)
#             d = float(np.mean([r["dice"] for r in rows.values()]))
#             say(f"  extra-region threshold {t}: val Dice {d:.3f}")
#             if d > best_d + 1e-4:
#                 best_t, best_d = t, d
#         rows, _ = eval_variant(ev_ids, entries, gts, lambda e: proba(model_all, e, cols_all), t_add=best_t, ctx_of=ctx_of)
#         d_snap = float(np.mean([r["dice"] for r in rows.values()]))
#         refine = bool(d_snap > best_d + 0.003)                   # keep the boundary snap only if it clearly helps
#         say(f"  boundary snap: val Dice {d_snap:.3f} vs {best_d:.3f} without -> {'KEPT' if refine else 'not used'}")
#         best = dict(kind=kind, params=p, t_add=best_t, refine=refine)
#         json.dump(best, open(cfg_path, "w"), indent=1)
#         say(f"chosen: {best}")
#     kind, p, t_add, refine = best["kind"], best["params"], best["t_add"], best.get("refine", False)

#     # ---------------- variants, oracles and baselines
#     say(f"\n=== {eval_name.upper()} RESULTS (full resolution 640x480) ===")
#     m_all = fit_model(kind, p, entries, fit_ids, cols_all)
#     m_se = fit_model(kind, p, entries, fit_ids, cols_se)
#     m_tx = fit_model(kind, p, entries, fit_ids, cols_tx)
#     chi_med = float(np.median(np.concatenate([entries[i]["feats"][:, IDX["lbp_chi2"]] for i in fit_ids if len(entries[i]["iou"])])))
#     f_all = lambda e: proba(m_all, e, cols_all)
#     full_name = "E+ FULL (classifier, several potholes" + (", boundary snap)" if refine else ")")
#     variants = {
#         "A edge/shape heuristic (no learning)": (lambda e: e["score_a"], None, "score", None),
#         "B A + texture rule": (lambda e: e["score_a"] * e["feats"][:, IDX["lbp_chi2"]] / (e["feats"][:, IDX["lbp_chi2"]] + chi_med + 1e-9), None, "score", None),
#         "C classifier on shape/edge only": (lambda e: proba(m_se, e, cols_se), None, "score", None),
#         "D classifier on texture only": (lambda e: proba(m_tx, e, cols_tx), None, "score", None),
#         "E FULL: classifier on all features (1 region)": (f_all, None, "score", None),
#         full_name: (f_all, t_add, "score", ctx_of if refine else None),
#         "oracle single (best one candidate)": (None, None, "oracle", None),
#         "oracle multi (best candidate per pothole)": (None, None, "oracle_multi", None),
#         "random candidate (lower bound)": (None, None, "random", None),
#     }
#     results, preds_all, table = {}, {}, []
#     for name, (fn, t, mode, cf) in variants.items():
#         rows, preds = eval_variant(ev_ids, entries, gts, fn, t_add=t, mode=mode, ctx_of=cf)
#         results[name], preds_all[name] = rows, preds
#         table.append(table_row(name, rows))
#     acc = np.zeros((pc.H, pc.W), np.float32)
#     tr_gts = []
#     for i in split_ids["train"]:
#         g = pc.gt_mask(paths.at[i, "label_path"])[0]
#         acc += g
#         tr_gts.append(g)
#     acc /= max(len(tr_gts), 1)
#     cands_mm = [(acc >= f * acc.max()).astype(np.uint8) for f in (0.3, 0.5, 0.7, 0.9)]
#     mm = max(cands_mm, key=lambda mk: np.mean([pc.dice_iou(mk, g)[0] for g in tr_gts]))   # best on TRAIN
#     rows = {i: dict(score_row(mm, gts[i]), n_regions=1) for i in ev_ids}
#     results["mean-mask baseline (fixed centred blob)"] = rows
#     table.append(table_row("mean-mask baseline (fixed centred blob)", rows))
#     m_sh = fit_model(kind, p, entries, fit_ids, cols_all, shuffle_labels=True)
#     rows, _ = eval_variant(ev_ids, entries, gts, lambda e: proba(m_sh, e, cols_all))
#     table.append(table_row("SANITY: trained on shuffled labels", rows))

#     T = pd.DataFrame(table).set_index("variant")
#     cols = ["n", "dice_mean", "dice_ci", "dice_median", "iou_mean", "pooled_iou", "precision", "recall", "hit50", "hit70"]
#     pd.set_option("display.width", 220)
#     say(T[cols].round(3).to_string())
#     T[cols].round(4).to_csv(os.path.join(pc.OUT, f"variants_{eval_name}.csv"))

#     # candidate coverage (where does the pipeline lose?)
#     best_u = np.array([e["iou_u"].max() if len(e["iou_u"]) else 0 for e in (entries[i] for i in ev_ids)])
#     inst_best = np.concatenate([e["iou_mat"].astype(np.float32).max(0) if e["iou_mat"].ndim == 2 and e["iou_mat"].shape[0] else
#                                 np.zeros(e["n_inst"]) for e in (entries[i] for i in ev_ids)])
#     say(f"\ncandidate coverage (work res): best single candidate vs whole GT has IoU>=0.5 in {100 * (best_u >= 0.5).mean():.0f}% of images, "
#         f">=0.7 in {100 * (best_u >= 0.7).mean():.0f}%, >=0.85 in {100 * (best_u >= 0.85).mean():.0f}%")
#     say(f"ground-truth potholes ({len(inst_best)} in {len(ev_ids)} images) that have a candidate with IoU>=0.5: {100 * (inst_best >= 0.5).mean():.0f}%, >=0.7: {100 * (inst_best >= 0.7).mean():.0f}%")
#     srcs = [SRC_NAMES[int(entries[i]["src"][int(np.argmax(entries[i]["iou_u"]))])] for i in ev_ids
#             if len(entries[i]["iou_u"]) and entries[i]["iou_u"].max() >= 0.5]
#     if srcs:
#         vc = pd.Series(srcs).value_counts()
#         say("source of the best candidate (IoU>=0.5):", {k: int(v) for k, v in vc.items()}, "<- sources that never win can be removed")
#     rank1 = []
#     for i in ev_ids[:60]:
#         e = entries[i]
#         ctx = ctx_of(i)
#         g = cv2.resize(gts[i], WORK, interpolation=cv2.INTER_NEAREST)
#         n, lab, st, _ = cv2.connectedComponentsWithStats(g, connectivity=8)
#         if n <= 1:
#             continue
#         g = lab == (1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))
#         fv = full_feats(g, ctx, base_feats(g, ctx), 0)[None]
#         s_gt = m_all.predict_proba(fv[:, cols_all])[0, 1]
#         rank1.append((s_gt, s_gt - (proba(m_all, e, cols_all).max() if len(e["iou"]) else 0)))
#     if rank1:
#         rk = np.array(rank1)
#         say(f"SANITY GT-as-candidate: the TRUE outline is accepted (probability>=0.5) in {100 * (rk[:, 0] >= 0.5).mean():.0f}% of {len(rk)} images "
#             f"and scores within 0.1 of the best generated candidate in {100 * (rk[:, 1] >= -0.1).mean():.0f}% "
#             "(should be high; if low, the features cannot recognise even a perfect region)")

#     # ---------------- per-image table, failure taxonomy, depth diagnostic
#     R = pd.DataFrame({i: results[full_name][i] for i in ev_ids}).T.astype(float)
#     R.index.name = "image_id"
#     R["dice_E"] = R["dice"]
#     for key, nm in (("A", "A edge/shape heuristic (no learning)"), ("C", "C classifier on shape/edge only"),
#                     ("D", "D classifier on texture only"), ("oracle", "oracle multi (best candidate per pothole)")):
#         R[f"dice_{key}"] = [results[nm][i]["dice"] for i in ev_ids]
#     R["oracle_iou"] = [results["oracle multi (best candidate per pothole)"][i]["iou"] for i in ev_ids]
#     R["gt_frac"] = [gts[i].mean() for i in ev_ids]
#     R["gt_blobs"] = [int(cv2.connectedComponents(gts[i])[0] - 1) for i in ev_ids]

#     def ftype(r):
#         if r.iou >= 0.75:
#             return "good"
#         if r.oracle_iou < 0.5:
#             return "candidate_generation_failed"
#         if r.iou < 0.5:
#             return "wrong_candidate_selected"
#         return "boundary_imprecise"
#     R["failure_type"] = R.apply(ftype, axis=1)
#     R["size_error"] = np.where(R.pred_frac > 1.5 * R.gt_frac, "over", np.where(R.pred_frac < R.gt_frac / 1.5, "under", "ok"))
#     dg, dp, okg, okp, mL = [], [], [], [], []
#     for i in ev_ids:
#         d = pc.load_depth_cm(paths.at[i, "depth_path"])
#         mL.append(float(cv2.cvtColor(pc.load_rgb(paths.at[i, "rgb_path"])[0], cv2.COLOR_BGR2LAB)[..., 0].mean()))
#         pred = preds_all[full_name][i]
#         rg, ogk, _, _ = pc.rel_depth(d, gts[i])
#         rp, opk, _, _ = pc.rel_depth(d, pred)
#         dg.append(pc.depth_stats(rg, gts[i])["p90"]); dp.append(pc.depth_stats(rp, pred)["p90"])
#         okg.append(ogk); okp.append(opk)
#     R["depth_p90_gt"], R["depth_p90_pred"], R["depth_ok_gt"], R["depth_ok_pred"], R["mean_L"] = dg, dp, okg, okp, mL
#     R.round(4).to_csv(os.path.join(pc.OUT, f"per_image_{eval_name}.csv"))
#     say("\nfailure types (final pipeline):", {k: int(v) for k, v in R.failure_type.value_counts().items()})
#     say("size errors:", {k: int(v) for k, v in R.size_error.value_counts().items()}, "| images with several GT blobs:", int((R.gt_blobs > 1).sum()))
#     ratio = (R.pred_frac / R.gt_frac).replace([np.inf, -np.inf], np.nan).dropna()
#     say(f"predicted/true area ratio, percentiles 10/50/90: {' / '.join(f'{np.percentile(ratio, q):.2f}' for q in (10, 50, 90))}; "
#         f"within +-25% of true area: {100 * ((ratio > 0.75) & (ratio < 1.25)).mean():.0f}%")
#     bd = []
#     for fac, nice in (("gt_frac", "pothole size"), ("mean_L", "image brightness"), ("depth_p90_gt", "depth (cm below road)")):
#         v = R[fac].astype(float)
#         if v.notna().sum() < 9 or v.nunique() < 3:
#             continue
#         for b, g in R.groupby(pd.qcut(v, 3, duplicates="drop"), observed=True):
#             bd.append(dict(factor=nice, bin=str(b), n=len(g), A_heuristic=g.dice_A.mean(), C_shape_edge=g.dice_C.mean(),
#                            D_texture=g.dice_D.mean(), E_full=g.dice_E.mean(), oracle=g.dice_oracle.mean()))
#     if bd:
#         BD = pd.DataFrame(bd)
#         BD.round(3).to_csv(os.path.join(pc.OUT, f"breakdown_{eval_name}.csv"), index=False)
#         say("\nmean Dice by condition (terciles):")
#         say(BD.round(3).to_string(index=False))
#     both = R[R.depth_ok_gt & R.depth_ok_pred].dropna(subset=["depth_p90_gt", "depth_p90_pred"])
#     if len(both) > 3:
#         from scipy.stats import spearmanr
#         rho = spearmanr(both.depth_p90_gt, both.depth_p90_pred)[0]
#         say(f"depth preview: on {len(both)} images with reliable depth, depth-below-road from the PREDICTED mask vs the GT mask: "
#             f"Spearman {rho:.2f}, mean abs difference {np.mean(np.abs(both.depth_p90_gt - both.depth_p90_pred)):.2f} cm")
#     if kind == "rf":
#         imp = pd.Series(m_all.feature_importances_, index=FEATURES).sort_values(ascending=False)
#         imp.round(4).to_csv(os.path.join(pc.OUT, "feature_importance.csv"), header=["importance"])
#         say("top-10 features:", ", ".join(f"{k} {v:.3f}" for k, v in imp.head(10).items()))

#     # ---------------- predicted masks for step3_features.py
#     ids_out = list(ev_ids)
#     packed = np.stack([np.packbits(preds_all[full_name][i].astype(np.uint8).ravel()) for i in ids_out])
#     np.savez_compressed(os.path.join(pc.OUT, f"pred_masks_{eval_name}.npz"), ids=np.array(ids_out), packed=packed)

#     # ---------------- overlays: worst and random cases
#     odir = os.path.join(pc.OUT, f"overlays_{eval_name}")
#     os.makedirs(odir, exist_ok=True)
#     for f_ in os.listdir(odir):
#         os.remove(os.path.join(odir, f_))
#     rng = np.random.default_rng(0)
#     worst = R.sort_values("dice").index[:16].tolist()
#     rest = [i for i in R.index if i not in worst]
#     rand = list(rng.choice(rest, size=min(8, len(rest)), replace=False)) if rest else []
#     for tag, lst in (("worst", worst), ("random", rand)):
#         for n, i in enumerate(lst):
#             rgb, _ = pc.load_rgb(paths.at[i, "rgb_path"])
#             ctx = ctx_of(i)
#             left = pc.overlay(rgb, gt=gts[i], pred=preds_all[full_name][i],
#                               text=f"{i} dice {R.at[i, 'dice']:.2f} {R.at[i, 'failure_type']}")
#             right = cv2.cvtColor(cv2.resize(ctx["E_ref"], (pc.W, pc.H), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
#             right = pc.overlay(right, gt=gts[i], pred=preds_all[full_name][i], text="edge map + GT(green)/pred(red)")
#             pc.imwrite(os.path.join(odir, f"{tag}_{n:02d}_{i}.jpg"), np.hstack([left, right]))
#     say(f"\n[ok] overlays in {odir}  (worst 16 + 8 random)")
#     try:
#         import matplotlib
#         matplotlib.use("Agg")
#         import matplotlib.pyplot as plt
#         fig, ax = plt.subplots(figsize=(7, 4))
#         for nm, lab_ in (("A edge/shape heuristic (no learning)", "A"), (full_name, "final"), ("oracle multi (best candidate per pothole)", "oracle")):
#             ax.hist([results[nm][i]["dice"] for i in ev_ids], bins=20, range=(0, 1), alpha=0.5, label=lab_)
#         ax.set_xlabel("Dice per image"); ax.set_ylabel("images"); ax.legend()
#         fig.tight_layout(); fig.savefig(os.path.join(pc.OUT, f"dice_hist_{eval_name}.png"), dpi=120)
#     except Exception as ex:
#         say("(matplotlib plot skipped:", ex, ")")
#     with open(os.path.join(pc.OUT, f"summary_{eval_name}.txt"), "w", encoding="utf-8") as f:
#         f.write("\n".join(log))


# def main():
#     ap = argparse.ArgumentParser()
#     sub = ap.add_subparsers(dest="cmd", required=True)
#     b = sub.add_parser("build")
#     b.add_argument("--workers", type=int, default=1)
#     b.add_argument("--limit", type=int, default=0, help="smoke test on a random subset")
#     r = sub.add_parser("run")
#     r.add_argument("--test", action="store_true", help="final evaluation on the test split (use once)")
#     r.add_argument("--smoke", action="store_true")
#     r.add_argument("--models", default="rf,svm", choices=["rf", "rf,svm"])
#     args = ap.parse_args()
#     (cmd_build if args.cmd == "build" else cmd_run)(args)


# if __name__ == "__main__":
#     main()

"""
step2_segment.py  --  Step 2 (version 2): classical pothole segmentation, threshold-hybrid pipeline.

Pipeline (RGB only, no pretrained deep models):
  1. Preprocess   CLAHE (contrast), bilateral denoise + unsharp mask ("blur removal"); Lab colour smoothing.
  2. Candidates   (a) histogram thresholding (Otsu and triangle, both polarities) of lightness, two colour
                      channels and local texture energy            [Koch & Brilakis 2011; Jakstys 2016]
                  (b) k-means colour clustering and local adaptive thresholding
                  (c) filled outlines of closed Canny edge groups   [proposal]
                  Every mask is cleaned with morphology; lobed blobs are split by watershed on the distance
                  transform [Chung & Khan 2019]; near-duplicates are removed (NMS).
  3. Features     shape + edge support, and texture/colour of the INSIDE compared with a ring of ROAD around it.
  4. Classifier   Random Forest / SVM scores every candidate; every candidate that scores high enough and does
                  not overlap a better one is kept (several potholes per image are allowed).
  5. Boundary     optional marker-controlled watershed that snaps the outline to Canny edges (kept only if it
                  helps on validation).

Usage (after step1_prepare.py):
    python step2_segment.py build --workers 4      # slow part, run once, cached in 'cache'
    python step2_segment.py run                    # train on TRAIN, tune/evaluate on VAL, write all reports
    python step2_segment.py run --test             # FINAL: refit on train+val, evaluate on TEST (do this once)
Quick smoke test on 40 images:  build --limit 40  then  run --smoke
"""
import argparse, os, sys, json, pickle, time, hashlib, warnings
import numpy as np, cv2, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pothole_common as pc

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ====================================================================== configuration
C = dict(
    work_w=320, work_h=240,            # candidates + features are computed at half resolution (speed, less noise)
    clahe_clip=2.0, clahe_tile=8,      # contrast adjustment
    bilateral=(5, 25, 5),              # edge-preserving denoise (d, sigmaColor, sigmaSpace)
    unsharp_sigma=2.0, unsharp_amount=0.6,   # mild "blur removal"
    thr_sigma=2.0,                     # smoothing of L/a/b before thresholding (removes single stones)
    tex_win=9,                         # window of the local-texture (std) channel
    kmeans_ks=(2, 3),                  # colour clusters
    adapt_blocks=(51, 121), adapt_c=6, # local adaptive threshold: block size (odd) and offset
    blur_sigmas=(1.5, 3.0), canny_pcts=(88, 94), ref_pct=90,   # Canny: two smoothing scales, strongest 12% / 6% of gradients
    close_ks=(5, 11),                  # closing for Canny outlines
    open_k=5,                          # opening that removes thin tails from filled regions
    mask_open=3, mask_close=7,         # clean-up of threshold masks (close joins stones inside a pothole)
    max_comps_per_mask=12,             # largest blobs kept per binary mask
    split_solidity=0.85, split_frac=0.5,   # watershed split of lobed blobs: only if solidity < 0.85; markers at 50% of max distance
    min_area_frac=0.003, max_area_frac=0.90,
    nms_iou=0.70, max_cands=120,       # FIX(attempt2a): was 0.85. The ellipse source (attempt1) adds a near-
                                        # duplicate of almost every candidate, which 0.85 let both survive --
                                        # doubling the pool with near-identical pairs, which the per-image
                                        # sanity check showed was confusing the classifier (0.85 -> see
                                        # chat results before this change). 0.70 collapses an ellipse and its
                                        # source candidate into one before the classifier ever sees both.
    area_prior_max_z=2.5,              # FIX(attempt2b): candidates whose log-area is more than this many prior
                                        # std-devs ABOVE the TRAIN area prior are dropped at the SOURCE, not just
                                        # downweighted. Targets the oversized tail (90th pct area ratio ~2.3-2.4x,
                                        # median Dice 0.58 on those images) that ellipse-fitting alone could not
                                        # fix, because fitting an ellipse to an already-too-big blob just gives
                                        # an equally-big ellipse.
    ring_scale=0.35, ring_min=5, ring_max=30, ring_gap=2,   # road ring around a candidate (pixels at work resolution)
    pos_iou=0.5, neg_iou=0.3,          # labels for training (see make_labels)
    rel_pos=0.10, rel_neg=0.20,
    lbp_P=8, lbp_R=1,
    max_regions=15, overlap_skip=0.3,  # several potholes per image; skip a region that overlaps a kept one by >30% of the smaller
    snap_frac=0.25,                    # boundary snap band = 25% of the region's equivalent radius
    svm_max_samples=20000,
)
PIPELINE_VERSION = 3          # bumped: ellipse-fit candidate source added
WORK = (C["work_w"], C["work_h"])
LBP_BINS = C["lbp_P"] + 2
GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")
SRC_NAMES = ("canny", "otsu", "triangle", "kmeans", "adaptive", "split", "ellipse")


def ell(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(k), int(k)))


def cfg_hash():
    s = json.dumps(dict(C, v=PIPELINE_VERSION), sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()[:8]


# ====================================================================== feature names and groups
FEATURES = (["area_frac", "solidity", "circ", "elong", "extent", "ell_res", "cx", "cy", "border_frac",
             "edge_support", "grad_b"] + [f"src_{s}" for s in SRC_NAMES])
N_SHAPE = len(FEATURES)
FEATURES += ["mean_in", "std_in", "dmean", "std_ratio", "da", "db", "lap_in", "lap_ratio", "grad_in", "grad_ratio",
             "ed_in", "ed_ring", "ed_diff", "entropy_in"]
FEATURES += [f"lbp_in_{i}" for i in range(LBP_BINS)] + [f"lbp_d_{i}" for i in range(LBP_BINS)] + ["lbp_chi2"]
FEATURES += [f"glcm_in_{p}" for p in GLCM_PROPS] + [f"glcm_r_{p}" for p in GLCM_PROPS] + [f"glcm_d_{p}" for p in GLCM_PROPS]
FEATURES += ["ring_ok"]
GROUPS = {"shape_edge": list(range(N_SHAPE)), "texture": list(range(N_SHAPE, len(FEATURES))),
          "all": list(range(len(FEATURES)))}
IDX = {n: i for i, n in enumerate(FEATURES)}


# ====================================================================== 1. preprocessing
def make_ctx(bgr):
    """Everything computed once per image at working resolution."""
    h, w = bgr.shape[:2]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    # --- for EDGES: contrast adjustment + denoise + mild deblur (proposal: "contrast adjustment, blur removal")
    clahe = cv2.createCLAHE(clipLimit=C["clahe_clip"], tileGridSize=(C["clahe_tile"],) * 2)
    Lc = clahe.apply(L)
    den = cv2.bilateralFilter(Lc, *C["bilateral"])
    blur = cv2.GaussianBlur(den, (0, 0), C["unsharp_sigma"])
    enh = cv2.addWeighted(den, 1 + C["unsharp_amount"], blur, -C["unsharp_amount"], 0)

    def canny_at(sigma, pct):
        g = cv2.GaussianBlur(enh, (0, 0), sigma)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        hi = float(np.percentile(np.hypot(gx, gy), pct))      # image-adaptive threshold (handles lighting changes)
        return cv2.Canny(g, 0.5 * hi, hi, L2gradient=True)

    edge_maps = [canny_at(s, p) for s in C["blur_sigmas"] for p in C["canny_pcts"]]
    E_ref = canny_at(C["blur_sigmas"][0], C["ref_pct"])
    edge_near = cv2.dilate(E_ref, np.ones((5, 5), np.uint8)) > 0
    edge_snap = cv2.dilate(canny_at(C["blur_sigmas"][0], C["canny_pcts"][-1]), np.ones((3, 3), np.uint8)) > 0
    # --- for THRESHOLDING: smoothed raw Lab channels + local texture energy
    sg = C["thr_sigma"]
    Ls, As, Bs = (cv2.GaussianBlur(x, (0, 0), sg) for x in (L, A, B))
    Lf = L.astype(np.float32)
    mu = cv2.blur(Lf, (C["tex_win"],) * 2)
    mu2 = cv2.blur(Lf * Lf, (C["tex_win"],) * 2)
    T = cv2.GaussianBlur(np.sqrt(np.maximum(mu2 - mu * mu, 0)), (0, 0), sg)
    lo, hi = np.percentile(T, 1), np.percentile(T, 99)
    T8 = np.clip((T - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    bgr_s = cv2.bilateralFilter(bgr, 5, 25, 5)               # smoothed colour image for the watershed steps
    # --- for FEATURES use the RAW lightness (CLAHE would equalise the very inside/outside contrast we want to measure)
    Lg = cv2.GaussianBlur(L, (0, 0), 1.0)
    gx = cv2.Sobel(Lg, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(Lg, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    lap = np.abs(cv2.Laplacian(Lg, cv2.CV_32F))
    lbp = local_binary_pattern(L, C["lbp_P"], C["lbp_R"], method="uniform").astype(np.uint8)
    return dict(h=h, w=w, L=L, A=A, B=B, edge_maps=edge_maps, E_ref=E_ref, edge_near=edge_near, edge_snap=edge_snap,
                chan=dict(L=Ls, A=As, B=Bs, T=T8), bgr_s=bgr_s,
                mag=mag, mag_mean=float(mag.mean()) + 1e-6, lap=lap, lbp=lbp)


# ====================================================================== 2. candidate generation
def fill_outer(m_u8):
    cs, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m_u8)
    cv2.drawContours(out, cs, -1, 1, -1)
    return out


def clean_region(m_u8):
    """Shape clean-up: remove thin tails, keep the largest blob, fill holes."""
    m = cv2.morphologyEx(m_u8, cv2.MORPH_OPEN, ell(C["open_k"]))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return fill_outer((lab == k).astype(np.uint8))


def ellipse_candidate(comp):
    """Fit an ellipse to the candidate's largest contour and return it as a FILLED mask -- a smooth,
    regularised version of the same region. Added as an EXTRA candidate alongside the original comp
    (never replacing it): a free-form thresholded blob can keep following a stray edge indefinitely
    (this is our main over-size failure mode -- predicted/true area ratio has a long heavy tail), while
    a fitted ellipse physically cannot. [literature: the candidate RoI is commonly modelled as an ellipse
    before texture confirmation, e.g. Koch & Brilakis-style pipelines]. Returns None if the contour has
    too few points to fit (cv2.fitEllipse needs >=5)."""
    cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    if len(c) < 5:
        return None
    try:
        e = cv2.fitEllipse(c)
    except cv2.error:
        return None
    em = np.zeros(comp.shape, np.uint8)
    cv2.ellipse(em, e, 1, -1)
    return em if em.sum() > 0 else None


def split_lobes(comp, ctx, amin):
    """Watershed on the distance transform: separates potholes that were merged into one blob."""
    cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = float(comp.sum())
    harea = max(cv2.contourArea(cv2.convexHull(max(cs, key=cv2.contourArea))), 1.0)
    if area / harea >= C["split_solidity"]:
        return []
    D = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
    sure_fg = (D > C["split_frac"] * D.max()).astype(np.uint8)
    n, mk = cv2.connectedComponents(sure_fg)
    if n - 1 < 2:
        return []
    markers = np.zeros(comp.shape, np.int32)
    markers[comp == 0] = 1
    markers[mk > 0] = mk[mk > 0] + 1
    cv2.watershed(ctx["bgr_s"], markers)
    parts = []
    for k in range(2, n + 1):
        part = ((markers == k) & (comp > 0)).astype(np.uint8)
        if part.sum() >= amin:
            parts.append(part)
    return parts


def binary_masks(ctx):
    """All thresholding-based binary masks, tagged with their source name."""
    h, w = ctx["h"], ctx["w"]
    out = []
    for ch in ctx["chan"].values():                     # (a) histogram thresholding, both polarities
        for src, flag in (("otsu", cv2.THRESH_OTSU), ("triangle", cv2.THRESH_TRIANGLE)):
            t, _ = cv2.threshold(ch, 0, 255, cv2.THRESH_BINARY + flag)
            out.append(((ch < t).astype(np.uint8), src))
            out.append(((ch > t).astype(np.uint8), src))
    feats = np.dstack([ctx["chan"]["L"], ctx["chan"]["A"], ctx["chan"]["B"]]).reshape(-1, 3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    for k in C["kmeans_ks"]:                            # (b) colour clustering
        cv2.setRNGSeed(0)
        _, labels, _ = cv2.kmeans(feats, k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
        labels = labels.reshape(h, w)
        for j in range(k):
            out.append(((labels == j).astype(np.uint8), "kmeans"))
    Ls = ctx["chan"]["L"]
    for b in C["adapt_blocks"]:                         # (b) local adaptive thresholding (small potholes on uniform road)
        out.append((cv2.adaptiveThreshold(Ls, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, b, C["adapt_c"]), "adaptive"))
        out.append((cv2.adaptiveThreshold(Ls, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, b, -C["adapt_c"]), "adaptive"))
    return out


def raw_candidates(ctx, prior=None):
    h, w = ctx["h"], ctx["w"]
    amin, amax = C["min_area_frac"] * h * w, C["max_area_frac"] * h * w
    area_cap = None
    if prior is not None:
        area_cap = np.exp(prior[0] + C["area_prior_max_z"] * prior[1]) * h * w
    out, seen = [], set()
    rejected_area = [0]          # FIX(attempt2b): count of candidates dropped by the area-prior cap (diagnostic)

    def add(m, src):
        if m is None:
            return
        a = int(m.sum())
        if not (amin <= a <= amax):
            return
        if area_cap is not None and a > area_cap:       # FIX(attempt2b): hard-reject implausibly large candidates
            rejected_area[0] += 1                        # (ellipse-fitting a too-big blob just gives a too-big ellipse)
            return
        key = (cv2.resize(m.astype(np.uint8), (64, 48), interpolation=cv2.INTER_AREA) > 0).tobytes()
        if key in seen:                                 # exact duplicates (same blob from different thresholds)
            return
        seen.add(key)
        out.append((m.astype(bool), SRC_NAMES.index(src)))

    for m, src in binary_masks(ctx):
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ell(C["mask_open"]))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ell(C["mask_close"]))
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        order = np.argsort(-st[1:, cv2.CC_STAT_AREA])[:C["max_comps_per_mask"]] + 1
        for j in order:
            if st[j, cv2.CC_STAT_AREA] < amin:
                continue
            comp = fill_outer((lab == j).astype(np.uint8))
            add(comp, src)
            add(ellipse_candidate(comp), "ellipse")
            for part in split_lobes(comp, ctx, amin):
                part = clean_region(part)
                add(part, "split")
                if part is not None:
                    add(ellipse_candidate(part), "ellipse")
    for E in ctx["edge_maps"]:                          # (c) filled outlines of closed edge groups (proposal)
        for k in C["close_ks"]:
            closed = cv2.morphologyEx(E, cv2.MORPH_CLOSE, ell(k))
            cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                if cv2.contourArea(c) < 0.5 * amin:
                    continue
                f = np.zeros((h, w), np.uint8)
                cv2.drawContours(f, [c], -1, 1, -1)
                f = clean_region(f)
                add(f, "canny")
                if f is not None:
                    add(ellipse_candidate(f), "ellipse")
    return out, rejected_area[0]


def base_feats(mask, ctx):
    """Shape + edge-support features (cheap; also used to rank candidates before the classifier exists)."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    area = len(xs)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    cm = np.pad(mask[y0:y1 + 1, x0:x1 + 1].astype(np.uint8), 1)
    cnts, _ = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    per = max(cv2.arcLength(c, True), 1.0)
    harea = max(cv2.contourArea(cv2.convexHull(c)), 1.0)
    rw, rh = cv2.minAreaRect(c)[1]
    ell_res = 1.0
    if len(c) >= 5:
        try:
            e = cv2.fitEllipse(c)
            em = np.zeros_like(cm)
            cv2.ellipse(em, e, 1, -1)
            ell_res = float((em.astype(bool) ^ cm.astype(bool)).sum()) / area
        except cv2.error:
            pass
    frame = int(mask[0, :].sum() + mask[-1, :].sum() + mask[:, 0].sum() + mask[:, -1].sum())
    bnd = cm.astype(bool) & ~(cv2.erode(cm, np.ones((3, 3), np.uint8)) > 0)
    en = np.pad(ctx["edge_near"][y0:y1 + 1, x0:x1 + 1], 1)
    mg = np.pad(ctx["mag"][y0:y1 + 1, x0:x1 + 1], 1)
    return dict(area_frac=area / (h * w), solidity=min(area / harea, 1.0), circ=min(4 * np.pi * area / per ** 2, 1.5),
                elong=min(max(rw, rh) / max(min(rw, rh), 1.0), 20.0), extent=area / ((x1 - x0 + 1) * (y1 - y0 + 1)),
                ell_res=min(ell_res, 5.0), cx=float(xs.mean()) / w, cy=float(ys.mean()) / h,
                border_frac=min(frame / per, 1.0),
                edge_support=float(en[bnd].mean()) if bnd.any() else 0.0,
                grad_b=float(mg[bnd].mean() / ctx["mag_mean"]) if bnd.any() else 0.0)


def heuristic_score(b, prior):
    """Technique 1 on its own (no learning): closed, edge-supported, compact, plausible-size, not cut by the border."""
    ap = max(np.exp(-0.5 * ((np.log(max(b["area_frac"], 1e-6)) - prior[0]) / prior[1]) ** 2), 0.05)
    return float(b["edge_support"] * (0.5 + 0.5 * b["solidity"]) * ap * (1 - 0.5 * b["border_frac"]))


def nms_and_cap(cands, prior):
    """Score, drop near-duplicates (IoU > nms_iou on 1/4-res masks), keep the best max_cands."""
    if not cands:
        return []
    scored = [(heuristic_score(b, prior), m, src, b) for m, src, b in cands]
    scored.sort(key=lambda t: -t[0])
    small = np.stack([cv2.resize(m.astype(np.uint8), (C["work_w"] // 4, C["work_h"] // 4),
                                 interpolation=cv2.INTER_AREA).ravel() > 0 for _, m, _, _ in scored]).astype(np.float32)
    inter = small @ small.T
    ar = small.sum(1)
    iou = inter / np.maximum(ar[:, None] + ar[None, :] - inter, 1.0)
    keep = []
    for i in range(len(scored)):
        if all(iou[i, j] <= C["nms_iou"] for j in keep):
            keep.append(i)
        if len(keep) >= C["max_cands"]:
            break
    return [scored[i] for i in keep]


def full_feats(mask, ctx, base, src):
    """Features: base shape/edge + inside-vs-road-ring texture comparison."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    area = len(xs)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    rw = int(np.clip(C["ring_scale"] * np.sqrt(area), C["ring_min"], C["ring_max"]))
    mg = rw + C["ring_gap"] + 2
    cx0, cy0, cx1, cy1 = max(0, x0 - mg), max(0, y0 - mg), min(w, x1 + mg + 1), min(h, y1 + mg + 1)
    m = mask[cy0:cy1, cx0:cx1].astype(np.uint8)
    ring = (cv2.dilate(m, ell(2 * rw + 1)) > 0) & ~(cv2.dilate(m, ell(2 * C["ring_gap"] + 1)) > 0)
    core = cv2.erode(m, ell(5)) > 0               # inside, away from the blurry boundary
    if core.sum() < 0.3 * m.sum():
        core = m > 0
    ring_ok = bool(ring.sum() >= 50)
    L = ctx["L"][cy0:cy1, cx0:cx1]
    Af = ctx["A"][cy0:cy1, cx0:cx1].astype(np.float32)
    Bf = ctx["B"][cy0:cy1, cx0:cx1].astype(np.float32)
    lap = ctx["lap"][cy0:cy1, cx0:cx1]
    mag = ctx["mag"][cy0:cy1, cx0:cx1]
    Er = (ctx["E_ref"][cy0:cy1, cx0:cx1] > 0)
    lb = ctx["lbp"][cy0:cy1, cx0:cx1]
    nan = np.nan
    Li = L[core].astype(np.float32)
    f = {k: base[k] for k in FEATURES[:11]}
    for n, s in enumerate(SRC_NAMES):
        f[f"src_{s}"] = 1.0 if src == n else 0.0
    f["mean_in"] = Li.mean() / 255.0
    f["std_in"] = Li.std() / 255.0
    f["lap_in"] = lap[core].mean()
    f["grad_in"] = mag[core].mean() / ctx["mag_mean"]
    f["ed_in"] = Er[core].mean()
    hist = np.bincount((Li.astype(np.int32) >> 4), minlength=16)[:16] / max(len(Li), 1)
    f["entropy_in"] = float(-(hist[hist > 0] * np.log(hist[hist > 0])).sum())
    hin = np.bincount(lb[core], minlength=LBP_BINS)[:LBP_BINS] / max(core.sum(), 1)
    for i in range(LBP_BINS):
        f[f"lbp_in_{i}"] = hin[i]

    def glcm_vec(sel):
        q = (L >> 4).astype(np.uint8)
        q = np.where(sel, q, 16).astype(np.uint8)          # level 16 = "ignore" (outside the region)
        g = graycomatrix(q, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=17, symmetric=True, normed=False)
        P = g[:16, :16].astype(np.float64)
        if P.sum() < 30:
            return [nan] * len(GLCM_PROPS)
        return [float(graycoprops(P, p).mean()) for p in GLCM_PROPS]

    gin = glcm_vec(core)
    for p, v in zip(GLCM_PROPS, gin):
        f[f"glcm_in_{p}"] = v
    if ring_ok:
        Lr = L[ring].astype(np.float32)
        f["dmean"] = (Li.mean() - Lr.mean()) / 255.0
        f["std_ratio"] = min(Li.std() / (Lr.std() + 1.0), 20.0)
        f["da"] = (Af[core].mean() - Af[ring].mean()) / 255.0
        f["db"] = (Bf[core].mean() - Bf[ring].mean()) / 255.0
        lr = lap[ring].mean()
        f["lap_ratio"] = min(f["lap_in"] / (lr + 0.5), 50.0)
        f["grad_ratio"] = min(f["grad_in"] / (mag[ring].mean() / ctx["mag_mean"] + 0.05), 50.0)
        f["ed_ring"] = Er[ring].mean()
        f["ed_diff"] = f["ed_in"] - f["ed_ring"]
        hr = np.bincount(lb[ring], minlength=LBP_BINS)[:LBP_BINS] / max(ring.sum(), 1)
        for i in range(LBP_BINS):
            f[f"lbp_d_{i}"] = hin[i] - hr[i]
        f["lbp_chi2"] = float(0.5 * np.sum((hin - hr) ** 2 / (hin + hr + 1e-9)))
        gr = glcm_vec(ring)
        for p, a, b in zip(GLCM_PROPS, gin, gr):
            f[f"glcm_r_{p}"] = b
            f[f"glcm_d_{p}"] = a - b
    f["ring_ok"] = 1.0 if ring_ok else 0.0
    v = np.array([f.get(n, nan) for n in FEATURES], np.float64)
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def rle_encode(m):
    f = m.ravel().astype(np.int8)
    d = np.diff(np.concatenate([[0], f, [0]]))
    return np.flatnonzero(d == 1).astype(np.int32), np.flatnonzero(d == -1).astype(np.int32)


def rle_decode(rle, shape):
    n = shape[0] * shape[1]
    d = np.zeros(n + 1, np.int8)
    d[rle[0]] += 1
    d[rle[1]] -= 1
    return (np.cumsum(d[:n]) > 0).reshape(shape)


def load_work(rgb_path):
    bgr, _ = pc.load_rgb(rgb_path)
    return cv2.resize(bgr, WORK, interpolation=cv2.INTER_AREA)


def instance_masks(label_path):
    """One work-resolution mask per ground-truth polygon (a pothole each)."""
    polys, _ = pc.read_polygons(label_path)
    out = []
    for p in polys:
        m = np.zeros((pc.H, pc.W), np.uint8)
        cv2.fillPoly(m, [p], 1)
        mw = cv2.resize(m, WORK, interpolation=cv2.INTER_NEAREST).astype(bool)
        if mw.any():
            out.append(mw)
    return out


def process_image(args):
    """One image -> candidates (RLE), feature matrix, IoU with every ground-truth pothole."""
    iid, rgb_path, label_path, prior = args
    cv2.setNumThreads(1)                          # avoid CPU oversubscription when several workers run
    cv2.setRNGSeed(0)
    try:
        ctx = make_ctx(load_work(rgb_path))
        inst = instance_masks(label_path)
        raw, n_area_rejected = raw_candidates(ctx, prior)
        pre = [(m, s, base_feats(m, ctx)) for m, s in raw]
        kept = nms_and_cap(pre, prior)
        feats, rles, srcs, sa, masks = [], [], [], [], []
        for score, m, src, base in kept:
            feats.append(full_feats(m, ctx, base, src))
            rles.append(rle_encode(m))
            srcs.append(src)
            sa.append(score)
            masks.append(m)
        n = len(masks)
        if n and inst:
            Cm = np.stack([m.ravel() for m in masks]).astype(np.float32)
            Im = np.stack([g.ravel() for g in inst]).astype(np.float32)
            inter = Cm @ Im.T
            iou_mat = inter / np.maximum(Cm.sum(1)[:, None] + Im.sum(1)[None, :] - inter, 1.0)
            U = np.any(np.stack(inst), axis=0).ravel().astype(np.float32)
            iu = (Cm @ U) / np.maximum(Cm.sum(1) + U.sum() - Cm @ U, 1.0)
        else:
            iou_mat = np.zeros((n, len(inst)), np.float32)
            iu = np.zeros(n, np.float32)
        iou = iou_mat.max(1) if iou_mat.shape[1] else np.zeros(n, np.float32)
        ref = (iou_mat.max(0)[iou_mat.argmax(1)] if iou_mat.shape[1] and n else np.zeros(n, np.float32))
        return dict(id=iid, ok=True, has_gt=bool(inst), n_inst=len(inst),
                    feats=np.array(feats, np.float32).reshape(n, len(FEATURES)),
                    iou=iou.astype(np.float32), ref=ref.astype(np.float32), iou_u=iu.astype(np.float32),
                    iou_mat=iou_mat.astype(np.float16), rle=rles, src=np.array(srcs, np.int8),
                    score_a=np.array(sa, np.float32), n_raw=len(raw), n_area_rejected=n_area_rejected)
    except Exception as ex:                       # never let one bad image kill a 1000-image run
        return dict(id=iid, ok=False, error=repr(ex))


# ====================================================================== build (cache)
def cache_path(smoke=False):
    return os.path.join(pc.CACHE, f"features_{cfg_hash()}{'_smoke' if smoke else ''}.pkl")


def train_prior(df, sp):
    """Area prior (log-normal) over individual TRAIN ground-truth potholes."""
    ids = set(sp.image_id[sp.split == "train"])
    la = []
    for _, r in df[df.image_id.isin(ids)].iterrows():
        polys, _ = pc.read_polygons(r.label_path)
        for p in polys:
            m = np.zeros((pc.H, pc.W), np.uint8)
            cv2.fillPoly(m, [p], 1)
            if m.sum() > 0:
                la.append(np.log(m.sum() / (pc.W * pc.H)))
    return float(np.mean(la)), float(max(np.std(la), 0.7))


def cmd_build(args):
    if not os.path.exists(pc.SPLITS):
        sys.exit("splits.csv not found. Run step1_prepare.py first.")
    os.makedirs(pc.CACHE, exist_ok=True)
    df, sp = pc.load_index(), pd.read_csv(pc.SPLITS)
    prior = train_prior(df, sp)
    print(f"area prior from TRAIN GT: mean log-area {prior[0]:.2f} (~{100 * np.exp(prior[0]):.1f}% of image), sd {prior[1]:.2f}")
    sub = df if not args.limit else df.sample(n=min(args.limit, len(df)), random_state=0)
    jobs = [(r.image_id, r.rgb_path, r.label_path, prior) for r in sub.itertuples()]
    t0 = time.time()
    results = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, res in enumerate(ex.map(process_image, jobs, chunksize=4), 1):
                results.append(res)
                if i % 50 == 0:
                    print(f"  {i}/{len(jobs)}  ({time.time() - t0:.0f}s)")
    else:
        for i, j in enumerate(jobs, 1):
            results.append(process_image(j))
            if i % 25 == 0:
                print(f"  {i}/{len(jobs)}  ({time.time() - t0:.0f}s)")
    bad = [r for r in results if not r["ok"]]
    good = {r["id"]: r for r in results if r["ok"]}
    if bad:
        print(f"!! {len(bad)} images FAILED and are excluded:", [(b['id'], b['error']) for b in bad[:5]])
    nc = np.array([len(r["iou"]) for r in good.values()])
    print(f"candidates per image: median {int(np.median(nc))}, min {nc.min()}, max {nc.max()}; images with 0 candidates: {(nc == 0).sum()}")
    nraw = np.array([r.get("n_raw", 0) for r in good.values()])
    narej = np.array([r.get("n_area_rejected", 0) for r in good.values()])
    print(f"  FIX attempt2a (nms_iou 0.85->0.70): kept-after-NMS median is {int(np.median(nc))} above; "
         f"compare against the PREVIOUS run's printed median to see how many near-duplicates this removed")
    print(f"  FIX attempt2b (area-prior hard cap): {int(narej.sum())} candidates rejected total across all images "
         f"(median {int(np.median(narej))}/image, {int((narej > 0).sum())} images had at least one rejection)")
    with open(cache_path(args.limit > 0), "wb") as f:
        pickle.dump(dict(meta=dict(cfg=C, features=FEATURES, prior=prior), entries=good, failed=bad), f)
    print(f"[ok] cached {len(good)} images in {time.time() - t0:.0f}s -> {cache_path(args.limit > 0)}")


# ====================================================================== model helpers
def make_model(kind, p):
    if kind == "rf":
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=3, max_features="sqrt",
                                      class_weight="balanced_subsample", n_jobs=-1, random_state=0)
    return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=p.get("C", 1.0), gamma="scale",
                                              class_weight="balanced", probability=True, random_state=0))


def make_labels(iou, ref):
    """1 = good outline of a ground-truth pothole (IoU>=0.5 and close to the best candidate for THAT pothole),
    0 = not a pothole outline (IoU<0.3, or clearly worse than the best candidate for its pothole),
    -1 = ambiguous, left out of training."""
    y = np.full(len(iou), -1, int)
    y[iou >= np.maximum(C["pos_iou"], ref - C["rel_pos"])] = 1
    neg = (iou < C["neg_iou"]) | ((ref >= C["pos_iou"]) & (iou < ref - C["rel_neg"]))
    y[neg & (y != 1)] = 0
    return y


def build_xy(entries, ids, cols, shuffle_labels=False):
    X, y = [], []
    for i in ids:
        e = entries[i]
        if not e["has_gt"] or len(e["iou"]) == 0:
            continue
        lab = make_labels(e["iou"], e["ref"])
        keep = lab >= 0
        X.append(e["feats"][keep][:, cols])
        y.append(lab[keep])
    X, y = np.vstack(X), np.concatenate(y)
    if y.sum() == 0 or y.sum() == len(y):
        raise RuntimeError("training set has only one class; candidate generation found no good candidates (see 'build' output)")
    if shuffle_labels:
        y = np.random.default_rng(0).permutation(y)
    return X, y


def fit_model(kind, p, entries, ids, cols, shuffle_labels=False):
    X, y = build_xy(entries, ids, cols, shuffle_labels)
    if kind == "svm" and len(y) > C["svm_max_samples"]:
        rng = np.random.default_rng(0)
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        n_neg = max(C["svm_max_samples"] - len(pos), 1000)
        sel = np.concatenate([pos, rng.choice(neg, min(n_neg, len(neg)), replace=False)])
        X, y = X[sel], y[sel]
    m = make_model(kind, p)
    m.fit(X, y)
    return m


def proba(model, e, cols):
    if len(e["iou"]) == 0:
        return np.zeros(0)
    return model.predict_proba(e["feats"][:, cols])[:, 1]


# ====================================================================== prediction + evaluation
def fill_holes(m):
    return fill_outer(m.astype(np.uint8))


def to_full(mask_work):
    """work-res bool mask -> clean 640x480 uint8 mask."""
    f = cv2.resize(mask_work.astype(np.float32), (pc.W, pc.H), interpolation=cv2.INTER_LINEAR) > 0.5
    return fill_holes(cv2.morphologyEx(f.astype(np.uint8), cv2.MORPH_OPEN, ell(3)))


def choose(e, scores, t_add):
    """Best candidate always; further candidates with score>=t_add that do not overlap a kept one by more than
    overlap_skip of the smaller region (several potholes per image), at most max_regions."""
    if len(scores) == 0:
        return []
    order = np.argsort(-scores)
    sel = [int(order[0])]
    if t_add is not None:
        shape = (WORK[1], WORK[0])
        masks = {sel[0]: rle_decode(e["rle"][sel[0]], shape)}
        for j in order[1:]:
            if len(sel) >= C["max_regions"] or scores[j] < t_add:
                break
            mj = rle_decode(e["rle"][int(j)], shape)
            aj = max(int(mj.sum()), 1)
            if all((mj & masks[k]).sum() / max(min(aj, int(masks[k].sum())), 1) <= C["overlap_skip"] for k in sel):
                sel.append(int(j))
                masks[int(j)] = mj
    return sel


def snap_region(region, ctx):
    """Marker-controlled watershed: the outline moves to the strongest nearby edge inside a band around the region.
    Canny edges are drawn as bright ridges that the flood cannot cross cheaply."""
    reg = region.astype(np.uint8)
    area = int(reg.sum())
    r = np.sqrt(area / np.pi)
    k = int(max(3, round(C["snap_frac"] * r)))
    inner = cv2.erode(reg, ell(2 * k + 1))
    outer = cv2.dilate(reg, ell(2 * k + 1))
    if inner.sum() < 20:
        return region
    img = ctx["bgr_s"].copy()
    img[ctx["edge_snap"]] = 255
    markers = np.zeros(reg.shape, np.int32)
    markers[outer == 0] = 1
    markers[inner > 0] = 2
    cv2.watershed(img, markers)
    out = markers == 2
    if out.sum() < 0.5 * area or out.sum() > 1.5 * area:
        return region
    return out


def pred_work_mask(e, sel, snap_ctx=None):
    shape = (WORK[1], WORK[0])
    m = np.zeros(shape, bool)
    for j in sel:
        r = rle_decode(e["rle"][j], shape)
        m |= snap_region(r, snap_ctx) if snap_ctx is not None else r
    return m


def score_row(pred_full, gt_full):
    d, i = pc.dice_iou(pred_full, gt_full)
    p, g = pred_full.astype(bool), gt_full.astype(bool)
    tp = float((p & g).sum())
    return dict(dice=d, iou=i, tp=tp, fp=float(p.sum() - tp), fn=float(g.sum() - tp), pred_frac=p.mean())


def oracle_multi_sel(e):
    """Best candidate for every ground-truth pothole (upper bound for several potholes per image)."""
    m = e["iou_mat"].astype(np.float32)
    sel = []
    for j in range(m.shape[1] if m.ndim == 2 else 0):
        if m.shape[0] == 0:
            break
        k = int(np.argmax(m[:, j]))
        if m[k, j] >= 0.3 and k not in sel:
            sel.append(k)
    return sel


def eval_variant(ids, entries, gts, scorer, t_add=None, mode="score", ctx_of=None):
    """scorer(entry)->scores. mode: score | oracle | oracle_multi | random.
    ctx_of(id) -> image context enables the boundary snap."""
    rows, preds = {}, {}
    rng = np.random.default_rng(0)
    for i in ids:
        e = entries[i]
        if len(e["iou"]) == 0:
            sel = []
        elif mode == "oracle":
            sel = [int(np.argmax(e["iou_u"]))]
        elif mode == "oracle_multi":
            sel = oracle_multi_sel(e) or [int(np.argmax(e["iou_u"]))]
        elif mode == "random":
            sel = [int(rng.integers(len(e["iou"])))]
        else:
            sel = choose(e, scorer(e), t_add)
        if sel:
            pred = to_full(pred_work_mask(e, sel, ctx_of(i) if ctx_of else None))
        else:
            pred = np.zeros((pc.H, pc.W), np.uint8)
        r = score_row(pred, gts[i])
        r["n_regions"] = len(sel)
        rows[i], preds[i] = r, pred
    return rows, preds


def table_row(name, rows):
    df = pd.DataFrame(rows.values())
    s = pc.summarize(df.dice, df.iou, df.tp, df.fp, df.fn)
    lo, hi = pc.boot_ci(df.dice.values)
    s.update(variant=name, dice_ci=f"{lo:.3f}-{hi:.3f}")
    return s


# ====================================================================== run
def cmd_run(args):
    path = cache_path(args.smoke)
    if not os.path.exists(path):
        sys.exit(f"cache not found ({path}). Run:  python step2_segment.py build")
    with open(path, "rb") as f:
        pk = pickle.load(f)
    entries = pk["entries"]
    df, sp = pc.load_index(), pd.read_csv(pc.SPLITS)
    paths = df.set_index("image_id")
    split_ids = {s: [i for i in sp.image_id[sp.split == s] if i in entries and entries[i]["has_gt"]]
                 for s in ("train", "val", "test")}
    for s, v in split_ids.items():
        print(f"{s}: {len(v)} usable images")
    n_empty = sum(1 for e in entries.values() if not e["has_gt"])
    if n_empty:
        print(f"({n_empty} images without ground truth are excluded)")
    os.makedirs(pc.OUT, exist_ok=True)
    cfg_path = os.path.join(pc.OUT, "best_config.json")
    log = []

    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        log.append(s)

    if args.test:
        if not os.path.exists(cfg_path):
            sys.exit("best_config.json missing. Run `run` (validation) first, then `run --test`.")
        best = json.load(open(cfg_path))
        fit_ids, eval_name = split_ids["train"] + split_ids["val"], "test"
        say(f"FINAL TEST RUN: refitting '{best['kind']}' on train+val, evaluating on test ONCE")
    else:
        best, fit_ids, eval_name = None, split_ids["train"], "val"
    ev_ids = split_ids[eval_name]
    gts = {i: pc.gt_mask(paths.at[i, "label_path"])[0] for i in ev_ids}
    cols_all, cols_se, cols_tx = GROUPS["all"], GROUPS["shape_edge"], GROUPS["texture"]
    ctx_cache = {}

    def ctx_of(i):
        if i not in ctx_cache:
            ctx_cache[i] = make_ctx(load_work(paths.at[i, "rgb_path"]))
        return ctx_cache[i]

    # ---------------- model selection on VAL (only in validation mode)
    if best is None:
        kinds = [("rf", {})] if args.models == "rf" else [("rf", {}), ("svm", {"C": 1.0}), ("svm", {"C": 10.0})]
        sel_rows = []
        for kind, p in kinds:
            t0 = time.time()
            m = fit_model(kind, p, entries, fit_ids, cols_all)
            rows, _ = eval_variant(ev_ids, entries, gts, lambda e, m=m: proba(m, e, cols_all))
            s = table_row(f"{kind} {p}", rows)
            say(f"model {kind} {p}: val Dice {s['dice_mean']:.3f} (top-1 only)  [{time.time() - t0:.0f}s]")
            sel_rows.append((s["dice_mean"], kind, p, m))
        sel_rows.sort(key=lambda t: -t[0])
        _, kind, p, model_all = sel_rows[0]
        best_t, best_d = None, -1
        for t in (None, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):          # probability needed to keep an extra region
            rows, _ = eval_variant(ev_ids, entries, gts, lambda e: proba(model_all, e, cols_all), t_add=t)
            d = float(np.mean([r["dice"] for r in rows.values()]))
            say(f"  extra-region threshold {t}: val Dice {d:.3f}")
            if d > best_d + 1e-4:
                best_t, best_d = t, d
        rows, _ = eval_variant(ev_ids, entries, gts, lambda e: proba(model_all, e, cols_all), t_add=best_t, ctx_of=ctx_of)
        d_snap = float(np.mean([r["dice"] for r in rows.values()]))
        refine = bool(d_snap > best_d + 0.003)                   # keep the boundary snap only if it clearly helps
        say(f"  boundary snap: val Dice {d_snap:.3f} vs {best_d:.3f} without -> {'KEPT' if refine else 'not used'}")
        best = dict(kind=kind, params=p, t_add=best_t, refine=refine)
        json.dump(best, open(cfg_path, "w"), indent=1)
        say(f"chosen: {best}")
    kind, p, t_add, refine = best["kind"], best["params"], best["t_add"], best.get("refine", False)

    # ---------------- variants, oracles and baselines
    say(f"\n=== {eval_name.upper()} RESULTS (full resolution 640x480) ===")
    m_all = fit_model(kind, p, entries, fit_ids, cols_all)
    m_se = fit_model(kind, p, entries, fit_ids, cols_se)
    m_tx = fit_model(kind, p, entries, fit_ids, cols_tx)
    chi_med = float(np.median(np.concatenate([entries[i]["feats"][:, IDX["lbp_chi2"]] for i in fit_ids if len(entries[i]["iou"])])))
    f_all = lambda e: proba(m_all, e, cols_all)
    full_name = "E+ FULL (classifier, several potholes" + (", boundary snap)" if refine else ")")
    variants = {
        "A edge/shape heuristic (no learning)": (lambda e: e["score_a"], None, "score", None),
        "B A + texture rule": (lambda e: e["score_a"] * e["feats"][:, IDX["lbp_chi2"]] / (e["feats"][:, IDX["lbp_chi2"]] + chi_med + 1e-9), None, "score", None),
        "C classifier on shape/edge only": (lambda e: proba(m_se, e, cols_se), None, "score", None),
        "D classifier on texture only": (lambda e: proba(m_tx, e, cols_tx), None, "score", None),
        "E FULL: classifier on all features (1 region)": (f_all, None, "score", None),
        full_name: (f_all, t_add, "score", ctx_of if refine else None),
        "oracle single (best one candidate)": (None, None, "oracle", None),
        "oracle multi (best candidate per pothole)": (None, None, "oracle_multi", None),
        "random candidate (lower bound)": (None, None, "random", None),
    }
    results, preds_all, table = {}, {}, []
    for name, (fn, t, mode, cf) in variants.items():
        rows, preds = eval_variant(ev_ids, entries, gts, fn, t_add=t, mode=mode, ctx_of=cf)
        results[name], preds_all[name] = rows, preds
        table.append(table_row(name, rows))
    acc = np.zeros((pc.H, pc.W), np.float32)
    tr_gts = []
    for i in split_ids["train"]:
        g = pc.gt_mask(paths.at[i, "label_path"])[0]
        acc += g
        tr_gts.append(g)
    acc /= max(len(tr_gts), 1)
    cands_mm = [(acc >= f * acc.max()).astype(np.uint8) for f in (0.3, 0.5, 0.7, 0.9)]
    mm = max(cands_mm, key=lambda mk: np.mean([pc.dice_iou(mk, g)[0] for g in tr_gts]))   # best on TRAIN
    rows = {i: dict(score_row(mm, gts[i]), n_regions=1) for i in ev_ids}
    results["mean-mask baseline (fixed centred blob)"] = rows
    table.append(table_row("mean-mask baseline (fixed centred blob)", rows))
    m_sh = fit_model(kind, p, entries, fit_ids, cols_all, shuffle_labels=True)
    rows, _ = eval_variant(ev_ids, entries, gts, lambda e: proba(m_sh, e, cols_all))
    table.append(table_row("SANITY: trained on shuffled labels", rows))

    T = pd.DataFrame(table).set_index("variant")
    cols = ["n", "dice_mean", "dice_ci", "dice_median", "iou_mean", "pooled_iou", "precision", "recall", "hit50", "hit70"]
    pd.set_option("display.width", 220)
    say(T[cols].round(3).to_string())
    T[cols].round(4).to_csv(os.path.join(pc.OUT, f"variants_{eval_name}.csv"))

    # candidate coverage (where does the pipeline lose?)
    best_u = np.array([e["iou_u"].max() if len(e["iou_u"]) else 0 for e in (entries[i] for i in ev_ids)])
    inst_best = np.concatenate([e["iou_mat"].astype(np.float32).max(0) if e["iou_mat"].ndim == 2 and e["iou_mat"].shape[0] else
                                np.zeros(e["n_inst"]) for e in (entries[i] for i in ev_ids)])
    say(f"\ncandidate coverage (work res): best single candidate vs whole GT has IoU>=0.5 in {100 * (best_u >= 0.5).mean():.0f}% of images, "
        f">=0.7 in {100 * (best_u >= 0.7).mean():.0f}%, >=0.85 in {100 * (best_u >= 0.85).mean():.0f}%")
    say(f"ground-truth potholes ({len(inst_best)} in {len(ev_ids)} images) that have a candidate with IoU>=0.5: {100 * (inst_best >= 0.5).mean():.0f}%, >=0.7: {100 * (inst_best >= 0.7).mean():.0f}%")
    srcs = [SRC_NAMES[int(entries[i]["src"][int(np.argmax(entries[i]["iou_u"]))])] for i in ev_ids
            if len(entries[i]["iou_u"]) and entries[i]["iou_u"].max() >= 0.5]
    if srcs:
        vc = pd.Series(srcs).value_counts()
        say("source of the best candidate (IoU>=0.5):", {k: int(v) for k, v in vc.items()}, "<- sources that never win can be removed")
    rank1 = []
    for i in ev_ids[:60]:
        e = entries[i]
        ctx = ctx_of(i)
        g = cv2.resize(gts[i], WORK, interpolation=cv2.INTER_NEAREST)
        n, lab, st, _ = cv2.connectedComponentsWithStats(g, connectivity=8)
        if n <= 1:
            continue
        g = lab == (1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))
        fv = full_feats(g, ctx, base_feats(g, ctx), 0)[None]
        s_gt = m_all.predict_proba(fv[:, cols_all])[0, 1]
        rank1.append((s_gt, s_gt - (proba(m_all, e, cols_all).max() if len(e["iou"]) else 0)))
    if rank1:
        rk = np.array(rank1)
        say(f"SANITY GT-as-candidate: the TRUE outline is accepted (probability>=0.5) in {100 * (rk[:, 0] >= 0.5).mean():.0f}% of {len(rk)} images "
            f"and scores within 0.1 of the best generated candidate in {100 * (rk[:, 1] >= -0.1).mean():.0f}% "
            "(should be high; if low, the features cannot recognise even a perfect region)")

    # ---------------- per-image table, failure taxonomy, depth diagnostic
    R = pd.DataFrame({i: results[full_name][i] for i in ev_ids}).T.astype(float)
    R.index.name = "image_id"
    R["dice_E"] = R["dice"]
    for key, nm in (("A", "A edge/shape heuristic (no learning)"), ("C", "C classifier on shape/edge only"),
                    ("D", "D classifier on texture only"), ("oracle", "oracle multi (best candidate per pothole)")):
        R[f"dice_{key}"] = [results[nm][i]["dice"] for i in ev_ids]
    R["oracle_iou"] = [results["oracle multi (best candidate per pothole)"][i]["iou"] for i in ev_ids]
    R["gt_frac"] = [gts[i].mean() for i in ev_ids]
    R["gt_blobs"] = [int(cv2.connectedComponents(gts[i])[0] - 1) for i in ev_ids]

    def ftype(r):
        if r.iou >= 0.75:
            return "good"
        if r.oracle_iou < 0.5:
            return "candidate_generation_failed"
        if r.iou < 0.5:
            return "wrong_candidate_selected"
        return "boundary_imprecise"
    R["failure_type"] = R.apply(ftype, axis=1)
    R["size_error"] = np.where(R.pred_frac > 1.5 * R.gt_frac, "over", np.where(R.pred_frac < R.gt_frac / 1.5, "under", "ok"))
    dg, dp, okg, okp, mL = [], [], [], [], []
    for i in ev_ids:
        d = pc.load_depth_cm(paths.at[i, "depth_path"])
        mL.append(float(cv2.cvtColor(pc.load_rgb(paths.at[i, "rgb_path"])[0], cv2.COLOR_BGR2LAB)[..., 0].mean()))
        pred = preds_all[full_name][i]
        rg, ogk, _, _ = pc.rel_depth(d, gts[i])
        rp, opk, _, _ = pc.rel_depth(d, pred)
        dg.append(pc.depth_stats(rg, gts[i])["p90"]); dp.append(pc.depth_stats(rp, pred)["p90"])
        okg.append(ogk); okp.append(opk)
    R["depth_p90_gt"], R["depth_p90_pred"], R["depth_ok_gt"], R["depth_ok_pred"], R["mean_L"] = dg, dp, okg, okp, mL
    R.round(4).to_csv(os.path.join(pc.OUT, f"per_image_{eval_name}.csv"))
    say("\nfailure types (final pipeline):", {k: int(v) for k, v in R.failure_type.value_counts().items()})
    say("size errors:", {k: int(v) for k, v in R.size_error.value_counts().items()}, "| images with several GT blobs:", int((R.gt_blobs > 1).sum()))
    ratio = (R.pred_frac / R.gt_frac).replace([np.inf, -np.inf], np.nan).dropna()
    say(f"predicted/true area ratio, percentiles 10/50/90: {' / '.join(f'{np.percentile(ratio, q):.2f}' for q in (10, 50, 90))}; "
        f"within +-25% of true area: {100 * ((ratio > 0.75) & (ratio < 1.25)).mean():.0f}%")
    bd = []
    for fac, nice in (("gt_frac", "pothole size"), ("mean_L", "image brightness"), ("depth_p90_gt", "depth (cm below road)")):
        v = R[fac].astype(float)
        if v.notna().sum() < 9 or v.nunique() < 3:
            continue
        for b, g in R.groupby(pd.qcut(v, 3, duplicates="drop"), observed=True):
            bd.append(dict(factor=nice, bin=str(b), n=len(g), A_heuristic=g.dice_A.mean(), C_shape_edge=g.dice_C.mean(),
                           D_texture=g.dice_D.mean(), E_full=g.dice_E.mean(), oracle=g.dice_oracle.mean()))
    if bd:
        BD = pd.DataFrame(bd)
        BD.round(3).to_csv(os.path.join(pc.OUT, f"breakdown_{eval_name}.csv"), index=False)
        say("\nmean Dice by condition (terciles):")
        say(BD.round(3).to_string(index=False))
    both = R[R.depth_ok_gt & R.depth_ok_pred].dropna(subset=["depth_p90_gt", "depth_p90_pred"])
    if len(both) > 3:
        from scipy.stats import spearmanr
        rho = spearmanr(both.depth_p90_gt, both.depth_p90_pred)[0]
        say(f"depth preview: on {len(both)} images with reliable depth, depth-below-road from the PREDICTED mask vs the GT mask: "
            f"Spearman {rho:.2f}, mean abs difference {np.mean(np.abs(both.depth_p90_gt - both.depth_p90_pred)):.2f} cm")
    if kind == "rf":
        imp = pd.Series(m_all.feature_importances_, index=FEATURES).sort_values(ascending=False)
        imp.round(4).to_csv(os.path.join(pc.OUT, "feature_importance.csv"), header=["importance"])
        say("top-10 features:", ", ".join(f"{k} {v:.3f}" for k, v in imp.head(10).items()))

    # ---------------- predicted masks for step3_features.py
    ids_out = list(ev_ids)
    packed = np.stack([np.packbits(preds_all[full_name][i].astype(np.uint8).ravel()) for i in ids_out])
    np.savez_compressed(os.path.join(pc.OUT, f"pred_masks_{eval_name}.npz"), ids=np.array(ids_out), packed=packed)

    # ---------------- overlays: worst and random cases
    odir = os.path.join(pc.OUT, f"overlays_{eval_name}")
    os.makedirs(odir, exist_ok=True)
    for f_ in os.listdir(odir):
        os.remove(os.path.join(odir, f_))
    rng = np.random.default_rng(0)
    worst = R.sort_values("dice").index[:16].tolist()
    rest = [i for i in R.index if i not in worst]
    rand = list(rng.choice(rest, size=min(8, len(rest)), replace=False)) if rest else []
    for tag, lst in (("worst", worst), ("random", rand)):
        for n, i in enumerate(lst):
            rgb, _ = pc.load_rgb(paths.at[i, "rgb_path"])
            ctx = ctx_of(i)
            left = pc.overlay(rgb, gt=gts[i], pred=preds_all[full_name][i],
                              text=f"{i} dice {R.at[i, 'dice']:.2f} {R.at[i, 'failure_type']}")
            right = cv2.cvtColor(cv2.resize(ctx["E_ref"], (pc.W, pc.H), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
            right = pc.overlay(right, gt=gts[i], pred=preds_all[full_name][i], text="edge map + GT(green)/pred(red)")
            pc.imwrite(os.path.join(odir, f"{tag}_{n:02d}_{i}.jpg"), np.hstack([left, right]))
    say(f"\n[ok] overlays in {odir}  (worst 16 + 8 random)")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        for nm, lab_ in (("A edge/shape heuristic (no learning)", "A"), (full_name, "final"), ("oracle multi (best candidate per pothole)", "oracle")):
            ax.hist([results[nm][i]["dice"] for i in ev_ids], bins=20, range=(0, 1), alpha=0.5, label=lab_)
        ax.set_xlabel("Dice per image"); ax.set_ylabel("images"); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(pc.OUT, f"dice_hist_{eval_name}.png"), dpi=120)
    except Exception as ex:
        say("(matplotlib plot skipped:", ex, ")")
    with open(os.path.join(pc.OUT, f"summary_{eval_name}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--workers", type=int, default=1)
    b.add_argument("--limit", type=int, default=0, help="smoke test on a random subset")
    r = sub.add_parser("run")
    r.add_argument("--test", action="store_true", help="final evaluation on the test split (use once)")
    r.add_argument("--smoke", action="store_true")
    r.add_argument("--models", default="rf,svm", choices=["rf", "rf,svm"])
    args = ap.parse_args()
    (cmd_build if args.cmd == "build" else cmd_run)(args)


if __name__ == "__main__":
    main()