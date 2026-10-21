# -*- coding: utf-8 -*-
"""PSA, nanocarbon (carbon black, soot) analysis in two steps, following the
soot TEM literature (e.g. Sipkens et al., Powder Technology 2021):

1. aggregate-level segmentation (background / carbon), then per aggregate:
   projected area, equivalent diameter, radius of gyration, number of
   primary particles;
2. primary particles by a StarDist model trained for TE images of carbon
   aggregates (synthetic aggregates of known primaries, Beer-Lambert
   transmission; validated on unseen synthetic images: F1 0.66, geometric
   mean primary diameter within about 2 %).

Primary diameter = 2 x 90th percentile of the 32 polygon rays: rays towards
an overlap are shortened, the free part of the contour gives the true
radius (the visible-area ECD underestimates overlapped primaries by ~19 %).
Fractal morphology: Np from the projected area (Köylü et al. 1995, ka = 1.15,
alpha = 1.09), then the ensemble fit Np = kf (2 Rg / dp)^Df.
SE images (bulk powder, no background): primaries only (generic model).
"""
from psa.learned import *  # noqa: F401,F403

from skimage import measure


def _epaisseur_mediane(masque):
    """Typical chain width of the aggregates (px): twice the median of the
    distance map along the skeleton, close to the primary diameter."""
    from skimage.morphology import skeletonize
    if not masque.any():
        return 30.0
    d = ndi.distance_transform_edt(masque)
    v = d[skeletonize(masque)]
    return float(2.0 * np.median(v)) if v.size else 30.0

MODELE_NANO_TE = "nanocarbon_te"
CIBLE_PRIMAIRE_PX = 35.0          # median primary diameter of the training set
KA, ALPHA = 1.15, 1.09            # Köylü et al. 1995, projected-area relation


def _detecter_te(g):
    """True when the particles are DARK on a bright background (TE)."""
    try:
        return not bool(nano.detecter_polarite(g))
    except Exception as e:                       # fallback, but traced
        journal("warning", f"nanocarbon: polarity detection failed ({e}), "
                           "median/mean rule used")
        return float(np.median(g)) > float(np.mean(g))


def fond_te(g, ordre=2, iterations=3):
    """Bright TE background as a smooth polynomial surface fitted on the
    background pixels (the brightest ones, refined iteratively), and the
    background noise (robust std). Returns (background, noise)."""
    H, W = g.shape
    pas = max(1, int(max(H, W) / 256))
    gs = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.0)[::pas, ::pas]
    yy, xx = np.mgrid[0:gs.shape[0], 0:gs.shape[1]]
    yy, xx = yy.ravel() / gs.shape[0], xx.ravel() / gs.shape[1]
    v = gs.ravel()
    termes = [np.ones_like(xx)] + [xx ** i * yy ** j for i in range(ordre + 1)
                                   for j in range(ordre + 1 - i) if i + j > 0]
    A_ = np.stack(termes, 1)
    sel = v >= np.percentile(v, 60)
    for _ in range(iterations):
        c, *_ = np.linalg.lstsq(A_[sel], v[sel], rcond=None)
        fit = A_ @ c
        r = v - fit
        s_ = 1.4826 * np.median(np.abs(r[sel] - np.median(r[sel])))
        sel = r > -2.5 * max(s_, 1.0)
    Yf, Xf = np.mgrid[0:H, 0:W]
    Yf, Xf = Yf.ravel() / H, Xf.ravel() / W
    termes = [np.ones_like(Xf)] + [Xf ** i * Yf ** j for i in range(ordre + 1)
                                   for j in range(ordre + 1 - i) if i + j > 0]
    fond = (np.stack(termes, 1) @ c).reshape(H, W).astype(np.float32)
    return fond, float(max(s_, 1.0))


def segmenter_agregats(g, d_primaire_px=None):
    """Aggregate mask of a TE image (dark carbon on a bright background):
    transmission t = I / I_background (Beer-Lambert); a pixel is carbon
    when t drops below 1 by more than the background noise. Cleaning, small
    holes (< ~one primary) filled, fragments smaller than half a primary
    removed. Returns a label image of the aggregates."""
    fond, bruit = fond_te(g)
    t = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.5) / np.maximum(fond, 1)
    seuil = 1.0 - max(3.5 * bruit / max(float(np.median(fond)), 1.0), 0.03)
    m = (t < seuil).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    dp = d_primaire_px or 20.0
    trou_max = 0.6 * np.pi * (dp / 2) ** 2
    plein = ndi.binary_fill_holes(m > 0)
    trous, nt = ndi.label(plein & ~(m > 0))
    if nt:
        a = ndi.sum(np.ones_like(trous), trous, np.arange(1, nt + 1))
        petits = np.isin(trous, np.nonzero(a < trou_max)[0] + 1)
        m = ((m > 0) | petits).astype(np.uint8) * 255
    lab, n = ndi.label(m > 0)
    if n:
        a = ndi.sum(np.ones_like(lab), lab, np.arange(1, n + 1))
        garde = np.zeros(n + 1, bool)
        garde[1:] = a >= 0.5 * np.pi * (dp / 2) ** 2
        lab = np.where(garde[lab], lab, 0)
        _, lab = np.unique(lab, return_inverse=True)
        lab = lab.reshape(m.shape).astype(np.int32)
    return lab


def primaires_stardist(gb, d_typ_px, nom=None, tta=True, params=None,
                       passes=2):
    """Primary particles on a BRIGHT-particle image. Returns a dict of
    arrays: x, y (px, full image), dp_px (2 x P90 of the rays), prob, and
    the polygons (n, 2, rays) in full-image pixels."""
    p = params or {}
    nom = nom or MODELE_NANO_TE
    if backend_stardist(nom) != "onnx":
        nom = STARDIST_MODELE
    pr0, nm0 = _seuils_modele(nom)
    prob_t = float(p.get("stardist_prob") or pr0)
    nms_t = float(p.get("stardist_nms") or nm0)
    H, W = gb.shape
    res = None
    ech = float(np.clip(CIBLE_PRIMAIRE_PX / max(d_typ_px, 1.0), 0.15, 1.75))
    ech = min(ech, 2600.0 / max(H, W))          # bounded working image size
    for _passe in range(passes):
        gs = cv2.resize(gb, (max(32, int(W * ech)), max(32, int(H * ech))),
                        interpolation=cv2.INTER_AREA if ech < 1 else
                        cv2.INTER_CUBIC)
        x = _normaliser(gs)
        cpts, cpr, cds = candidats_stardist(x, nom, prob_t, tta)
        pr, co, pts = nms_candidats(cpts, cpr, cds, x.shape, nms_t)
        ray = np.linalg.norm(co - pts[:, :, None], axis=1) if len(co) else \
            np.zeros((0, 32))
        dp_w = 2.0 * np.percentile(ray, 90, axis=1) if len(ray) else np.array([])
        res = {"x": pts[:, 1] / ech if len(pts) else np.array([]),
               "y": pts[:, 0] / ech if len(pts) else np.array([]),
               "dp_px": dp_w / ech, "prob": pr, "poly": co / ech,
               "echelle": ech, "modele": nom, "prob_thresh": prob_t,
               "nms_thresh": nms_t}
        if not len(dp_w):
            break
        ecart = float(np.median(dp_w)) / CIBLE_PRIMAIRE_PX
        if 0.75 <= ecart <= 1.33:
            break
        ech = float(np.clip(ech / ecart, 0.15, min(1.75, 2600.0 / max(H, W))))
    return res


def chercher_echelle(gb, nom, params=None, candidats=None, cote_max=900):
    """Working-scale search (SE images, no aggregate mask): the model is run
    (no augmentation) on a central region for several assumed primary
    sizes; the retained size has the highest mean network confidence among
    those whose median size at the working scale matches the training
    (0.7-1.4 x target). Returns (size_px, confidence)."""
    H, W = gb.shape
    h, w = min(H, cote_max), min(W, cote_max)
    y0, x0 = (H - h) // 2, (W - w) // 2
    zone = gb[y0:y0 + h, x0:x0 + w]
    candidats = candidats or np.geomspace(20, 110, 6)
    meilleur, n_max = None, 0
    for dt in candidats:
        r = primaires_stardist(zone, float(dt), nom, tta=False, params=params,
                               passes=1)
        if not len(r["dp_px"]):
            continue
        med_w = float(np.median(r["dp_px"])) * r["echelle"]
        conf = float(np.mean(r["prob"]))
        # criterion: mean network confidence (the count favours over-split
        # interpretations; validated on synthetic aggregates of known size)
        if 0.7 * CIBLE_PRIMAIRE_PX <= med_w <= 1.4 * CIBLE_PRIMAIRE_PX and \
                len(r["dp_px"]) >= 10 and conf > n_max:
            meilleur, n_max = float(np.median(r["dp_px"])), conf
    return meilleur, n_max


def _rg(ys, xs):
    return float(np.sqrt(np.mean((ys - ys.mean()) ** 2 + (xs - xs.mean()) ** 2)))


def analyser_nanocarbone(image_bgr, ppm, params=None, zones=None,
                         progress=None):
    """Two-step nanocarbon analysis. ppm in px/µm. Same result layout as
    the other detectors (diametres = primary diameters in nm)."""
    p = dict(nano.NANO_PARAMS)
    if params:
        p.update(params)

    def pg(v, m):
        if progress:
            progress(v, m)
    g0 = image_bgr if image_bgr.ndim == 2 else cv2.cvtColor(image_bgr,
                                                            cv2.COLOR_BGR2GRAY)
    y_bar = g0.shape[0]
    if p.get("couper_databar", True):
        try:
            y_bar = int(nano.detecter_databar(g0)) or y_bar
        except Exception as e:
            journal("warning", f"nanocarbon: databar not detected ({e}); the "
                               "whole image is analysed: mask the databar with "
                               "an exclusion zone if present")
    g = g0[:max(16, y_bar - 2)].copy()
    if zones:
        med = int(np.median(g))
        for z in zones:
            cv2.fillPoly(g, [np.asarray(z, np.int32)], med)
    ppnm = ppm / 1000.0
    te = _detecter_te(g) if p.get("polarite", "auto") == "auto" else \
        p.get("polarite") == "sombres"
    gb = cv2.bitwise_not(g) if te else g
    H, W = g.shape
    pg(15, "Aggregate segmentation…")
    lab_agg = segmenter_agregats(g) if te else \
        (np.ones((H, W), np.int32))
    # typical primary size: aggregate chain width (TE); for SE (no
    # background) the area-weighted estimator of the generic engine
    # Typical primary size. TE: width of the aggregate chains, refined by
    # the second pass (best on synthetic aggregates of known size: median
    # error -1.3 %, mean |error| 4.9 %). SE (no background): working-scale
    # search on the network confidence.
    if te and (lab_agg > 0).any():
        d_typ = _epaisseur_mediane(lab_agg > 0)
    else:
        pg(25, "Working-scale search (SE image)…")
        d_typ = chercher_echelle(gb, STARDIST_MODELE, p)[0] or \
            estimer_diametre_px(gb)
    pri = primaires_stardist(gb, d_typ, MODELE_NANO_TE if te else
                             STARDIST_MODELE, tta=p.get("stardist_tta", True),
                             params=p)
    # keep primaries inside aggregates and not cut by the image border
    xi = np.clip(np.round(pri["x"]).astype(int), 0, W - 1)
    yi = np.clip(np.round(pri["y"]).astype(int), 0, H - 1)
    r = pri["dp_px"] / 2
    dedans = lab_agg[yi, xi] > 0
    hors_bord = (pri["x"] - r > 2) & (pri["y"] - r > 2) & \
        (pri["x"] + r < W - 3) & (pri["y"] + r < H - 3)
    ok = dedans & hors_bord
    n_bord = int((dedans & ~hors_bord).sum())
    dp_px = pri["dp_px"][ok]
    dp_nm = dp_px / ppnm
    x_ok, y_ok = pri["x"][ok], pri["y"][ok]
    pg(70, "Aggregate morphology…")
    agg = []
    dp_moy = float(np.mean(dp_px)) if dp_px.size else d_typ
    for rp in measure.regionprops(lab_agg):
        minr, minc, maxr, maxc = rp.bbox
        bord = minr <= 0 or minc <= 0 or maxr >= H or maxc >= W
        ys, xs = rp.coords[:, 0].astype(float), rp.coords[:, 1].astype(float)
        dans = (lab_agg[np.clip(np.round(y_ok).astype(int), 0, H - 1),
                        np.clip(np.round(x_ok).astype(int), 0, W - 1)] == rp.label)
        aa = float(rp.area)
        agg.append({"label": int(rp.label), "bord": bool(bord),
                    "aire_nm2": aa / ppnm ** 2,
                    "ecd_nm": 2 * np.sqrt(aa / np.pi) / ppnm,
                    "rg_nm": _rg(ys, xs) / ppnm,
                    "np_detectes": int(dans.sum()),
                    "np_aire": float(KA * (aa / (np.pi * dp_moy ** 2 / 4)) ** ALPHA),
                    "centre": (float(rp.centroid[1]), float(rp.centroid[0]))})
    complets = [a for a in agg if not a["bord"]]
    # ensemble fractal fit Np = kf (2 Rg / dp)^Df (aggregates with Np >= 5)
    df = kf = r2 = None
    pts_f = [(a["np_aire"], 2 * a["rg_nm"] / (dp_moy / ppnm)) for a in complets
             if a["np_aire"] >= 5 and a["rg_nm"] > 0]
    if len(pts_f) >= 5:
        npa, xr = np.log([q[0] for q in pts_f]), np.log([q[1] for q in pts_f])
        A_ = np.vstack([xr, np.ones_like(xr)]).T
        (df, lkf), *_ = np.linalg.lstsq(A_, npa, rcond=None)
        pred = A_ @ np.array([df, lkf])
        r2 = float(1 - np.sum((npa - pred) ** 2) / max(np.sum((npa - npa.mean()) ** 2), 1e-12))
        df, kf = float(df), float(np.exp(lkf))
    # quality: mean network confidence and number of primaries
    conf = float(np.mean(pri["prob"][ok])) if ok.any() else 0.0
    n = int(dp_nm.size)
    score = float(np.clip(100 * (0.6 * conf + 0.4 * min(1.0, n / 200)), 0, 100))
    st = "good" if score >= 70 and n >= 100 else ("warn" if n >= 30 else "bad")
    msg = (f"{n} primary particles in {len(complets)} complete aggregates"
           if te else f"{n} primary particles (SE image: bulk powder, "
           "aggregates not separable)")
    pg(95, "Summary…")
    d_agg = np.array([a["ecd_nm"] for a in complets], float)
    cercles = [(float(x), float(y), float(d / 2)) for x, y, d in
               zip(x_ok, y_ok, dp_px)]
    aires = np.array([a["aire_nm2"] for a in complets], float)
    nano_res = {
        "methode": "stardist_deux_etapes", "imagerie": "TE" if te else "SE",
        "gray_pre": g, "seg": {"d_aggl_nm": d_agg}, "granulo": None,
        "d_hough_nm": np.array([]), "cercles_hough": cercles,
        "df": df, "df_r2": r2, "kf": kf, "agregats": agg,
        "primaires": {"x": x_ok, "y": y_ok, "dp_nm": dp_nm,
                      "poly": pri["poly"][ok], "prob": pri["prob"][ok]},
        "stardist": {k: pri[k] for k in ("echelle", "modele", "prob_thresh",
                                         "nms_thresh")},
        "aire_moy_nm2": float(np.mean(aires)) if aires.size else 0.0,
    }
    ids = [a["label"] for a in complets]
    return {
        "mode": "nano", "unite": "nm", "nano": nano_res,
        "moteur": "stardist",
        "masque": (lab_agg > 0).astype(np.uint8) * 255,
        "labels_ws": np.where(np.isin(lab_agg, ids), lab_agg, 0),
        "diametres": dp_nm,
        "populations": {"primaires": dp_nm, "agregats": d_agg},
        "population": "primaires",
        "n_regions": len(complets),
        "circularites": np.array([]), "aspects": np.array([]),
        "solidites": np.array([]),
        "ids": ids, "centres_px": [(int(a["centre"][0]), int(a["centre"][1]))
                                   for a in complets],
        "n_particules": n, "n_bord": n_bord,
        "score_ws": score, "msg_ws": msg, "statut_ws": st,
    }


__all__ = [_n for _n in dir() if not _n.startswith("__")]
