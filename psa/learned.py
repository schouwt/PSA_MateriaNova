# -*- coding: utf-8 -*-
"""PSA, learned segmentation of particles (StarDist, deep learning).

StarDist (Schmidt et al., MICCAI 2018; Weigert et al., WACV 2020) predicts,
for every pixel, the probability of being inside an object and the distances
to its border along 32 rays: each particle is a star-convex polygon. This
suits compact particles and spheres, including their shaded bottom in SEM and
textured surfaces, where the threshold + watershed chain fragments or merges.

Default backend: the network exported to ONNX (Models/stardist/<model>/
model.onnx), run by onnxruntime on every CPU core, with our own validated
re-implementation of the StarDist post-processing: no TensorFlow needed.
Fallback: the original stardist + tensorflow packages when installed. The regions are then measured
by the same code as the watershed (psa_nano_meb.mesurer_regions).
"""
from psa.shapes import *  # noqa: F401,F403

import importlib.util

STARDIST_MODELE = "2D_versatile_fluo"   # pretrained, bright objects on dark
STARDIST_CIBLE_PX = 45.0                # object diameter where it works best
_MODELES = {}


def _dossier_modele(nom=STARDIST_MODELE):
    return os.path.join(_DOSSIER_APP, "Models", "stardist", nom)


def backend_stardist(nom=STARDIST_MODELE):
    """'onnx' (default: onnxruntime + model.onnx, no TensorFlow needed),
    'tensorflow' (stardist + tensorflow installed) or None."""
    try:
        if importlib.util.find_spec("onnxruntime") is not None and \
                os.path.exists(os.path.join(_dossier_modele(nom), "model.onnx")):
            return "onnx"
        if all(importlib.util.find_spec(m) is not None
               for m in ("stardist", "tensorflow", "csbdeep")):
            return "tensorflow"
    except (ImportError, ValueError):
        pass
    return None


def stardist_disponible():
    """True when a StarDist backend is available (no heavy import)."""
    return backend_stardist() is not None


def _seuils_modele(nom=STARDIST_MODELE):
    try:
        with open(os.path.join(_dossier_modele(nom), "thresholds.json")) as f:
            t = json.load(f)
        return float(t.get("prob", 0.479)), float(t.get("nms", 0.3))
    except (OSError, ValueError):
        return 0.479, 0.3


_SESSIONS = {}


def session_onnx(nom=STARDIST_MODELE):
    """onnxruntime session using every logical core."""
    if nom not in _SESSIONS:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, os.cpu_count() or 1)
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _SESSIONS[nom] = ort.InferenceSession(
            os.path.join(_dossier_modele(nom), "model.onnx"), so,
            providers=["CPUExecutionProvider"])
        journal("info", f"StarDist ONNX model loaded: {nom} "
                        f"({so.intra_op_num_threads} threads)")
    return _SESSIONS[nom]


def _normaliser(x, pmin=1.0, pmax=99.8):
    lo, hi = np.percentile(x, (pmin, pmax))
    return ((x.astype(np.float32) - lo) / max(float(hi - lo), 1e-6)).astype(
        np.float32)


def _predire_onnx(x, nom=STARDIST_MODELE):
    """prob (h, w) and dist (h, w, rays) on the 2x2 grid, for a normalised
    image; dimensions padded to a multiple of 16 (U-Net)."""
    sess = session_onnx(nom)
    H, W = x.shape
    xp = np.pad(x, ((0, (-H) % 16), (0, (-W) % 16)), mode="reflect")
    prob, dist = sess.run(None, {sess.get_inputs()[0].name: xp[None, ..., None]})
    h, w = (H + 1) // 2, (W + 1) // 2
    return prob[0, :h, :w, 0], dist[0, :h, :w]


def _predire_tta(x, nom=STARDIST_MODELE):
    """Test-time augmentation: the image and its 3 mirror images; each
    prediction is mapped back (rays re-indexed: horizontal mirror
    phi -> pi - phi, vertical mirror phi -> -phi) and averaged."""
    n = None
    acc_p, acc_d = 0.0, 0.0
    for fy, fx in ((False, False), (False, True), (True, False), (True, True)):
        xi = x[::-1] if fy else x
        xi = xi[:, ::-1] if fx else xi
        p, d = _predire_onnx(np.ascontiguousarray(xi), nom)
        n = d.shape[-1]
        k = np.arange(n)
        if fx:
            p, d = p[:, ::-1], d[:, ::-1][..., (n // 2 - k) % n]
        if fy:
            p, d = p[::-1], d[::-1][..., (-k) % n]
        acc_p = acc_p + p
        acc_d = acc_d + d
    return acc_p / 4.0, acc_d / 4.0


TUILE_PX, MARGE_PX = 1024, 128       # tiling of the network input (memory)


def candidats_stardist(x, nom=STARDIST_MODELE, prob_thresh=0.5, tta=True,
                       grid=(2, 2), b=2, tuile=TUILE_PX, marge=MARGE_PX):
    """Network prediction by TILES (1024 px + 128 px context margin) and
    extraction of the CANDIDATES only (grid points above prob_thresh with
    their ray distances): the memory no longer depends on the image size.
    Returns (points (n, 2) in grid coordinates, prob (n,), dist (n, rays))."""
    H, W = x.shape
    gy_tot, gx_tot = (H + 1) // 2, (W + 1) // 2
    pred = _predire_tta if tta else _predire_onnx
    P, Q, D = [], [], []
    pas = tuile if max(H, W) > tuile + 2 * marge else max(H, W)
    for y0 in range(0, H, pas):
        for x0 in range(0, W, pas):
            ya, xa = max(0, y0 - marge), max(0, x0 - marge)
            yb, xb = min(H, y0 + pas + marge), min(W, x0 + pas + marge)
            p, d = pred(np.ascontiguousarray(x[ya:yb, xa:xb]), nom)
            # core of the tile in grid coordinates (all offsets are even)
            cy0, cx0 = (y0 - ya) // grid[0], (x0 - xa) // grid[1]
            ny = (min(y0 + pas, H) - y0 + 1) // grid[0]
            nx = (min(x0 + pas, W) - x0 + 1) // grid[1]
            pc = p[cy0:cy0 + ny, cx0:cx0 + nx]
            ii, jj = np.nonzero(pc > prob_thresh)
            if not ii.size:
                continue
            gi, gj = ii + y0 // grid[0], jj + x0 // grid[1]
            ok = (gi >= b) & (gj >= b) & (gi < gy_tot - b) & (gj < gx_tot - b)
            P.append(np.stack([gi[ok], gj[ok]], 1))
            Q.append(pc[ii[ok], jj[ok]])
            D.append(d[cy0:cy0 + ny, cx0:cx0 + nx][ii[ok], jj[ok]])
    if not P:
        return np.zeros((0, 2), int), np.zeros(0), np.zeros((0, 32))
    return np.concatenate(P), np.concatenate(Q), np.concatenate(D)


def nms_candidats(pts, p, d, shape, nms_thresh, grid=(2, 2), exact=False):
    """Greedy non-maximum suppression of star-convex candidates: a candidate
    is dropped when its overlap with an already kept polygon, divided by the
    smaller area, exceeds nms_thresh. Returns (prob, polygons (n, 2, rays),
    points) of the kept objects, by decreasing probability.

    v1.7.8: a candidate whose CENTRE already lies inside a kept polygon is
    dropped without rasterising it (exact=False). A dense field gives about
    150 candidates per object (196 000 for 1 352 particles on a x100 powder),
    almost all centred inside an object already kept: the suppression took
    22 s, it now takes about a third. Validated against the official StarDist
    NMS on the same network outputs (powder x100, spheres, resin x20k, TCP):
    same objects as the exhaustive version (757 / 759, 159 / 159,
    1388 / 1404, 130 / 130; F1 vs official 0.9985, 1, 0.9943, 1)."""
    n_rays = d.shape[-1] if d.ndim == 2 else 32
    phi = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    if not len(p):
        return np.zeros(0), np.zeros((0, 2, n_rays)), np.zeros((0, 2))
    o = np.argsort(-p, kind="stable")
    pts, p, d = pts[o], p[o], d[o]
    ii, jj = pts[:, 0], pts[:, 1]
    coords = np.stack([ii[:, None] * grid[0] + d * np.sin(phi)[None],
                       jj[:, None] * grid[1] + d * np.cos(phi)[None]], 1)
    H, W = shape
    occ = np.zeros((H, W), np.int32)
    aires = [0.0]
    gardes = []
    for k in range(len(p)):
        c = coords[k]
        y0, y1 = int(max(0, np.floor(c[0].min()))), int(min(H, np.ceil(c[0].max()) + 1))
        x0, x1 = int(max(0, np.floor(c[1].min()))), int(min(W, np.ceil(c[1].max()) + 1))
        if y1 <= y0 or x1 <= x0:
            continue
        if not exact:
            cyi, cxi = int(ii[k] * grid[0]), int(jj[k] * grid[1])
            if 0 <= cyi < H and 0 <= cxi < W and occ[cyi, cxi]:
                continue                  # centre déjà couvert : supprimé
        loc = np.zeros((y1 - y0, x1 - x0), np.uint8)
        q = np.round(np.stack([c[1] - x0, c[0] - y0], 1)).astype(np.int32)
        cv2.fillPoly(loc, [q], 1)
        a = int(loc.sum())
        if a == 0:
            continue
        sous = occ[y0:y1, x0:x1][loc > 0]
        sous = sous[sous > 0]
        if sous.size:
            ids, inter = np.unique(sous, return_counts=True)
            if np.any(inter / np.minimum(a, np.asarray(aires)[ids]) > nms_thresh):
                continue
        gardes.append(k)
        aires.append(float(a))
        z = occ[y0:y1, x0:x1]
        z[(loc > 0) & (z == 0)] = len(aires) - 1
    g = np.asarray(gardes, int)
    points = np.stack([ii * grid[0], jj * grid[1]], 1).astype(float)
    return p[g], coords[g], points[g]


def charger_modele(nom=STARDIST_MODELE):
    """Loads (once) the StarDist model. A local copy in
    Models/stardist/<name>/ next to the program is used first (offline use,
    executable); otherwise the pretrained model is downloaded once from
    GitHub into the Keras cache."""
    if nom in _MODELES:
        return _MODELES[nom]
    if not stardist_disponible():
        raise RuntimeError("StarDist is not installed: pip install stardist "
                           "tensorflow")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from stardist.models import StarDist2D
    local = os.path.join(_DOSSIER_APP, "Models", "stardist")
    if os.path.isdir(os.path.join(local, nom)):
        modele = StarDist2D(None, name=nom, basedir=local)
    else:
        modele = StarDist2D.from_pretrained(nom)
    _MODELES[nom] = modele
    journal("info", f"StarDist model loaded: {nom}")
    return modele


def moteur_effectif(params):
    """Engine actually used: StarDist when requested AND available,
    otherwise the classic watershed (logged once)."""
    m = (params or {}).get("moteur", "stardist")
    if m == "stardist" and not stardist_disponible():
        if not _MODELES.get("_avert"):
            journal("warning", "StarDist unavailable (pip install onnxruntime): "
                               "classic watershed engine used")
            _MODELES["_avert"] = True
        return "watershed"
    return m


def _diametre_p90(m):
    d = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    v = d[m > 0]
    if v.size < 50 or not (m == 0).any():
        return None                       # masque vide ou image entière
    r = float(2.0 * np.percentile(v, 90) / 0.68)
    return r if np.isfinite(r) else None


def estimer_diametre_px(g):
    """Typical particle diameter (px), used to choose the working scale.
    Quick mask (Otsu on a smoothed image), then the 90th percentile of the
    distance map over the mask: in a disc of radius R, 10 % of the area lies
    deeper than 0.68 R, hence R = p90 / 0.68. This area-weighted estimate is
    dominated by the real particles, not by the small fragments of shaded
    or textured surfaces.
    Holes are filled (dark centre of shaded SEM spheres), EXCEPT when the
    filling changes the estimate by more than 1.6 times (v1.7.3): in a dense
    field the particles touch and form a network, filling then turns the
    matrix pockets between them into huge solid blobs (226 px instead of
    about 25 px on a dense powder at x100, 217 px instead of about 25 px on
    a resin at x20k: the image was shrunk 5 times and NO particle was
    found). Underestimating the size is harmless (particles slightly larger
    than optimal for the network), overestimating it is not."""
    gs = cv2.GaussianBlur(g, (0, 0), 2.0)
    _, m0 = cv2.threshold(gs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    noyau = np.ones((5, 5), np.uint8)
    m_brut = cv2.morphologyEx(m0, cv2.MORPH_OPEN, noyau)
    m_plein = cv2.morphologyEx(
        ndi.binary_fill_holes(m0 > 0).astype(np.uint8) * 255,
        cv2.MORPH_OPEN, noyau)
    d_plein, d_brut = _diametre_p90(m_plein), _diametre_p90(m_brut)
    if d_plein is None and d_brut is None:
        return 40.0
    if d_plein is None:
        return d_brut
    if d_brut is not None and d_plein > 1.6 * d_brut:
        return d_brut                     # réseau percolant : pas de remplissage
    return d_plein


def labels_stardist(gray, params=None):
    """Instance labels (full resolution) predicted by StarDist on an 8-bit
    grey image with BRIGHT particles. Returns (labels, info)."""
    p = params or {}
    nom = p.get("stardist_modele", STARDIST_MODELE)
    backend = backend_stardist(nom)
    if backend is None:
        raise RuntimeError("Deep learning engine unavailable: pip install "
                           "onnxruntime (or stardist tensorflow)")
    d_typ = float(p.get("stardist_diametre_px") or 0.0) or \
        estimer_diametre_px(gray)
    ech = float(np.clip(STARDIST_CIBLE_PX / max(d_typ, 1.0), 0.05, 2.0))
    H, W = gray.shape[:2]
    gs = cv2.resize(gray, (max(16, int(W * ech)), max(16, int(H * ech))),
                    interpolation=cv2.INTER_AREA if ech < 1 else
                    cv2.INTER_LINEAR)
    pr0, nm0 = _seuils_modele(nom)
    prob_t = float(p["stardist_prob"]) if p.get("stardist_prob") is not None else pr0
    nms_t = float(p["stardist_nms"]) if p.get("stardist_nms") is not None else nm0
    tta = bool(p.get("stardist_tta", True))
    if backend == "onnx":
        x = _normaliser(gs)
        pts, pr_, ds = candidats_stardist(x, nom, prob_t, tta)
        probs, coords, _pts = nms_candidats(pts, pr_, ds, x.shape, nms_t)
    else:
        from csbdeep.utils import normalize
        modele = charger_modele(nom)
        tuiles = tuple(max(1, int(math.ceil(n / 1024.0))) for n in gs.shape[:2])
        _, det = modele.predict_instances(
            normalize(gs, 1, 99.8), n_tiles=tuiles if max(tuiles) > 1 else None,
            prob_thresh=prob_t, nms_thresh=nms_t, show_tile_progress=False,
            verbose=False)
        coords, probs = det["coord"], det["prob"]
        tta = False
    # Full-resolution labels from the star-convex POLYGONS (smooth borders)
    lab = np.zeros((H, W), np.int32)
    for i in np.argsort(probs):                    # most probable drawn last
        yx = np.asarray(coords[i], float) / ech
        pts = np.round(np.stack([yx[1], yx[0]], 1)).astype(np.int32)
        cv2.fillPoly(lab, [pts], int(i) + 1)
    return lab, {"echelle": ech, "diametre_typique_px": d_typ, "modele": nom,
                 "backend": backend, "tta": tta, "prob_thresh": prob_t,
                 "nms_thresh": nms_t}


def segmenter_stardist(gray_pre, ppnm, params=None, gray_brut=None):
    """Same output as psa_nano_meb.segmenter_hmax (masque, labels, dist,
    d_aggl_nm, d_regs_nm, circs, centres_px, ids, aspects, solidites,
    n_bord), so that every downstream step works unchanged.
    gray_brut: raw grey image of the same field (preferred: the network was
    trained on natural intensities, the SEM contrast enhancement exaggerates
    the surface texture)."""
    p = dict(nano.NANO_PARAMS)
    if params:
        p.update(params)
    # taille typique imposée par l'utilisateur (v1.7.3), en nm : convertie
    # en px à l'échelle de l'image reçue (aperçu réduit ou plein format)
    if p.get("stardist_diametre_nm"):
        p["stardist_diametre_px"] = float(p["stardist_diametre_nm"]) * ppnm
    src = gray_pre
    if gray_brut is not None:
        src = gray_brut[:gray_pre.shape[0], :gray_pre.shape[1]]
    g = src if p.get("particules_claires", True) else cv2.bitwise_not(src)
    lab, info = labels_stardist(g, p)
    masque = (lab > 0).astype(np.uint8) * 255
    dist = cv2.distanceTransform(masque, cv2.DIST_L2, 5)
    lab_cc, n_cc = ndi.label(masque > 0)
    aires = ndi.sum(np.ones_like(lab_cc), lab_cc, np.arange(1, n_cc + 1)) \
        if n_cc else np.array([])
    d_aggl_nm = [2.0 * math.sqrt(a / math.pi) / ppnm for a in aires]
    aire_min_px = max(4, int(p.get("aire_min_px", 20)))
    # predicted polygons stop 1-2 px short of the image edge: a 3 px margin
    # keeps the "touching the border = excluded" rule of the watershed
    seg = nano.mesurer_regions(lab, masque, dist, d_aggl_nm, ppnm, p,
                               aire_min_px, combler=False, marge_bord=3)
    seg["moteur"] = "stardist"
    seg["stardist"] = info
    return seg


def labels_stardist_optique(gris, params):
    """Labels for the optical mode: the particle polarity follows the
    optical pipeline (inverted colours option already applied upstream)."""
    g = gris.copy()
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # The background is what touches the image frame: if the frame is mostly
    # bright, the particles are dark (transmitted light) and are inverted.
    cadre = np.concatenate([m[:3].ravel(), m[-3:].ravel(),
                            m[:, :3].ravel(), m[:, -3:].ravel()])
    if (cadre > 0).mean() > 0.5:
        g = cv2.bitwise_not(g)
    return labels_stardist(g, params)[0]


__all__ = [_n for _n in dir() if not _n.startswith("__")]
