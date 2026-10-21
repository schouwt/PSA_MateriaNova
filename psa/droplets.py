# -*- coding: utf-8 -*-
"""PSA, droplets in microfluidics (bright-field / transmitted light).

Why the usual PSA pipelines fail on these images: a droplet is seen as a
DARK RING (refraction at the interface), and its inside has the SAME grey
level as the continuous phase. A threshold therefore picks the rings and
the gaps between droplets, never the droplets; StarDist (trained on filled
bright objects) finds nothing.

Method used here, no learning, CPU only:
  1. dark ring map: high-pass image below a noise-scaled threshold;
  2. candidate droplets = regions ENCLOSED by the rings (connected
     components of the complement), kept if round and convex; letters,
     channel walls and gaps between droplets are rejected by shape;
  3. a Hough circle search with the radius found in step 2 recovers the
     droplets whose ring is broken (their inside leaks into the outside);
  4. each droplet is refined on radial intensity profiles (72 rays): the
     ring MIDDLE (intensity minimum) and the OUTER EDGE (steepest rise
     after the minimum) are located with sub-pixel precision, and a circle
     is fitted to each set of points with outlier rejection.

Diameter convention: the dark ring is an optical effect, the interface lies
inside it. "middle" (default) is the natural choice for packed emulsions,
where neighbours share one ring; "outer" suits isolated droplets with a
thick dark rim. Both are returned; their difference is the convention
uncertainty and is reported.

Volume: sphere if the droplet is smaller than the channel height h; if it
is squeezed (D > h), disc of height h with a half-torus rim (Pappus):
V = pi a^2 h + pi^2 h^2 a / 4 + pi h^3 / 6, a = (D - h) / 2.
"""
import math

import numpy as np
import cv2
from scipy import ndimage as ndi


# =============================================================================
# 1-2. ANNEAUX SOMBRES ET RÉGIONS ENCLOSES
# =============================================================================
def _gris(image):
    g = image
    if g.ndim == 3:
        g = cv2.cvtColor(g[..., :3], cv2.COLOR_BGR2GRAY)
    return g.astype(np.float32)


def carte_anneaux(gris, k=0.5):
    """Dark-ring mask: pixels darker than the local background by more than
    k times the robust spread of the high-pass image (the rings ARE the
    structure, so a pixel-noise threshold would depend on the compression
    and on the camera). The local background is a wide blur, so slow
    illumination gradients do not matter. Returns (mask, high-pass, spread)."""
    H, W = gris.shape
    s_fond = max(8.0, 0.06 * min(H, W))
    lisse = cv2.GaussianBlur(gris, (0, 0), 0.8)
    hp = lisse - cv2.GaussianBlur(gris, (0, 0), s_fond)
    s_hp = 1.4826 * np.median(np.abs(hp - np.median(hp))) + 1e-6
    sombre = hp < -k * s_hp
    sombre = cv2.morphologyEx(sombre.astype(np.uint8), cv2.MORPH_CLOSE,
                              np.ones((3, 3), np.uint8))
    return sombre.astype(bool), hp, float(s_hp)


def regions_encloses(sombre, aire_min=20, circ_min=0.72, sol_min=0.90):
    """Round convex regions enclosed by the dark rings: (cx, cy, r_equiv)."""
    interieur = (~sombre).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(interieur, connectivity=4)
    H, W = sombre.shape
    cands = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < aire_min or a > 0.25 * H * W:
            continue
        if x == 0 or y == 0 or x + w >= W or y + h >= H:
            continue                     # coupée par le bord : non mesurable
        m = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cnts, key=cv2.contourArea)
        per = cv2.arcLength(c, True)
        hull = cv2.contourArea(cv2.convexHull(c))
        if per <= 0 or hull <= 0:
            continue
        circ = 4 * math.pi * a / per ** 2
        sol = a / hull
        if circ >= circ_min and sol >= sol_min:
            cands.append((cen[i][0], cen[i][1], math.sqrt(a / math.pi)))
    c = np.array(cands, float).reshape(-1, 3)
    if len(c) >= 3:
        # les interstices entre gouttes tassées (petits triangles convexes
        # à cause de la discrétisation) sont 2 fois plus nombreux que les
        # gouttes en empilement compact : on garde la population qui porte
        # la SURFACE, et les régions d'au moins 0,35 fois son rayon
        r = c[:, 2]
        o = np.argsort(r)
        cum = np.cumsum(r[o] ** 2)
        r_surf = r[o][np.searchsorted(cum, 0.5 * cum[-1])]
        c = c[r >= 0.35 * r_surf]
    return c


# =============================================================================
# 4. AFFINAGE SUR PROFILS RADIAUX
# =============================================================================
def _ajuster_cercle(x, y):
    """Algebraic (Kasa) circle fit with 2 passes of outlier rejection."""
    ok = np.isfinite(x) & np.isfinite(y)
    for _ in range(3):
        if ok.sum() < 8:
            return None
        A = np.c_[2 * x[ok], 2 * y[ok], np.ones(ok.sum())]
        b = x[ok] ** 2 + y[ok] ** 2
        (cx, cy, c), *_ = np.linalg.lstsq(A, b, rcond=None)
        r = math.sqrt(max(c + cx ** 2 + cy ** 2, 0))
        res = np.hypot(x - cx, y - cy) - r
        mad = 1.4826 * np.median(np.abs(res[ok])) + 1e-3
        nouv = ok & (np.abs(res) < max(2.5 * mad, 0.75))
        if nouv.sum() == ok.sum():
            break
        ok = nouv
    return cx, cy, r, float(np.sqrt(np.mean(res[ok] ** 2))), float(ok.mean())


def _parabole(v, i):
    if 0 < i < len(v) - 1:
        d = v[i - 1] - 2 * v[i] + v[i + 1]
        if d != 0:
            return i + 0.5 * (v[i - 1] - v[i + 1]) / d
    return float(i)


def _passe(gris, cx, cy, r0, lo, hi, n_rayons):
    """One radial pass: per ray, centre of the intensity DIP (centroid of
    the part darker than half-way between the dip and the inside, more
    stable than the minimum on thick flat-bottomed rings) and outer edge
    (steepest rise after the dip)."""
    pas = 0.25
    s = np.arange(0.5 * r0, hi * r0 + 0.3 * r0 + 8, pas)
    th = np.linspace(0, 2 * np.pi, n_rayons, endpoint=False)
    X = cx + np.outer(np.cos(th), s)
    Y = cy + np.outer(np.sin(th), s)
    P = ndi.map_coordinates(gris, [Y, X], order=1, mode="nearest")
    P = ndi.gaussian_filter1d(P, 1.5, axis=1)
    G = np.gradient(P, axis=1)
    sel = np.where((s >= lo * r0) & (s <= hi * r0 + 2))[0]
    rm, re, contr = np.full(n_rayons, np.nan), np.full(n_rayons, np.nan), []
    ri = np.full(n_rayons, np.nan)
    for k in range(n_rayons):
        p = P[k]
        j = sel[0] + int(np.argmin(p[sel]))
        dedans = float(np.median(p[s < 0.7 * s[j]])) if (s < 0.7 * s[j]).any() else float(p[0])
        c = dedans - p[j]
        contr.append(c)
        if c <= 0:
            continue
        seuil = p[j] + 0.5 * c
        # creux limité à ±max(2 px, 12 % du rayon) autour du minimum : sur
        # des gouttes tassées, l'anneau se prolonge dans l'interstice sombre
        lim = int(max(2.0, 0.12 * r0) / pas)
        a = j
        while a > max(0, j - lim) and p[a - 1] < seuil:
            a -= 1
        b = j
        while b < min(len(p) - 1, j + lim) and p[b + 1] < seuil:
            b += 1
        w = seuil - p[a:b + 1]
        rm[k] = s[0] + pas * (a + float((w * np.arange(b - a + 1)).sum() / w.sum()))
        # bord intérieur : plus forte descente avant le minimum (côté goutte,
        # jamais perturbé par l'anneau d'une voisine)
        deb = max(1, j - int((0.25 * r0 + 4) / pas))
        if j - deb > 1:
            ji = deb + int(np.argmin(G[k, deb:j]))
            ji = _parabole(G[k], ji) if 0 < ji < len(p) - 1 else ji
            ri[k] = s[0] + pas * ji
        fin = min(len(p) - 1, b + int((0.25 * r0 + 4) / pas))
        if fin > j + 1:
            je = j + int(np.argmax(G[k, j:fin]))
            je = _parabole(-G[k], je) if 0 < je < len(p) - 1 else je
            re[k] = s[0] + pas * je
    xm, ym = cx + rm * np.cos(th), cy + rm * np.sin(th)
    xe, ye = cx + re * np.cos(th), cy + re * np.sin(th)
    xi, yi = cx + ri * np.cos(th), cy + ri * np.sin(th)
    # demi-épaisseur de l'anneau sur les rayons où le creux est le plus
    # étroit (pas de voisine collée dans cette direction)
    dmi = rm - ri
    ok = np.isfinite(dmi)
    hw = float(np.percentile(dmi[ok], 25)) if ok.sum() >= 8 else np.nan
    return (_ajuster_cercle(xm, ym), _ajuster_cercle(xe, ye),
            float(np.median(contr)), _ajuster_cercle(xi, yi), hw, (xm, ym))


def affiner(gris, cx, cy, r0, n_rayons=72):
    """Sub-pixel ring middle and outer edge, two passes (wide search, then
    narrow around the first estimate). Returns a dict, or None if no ring
    is found around (cx, cy)."""
    fm = _passe(gris, cx, cy, r0, 0.75, 1.35, n_rayons)[0]
    if fm is None or not (0.6 < fm[2] / r0 < 1.6):
        return None
    cx, cy, r1 = fm[0], fm[1], fm[2]
    fm, fe, contr, fi, hw, pts = _passe(gris, cx, cy, r1, 0.85, 1.15, n_rayons)
    if fm is None or not (0.8 < fm[2] / r1 < 1.2):
        return None
    # milieu de l'anneau = bord intérieur + demi-épaisseur : insensible à la
    # fusion avec l'anneau d'une goutte voisine dans un empilement compact
    r_mil = fm[2]
    if fi is not None and np.isfinite(hw) and 0 <= hw < 0.3 * fm[2]:
        r_mil = fi[2] + hw
    # forme : ellipse ajustée sur le milieu de l'anneau (déformation des
    # gouttes tassées ou écrasées). La circularité est celle de l'ellipse
    # (4πA/P², périmètre de Ramanujan) : celle du polygone des 72 points
    # serait faussée par leur bruit (périmètre en zigzag)
    asp, circ = 1.0, 1.0
    xm, ym = pts
    ok = np.isfinite(xm) & np.isfinite(ym)
    if ok.sum() >= 12:
        q = np.c_[xm[ok], ym[ok]].astype(np.float32)
        try:
            (_, _), (a1, a2), _ = cv2.fitEllipse(q)
            a, b = max(a1, a2) / 2, min(a1, a2) / 2
            if b > 0:
                asp = float(a / b)
                h = ((a - b) / (a + b)) ** 2
                per = np.pi * (a + b) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))
                circ = float(min(1.0, 4 * np.pi * (np.pi * a * b) / per ** 2))
        except cv2.error:
            pass
    return {"cx": fm[0], "cy": fm[1], "r_milieu": r_mil, "rms_milieu": fm[3],
            "frac_rayons": fm[4], "aspect": asp, "circularite": circ,
            "r_ext": fe[2] if fe else np.nan, "rms_ext": fe[3] if fe else np.nan,
            "contraste": contr}


# =============================================================================
# DÉTECTION COMPLÈTE
# =============================================================================
def detecter_gouttes(image, k_anneau=0.5, hough=True, zones=None):
    """Detects droplets. Returns a dict with per-droplet arrays (pixels):
    cx, cy, d_milieu, d_ext, rms, contraste, bord (circle leaves the
    image), source ('region' or 'hough'), plus diagnostics."""
    g = _gris(image)
    H, W = g.shape
    sombre, hp, s_hp = carte_anneaux(g, k_anneau)
    cands = regions_encloses(sombre)
    if zones is not None:                         # masque des zones utiles
        z = np.asarray(zones, bool)
        if len(cands):
            keep = z[np.clip(cands[:, 1].astype(int), 0, H - 1),
                     np.clip(cands[:, 0].astype(int), 0, W - 1)]
            cands = cands[keep]
    gl = cv2.GaussianBlur(g, (0, 0), 0.8)
    gouttes, sources = [], []
    for cx, cy, re_ in cands:
        # l'intérieur enclos s'arrête au bord intérieur de l'anneau : on
        # cherche l'anneau un peu plus loin
        f = affiner(gl, cx, cy, re_ * 1.08 + 1)
        if f is not None:
            gouttes.append(f)
            sources.append("region")
    # 3. Hough avec le rayon typique, pour les anneaux interrompus
    if hough and len(gouttes) >= 3:
        rt = float(np.median([f["r_milieu"] for f in gouttes]))
        g8 = cv2.normalize(gl, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cercles = cv2.HoughCircles(g8, cv2.HOUGH_GRADIENT_ALT, dp=1.5,
                                   minDist=max(4, 1.2 * rt), param1=60,
                                   param2=0.6, minRadius=max(3, int(0.6 * rt)),
                                   maxRadius=int(1.6 * rt) + 2)
        if cercles is not None:
            deja = np.array([[f["cx"], f["cy"], f["r_milieu"]] for f in gouttes])
            for x, y, r in cercles[0]:
                d = np.hypot(deja[:, 0] - x, deja[:, 1] - y)
                if (d < 0.6 * (deja[:, 2] + r)).any():
                    continue
                if zones is not None and not np.asarray(zones, bool)[
                        min(int(y), H - 1), min(int(x), W - 1)]:
                    continue
                f = affiner(gl, x, y, r)
                if f is not None:
                    gouttes.append(f)
                    sources.append("hough")
                    deja = np.vstack([deja, [f["cx"], f["cy"], f["r_milieu"]]])
    # Contrôle qualité : vrai anneau, cercle bien ajusté, pas de doublon.
    # Les candidats Hough (lettres gravées, parois, bulles) doivent en plus
    # ressembler à la population trouvée par les régions encloses
    ref = [f["contraste"] for f, src_ in zip(gouttes, sources) if src_ == "region"]
    c_ref = float(np.median(ref)) if ref else 0.0
    garde = []
    for i, f in enumerate(gouttes):
        r = f["r_milieu"]
        if not (2.5 <= r <= 0.5 * min(H, W)):
            continue
        if f["contraste"] < 0.5 * s_hp or f["frac_rayons"] < 0.6:
            continue
        if f["rms_milieu"] > max(0.12 * r, 0.8):   # gouttes tassées : facettes
            continue
        if f["contraste"] < 0.4 * c_ref:          # trou de lettre, défaut
            continue
        if sources[i] == "hough" and (f["contraste"] < 0.6 * c_ref or
                                      f["frac_rayons"] < 0.85 or
                                      f["rms_milieu"] > max(0.07 * r, 0.8)):
            continue
        if not (0.9 * r <= f["r_ext"] <= 1.5 * r + 2):
            f["r_ext"] = np.nan                   # bord extérieur non trouvé
        garde.append(i)
    gouttes = [gouttes[i] for i in garde]
    sources = [sources[i] for i in garde]
    ordre = np.argsort([-f["contraste"] for f in gouttes])
    final, src = [], []
    for i in ordre:
        f = gouttes[i]
        if any(math.hypot(f["cx"] - q["cx"], f["cy"] - q["cy"])
               < 0.5 * (f["r_milieu"] + q["r_milieu"]) for q in final):
            continue
        final.append(f)
        src.append(sources[i])
    cx = np.array([f["cx"] for f in final])
    cy = np.array([f["cy"] for f in final])
    rm = np.array([f["r_milieu"] for f in final])
    rext = np.array([f["r_ext"] for f in final])
    rb = np.where(np.isfinite(rext), rext, rm)
    bord = (cx - rb < 1) | (cy - rb < 1) | (cx + rb > W - 2) | \
        (cy + rb > H - 2) if len(final) else np.zeros(0, bool)
    return {"cx": cx, "cy": cy, "d_milieu": 2 * rm, "d_ext": 2 * rext,
            "rms": np.array([f["rms_milieu"] for f in final]),
            "aspect": np.array([f["aspect"] for f in final]),
            "circularite": np.array([f["circularite"] for f in final]),
            "contraste": np.array([f["contraste"] for f in final]),
            "bord": np.asarray(bord, bool), "source": np.array(src),
            "n_regions": int(len(cands)), "etalement_hp": s_hp,
            "anneaux": sombre}


# =============================================================================
# STATISTIQUES, VOLUME
# =============================================================================
def volume_goutte(d, h=None):
    """Droplet volume (same length unit cubed). Sphere if h is None or
    D <= h; else squeezed disc of height h with a half-torus rim."""
    d = np.asarray(d, float)
    if h is None:
        return np.pi * d ** 3 / 6
    a = np.clip((d - h) / 2, 0, None)
    v_disque = np.pi * a ** 2 * h + np.pi ** 2 * h ** 2 * a / 4 + np.pi * h ** 3 / 6
    return np.where(d <= h, np.pi * d ** 3 / 6, v_disque)


def statistiques(res, um_par_px=1.0, convention="milieu", h_um=None):
    """Summary statistics on the droplets fully inside the image."""
    ok = ~res["bord"]
    d = res["d_milieu" if convention == "milieu" else "d_ext"][ok] * um_par_px
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return {"n": 0}
    other = res["d_ext" if convention == "milieu" else "d_milieu"][ok] * um_par_px
    v = volume_goutte(d, h_um)
    return {"n": int(len(d)), "n_bord": int((~ok).sum()),
            "d_moy": float(d.mean()), "d_std": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
            "cv_pct": float(100 * d.std(ddof=1) / d.mean()) if len(d) > 1 else 0.0,
            "d10": float(np.percentile(d, 10)), "d50": float(np.median(d)),
            "d90": float(np.percentile(d, 90)),
            "ecart_convention": float(np.nanmedian(other - d))
            if np.isfinite(other).any() else np.nan,
            "v_moy": float(v.mean()), "convention": convention,
            "h_um": h_um, "unite": "um" if um_par_px != 1.0 else "px"}


# =============================================================================
# RECONNAISSANCE DU TYPE D'IMAGE
# =============================================================================
def ressemble_a_des_gouttes(image, res=None):
    """Does the image look like droplets seen as dark rings, their inside
    as bright as the outside (the case the usual PSA pipelines cannot
    handle)? Returns (bool, message). Meant for the PSA failure diagnosis:
    suggest the droplet mode."""
    res = res or detecter_gouttes(image)
    g = _gris(image)
    n = len(res["cx"])
    if n < 5:
        return False, f"only {n} ring-shaped droplets found"
    couv = float((np.pi * (res["d_milieu"] / 2) ** 2).sum() / g.size)
    d = res["d_milieu"]
    cv = float(d.std() / d.mean())
    oui = couv > 0.05 and cv < 0.35
    return oui, (f"{n} droplets seen as dark rings, {100 * couv:.0f} % of "
                 f"the image, size CV {100 * cv:.0f} %")


# =============================================================================
# IMAGES SYNTHÉTIQUES DE VALIDATION
# =============================================================================
def gouttes_synthetiques(H=480, W=640, d_moy=40.0, cv=0.03, tassees=True,
                         epaisseur=3.0, coeur_clair=False, seed=0):
    """Bright-field droplets with KNOWN diameters (ring middle): dark ring
    of given thickness, inside as bright as the continuous phase (optionally
    a bright core, as for spherical droplets acting as lenses). Packed
    (hexagonal, jittered) or sparse. Returns (image BGR uint8, truth array
    of (cx, cy, d) for droplets fully inside the image)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img = np.full((H, W), 170.0, np.float32)
    img += 10 * (xx / W - 0.5)                         # léger gradient
    gouttes = []
    if tassees:
        pas = d_moy * 1.02
        for j, y in enumerate(np.arange(pas / 2, H + pas, pas * math.sqrt(3) / 2)):
            for x in np.arange((j % 2) * pas / 2, W + pas, pas):
                d = d_moy * (1 + cv * rng.normal())
                gouttes.append((x + rng.normal(0, 0.03 * pas),
                                y + rng.normal(0, 0.03 * pas), d))
    else:
        essais = 0
        while len(gouttes) < int(0.25 * H * W / d_moy ** 2) and essais < 5000:
            essais += 1
            d = d_moy * (1 + cv * rng.normal())
            x, y = rng.uniform(0, W), rng.uniform(0, H)
            if all(math.hypot(x - a, y - b) > 0.5 * (d + c) + 4 for a, b, c in gouttes):
                gouttes.append((x, y, d))
    for x, y, d in gouttes:
        r = np.hypot(xx - x, yy - y)
        anneau = np.exp(-0.5 * ((r - d / 2) / (epaisseur / 2.355)) ** 2)
        img -= 110 * anneau
        if coeur_clair:
            img += 40 * np.clip(1 - r / (0.3 * d), 0, 1)
    img = cv2.GaussianBlur(img, (0, 0), 0.7) + rng.normal(0, 4, (H, W))
    img = np.clip(img, 0, 255).astype(np.uint8)
    v = np.array([(x, y, d) for x, y, d in gouttes
                  if x - d / 2 > 2 and y - d / 2 > 2 and x + d / 2 < W - 3
                  and y + d / 2 < H - 3], float).reshape(-1, 3)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), v
