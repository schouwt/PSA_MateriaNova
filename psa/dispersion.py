# -*- coding: utf-8 -*-
"""PSA, dispersion of a filler in a matrix (polymer, resin) from a SECTION.

The question is not only "how big are the particles" but "how are they
spread in the matrix". Two levels:

  MASK level (filler / matrix, no instance needed)
    - area fraction (Delesse: unbiased estimate of the volume fraction on a
      random plane section);
    - local area fraction in windows of growing size: its coefficient of
      variation drops fast for a good dispersion, slowly with agglomerates;
    - spherical contact distribution H(r): distance from a matrix point to
      the nearest filler ("free space", same idea as the free-space length
      of Khare & Burris, Polymer 2010, but a standard estimator of
      stochastic geometry, border-corrected by reduced sample);
    - agglomerates = connected components of the filler.

  PARTICLE level (instances, e.g. StarDist)
    - nearest-neighbour distances, Clark-Evans ratio (Donnelly correction);
    - pair correlation g(r) and Besag L(r) - r (reduced-sample estimator);
    - Voronoi cell areas (coefficient of variation);
    - particles per agglomerate.

Every indicator is compared with a NULL MODEL computed by Monte Carlo: the
SAME particles (shapes, sizes) placed at random in the same field without
overlap. Hard-core particles are never Poisson points, so the textbook
values (Clark-Evans R = 1, g = 1) are not the right reference. Global test:
studentized maximum absolute deviation of L(r) (Diggle 2003; Myllymaki
et al. 2017),
p = (1 + #sims >= obs) / (n + 1).

Stereology: Saltykov unfolding of the section diameters into sphere
diameters (assumes spheres and a random plane section: microtome or ion
polishing, NOT a fracture surface, which follows weak interfaces).
"""
import math

import numpy as np
import cv2
from scipy import ndimage as ndi
from scipy.spatial import cKDTree, Voronoi


# =============================================================================
# OUTILS
# =============================================================================
def _reetiqueter(lab):
    lab = np.asarray(lab)
    vals = np.unique(lab)
    vals = vals[vals != 0]
    lut = np.zeros(int(lab.max()) + 1, np.int32)
    lut[vals.astype(np.int64)] = np.arange(1, len(vals) + 1, dtype=np.int32)
    return lut[lab.astype(np.int64)]


def _proprietes(lab):
    """Area (px), centroid (y, x), ECD (px) per label 1..N."""
    n = int(lab.max())
    if n == 0:
        z = np.zeros(0)
        return z, np.zeros((0, 2)), z
    idx = np.arange(1, n + 1)
    un = np.ones(lab.shape, np.uint8)
    aire = ndi.sum(un, lab, idx).astype(float)
    cyx = np.array(ndi.center_of_mass(un, lab, idx), float).reshape(-1, 2)
    return aire, cyx, 2.0 * np.sqrt(aire / np.pi)


def _distance_bord(roi):
    """Distance (px) of each pixel to the outside of the ROI."""
    pad = np.pad(roi, 1, constant_values=False)
    return ndi.distance_transform_edt(pad)[1:-1, 1:-1]


# =============================================================================
# NIVEAU MASQUE
# =============================================================================
def fraction_locale_cv(masque, roi, tailles_px):
    """Coefficient of variation of the local area fraction in square windows
    (non-overlapping tiling, windows fully inside the ROI)."""
    m = (masque & roi).astype(np.float64)
    r = roi.astype(np.float64)
    Im = cv2.integral(m)
    Ir = cv2.integral(r)
    H, W = masque.shape
    cvs = []
    for s in tailles_px:
        s = int(s)
        if s < 2 or s > min(H, W):
            cvs.append(np.nan)
            continue
        ny, nx = H // s, W // s
        y0, x0 = (H - ny * s) // 2, (W - nx * s) // 2
        ys = y0 + np.arange(ny + 1) * s
        xs = x0 + np.arange(nx + 1) * s

        def boite(I):
            return (I[ys[1:]][:, xs[1:]] - I[ys[:-1]][:, xs[1:]]
                    - I[ys[1:]][:, xs[:-1]] + I[ys[:-1]][:, xs[:-1]])
        a_m, a_r = boite(Im), boite(Ir)
        ok = a_r >= 0.99 * s * s
        if ok.sum() < 4:
            cvs.append(np.nan)
            continue
        f = a_m[ok] / a_r[ok]
        cvs.append(float(f.std(ddof=1) / f.mean()) if f.mean() > 0 else np.nan)
    return np.array(cvs, float)


def contact_spherique(masque, roi, r_px, d_bord=None):
    """Spherical contact distribution H(r) = P(distance to filler <= r | point
    in the matrix), reduced-sample estimator (only matrix points at least r
    from the ROI border are used for each r)."""
    if d_bord is None:
        d_bord = _distance_bord(roi)
    matrice = roi & ~masque
    if not masque.any() or not matrice.any():
        return np.full(len(r_px), np.nan)
    d = ndi.distance_transform_edt(~masque)[matrice]
    b = d_bord[matrice]
    H = []
    for r in r_px:
        sel = b >= r
        H.append(float((d[sel] <= r).mean()) if sel.sum() > 50 else np.nan)
    return np.array(H, float)


def agglomerats(labels, masque):
    """Connected components of the filler (8-connectivity) and the number
    of particles in each."""
    comp, nc = ndi.label(masque, structure=np.ones((3, 3), int))
    n = int(labels.max())
    if n == 0 or nc == 0:
        return comp, np.zeros(nc, int), np.zeros(n, int)
    # composante de chaque particule = composante de son pixel majoritaire
    m = labels > 0
    pairs = np.stack([labels[m], comp[m]], 1)
    u, cnt = np.unique(pairs, axis=0, return_counts=True)
    ordre = np.lexsort((-cnt, u[:, 0]))
    u = u[ordre]
    prem = np.r_[True, u[1:, 0] != u[:-1, 0]]
    comp_de = np.zeros(n + 1, int)
    comp_de[u[prem, 0]] = u[prem, 1]
    n_par_comp = np.bincount(comp_de[1:], minlength=nc + 1)[1:]
    return comp, n_par_comp, comp_de[1:]


# =============================================================================
# NIVEAU PARTICULES (semis de points = centroïdes)
# =============================================================================
def clark_evans(pts, aire, perimetre):
    """Mean nearest-neighbour distance and Clark-Evans ratio with the
    Donnelly (1978) edge correction of the expected value."""
    n = len(pts)
    if n < 3:
        return np.nan, np.nan, np.array([])
    d, _ = cKDTree(pts).query(pts, k=2)
    nnd = d[:, 1]
    attendu = 0.5 * math.sqrt(aire / n) + (0.0514 + 0.041 / math.sqrt(n)) * \
        perimetre / n
    return float(nnd.mean()), float(nnd.mean() / attendu), nnd


def correlation_paires(pts, d_bord_pts, aire, r_bornes):
    """g(r) per annulus and L(r) - r, reduced-sample estimators: a point is
    used as centre for radius r only if it lies at least r from the border."""
    n = len(pts)
    nb = len(r_bornes) - 1
    if n < 3:
        return np.full(nb, np.nan), np.full(nb, np.nan)
    lam = n / aire
    paires = cKDTree(pts).query_pairs(r_bornes[-1], output_type="ndarray")
    if len(paires):
        dist = np.hypot(*(pts[paires[:, 0]] - pts[paires[:, 1]]).T)
        k = np.searchsorted(r_bornes, dist, side="right") - 1
    else:
        dist = k = np.zeros(0, int)
    g = np.full(nb, np.nan)
    L = np.full(nb, np.nan)
    for b in range(nb):
        r_in, r_out = r_bornes[b], r_bornes[b + 1]
        ok = d_bord_pts >= r_out
        n_c = int(ok.sum())
        if n_c < 10:
            continue
        sel = k == b
        c = ok[paires[sel, 0]].sum() + ok[paires[sel, 1]].sum() \
            if len(paires) else 0
        g[b] = c / (n_c * lam * math.pi * (r_out ** 2 - r_in ** 2))
        selk = dist <= r_out
        ck = ok[paires[selk, 0]].sum() + ok[paires[selk, 1]].sum() \
            if len(paires) else 0
        K = ck / (n_c * lam)
        L[b] = math.sqrt(K / math.pi) - r_out
    return g, L


def voronoi_cv(pts, roi):
    """Coefficient of variation of the Voronoi cell areas, cells fully
    inside the ROI only."""
    if len(pts) < 10:
        return np.nan
    H, W = roi.shape
    vor = Voronoi(pts[:, ::-1])               # (x, y)
    aires = []
    for ir in vor.point_region:
        reg = vor.regions[ir]
        if not reg or -1 in reg:
            continue
        v = vor.vertices[reg]
        if (v[:, 0] < 0).any() or (v[:, 1] < 0).any() or \
                (v[:, 0] >= W).any() or (v[:, 1] >= H).any():
            continue
        if not roi[v[:, 1].astype(int), v[:, 0].astype(int)].all():
            continue
        x, y = v[:, 0], v[:, 1]
        aires.append(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
    if len(aires) < 10:
        return np.nan
    a = np.array(aires)
    return float(a.std(ddof=1) / a.mean())


# =============================================================================
# MODÈLE NUL : MÊMES PARTICULES PLACÉES AU HASARD, SANS RECOUVREMENT
# =============================================================================
def placer_au_hasard(labels, roi, rng, essais=300):
    """Random sequential placement of the SAME particle shapes, in random
    order, without overlap: the CENTROID of each shape is uniform in the
    ROI and the parts falling outside are clipped, as for real particles
    cut by the edge of the field (keeps a uniform density up to the
    border). Returns (labels, n_failed)."""
    H, W = labels.shape
    formes = []
    for i, sl in enumerate(ndi.find_objects(labels), 1):
        if sl is None:
            continue
        f = labels[sl] == i
        fy, fx = ndi.center_of_mass(f)
        formes.append((f, fy, fx))
    # ordre ALÉATOIRE (et non « plus grosses d'abord ») : placer les grosses
    # d'abord les espace artificiellement et rend le modèle nul trop régulier
    formes = [formes[i] for i in rng.permutation(len(formes))]
    pix = np.flatnonzero(roi)
    occupe = np.zeros((H, W), bool)
    sortie = np.zeros((H, W), np.int32)
    echecs = 0
    for k, (f, fy, fx) in enumerate(formes, 1):
        h, w = f.shape
        place = False
        tir = pix[rng.integers(0, len(pix), essais)]
        for p in tir:
            cy, cx = divmod(int(p), W)
            y0 = int(round(cy + rng.random() - 0.5 - fy))
            x0 = int(round(cx + rng.random() - 0.5 - fx))
            ya, yb = max(0, y0), min(H, y0 + h)
            xa, xb = max(0, x0), min(W, x0 + w)
            if ya >= yb or xa >= xb:
                continue
            fc = f[ya - y0:yb - y0, xa - x0:xb - x0] & roi[ya:yb, xa:xb]
            if not fc.any() or (occupe[ya:yb, xa:xb] & fc).any():
                continue
            occupe[ya:yb, xa:xb] |= fc
            sortie[ya:yb, xa:xb][fc] = k
            place = True
            break
        if not place:
            echecs += 1
    # étiquettes consécutives : une particule non placée laisserait un trou
    return _reetiqueter(sortie), echecs


# =============================================================================
# STÉRÉOLOGIE : DÉPLIAGE DE SALTYKOV
# =============================================================================
def saltykov(d_sections, n_classes=12, d_max=None):
    """Scheil-Schwartz-Saltykov unfolding of section diameters (spheres,
    random plane section). Returns (edges, n_v): relative NUMBER of spheres
    per class (negative values from noise are set to 0, sum = 1). The
    sphere diameter of class j is its UPPER edge (classic convention): the
    mean sphere diameter is sum(n_v * edges[1:]), within a few % on
    log-normal spheres (sigma <= 0.4) with 15-20 classes."""
    d = np.asarray(d_sections, float)
    d = d[d > 0]
    if len(d) < 20:
        return None, None
    d_max = float(d_max or d.max() * 1.0001)
    bords = np.linspace(0, d_max, n_classes + 1)
    n_a, _ = np.histogram(d, bords)
    # P[i, j] : probabilité qu'une sphère de la classe j (diamètre = borne
    # haute) donne une section dans la classe i ; nombre de sections par
    # unité de surface = N_V * D_j * P[i, j]
    D = bords[1:]
    P = np.zeros((n_classes, n_classes))
    for j in range(n_classes):
        for i in range(j + 1):
            a, b = bords[i], bords[i + 1]
            P[i, j] = (math.sqrt(max(D[j] ** 2 - a ** 2, 0))
                       - math.sqrt(max(D[j] ** 2 - b ** 2, 0))) / D[j]
    n_v = np.zeros(n_classes)
    for j in range(n_classes - 1, -1, -1):        # de la plus grosse classe
        reste = n_a[j] - sum(n_v[k] * D[k] * P[j, k]
                             for k in range(j + 1, n_classes))
        n_v[j] = max(reste, 0.0) / (D[j] * P[j, j])
    s = n_v.sum()
    return bords, (n_v / s if s > 0 else n_v)


# =============================================================================
# ANALYSE COMPLÈTE
# =============================================================================
def _indicateurs(labels, roi, tailles_q, r_c, r_g, d_bord):
    masque = (labels > 0) & roi
    A = float(roi.sum())
    aire, cyx, ecd = _proprietes(labels)
    yi = np.clip(np.round(cyx[:, 0]).astype(int), 0, roi.shape[0] - 1)
    xi = np.clip(np.round(cyx[:, 1]).astype(int), 0, roi.shape[1] - 1)
    dans = roi[yi, xi] if len(cyx) else np.zeros(0, bool)
    pts = cyx[dans]
    per = float(np.count_nonzero(roi ^ ndi.binary_erosion(roi)))
    nnd_moy, R, nnd = clark_evans(pts, A, per)
    g, L = correlation_paires(pts, d_bord[yi[dans], xi[dans]], A, r_g)
    return {"phi": float(masque.sum() / A),
            "cv_local": fraction_locale_cv(masque, roi, tailles_q),
            "H": contact_spherique(masque, roi, r_c, d_bord),
            "nnd_moy": nnd_moy, "clark_evans": R, "nnd": nnd,
            "g": g, "L": L, "voronoi_cv": voronoi_cv(pts, roi),
            "n": int(len(pts)), "ecd": ecd}


def analyser_dispersion(labels, nm_par_px=1.0, roi=None, n_mc=39, seed=0,
                        progress=None):
    """Full dispersion analysis of a label image of filler particles (or
    of agglomerates if only a binary mask was segmented: then pass the
    connected components). Distances returned in nm."""
    labels = _reetiqueter(labels)
    H, W = labels.shape
    roi = np.ones((H, W), bool) if roi is None else np.asarray(roi, bool)
    labels = np.where(roi, labels, 0)
    labels = _reetiqueter(labels)
    aire, cyx, ecd = _proprietes(labels)
    if len(ecd) < 10:
        raise ValueError("Too few particles for a dispersion analysis (< 10)")
    d50 = float(np.median(ecd))
    # échelles d'analyse, en px
    cote = min(H, W)
    tailles_q = np.unique(np.geomspace(max(4 * d50, 8), cote / 3, 10).astype(int))
    pts = cyx
    nnd_med = float(np.median(cKDTree(pts).query(pts, k=2)[0][:, 1]))
    r_max = min(0.25 * cote, max(10 * nnd_med, 4 * d50))
    r_g = np.linspace(0, r_max, 41)
    d_bord = _distance_bord(roi)
    dt = ndi.distance_transform_edt(~((labels > 0) & roi))[roi & (labels == 0)]
    r_c = np.linspace(0, max(float(np.percentile(dt, 99)), 2.0), 40) \
        if dt.size else np.linspace(0, 10, 40)

    obs = _indicateurs(labels, roi, tailles_q, r_c, r_g, d_bord)
    comp, n_par_comp, comp_de = agglomerats(labels, (labels > 0) & roi)
    rng = np.random.default_rng(seed)
    sims, echecs = [], []
    for s in range(n_mc):
        if progress:
            progress(int(100 * s / max(n_mc, 1)), f"Null model {s + 1}/{n_mc}")
        lab_s, e = placer_au_hasard(labels, roi, rng)
        echecs.append(e)
        sims.append(_indicateurs(lab_s, roi, tailles_q, r_c, r_g, d_bord))

    def env(cle):
        a = np.array([s[cle] for s in sims], float)
        return np.nanmin(a, 0), np.nanmedian(a, 0), np.nanmax(a, 0), a

    res = {"nm_par_px": nm_par_px, "n_particules": obs["n"],
           "phi": obs["phi"], "d50_nm": d50 * nm_par_px,
           "ecd_nm": ecd * nm_par_px, "n_mc": n_mc,
           "nul_echecs_max": int(max(echecs) if echecs else 0),
           "tailles_q_nm": tailles_q * nm_par_px,
           "r_c_nm": r_c * nm_par_px,
           "r_g_nm": 0.5 * (r_g[1:] + r_g[:-1]) * nm_par_px,
           "observe": obs, "labels": labels, "roi": roi,
           "agglomerats": {"n": int(len(n_par_comp)),
                           "particules_par_agglomerat": n_par_comp,
                           "frac_part_agglomerees":
                               float((n_par_comp[comp_de - 1] >= 2).mean())
                               if len(comp_de) else np.nan,
                           "ecd_nm": (2 * np.sqrt(np.bincount(comp.ravel())[1:]
                                                  / np.pi) * nm_par_px)}}
    if n_mc:
        for cle in ("cv_local", "H", "g", "L"):
            lo, med, hi, a = env(cle)
            res[f"env_{cle}"] = (lo, med, hi)
        for cle in ("nnd_moy", "clark_evans", "voronoi_cv"):
            a = np.array([s[cle] for s in sims], float)
            res[f"nul_{cle}"] = (float(np.nanpercentile(a, 2.5)),
                                 float(np.nanmedian(a)),
                                 float(np.nanpercentile(a, 97.5)))
        # Test global MAD STUDENTISÉ sur L(r) - r (Myllymäki et al. 2017) :
        # l'écart à la médiane des simulations est divisé par leur écart-type
        # à chaque r, sinon les grands r (les plus variables) dominent le test
        _, med, _, a = env("L")
        sd = np.nanstd(a, 0, ddof=1)
        sd = np.where(sd > 0, sd, np.nan)
        dev = lambda v: float(np.nanmax(np.abs(v - med) / sd))
        u_obs = dev(obs["L"])
        u_sim = np.array([dev(v) for v in a])
        res["p_mad_L"] = float((1 + (u_sim >= u_obs).sum()) / (n_mc + 1))
        # sens de l'écart à l'échelle où il est maximal
        k = int(np.nanargmax(np.abs(obs["L"] - med) / sd))
        res["r_ecart_max_nm"] = float(res["r_g_nm"][k])
        signe = obs["L"][k] - med[k]
        if res["p_mad_L"] > 0.05:            # rejet si p <= alpha
            verdict = "random"
        else:
            verdict = "clustered" if signe > 0 else "regular"
        res["verdict"] = verdict
        # échelles où g(r) sort de l'enveloppe
        lo, _, hi = res["env_g"]
        res["r_g_au_dessus_nm"] = res["r_g_nm"][obs["g"] > hi].tolist()
        res["r_g_en_dessous_nm"] = res["r_g_nm"][obs["g"] < lo].tolist()
        # distance libre médiane (H = 0.5), observée et nulle
        res["libre_median_nm"] = _r_demi(r_c, obs["H"]) * nm_par_px
        res["libre_median_nul_nm"] = _r_demi(r_c, res["env_H"][1]) * nm_par_px
    bords, n_v = saltykov(ecd * nm_par_px, n_classes=15)
    res["saltykov"] = (bords, n_v)
    res["d_sphere_moy_nm"] = float((n_v * bords[1:]).sum()) \
        if n_v is not None else np.nan
    return res


def _r_demi(r, H):
    ok = np.isfinite(H)
    if ok.sum() < 2 or np.nanmax(H) < 0.5:
        return np.nan
    return float(np.interp(0.5, H[ok], r[ok]))


def resume_texte(res):
    """Short English summary of a dispersion analysis."""
    o = res["observe"]
    l = [f"Particles: {res['n_particules']}   area fraction: "
         f"{100 * res['phi']:.1f} %   median ECD (section): "
         f"{res['d50_nm']:.1f} nm"]
    if np.isfinite(res.get("d_sphere_moy_nm", np.nan)):
        l.append(f"Mean sphere diameter (Saltykov unfolding, spheres and "
                 f"random plane section assumed): "
                 f"{res['d_sphere_moy_nm']:.1f} nm (mean section ECD "
                 f"{np.mean(res['ecd_nm']):.1f} nm)")
    ag = res["agglomerats"]
    l.append(f"Agglomerates (connected filler): {ag['n']}, particles in an "
             f"agglomerate of >= 2: {100 * ag['frac_part_agglomerees']:.0f} %")
    if res.get("n_mc"):
        lo, med, hi = res["nul_clark_evans"]
        l.append(f"Clark-Evans R: {o['clark_evans']:.3f}  (random same "
                 f"particles: {med:.3f}, 95 % {lo:.3f}-{hi:.3f})")
        lo, med, hi = res["nul_voronoi_cv"]
        l.append(f"Voronoi cell CV: {o['voronoi_cv']:.3f}  (random: "
                 f"{med:.3f}, 95 % {lo:.3f}-{hi:.3f})")
        l.append(f"Median free distance to filler: "
                 f"{res['libre_median_nm']:.1f} nm  (random: "
                 f"{res['libre_median_nul_nm']:.1f} nm)")
        l.append(f"Global test on L(r): p = {res['p_mad_L']:.3f} "
                 f"({res['n_mc']} simulations), largest deviation at "
                 f"r = {res['r_ecart_max_nm']:.0f} nm  ->  "
                 f"{res['verdict'].upper()}")
        if res["nul_echecs_max"]:
            l.append(f"WARNING: up to {res['nul_echecs_max']} particles "
                     f"could not be placed in the null model (field too "
                     f"crowded): reference slightly biased")
    return "\n".join(l)


# =============================================================================
# HOMOGÉNÉITÉ SANS SEUIL (signal BSE de composition)
# =============================================================================
def homogeneite_intensite(gray, roi=None, tailles_px=None):
    """Threshold-free homogeneity of a composition-contrast (BSE) image:
    variance of the mean grey level in square windows of side L, divided by
    the pixel variance (invariant to brightness / contrast settings).

    For a material homogeneous beyond a correlation length xi, the ratio
    falls as L^-2 once L >> xi (slope -2 in log-log). A slower fall, or a
    plateau, reveals composition variations at that scale (matrix-rich
    zones, agglomerates, flow lines). Returns (sizes_px, ratio, slope over
    the upper half of the sizes)."""
    g = np.asarray(gray, np.float64)
    roi = np.ones(g.shape, bool) if roi is None else np.asarray(roi, bool)
    H, W = g.shape
    if tailles_px is None:
        tailles_px = np.unique(np.geomspace(4, min(H, W) / 4, 12).astype(int))
    v_pix = float(g[roi].var())
    I = cv2.integral(np.where(roi, g, 0.0))
    Ir = cv2.integral(roi.astype(np.float64))
    ratios = []
    for s in tailles_px:
        ny, nx = H // s, W // s
        y0, x0 = (H - ny * s) // 2, (W - nx * s) // 2
        ys = y0 + np.arange(ny + 1) * s
        xs = x0 + np.arange(nx + 1) * s

        def boite(J):
            return (J[ys[1:]][:, xs[1:]] - J[ys[:-1]][:, xs[1:]]
                    - J[ys[1:]][:, xs[:-1]] + J[ys[:-1]][:, xs[:-1]])
        a, n = boite(I), boite(Ir)
        ok = n >= 0.99 * s * s
        ratios.append(float((a[ok] / n[ok]).var(ddof=1) / v_pix)
                      if ok.sum() >= 4 else np.nan)
    ratios = np.array(ratios)
    t = np.asarray(tailles_px, float)
    sel = (t >= np.median(t)) & np.isfinite(ratios) & (ratios > 0)
    pente = float(np.polyfit(np.log(t[sel]), np.log(ratios[sel]), 1)[0]) \
        if sel.sum() >= 3 else np.nan
    return t, ratios, pente
