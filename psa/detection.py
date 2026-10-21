# -*- coding: utf-8 -*-
"""PSA, Image processing and particle / fibre detectors."""
from psa.nanocarbon import *  # noqa: F401,F403

# =============================================================================
# TRAITEMENT IMAGE  (noyau v0.5, épuré et testable sans GUI)
# =============================================================================
_LUT_CACHE: dict = {}


def rolling_ball(image_bgr, params):
    if not params.get("rolling_ball_enabled", False):
        return image_bgr.copy()
    r = max(5, int(params.get("rolling_ball_radius", 50)))
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    noy  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    fond = cv2.morphologyEx(l_ch, cv2.MORPH_CLOSE, noy)
    diff = cv2.subtract(fond, l_ch)
    if diff.max() == 0:
        return image_bgr.copy()
    lc = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(cv2.merge([np.uint8(255 - lc), a_ch, b_ch]),
                        cv2.COLOR_LAB2BGR)


def ameliorer(image_bgr, params):
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=params["clahe_clip"],
                            tileGridSize=(params["clahe_grid"],
                                          params["clahe_grid"]))
    l_ch = clahe.apply(l_ch)
    g = params["gamma"]
    if g not in _LUT_CACHE:
        _LUT_CACHE[g] = np.array(
            [min(255, int((i / 255.0) ** g * 255)) for i in range(256)],
            dtype=np.uint8)
    l_ch = cv2.LUT(l_ch, _LUT_CACHE[g])
    return cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


def seuiller(flou, params):
    """
    Otsu avec décalage relatif (otsu_offset, piloté par le curseur
    Sensibilité) ou seuil manuel. Masque blanc = particules dans les
    deux conventions de sens.
    """
    inv  = params.get("inverser_couleurs", False)
    base = cv2.THRESH_BINARY if inv else cv2.THRESH_BINARY_INV
    if params.get("threshold_mode", "otsu") == "manual":
        _, b = cv2.threshold(flou, int(params.get("threshold_manual", 128)),
                             255, base)
        return b
    t_otsu, _ = cv2.threshold(flou, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    off = float(params.get("otsu_offset", 0.0))
    t = int(np.clip(t_otsu * (1.0 + off), 1, 254))
    _, b = cv2.threshold(flou, t, 255, base)
    return b


def appliquer_zones(masque, zones):
    if not zones:
        return masque
    m = masque.copy()
    cv2.fillPoly(m, zones, 0)
    return m


def corde_max_v2(cnt):
    """Diamètre de Feret exact — O(n²) vectorisé sur l'enveloppe convexe."""
    hull = cv2.convexHull(cnt).reshape(-1, 2).astype(float)
    n = len(hull)
    if n < 2:
        return None, None, 0.0
    if n > 200:
        hull = hull[np.linspace(0, n - 1, 200, dtype=int)]
    dif = hull[:, None, :] - hull[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", dif, dif)
    i, j = np.unravel_index(int(np.argmax(d2)), d2.shape)
    return hull[i].astype(int), hull[j].astype(int), float(math.sqrt(d2[i, j]))


def evaluer_ws(masque, dist_norm, seuil_ws):
    """Score 0-100 de fiabilité de la segmentation (cf. v0.5)."""
    total = masque.size
    score, raisons = 0, []
    pct = (masque > 0).sum() / total
    if 0.03 <= pct <= 0.60:   score += 30
    elif pct < 0.01:          raisons.append("empty mask")
    elif pct > 0.80:          raisons.append("saturated mask")
    else:                     score += 15; raisons.append("partial detection")
    std_d = float(np.std(dist_norm[masque > 0]) if masque.any() else 0)
    if std_d >= 0.20:         score += 35
    elif std_d >= 0.12:       score += 20; raisons.append("weak maxima")
    else:                     raisons.append("flat map")
    pa = (dist_norm[masque > 0] >= seuil_ws).mean() if masque.any() else 0
    if 0.05 <= pa <= 0.80:    score += 25
    elif pa < 0.02:           raisons.append("separation too strong")
    else:                     score += 10; raisons.append("separation too weak")
    _, _, stats, _ = cv2.connectedComponentsWithStats(masque, connectivity=8)
    if len(stats) > 1 and float(np.median(stats[1:, cv2.CC_STAT_AREA])) >= 50:
        score += 10
    elif len(stats) > 1:
        raisons.append("small components")
    if score >= 75:
        st, msg = "good", ("Excellent separation" if not raisons
                           else f"Good, {raisons[0]}")
    elif score >= 45:
        st, msg = "warn", (" | ".join(raisons) or "Acceptable quality")
    else:
        st, msg = "bad",  (" | ".join(raisons[:2]) or "Unreliable segmentation")
    return score, msg, st


def evaluer_repr(d):
    n = len(d)
    if n == 0:
        return {"n": 0, "verdict": "bad", "message": "No particle.",
                "ic95": 0, "precision_rel": 0, "cv": 0, "cv_msg": "-",
                "n_min_10pct": 999, "mu": 0, "sigma": 0}
    mu = float(np.mean(d)); sigma = float(np.std(d, ddof=1)) if n > 1 else 0.0
    cv = sigma / mu * 100 if mu > 0 else 0.0
    tv = float(_sps.t.ppf(0.975, n - 1)) if n > 1 else float("inf")
    ic = tv * sigma / math.sqrt(n) if n > 1 else float("inf")
    pr = ic / mu * 100 if mu > 0 else float("inf")
    nm = math.ceil((tv * sigma / (0.10 * mu)) ** 2) if mu > 0 and sigma > 0 else n
    if n >= 30 and pr <= 10: v, m = "good", f"Excellent (n={n}) CI95 = ±{ic:.1f} (±{pr:.1f}%)"
    elif n >= 30:            v, m = "good", f"Representative (n={n}) CI95 = ±{ic:.1f}"
    elif n >= 15:            v, m = "warn", f"Acceptable (n={n}), aim for n ≥ {max(30, nm)}"
    else:                    v, m = "bad",  f"Not representative (n={n}<30), aim for n ≥ {max(30, nm)}"
    cvm = ("Homogeneous" if cv <= 20 else
           ("Moderately dispersed" if cv <= 50 else "Very heterogeneous"))
    return {"n": n, "mu": mu, "sigma": sigma, "cv": cv, "cv_msg": cvm,
            "ic95": ic, "precision_rel": pr, "n_min_10pct": nm,
            "verdict": v, "message": m}


def dx_volume(diams):
    """Dx10/50/90 volumiques (ISO 9276-2, pondération d³, hypothèse sphères)."""
    if len(diams) == 0:
        return 0., 0., 0.
    d_s = np.sort(np.asarray(diams, float))
    v = np.cumsum(d_s ** 3)
    v /= v[-1]

    def dx(p):
        i = np.searchsorted(v, p)
        if i == 0:        return float(d_s[0])
        if i >= len(d_s): return float(d_s[-1])
        t = (p - v[i - 1]) / (v[i] - v[i - 1] + 1e-12)
        return float(d_s[i - 1] + t * (d_s[i] - d_s[i - 1]))

    return dx(0.10), dx(0.50), dx(0.90)


def comparer_bet(d32, unite, rho_gcm3, bet_m2g):
    """
    Validation croisée image ↔ BET (adsorption N₂), usage nanocarbones.

    Principe : l'azote accède aux surfaces entre primaires d'un agrégat,
    donc le BET sonde la taille des particules PRIMAIRES — une mesure
    indépendante de l'analyse d'image.
      • SSA géométrique = 6/(ρ·D[3,2])   (sphères denses et lisses)
      • d_BET           = 6/(ρ·SSA_BET)  (diamètre équivalent surface)
      • F = SSA_BET / SSA_géom :
          F ≈ 1     → tailles image confirmées par le BET
          F > 1.25  → microporosité/rugosité, OU sous-segmentation
                       (agrégats comptés comme particules → D[3,2] gonflé)
          F < 0.8   → cous de frittage (surface inaccessible au gaz),
                       OU sur-segmentation (D[3,2] image sous-estimé)
    Unités : ρ en g/cm³, d en µm ou nm selon `unite`, SSA en m²/g.
    """
    d32_um = d32 / 1000.0 if unite == "nm" else d32
    if d32_um <= 0 or rho_gcm3 <= 0 or bet_m2g <= 0:
        raise ValueError("d32, rho and BET must be > 0")
    ssa_geom = 6.0 / (rho_gcm3 * d32_um)
    d_bet_um = 6.0 / (rho_gcm3 * bet_m2g)
    d_bet = d_bet_um * 1000.0 if unite == "nm" else d_bet_um
    f = bet_m2g / ssa_geom
    if 0.8 <= f <= 1.25:
        verdict = "good"
        message = ("Consistent: the primary size measured on the image "
                   "is confirmed by BET (dense, smooth particles).")
    elif f > 1.25:
        verdict = "warn"
        message = (f"BET {f:.1f}× the geometric surface: probable microporosity or "
                   "surface roughness, or under-segmentation "
                   "(aggregates counted as particles, D[3,2] overestimated). "
                   "Check with the Separation slider or an STSA measurement.")
    else:
        verdict = "warn"
        message = (f"BET {1/f:.1f}× lower than the geometric surface: "
                   "sintering necks between primaries (surface inaccessible "
                   "to the gas), or over-segmentation (image D[3,2] underestimated).")
    return {"ssa_geom": ssa_geom, "d_bet": d_bet, "facteur": f,
            "verdict": verdict, "message": message}


def profil_epaisseur(gris, p1, p2, n=400, lissage=2.0, amp_min=1.5):
    """
    Mesure d'épaisseur de couche sur coupe MEB (coating PVD, coquille
    core-shell…) : extrait le profil de gris le long du segment p1→p2,
    puis détecte les DEUX interfaces comme les extrema de gradient de
    signes opposés les plus marqués (montée puis descente du niveau de
    gris en traversant la couche).

    Retourne (profil, f_a, f_b, epaisseur_px) où f_a/f_b sont les positions
    fractionnaires [0–1] des interfaces le long du segment, ou
    (profil, None, None, None) si les interfaces ne sont pas détectables
    (contraste insuffisant) — on retombe alors sur la longueur tracée.
    """
    from scipy.ndimage import map_coordinates, gaussian_filter1d
    x = np.linspace(p1[0], p2[0], n)
    y = np.linspace(p1[1], p2[1], n)
    prof = map_coordinates(gris.astype(np.float32), [y, x], order=1)
    lis = gaussian_filter1d(prof, lissage)
    g = np.gradient(lis)
    i_hi, i_lo = int(np.argmax(g)), int(np.argmin(g))
    if min(abs(g[i_hi]), abs(g[i_lo])) < amp_min or abs(i_hi - i_lo) < 3:
        return prof, None, None, None
    a, b = sorted((i_hi, i_lo))
    seg = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    return prof, a / (n - 1), b / (n - 1), (b - a) * seg / (n - 1)


def profils_epaisseur_auto(gris, p1, p2, n_profils=400, lissage=2.0,
                           amp_min=1.5, pas_min=2.0):
    """
    Mesure AUTOMATIQUE de l'épaisseur d'une couche le long d'une coupe.

    À partir d'un seul trait de référence p1→p2 tracé EN TRAVERS de la
    couche (qui fixe la direction de mesure et la fenêtre de recherche),
    balaie n_profils parallèles le long de la couche — perpendiculairement
    au trait — et mesure l'épaisseur de chacun par la même détection des
    deux interfaces que `profil_epaisseur`. Les profils sans interfaces
    nettes (hors couche, défaut) sont simplement ignorés.

    Retourne (eps_px, interfaces, taux) :
      • eps_px      : tableau des épaisseurs mesurées (pixels)
      • interfaces  : liste de paires ((x_haut,y_haut),(x_bas,y_bas)) pour
                      tracer l'enveloppe de la couche mesurée
      • taux        : fraction de profils exploitables (indice de fiabilité)
    """
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    u = p2 - p1
    L = float(np.hypot(*u))
    if L < 5:
        return np.array([]), [], 0.0
    u /= L
    v = np.array([-u[1], u[0]])                 # le long de la couche
    H, W = gris.shape
    c0 = (p1 + p2) / 2.0
    # plage de déplacement t (le long de v) gardant le centre dans l'image
    tmin, tmax = -1e9, 1e9
    for comp, ci, lim in ((v[0], c0[0], W), (v[1], c0[1], H)):
        if abs(comp) > 1e-9:
            ta, tb = (0 - ci) / comp, (lim - ci) / comp
            tmin = max(tmin, min(ta, tb))
            tmax = min(tmax, max(ta, tb))
    if not (np.isfinite(tmin) and np.isfinite(tmax)) or tmax - tmin < pas_min:
        return np.array([]), [], 0.0
    n = int(min(n_profils, max(10, (tmax - tmin) / pas_min)))
    eps, interfaces = [], []
    for t in np.linspace(tmin + 1.0, tmax - 1.0, n):
        c = c0 + t * v
        a = c - (L / 2.0) * u
        b = c + (L / 2.0) * u
        _, fa, fb, ep = profil_epaisseur(gris, a, b, lissage=lissage,
                                         amp_min=amp_min)
        if ep is not None:
            eps.append(ep)
            interfaces.append((tuple(a + fa * (b - a)),
                               tuple(a + fb * (b - a))))
    taux = len(eps) / n if n else 0.0
    return np.array(eps), interfaces, taux


def stats_epaisseur(eps_px, ppm):
    """Statistiques robustes d'épaisseur (en µm) à partir des épaisseurs en
    pixels. Rejette les valeurs aberrantes (interfaces parasites sur défaut,
    fissure, particule) au-delà de 3·MAD de la médiane, et renvoie aussi le
    nombre total et le nombre retenu."""
    e = np.asarray(eps_px, float)
    e = e[e > 0] / ppm
    if len(e) == 0:
        return None
    med0 = float(np.median(e))
    mad = float(np.median(np.abs(e - med0))) * 1.4826
    e_f = e[np.abs(e - med0) <= 3.0 * mad] if mad > 0 else e
    if len(e_f) == 0:
        e_f = e
    moy = float(np.mean(e_f))
    ec = float(np.std(e_f, ddof=1)) if len(e_f) > 1 else 0.0
    return {
        "valeurs": e_f, "n": int(len(e_f)), "n_total": int(len(e)),
        "moy": moy, "ec": ec,
        "cv": (ec / moy * 100.0) if moy > 0 else 0.0,
        "med": float(np.median(e_f)),
        "min": float(np.min(e_f)), "max": float(np.max(e_f)),
        "p10": float(np.percentile(e_f, 10)),
        "p90": float(np.percentile(e_f, 90)),
    }


def epaisseur_ruban_guide(gris, p1, p2, ppm, pct=90.0):
    """
    Épaisseur d'un coating COURBE (ruban brillant, ex. Zn/Zr sur grain) :
    segmente la bande brillante que le trait de référence p1→p2 TRAVERSE,
    en extrait le squelette (ligne centrale) et mesure l'épaisseur locale
    = 2 × distance au bord en chaque point du squelette.

    Le trait sert à désigner le BON ruban : on ne garde que la composante
    brillante qu'il croise, ce qui évite les zones brillantes parasites
    (intérieur de grain fracturé, etc.). Robuste à la courbure du coating.

    Retourne (eps_px, band, skel) — eps_px le tableau des épaisseurs (px),
    band le masque du ruban, skel le squelette nettoyé — ou (None, None,
    None) si le trait ne croise aucune bande brillante.
    """
    from skimage.morphology import skeletonize
    g = gris if gris.ndim == 2 else cv2.cvtColor(gris, cv2.COLOR_BGR2GRAY)
    gb = cv2.GaussianBlur(g, (3, 3), 0)
    band = (gb >= np.percentile(gb, pct)).astype(np.uint8)
    band = cv2.morphologyEx(
        band, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lab = cv2.connectedComponents(band, 8)
    if n <= 1:
        return None, None, None
    xs = np.linspace(p1[0], p2[0], 150).astype(int)
    ys = np.linspace(p1[1], p2[1], 150).astype(int)
    ok = (xs >= 0) & (xs < lab.shape[1]) & (ys >= 0) & (ys < lab.shape[0])
    labs = lab[ys[ok], xs[ok]]
    labs = labs[labs > 0]
    if labs.size == 0:
        return None, None, None
    cible = int(np.bincount(labs).argmax())          # bande la plus traversée
    band = (lab == cible).astype(np.uint8)
    band = ndi.binary_fill_holes(band).astype(np.uint8)
    dist = cv2.distanceTransform(band, cv2.DIST_L2, 5)
    skel = skeletonize(band > 0)
    # nettoyage : retirer jonctions (spurs) puis petits fragments + pointes
    K = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
    vois = cv2.filter2D(skel.astype(np.uint8), -1, K)
    skel = skel & ~ndi.binary_dilation(skel & (vois >= 3), iterations=2)
    lb, nf = ndi.label(skel, structure=np.ones((3, 3)))
    for i in range(1, nf + 1):
        if (lb == i).sum() < 15:
            skel[lb == i] = False
    eps = 2.0 * dist[skel]
    return eps[eps > 1], band, skel.astype(bool)


# =============================================================================
# MAXIMUM SIZE FILTER (particle modes: optical, SEM powders, SEM nano)
# =============================================================================
# Log-scale slider: 0..100, the right end (>= 99.5) means "no limit".
TAILLE_MAX_BORNES = {"optique": (1.0, 3000.0), "meb": (1.0, 3000.0),
                     "nano": (5.0, 5000.0)}


def pos_vers_taille_max(pos, mode):
    lo, hi = TAILLE_MAX_BORNES.get(mode, (1.0, 3000.0))
    if pos >= 99.5:
        return 0.0
    return float(10 ** (math.log10(lo) + pos / 100.0 *
                        (math.log10(hi) - math.log10(lo))))


def taille_max_vers_pos(v, mode):
    lo, hi = TAILLE_MAX_BORNES.get(mode, (1.0, 3000.0))
    if not v or v <= 0:
        return 100.0
    v = min(max(float(v), lo), hi)
    return 100.0 * (math.log10(v) - math.log10(lo)) / (math.log10(hi) -
                                                        math.log10(lo))


def fmt_taille_max(v):
    if not v or v <= 0:
        return "off"
    return f"{v:.0f}" if v >= 10 else f"{v:.1f}"


def filtrer_taille_max(res, params):
    """Removes the particles whose diameter exceeds params["taille_max"]
    (mode unit, 0 = no limit) from every per-particle list, so that the
    statistics, contours, manual editing and exports stay consistent.
    The number of removed objects is kept in res["n_trop_grands"]."""
    dmax = float((params or {}).get("taille_max", 0.0) or 0.0)
    res["taille_max"] = dmax
    res["n_trop_grands"] = 0
    if dmax <= 0 or res.get("mode") == "fibres":
        return res
    d = np.asarray(res.get("diametres", []), float)
    n = len(d)
    garde = d <= dmax
    if res.get("mode") == "nano":
        pops = res.get("populations") or {}
        for k, v in list(pops.items()):
            v = np.asarray(v, float)
            pops[k] = v[v <= dmax]
        res["diametres"] = d[garde]
        res["n_particules"] = int(garde.sum())
        res["n_trop_grands"] = int(n - garde.sum())
        return res
    if garde.all():
        return res
    ids = res.get("ids")
    retires = ([ids[i] for i in range(n) if not garde[i]]
               if ids is not None and len(ids) == n else [])
    for k in ("diametres", "circularites", "aspects", "solidites"):
        a = np.asarray(res.get(k, []))
        if len(a) == n:
            res[k] = a[garde]
    for k in ("centres_px", "contours", "cordes", "ids"):
        v = res.get(k)
        if v is not None and len(v) == n:
            res[k] = [x for x, g in zip(v, garde) if g]
    if retires and res.get("labels_ws") is not None:
        lab = res["labels_ws"].copy()
        lab[np.isin(lab, retires)] = 0
        res["labels_ws"] = lab
    res["n_particules"] = int(garde.sum())
    res["n_trop_grands"] = int(n - garde.sum())
    return res


def detecter_gouttes_mode(image_bgr, ppm, params, zones=None, progress=None):
    """Droplets & capsules mode (microfluidics, v1.7.5). Detection of the
    dark rings and sub-pixel circle fit (psa/droplets.py), returned in the
    common result format so that statistics, graphs, report and exports work
    as for particles. Diameter at the outer edge or at the ring middle
    (params["convention"]); droplets cut by the image edge are excluded."""
    from psa import droplets as GT
    p = dict(PARAMS_GOUTTES)
    p.update(params or {})
    H, W = image_bgr.shape[:2]
    zm = None
    if zones:
        zm = np.zeros((H, W), np.uint8)
        for z in zones:
            cv2.fillPoly(zm, [np.asarray(z, np.int32)], 1)
        zm = zm > 0
    if progress:
        progress(10, "Dark-ring detection…")
    g = GT.detecter_gouttes(image_bgr, k_anneau=float(p["k_anneau"]),
                            hough=bool(p.get("hough", True)), zones=zm)
    if progress:
        progress(80, "Statistics…")
    ump = 1.0 / ppm if ppm > 0 else 1.0
    conv = p.get("convention", "ext")
    h = float(p.get("hauteur_canal_um") or 0.0) or None
    d_px = g["d_milieu"] if conv == "milieu" else np.where(
        np.isfinite(g["d_ext"]), g["d_ext"], g["d_milieu"])
    ok = ~g["bord"]
    idx = np.where(ok)[0]
    labels = np.zeros((H, W), np.int32)
    contours, centres = [], []
    for k, i in enumerate(idx, 1):
        c = (int(round(g["cx"][i])), int(round(g["cy"][i])))
        r = max(1, int(round(d_px[i] / 2)))
        cv2.circle(labels, c, r, k, -1)
        contours.append(cv2.ellipse2Poly(c, (r, r), 0, 0, 360, 4).reshape(-1, 1, 2))
        centres.append((float(g["cx"][i]), float(g["cy"][i])))
    base = image_bgr.copy() if image_bgr.ndim == 3 else \
        cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    ep = max(1, int(round(max(H, W) / 700)))
    for i in range(len(g["cx"])):
        cv2.circle(base, (int(round(g["cx"][i])), int(round(g["cy"][i]))),
                   int(round(d_px[i] / 2)),
                   (60, 60, 230) if g["bord"][i] else (90, 210, 40), ep,
                   cv2.LINE_AA)
    # qualité : ajustement des anneaux (résidu relatif) et reconnaissance
    rel = g["rms"][ok] / np.maximum(g["d_milieu"][ok] / 2, 1e-6)
    oui, msg_type = GT.ressemble_a_des_gouttes(image_bgr, g)
    if len(rel):
        score = float(np.clip(100 * (1 - np.median(rel) / 0.12), 0, 100))
        if not oui:
            score = min(score, 40.0)
    else:
        score = 0.0
    st = "good" if score >= 70 else ("warn" if score >= 45 else "bad")
    msg = (f"Ring-fit residual {100 * np.median(rel):.1f} % of the radius"
           if len(rel) else msg_type)
    stats = GT.statistiques(g, ump, conv, h)
    return {
        "mode": "gouttes", "moteur": "rings", "unite": "µm",
        "diametres": d_px[ok] * ump,
        "circularites": g["circularite"][ok], "aspects": g["aspect"][ok],
        "solidites": np.ones(len(idx)), "cordes": [],
        "centres_px": centres, "contours": contours,
        "ids": list(range(1, len(idx) + 1)), "labels_ws": labels,
        "masque": (labels > 0).astype(np.uint8) * 255,
        "image_base": base,
        "gray_pre": cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        if image_bgr.ndim == 3 else image_bgr,
        "n_particules": int(len(idx)), "n_bord": int((~ok).sum()),
        "score_ws": score, "statut_ws": st, "msg_ws": msg,
        "gouttes": g, "gouttes_stats": stats, "convention": conv,
        "hauteur_canal_um": h, "um_par_px": ump, "ressemble": oui,
    }


def detecter_optique(image_orig, image_am, ppm, params, zones=None,
                     progress=None):
    """
    Pipeline optique complet (v0.5 corrigé : corde_max_v2, otsu_offset,
    filtre circularité désactivé sous 80 px² — biais de discrétisation).
    progress(pct, message) optionnel pour la barre de progression GUI.
    """
    def pg(p, m):
        if progress:
            progress(p, m)

    params = appliquer_taille_min_off(params, "optique")
    pg(5, "Preprocessing…")
    gris = cv2.cvtColor(image_am, cv2.COLOR_BGR2GRAY)
    k = params["blur_ksize"]; k = k if k % 2 == 1 else k + 1
    flou = cv2.GaussianBlur(gris, (k, k), sigmaX=1.5)
    binaire = seuiller(flou, params)
    t = max(3, int(min(gris.shape) * 0.012)); t = t if t % 2 == 1 else t + 1
    noy = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (t, t))
    masque = cv2.morphologyEx(binaire, cv2.MORPH_CLOSE, noy, iterations=2)
    masque = cv2.morphologyEx(masque, cv2.MORPH_OPEN, noy, iterations=1)
    masque = appliquer_zones(masque, zones or [])
    H, W = masque.shape

    pg(30, "Distance map + watershed…")
    dist = cv2.distanceTransform(masque, cv2.DIST_L2, 5)
    dist_n = dist / (dist.max() + 1e-9)
    sc_ws, msg_ws, st_ws = evaluer_ws(masque, dist_n,
                                      params["watershed_dist_pct"])
    rmin = max(3, int(params["diametre_min_um"] * ppm / 2))
    coords = peak_local_max(dist_n, min_distance=rmin,
                            threshold_abs=params["watershed_dist_pct"],
                            labels=masque.astype(bool))
    mm = np.zeros(dist_n.shape, bool)
    if len(coords):
        mm[tuple(coords.T)] = True
    ml, _ = ndi.label(mm)
    lws = watershed(-dist, ml, mask=masque.astype(bool))
    if moteur_effectif(params) == "stardist":
        # learned segmentation: StarDist labels replace the watershed; the
        # measurement below (contours, ECD, shape) is unchanged
        pg(35, "StarDist (deep learning)…")
        gris_sd = cv2.cvtColor(image_am, cv2.COLOR_BGR2GRAY) \
            if image_am.ndim == 3 else image_am
        lws = labels_stardist_optique(gris_sd, params)
        if zones:
            lws[appliquer_zones(np.full(lws.shape, 255, np.uint8), zones) == 0] = 0

    pg(55, "Filtering regions…")
    aire_min = math.pi * (params["diametre_min_um"] * ppm / 2) ** 2
    diams, circs, cnts_ok, centres, cordes = [], [], [], [], []
    aspects, solidites = [], []
    n_bord = 0
    ids = [r for r in np.unique(lws) if r != 0]
    for idx, rid in enumerate(ids):
        if progress and idx % 25 == 0 and ids:
            pg(55 + 35 * idx / len(ids), "Filtering regions…")
        reg = np.uint8(lws == rid) * 255
        cs, _ = cv2.findContours(reg, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            continue
        cnt = max(cs, key=cv2.contourArea)
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bx <= 0 or by <= 0 or (bx + bw) >= W or (by + bh) >= H:
            n_bord += 1
            continue
        aire = cv2.contourArea(cnt)
        if aire < aire_min:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue
        circ = min(1.0, 4 * math.pi * aire / peri ** 2)
        # Sous ~80 px², le périmètre pixelisé sous-estime la circularité :
        # on ne filtre que les objets assez grands pour une mesure fiable.
        if aire >= 80 and circ < params["circularite_min"]:
            continue
        diams.append(2.0 * math.sqrt(aire / math.pi) / ppm)
        circs.append(circ)
        cnts_ok.append(cnt)
        asp, sol = forme_contour(cnt, aire)
        aspects.append(asp)
        solidites.append(sol)
        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"]) if M["m00"] else bx + bw // 2
        cy = int(M["m01"] / M["m00"]) if M["m00"] else by + bh // 2
        centres.append((cx, cy))
        p1, p2, cp = corde_max_v2(cnt)
        cordes.append((p1, p2, cp))   # corde en PIXELS (conversion à l'export)

    pg(92, "Annotation…")
    image_base = rendre_annotation(image_orig, cnts_ok, centres, cordes,
                                   zones)

    pg(100, "Done")
    return {
        "moteur":          moteur_effectif(params),
        "mode":            "optique",
        "unite":           "µm",
        "image_base":      image_base,
        "image_amelioree": image_am,
        "masque":          masque,
        "labels_ws":       lws,
        "dist":            dist,
        "contours":        cnts_ok,
        "centres_px":      centres,
        "cordes":          cordes,
        "diametres":       np.array(diams),
        "circularites":    np.array(circs),
        "aspects":         np.array(aspects),
        "solidites":       np.array(solidites),
        "n_particules":    len(cnts_ok),
        "n_bord":          n_bord,
        "score_ws":        sc_ws,
        "msg_ws":          msg_ws,
        "statut_ws":       st_ws,
    }


def detecter_nano(image_bgr, ppm, params, zones=None, progress=None):
    """Nano analysis. Default engine: the two-step nanocarbon pipeline
    (aggregates, then StarDist primaries, psa/nanocarbon.py). Classic engine
    (moteur = "watershed"): h-maxima watershed + Hough + granulometry."""
    if not NANO_OK:
        raise RuntimeError("Module psa_nano_meb not found.")
    params = appliquer_taille_min_off(params or {}, "nano")
    if (params or {}).get("moteur", "stardist") == "stardist" and \
            stardist_disponible():
        return analyser_nanocarbone(image_bgr, ppm, params, zones, progress)
    img = image_bgr.copy()
    if zones:
        # Le module nano ne gère pas les zones : on les neutralise en les
        # remplissant avec la médiane du fond avant l'analyse.
        med = int(np.median(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))) \
              if img.ndim == 3 else int(np.median(img))
        for z in zones:
            cv2.fillPoly(img, [z], (med, med, med) if img.ndim == 3 else med)
    if progress:
        progress(10, "SEM preprocessing…")
    res = nano.analyser_meb(img, ppm, params, verbose=False)
    if progress:
        progress(95, "Summary…")
    seg = res["seg"]
    dist_n = seg["dist"] / (seg["dist"].max() + 1e-9)
    sc, msg, st = evaluer_ws(seg["masque"], dist_n, 0.3)
    if progress:
        progress(100, "Done")
    # Trois populations mesurables (terminologie ISO 26824). Pour un
    # nanopoudre, la STATISTIQUE pertinente porte sur les PARTICULES
    # PRIMAIRES (distribution en nombre par transformée de Hough) : les
    # régions watershed ne donnent que les quelques agrégats, trop peu
    # nombreux pour une statistique. L'opérateur peut basculer dans les
    # résultats.
    d_primaires = np.asarray(res["d_hough_nm"], float)
    d_agregats = np.asarray(seg["d_regs_nm"], float)
    d_agglo = np.asarray(seg["d_aggl_nm"], float)
    populations = {"primaires": d_primaires,
                   "agregats": d_agregats,
                   "agglomerats": d_agglo}
    pop = "primaires" if len(d_primaires) >= 10 else "agregats"
    d = populations[pop]
    return {
        "mode":         "nano",
        "unite":        "nm",
        "nano":         res,
        "masque":       seg["masque"],
        "labels_ws":    seg["labels"],
        "diametres":    d,
        "populations":  populations,
        "population":   pop,
        "n_regions":    int(len(d_agregats)),
        "circularites": seg["circs"],
        "aspects":      seg["aspects"],
        "solidites":    seg["solidites"],
        "ids":          list(seg["ids"]),
        "centres_px":   seg["centres_px"],
        "n_particules": len(d),
        "n_bord":       seg["n_bord"],
        "score_ws":     sc,
        "msg_ws":       msg,
        "statut_ws":    st,
    }


def fmt_diam_fibre(um):
    """Diamètre de fibre lisible : nm sous 1 µm, µm au-delà."""
    um = float(um)
    return f"{um * 1000:.0f} nm" if um < 1.0 else f"{um:.2f} µm"


def detecter_fibres(image_bgr, ppm, params, zones=None, progress=None):
    """Analyse de fibres électrospinnées via psa_fibres, format commun.

    Les diamètres de fibre (exprimés en µm) jouent le rôle de la population
    mesurée : toute la machinerie statistique (Dn50, distribution, export)
    s'applique alors directement. Les grandeurs propres aux fibres (billes,
    alignement, porosité) sont jointes dans la clé « fibres »."""
    if not FIB_OK:
        raise RuntimeError("Module psa_fibres not found.")
    img = image_bgr.copy()
    if zones:
        med = (int(np.median(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
               if img.ndim == 3 else int(np.median(img)))
        for z in zones:
            cv2.fillPoly(img, [z], (med, med, med) if img.ndim == 3 else med)
    if progress:
        progress(10, "Segmenting the fibre network…")
    fr = fibres.analyser_fibres(img, ppm, params)
    if progress:
        progress(95, "Summary…")
    diam = np.asarray(fr.get("diam_um", []), float)        # unité : µm
    # Qualité : assez de SITES de mesure indépendants, et une part suffisante
    # de la longueur de fibre effectivement mesurée de bord à bord (chaque
    # pixel de bord lance un segment : ~2 segments par pixel de fibre).
    n_sites = int(len(diam))
    lg = float(np.count_nonzero(fr.get("skel_fib", np.zeros(1))))
    part = (min(1.0, fr.get("n_rayons", 0) / (2.0 * lg)) if lg > 0 else 0.0)
    score = (60.0 * min(1.0, n_sites / 50.0) +
             40.0 * min(1.0, part / 0.25))
    st = "good" if score >= 70 else ("warn" if score >= 40 else "bad")
    msg = (f"{n_sites} measurement sites ({fr.get('n_rayons', 0)} edge-to-edge "
           f"segments) · ~{100 * part:.0f} % of the fibre length "
           f"measured · {fr['n_billes']} beads")
    base = fibres.image_annotee_fibres(img, fr)
    if progress:
        progress(100, "Done")
    return {
        "mode":         "fibres",
        "unite":        "µm",
        "fibres":       fr,
        "image_base":   base,
        "masque":       (fr["masque"].astype(np.uint8) * 255),
        "labels_ws":    None,
        "diametres":    diam,
        "circularites": np.array([]),
        "aspects":      np.array([]),
        "solidites":    np.array([]),
        "ids":          [],
        "centres_px":   [],
        "n_particules": int(len(diam)),
        "n_bord":       0,
        "score_ws":     score,
        "msg_ws":       msg,
        "statut_ws":    st,
    }


def detecter_meb(image_bgr, ppm, params, zones=None, progress=None):
    """
    Analyse MEB à l'échelle MICRONIQUE (poudres métalliques 5-200 µm…) :
    mêmes algorithmes que le mode nano (prétraitement MEB, Sauvola +
    plancher, watershed h-maxima) mais paramètres et résultats en µm.
    """
    if not NANO_OK:
        raise RuntimeError("Module psa_nano_meb not found.")
    params = appliquer_taille_min_off(params, "meb")
    img = image_bgr.copy()
    if zones:
        med = int(np.median(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))) \
              if img.ndim == 3 else int(np.median(img))
        for z in zones:
            cv2.fillPoly(img, [z], (med, med, med) if img.ndim == 3 else med)
    p_n = params_vers_nano(params)
    if progress:
        progress(10, "SEM preprocessing…")
    gray, _ = nano.pretraiter_meb(img, p_n)
    if progress:
        progress(45, "StarDist (deep learning)…"
                 if moteur_effectif(p_n) == "stardist"
                 else "Sauvola + watershed h-maxima…")
    if moteur_effectif(p_n) == "stardist":
        brut = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        seg = segmenter_stardist(gray, ppm / 1000.0, p_n, gray_brut=brut)
    else:
        seg = nano.segmenter_hmax(gray, ppm / 1000.0, p_n)
    if progress:
        progress(90, "Summary…")
    dist_n = seg["dist"] / (seg["dist"].max() + 1e-9)
    sc, msg, st = evaluer_ws(seg["masque"], dist_n, 0.3)
    if progress:
        progress(100, "Done")
    return {
        "moteur":       seg.get("moteur", "watershed"),
        "stardist":     seg.get("stardist", {}),
        "mode":         "meb",
        "unite":        "µm",
        "gray_pre":     gray,
        "masque":       seg["masque"],
        "labels_ws":    seg["labels"],
        "dist":         seg["dist"],
        "diametres":    seg["d_regs_nm"] / 1000.0,
        "circularites": seg["circs"],
        "aspects":      seg["aspects"],
        "solidites":    seg["solidites"],
        "ids":          list(seg["ids"]),
        "centres_px":   seg["centres_px"],
        "d_aggl_um":    seg["d_aggl_nm"] / 1000.0,
        "n_particules": len(seg["d_regs_nm"]),
        "n_bord":       seg["n_bord"],
        "score_ws":     sc,
        "msg_ws":       msg,
        "statut_ws":    st,
    }


def charger_image(chemin):
    """
    Lecture robuste pour images réelles :
      • chemins accentués/Unicode sous Windows (cv2.imread y échoue
        silencieusement) → np.fromfile + imdecode ;
      • TIFF 16 bits (caméras 12/16 bits) → étirement robuste sur les
        percentiles 0.1–99.9 (la conversion naïve d'OpenCV divise par 256
        et rend l'image quasi noire) ;
      • niveaux de gris et canaux alpha normalisés en BGR 8 bits.
    Retourne (image_bgr, note) — note décrit la conversion appliquée.
    """
    note = ""
    try:
        donnees = np.fromfile(str(chemin), dtype=np.uint8)
        img = cv2.imdecode(donnees, cv2.IMREAD_UNCHANGED)
    except Exception:
        img = None
    if img is None:
        img = cv2.imread(str(chemin), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, ""
    if img.dtype == np.uint16:
        lo, hi = np.percentile(img, (0.1, 99.9))
        if hi <= lo:
            lo, hi = float(img.min()), float(max(img.max(), img.min() + 1))
        img = np.clip((img.astype(np.float32) - lo) / (hi - lo) * 255.0,
                      0, 255).astype(np.uint8)
        note = "16-bit TIFF → 8 bits (percentile stretch 0.1–99.9)"
    elif img.dtype != np.uint8:
        img = cv2.normalize(img.astype(np.float32), None, 0, 255,
                            cv2.NORM_MINMAX).astype(np.uint8)
        note = f"converted from {img.dtype}"
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img, note


def calibration_hitachi(chemin):
    """px/µm from the Hitachi SEM text file saved with the image (same name,
    .txt): PixelSize in nm. None if absent or unreadable."""
    import re as _re
    base = os.path.splitext(str(chemin))[0]
    for ext in (".txt", ".TXT"):
        f = base + ext
        if not os.path.exists(f):
            continue
        for enc in ("utf-8", "cp1252", "latin-1"):
            try:
                with open(f, encoding=enc) as fh:
                    txt = fh.read()
                break
            except (UnicodeDecodeError, OSError):
                txt = ""
        if "[SemImageFile]" not in txt and "SemImageFile" not in txt:
            continue                      # pas un fichier Hitachi
        m = _re.search(r"^\s*PixelSize\s*=\s*([0-9.eE+-]+)", txt, _re.M)
        try:
            nm = float(m.group(1)) if m else 0.0
        except ValueError:
            nm = 0.0
        if nm > 0:
            return 1000.0 / nm
    return None


def calibration_metadonnees(chemin):
    """
    Best-effort : extrait la taille de pixel des métadonnées TIFF.
    Couvre FEI/Thermo (PixelWidth), Zeiss SmartSEM (image pixel size),
    et les tags génériques ImageJ/résolution TIFF. Tout est défensif :
    au moindre doute → (None, None), la calibration manuelle reste reine.
    Retourne (ppm en px/µm, source) — TOUJOURS à confirmer par l'opérateur.
    v1.7.8 : lit d'abord le fichier texte Hitachi enregistré avec l'image
    (même nom, .txt, PixelSize en nm), comme celui du SU8020.
    """
    ppm_h = calibration_hitachi(chemin)
    if ppm_h:
        return ppm_h, "Hitachi .txt (PixelSize)"
    if not TIF_OK or not str(chemin).lower().endswith((".tif", ".tiff")):
        return None, None
    EN_UM = {"m": 1e6, "cm": 1e4, "mm": 1e3, "um": 1.0, "µm": 1.0,
             "micron": 1.0, "nm": 1e-3, "pm": 1e-6}
    try:
        with tifffile.TiffFile(str(chemin)) as tf:
            # FEI / Thermo Fisher : PixelWidth en mètres/pixel
            try:
                fei = tf.fei_metadata
                if fei:
                    for bloc in ("Scan", "EScan", "EBeam"):
                        pw = fei.get(bloc, {}).get("PixelWidth")
                        if pw:
                            return 1.0 / (float(pw) * 1e6), \
                                   "FEI/Thermo (PixelWidth)"
            except Exception as e:
                journal("warning", f"calibration metadata: FEI tags unreadable: {e}")
            # Zeiss SmartSEM : ap_image_pixel_size = (libellé, valeur, unité)
            try:
                sem = tf.sem_metadata
                if sem:
                    for cle, val in sem.items():
                        if "pixel" in str(cle).lower() and "size" in \
                                str(cle).lower() and isinstance(val, tuple) \
                                and len(val) >= 3:
                            v, u = float(val[1]), str(val[2]).lower()
                            if u in EN_UM and v > 0:
                                return 1.0 / (v * EN_UM[u]), \
                                       "Zeiss SmartSEM (pixel size)"
            except Exception as e:
                journal("warning", f"calibration metadata: Zeiss tags unreadable: {e}")
            # Générique : XResolution (px par unité) + unité ImageJ ou TIFF
            try:
                page = tf.pages[0]
                xres = page.tags.get("XResolution")
                if xres:
                    num, den = xres.value
                    px_par_unite = float(num) / float(den or 1)
                    unite = None
                    ij = getattr(tf, "imagej_metadata", None) or {}
                    if ij.get("unit"):
                        unite = str(ij["unit"]).lower()
                    else:
                        ru = page.tags.get("ResolutionUnit")
                        if ru and getattr(ru.value, "value", ru.value) == 3:
                            unite = "cm"
                    if unite in EN_UM and px_par_unite > 0:
                        ppm = px_par_unite / EN_UM[unite]
                        if 1e-4 < ppm < 1e6:
                            return ppm, f"tags TIFF (XResolution, {unite})"
            except Exception as e:
                journal("warning", f"calibration metadata: TIFF resolution tags unreadable: {e}")
    except Exception as e:
        journal("warning", f"calibration metadata unreadable: {e}")
    return None, None


def diagnostiquer_echec(res, session):
    """Message actionnable quand l'analyse retient 0 particule."""
    couv = float(res["masque"].mean()) if res.get("masque") is not None else 0
    couv = couv / 255.0 if couv > 1 else couv
    if couv < 0.02:
        return ("Nearly empty mask (%.1f %% of the image): Sensitivity too "
                "low, or DARK particles on a bright background in SEM mode "
                "(designed for the opposite), or insufficient contrast; try "
                "the advanced mode / inversion." % (100 * couv))
    if couv > 0.70:
        return ("Saturated mask (%.0f %% of the image): Sensitivity too "
                "high, the background is detected as particles. Lower the "
                "Sensitivity or check the background flattening."
                % (100 * couv))
    return ("Structures are detected (%.0f %% of the image) but all "
            "filtered out: Minimum size too large for this image, or "
            "wrong CALIBRATION (check the px/µm: the number one cause), "
            "mode %s, %.4g px/µm." % (100 * couv, session.mode, session.ppm))


__all__ = [_n for _n in dir() if not _n.startswith("__")]
