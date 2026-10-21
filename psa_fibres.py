# -*- coding: utf-8 -*-
"""
=============================================================================
PSA FIBRES — Module d'analyse de fibres électrospinnées (MEB)
Centre de Recherche Materia Nova — Mons, Belgique — 2026
=============================================================================
Mesure le DIAMÈTRE DE FIBRE d'un non-tissé électrospinné, métrique reine de
l'électrospinning.

Méthode par défaut, « bords » (indépendante du seuil) : un réseau de fibres
n'est pas séparable en objets, mais chaque fibre est bordée de deux bords
nets, parallèles et de sens opposés. Depuis chaque pixel de bord, un rayon
traverse la fibre vers le côté clair et s'arrête au premier bord opposé qui
satisfait le critère de l'opérateur (même niveau extérieur des deux côtés,
intérieur plus clair et sans trou) ; c'est le principe de la Stroke Width
Transform (Epshtein et al., CVPR 2010), soit le geste de l'opérateur qui pose
ses curseurs. Des filtres écartent les liserés brillants (effet de bord SE),
la texture, les rayons obliques, ceux qui filent le long d'une fibre fine ou
enjambent un faisceau, et vérifient la cohérence avec l'échelle locale du
masque. Les mesures sont regroupées en SITES indépendants le long des fibres
(distribution pondérée par la longueur, comme DiameterJ ; n honnête).

Validation : fibres synthétiques de diamètre connu (6 à 48 px, écart < 5 %
dès 12 px) ; deux mesures au curseur MEB d'un opérateur (1,49 et 1,06 µm,
retrouvées à +3 %). Le diamètre ne varie plus avec le seuil.

Le masque seuillé reste utilisé pour les métriques de surface : couverture,
porosité du mat, billes (« beads », renflements compacts), pores des fibres
poreuses, et comme garde-fou d'échelle. L'ancienne méthode (squelette + carte
de distance, dépendante du seuil) reste disponible : methode_diametre="masque".

Limites : fibres de moins de ~4 px non mesurables (augmenter le
grandissement) ; profils sans bord marqué (flancs qui s'assombrissent
progressivement jusqu'au fond, rare en électrons secondaires) mesurés entre
les points de plus forte pente, donc sous-estimés.

Dépendances : opencv-python, numpy, scipy, scikit-image.
=============================================================================
"""

import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from skimage import measure
from skimage.feature import structure_tensor
from scipy.spatial import cKDTree

# Recadrage du bandeau MEB : on réutilise le détecteur du module nano s'il
# est présent, sinon repli simple.
try:
    from psa_nano_meb import detecter_databar as _detecter_databar
except Exception:
    _detecter_databar = None


FIBRES_PARAMS = {
    "couper_databar":  True,
    "denoise_h":       5.0,      # débruitage non-local (0 = off)
    "clahe_clip":      2.0,
    "fibres_claires":  True,     # fibres claires sur fond sombre (SE)
    "seuil_methode":   "otsu",   # "otsu" | "sauvola"
    "seuil_facteur":   0.65,      # multiplie le seuil Otsu : <1 capte le CORPS
                                  # de la fibre (pas que la crête brillante)
    "sauvola_window":  51,
    "sauvola_k":       0.10,
    "bead_facteur":    3.0,      # bille si épaisseur locale > facteur × rayon médian
    "exclure_jonctions": True,
    "aire_min_px":     30,
    "methode_diametre": "bords",  # "bords" (appariement, sans seuil) | "masque"
    "bords_k_bruit":   3.0,      # seuil de bord en multiples du bruit
    "bords_symetrie":  0.35,     # tolérance d'asymétrie des niveaux extérieurs
    "bords_res_min_px": 3.5,     # limite de résolution (px)
    "bords_sauts":     2,        # bords opposés invalides enjambables
    "bords_max_traverses": 5,    # bords forts internes tolérés (liserés)
    "coherence_echelle": 3.0,    # tolérance bords / échelle du masque (×)
    "combler_pores":   False,    # fibres POREUSES : comble les pores (mesure
                                 # le Ø du ruban plein) et quantifie la porosité
    "pore_fermeture_px": 7,      # fermeture pour reconnecter les bords du ruban
    "pore_min_px":     8,        # aire mini d'un pore réel (px)
}


def _databar_y(gray, couper):
    if not couper:
        return gray.shape[0]
    if _detecter_databar is not None:
        try:
            y = int(_detecter_databar(gray))
            if 0 < y <= gray.shape[0]:
                return y
        except Exception:
            pass
    return int(gray.shape[0] * 0.92)


def segmenter_fibres(gray, params=None):
    """Masque binaire du réseau de fibres (claires sur fond sombre)."""
    p = dict(FIBRES_PARAMS)
    if params:
        p.update(params)
    g = gray if p.get("fibres_claires", True) else cv2.bitwise_not(gray)
    g = cv2.createCLAHE(float(p.get("clahe_clip", 2.0)), (8, 8)).apply(g)
    h = float(p.get("denoise_h", 5.0))
    if h > 0:
        g = cv2.fastNlMeansDenoising(g, None, h, 7, 21)
    if str(p.get("seuil_methode", "otsu")) == "sauvola":
        from skimage.filters import threshold_sauvola
        win = int(p.get("sauvola_window", 51))
        win = win if win % 2 == 1 else win + 1
        th = threshold_sauvola(g, window_size=win,
                               k=float(p.get("sauvola_k", 0.10)))
        mask = g > th
    else:
        otsu_val, _ = cv2.threshold(g, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Facteur < 1 → seuil plus bas → on capte le corps grisé de la fibre
        # jusqu'au fond sombre, pas seulement la crête brillante centrale
        # (sinon le Ø des fibres épaisses est largement sous-estimé).
        fac = float(p.get("seuil_facteur", 0.65))
        mask = g > (otsu_val * fac)
    mask = ndi.binary_closing(mask, iterations=1)
    mask = ndi.binary_opening(mask, iterations=1)
    lbl, n = ndi.label(mask)
    if n:
        aires = ndi.sum(np.ones_like(lbl), lbl, range(1, n + 1))
        garde = np.zeros(n + 1, bool)
        garde[1:] = aires >= int(p.get("aire_min_px", 30))
        mask = garde[lbl]
    return mask.astype(bool)


# =============================================================================
# MESURE DU DIAMÈTRE PAR APPARIEMENT DE BORDS (méthode par défaut)
# =============================================================================
# Le diamètre ne dépend plus d'un seuil : depuis chaque pixel de bord on lance
# un rayon vers l'intérieur clair de la fibre et l'on s'arrête au premier bord
# fort ; la largeur n'est retenue que si ce bord est OPPOSÉ (anti-parallèle)
# et de force comparable (principe de la Stroke Width Transform, Epshtein et
# al., CVPR 2010). C'est le geste de l'opérateur qui pose ses curseurs sur les
# deux bords nets. Des filtres écartent ensuite ce qui n'est pas une fibre :
# liserés brillants, texture, croisements, rayons obliques.

def _bords_nms(gs, k_bruit):
    gx = ndi.sobel(gs, axis=1) / 8.0
    gy = ndi.sobel(gs, axis=0) / 8.0
    mag = np.hypot(gx, gy)
    q = (np.round(np.arctan2(gy, gx) / (np.pi / 4)) % 4).astype(int)
    nms = np.zeros(mag.shape, bool)
    for k, (dy, dx) in {0: (0, 1), 1: (1, 1), 2: (1, 0), 3: (1, -1)}.items():
        nms |= ((q == k) & (mag >= np.roll(mag, (dy, dx), (0, 1))) &
                (mag >= np.roll(mag, (-dy, -dx), (0, 1))))
    bruit = 1.4826 * np.median(np.abs(mag - np.median(mag)))
    nms &= mag > max(k_bruit * bruit, 2.0)
    nms[:2, :] = nms[-2:, :] = nms[:, :2] = nms[:, -2:] = False
    return mag, gx, gy, nms


def epaisseur_locale(mask, dist=None, skel=None, n_classes=14):
    """Épaisseur locale (px) de chaque pixel du masque : diamètre de la plus
    grande boule inscrite qui le contient (Hildebrand & Rüegsegger, 1997),
    approchée par classes de rayon. Donne l'ÉCHELLE de l'objet où l'on est."""
    if dist is None:
        dist = ndi.distance_transform_edt(mask)
    if skel is None:
        skel = skeletonize(mask)
    T = np.zeros(mask.shape, np.float32)
    r_sk = dist[skel]
    if r_sk.size == 0:
        return T
    rs = np.unique(np.round(np.geomspace(1.0, max(1.5, r_sk.max()),
                                         n_classes), 1))
    for r in rs:
        graines = (skel & (dist >= r)).astype(np.uint8)
        if not graines.any():
            continue
        k = int(2 * np.ceil(r) + 1)
        disque = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        T[cv2.dilate(graines, disque).astype(bool)] = 2.0 * r
    return T


def largeurs_bords(g, p=None):
    """
    Largeurs locales des fibres (px) par appariement de bords.
      g : image 8 bits, fibres CLAIRES (inverser au préalable sinon), débruitée.
    Retourne un dict : larg (px), dep / arr (extrémités y,x), axe (vecteurs
    unitaires le long de la fibre). Vide si aucune fibre n'est trouvée.
    """
    p = p or {}
    sigma = float(p.get("bords_sigma", 1.2))
    cos_anti = float(p.get("bords_cos_anti", 0.80))
    ratio = float(p.get("bords_ratio_force", 0.45))
    sym_max = float(p.get("bords_symetrie", 0.35))
    res_min = float(p.get("bords_res_min_px", 3.5))
    max_sauts = int(p.get("bords_sauts", 2))
    H, W = g.shape
    gs = ndi.gaussian_filter(g.astype(np.float32), sigma)
    mag, gx, gy, nms = _bords_nms(gs, float(p.get("bords_k_bruit", 3.0)))
    vide = {"larg": np.array([]), "dep": np.zeros((0, 2)),
            "arr": np.zeros((0, 2)), "axe": np.zeros((0, 2))}
    ys, xs = np.nonzero(nms)
    if ys.size == 0:
        return vide

    def val(yy, xx):
        return ndi.map_coordinates(gs, [yy, xx], order=1, mode="nearest")

    m0 = mag[ys, xs]
    dy, dx = gy[ys, xs] / m0, gx[ys, xs] / m0          # vers le côté clair
    y0, x0 = ys.astype(np.float32), xs.astype(np.float32)
    n = ys.size
    off = 2.0 + 2.0 * sigma
    e0 = val(y0 - off * dy, x0 - off * dx)              # niveau extérieur
    actif = np.ones(n, bool)
    fin = np.full(n, -1.0, np.float32)
    fy = np.zeros(n, np.int32)
    fx = np.zeros(n, np.int32)
    somme = np.zeros(n, np.float32)
    cpt = np.zeros(n, np.float32)
    mini = np.full(n, 1e9, np.float32)
    sauts = np.zeros(n, np.int16)
    trav = np.zeros(n, np.int16)          # bords forts traversés
    max_trav = int(p.get("bords_max_traverses", 5))
    t, larg_max = 1.5, 0.35 * min(H, W)
    # 1. Lancer de rayons (vectorisé). Le rayon s'arrête au premier bord
    #    OPPOSÉ qui satisfait le critère de l'opérateur : même niveau
    #    extérieur des deux côtés, intérieur plus clair et sans trou. Il peut
    #    enjamber quelques bords opposés invalides (liseré à sillon sombre),
    #    mais abandonne s'il retombe dans le fond ou entre dans un autre objet.
    while t < larg_max and actif.any():
        idx = np.nonzero(actif)[0]
        py = np.rint(y0[idx] + t * dy[idx]).astype(np.int32)
        px = np.rint(x0[idx] + t * dx[idx]).astype(np.int32)
        dehors = (py < 0) | (px < 0) | (py >= H) | (px >= W)
        actif[idx[dehors]] = False
        idx, py, px = idx[~dehors], py[~dehors], px[~dehors]
        v = gs[py, px]
        somme[idx] += v
        cpt[idx] += 1
        mini[idx] = np.minimum(mini[idx], v)
        moy = somme[idx] / cpt[idx]
        cosg = ((gx[py, px] * dx[idx] + gy[py, px] * dy[idx]) /
                np.maximum(mag[py, px], 1e-9))
        fort = nms[py, px] & (mag[py, px] >= ratio * m0[idx])
        cand = fort & (cosg < -cos_anti)
        if cand.any():
            j = idx[cand]
            e1 = val(py[cand] + off * dy[j], px[cand] + off * dx[j])
            inter = moy[cand]
            hi, lo = np.maximum(e0[j], e1), np.minimum(e0[j], e1)
            ok = ((inter - hi > 6.0) &
                  (np.abs(e1 - e0[j]) <= sym_max * (inter - lo)) &
                  (mini[j] >= lo + 0.25 * (inter - lo)))
            fin[j[ok]] = t
            fy[j[ok]], fx[j[ok]] = py[cand][ok], px[cand][ok]
            actif[j[ok]] = False
            sauts[j[~ok]] += 1
            actif[j[~ok][sauts[j[~ok]] > max_sauts]] = False
        # une section d'UNE fibre ne croise que quelques bords internes
        # (liserés et sillons) ; un rayon qui traverse un faisceau en croise
        # beaucoup → abandon
        trav[idx] += (nms[py, px] & (mag[py, px] >= 0.5 * m0[idx]) &
                      actif[idx]).astype(np.int16)
        trop = trav[idx] > max_trav
        sorti = (t > 3.0) & (v <= e0[idx] + 0.15 * (moy - e0[idx]))
        entre = nms[py, px] & (mag[py, px] >= 0.8 * m0[idx]) & \
            (cosg > cos_anti)
        actif[idx[sorti | entre | trop]] = False
        t += 0.5
    k = np.nonzero(fin > 0)[0]
    if k.size:
        m1 = mag[fy[k], fx[k]]
        k = k[np.minimum(m1, m0[k]) / np.maximum(m1, m0[k]) >= ratio]
    if k.size == 0:
        return vide
    # 2. Obliquité : projection sur la normale moyenne des deux bords
    n0 = np.column_stack([dy[k], dx[k]])
    g1 = np.column_stack([gy[fy[k], fx[k]], gx[fy[k], fx[k]]])
    nm = n0 - g1 / np.linalg.norm(g1, axis=1, keepdims=True)
    nm /= np.maximum(np.linalg.norm(nm, axis=1, keepdims=True), 1e-9)
    vec = np.column_stack([fy[k] - y0[k], fx[k] - x0[k]])
    Lr = np.linalg.norm(vec, axis=1)
    cosr = np.abs(np.sum(vec * nm, axis=1)) / np.maximum(Lr, 1e-9)
    ok = cosr >= np.cos(np.radians(25))
    k, larg, nm = k[ok], (Lr * cosr)[ok], nm[ok]
    if k.size == 0:
        return vide
    mid = np.column_stack([(y0[k] + fy[k]) / 2, (x0[k] + fx[k]) / 2])
    axe = np.column_stack([-nm[:, 1], nm[:, 0]])
    # 3. Contrôle croisé : une vraie section se prolonge le long de l'axe de
    #    la fibre (± une demi-largeur reste dans la fibre). Un rayon qui file
    #    le long d'une fibre fine retombe aussitôt dans le fond.
    e_lo = np.minimum(e0[k], val(fy[k] + off * dy[k], fx[k] + off * dx[k]))
    c_mid = val(mid[:, 0], mid[:, 1])
    seuil = e_lo + 0.5 * (c_mid - e_lo)
    h = 0.5 * larg
    ok = ((val(mid[:, 0] + h * axe[:, 0], mid[:, 1] + h * axe[:, 1]) >= seuil) &
          (val(mid[:, 0] - h * axe[:, 0], mid[:, 1] - h * axe[:, 1]) >= seuil))
    k, larg, nm, mid, axe = k[ok], larg[ok], nm[ok], mid[ok], axe[ok]
    if k.size == 0:
        return vide
    # 4. Imbrication : un trait contenu dans un trait ≥ 1,8× plus large est un
    #    détail de surface (texture, liseré, pore), pas une fibre
    carte = np.zeros((H, W), np.float32)
    for i in np.argsort(larg):
        cv2.line(carte, (int(x0[k[i]]), int(y0[k[i]])),
                 (int(fx[k[i]]), int(fy[k[i]])), float(larg[i]),
                 max(1, int(0.35 * larg[i])))
    ok = carte[mid[:, 0].astype(int), mid[:, 1].astype(int)] < 1.8 * larg
    k, larg, nm, mid, axe = k[ok], larg[ok], nm[ok], mid[ok], axe[ok]
    # 5. Étendue : une fibre se prolonge sur ≥ 2 largeurs le long de son axe
    if k.size:
        R = np.minimum(np.maximum(20.0, 4.0 * larg), 400.0)
        voisins = cKDTree(mid).query_ball_point(mid, R)
        etendue = np.zeros(k.size)
        for i, vs in enumerate(voisins):
            vs = np.asarray(vs)
            sem = vs[(np.abs(larg[vs] - larg[i]) <= 0.25 * larg[i]) &
                     (np.abs(nm[vs] @ nm[i]) >= 0.9)]
            if sem.size > 1:
                pr = (mid[sem] - mid[i]) @ axe[i]
                etendue[i] = pr.max() - pr.min()
        ok = (etendue >= np.maximum(2.0 * larg, 15.0)) & (larg >= res_min)
        k, larg, axe = k[ok], larg[ok], axe[ok]
    return {"larg": larg,
            "dep": np.column_stack([y0[k], x0[k]]),
            "arr": np.column_stack([fy[k], fx[k]]).astype(np.float32),
            "axe": axe}


def sites_mesure(b, pas_px):
    """Regroupe les rayons en SITES de mesure le long des fibres (cellules de
    taille fixe centrées sur l'axe) : distribution pondérée par la longueur
    de fibre, comme DiameterJ, et n ≈ nombre de mesures indépendantes (et non
    des milliers de pixels voisins, qui gonfleraient la précision affichée).
    Retourne (largeur médiane par site, position y,x, indices des rayons)."""
    L = b["larg"]
    if L.size == 0:
        return np.array([]), np.zeros((0, 2)), []
    mid = (b["dep"] + b["arr"]) / 2.0
    s = max(4.0, float(pas_px))
    cle = np.floor(mid / s).astype(np.int64)
    _, inv = np.unique(cle, axis=0, return_inverse=True)
    inv = inv.ravel()
    ordre = np.argsort(inv, kind="stable")
    bornes = np.searchsorted(inv[ordre], np.arange(inv.max() + 2))
    groupes = [ordre[bornes[i]:bornes[i + 1]] for i in range(inv.max() + 1)]
    d = np.array([np.median(L[gr]) for gr in groupes])
    pos = np.array([mid[gr].mean(0) for gr in groupes])
    return d, pos, groupes


def analyser_fibres(image_bgr, ppm, params=None):
    """
    Analyse complète d'un réseau de fibres électrospinnées.
      image_bgr : image OpenCV (BGR ou gris)
      ppm       : calibration en pixels par micromètre

    Diamètre : par défaut par APPARIEMENT DE BORDS (« bords », indépendant du
    seuil) ; l'ancienne mesure squelette + carte de distance reste disponible
    (methode_diametre = "masque"). Le masque seuillé sert aux métriques de
    surface : couverture, porosité du mat, billes, pores.

    Retourne un dict : diam_px / diam_um / diam_nm (un diamètre par SITE de
    mesure le long des fibres, hors billes), n_rayons, rayons (pour le
    tracé), masque, billes, alignement, orientation, porosités, y_databar.
    """
    p = dict(FIBRES_PARAMS)
    if params:
        p.update(params)
    ppm = float(ppm) if ppm else 1.0
    gray = (image_bgr if image_bgr.ndim == 2
            else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY))
    y = _databar_y(gray, p.get("couper_databar", True))
    g = gray[:y]

    mask = segmenter_fibres(g, p)
    # Fibres POREUSES : on referme les bords du ruban et on comble les pores
    # (trous internes) ; les pores sont quantifiés à part.
    pores_mask = None
    if p.get("combler_pores", False):
        k = int(p.get("pore_fermeture_px", 7))
        k = k if k % 2 == 1 else k + 1
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mc = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                              ker).astype(bool)
        mask_plein = ndi.binary_fill_holes(mc)
        pores_mask = mask_plein & ~mc
        mask = mask_plein
    dist = ndi.distance_transform_edt(mask)
    skel = skeletonize(mask)

    # --- Diamètres ---------------------------------------------------------
    methode = str(p.get("methode_diametre", "bords"))
    rayons = None
    if methode == "bords":
        gb = g if p.get("fibres_claires", True) else cv2.bitwise_not(g)
        h = float(p.get("denoise_h", 5.0))
        if h > 0:
            gb = cv2.fastNlMeansDenoising(gb, None, h, 7, 21)
        if pores_mask is not None and pores_mask.any():
            # pores rebouchés au niveau de la fibre : on mesure le ruban plein
            gb = cv2.inpaint(gb, pores_mask.astype(np.uint8), 3,
                             cv2.INPAINT_TELEA)
        pb = p
        if pores_mask is not None:
            # Rubans poreux : contour à grande échelle, texture interne fine.
            # On cherche les bords à l'échelle du ruban (lissage ∝ épaisseur).
            T = epaisseur_locale(mask, dist, skel)
            t_med = float(np.median(T[mask])) if mask.any() else 0.0
            pb = dict(p)
            pb["bords_sigma"] = max(float(p.get("bords_sigma", 1.2)),
                                    float(np.clip(0.02 * t_med, 1.2, 5.0)))
            pb["bords_max_traverses"] = max(10, int(p.get(
                "bords_max_traverses", 5)))
        rayons = largeurs_bords(gb, pb)
        r_ref = (0.5 * float(np.median(rayons["larg"]))
                 if rayons["larg"].size else 0.0)
    else:
        r_sk = dist[skel]
        r_sk = r_sk[r_sk > 0]
        r_ref = float(np.median(r_sk)) if r_sk.size else 0.0

    if r_ref <= 0:
        return _resultat_vide(mask, skel, dist, y, ppm, pores_mask,
                              "No measurable fibre (check the polarity, "
                              "the contrast or the magnification).")

    # --- Billes (beads) : épaississements locaux du réseau ----------------
    bead_mask = dist > float(p.get("bead_facteur", 3.0)) * r_ref
    # Une bille est un renflement COMPACT ; le cœur d'une fibre simplement
    # plus épaisse que la moyenne est allongé et n'en est pas une.
    billes_d_um = []
    garde = np.zeros_like(bead_mask)
    seuil_a = np.pi * (2.0 * r_ref) ** 2           # au moins ~2× le Ø fibre
    for rp in measure.regionprops(ndi.label(bead_mask)[0]):
        majo = getattr(rp, "axis_major_length", None) or rp.major_axis_length
        mino = getattr(rp, "axis_minor_length", None) or rp.minor_axis_length
        if rp.area > seuil_a and majo / max(mino, 1e-9) <= 2.5:
            billes_d_um.append(2.0 * np.sqrt(rp.area / np.pi) / ppm)
            garde[rp.slice] |= rp.image
    bead_mask = garde
    billes_d_um = np.array(billes_d_um, float)
    excl = ndi.binary_dilation(bead_mask, iterations=2)

    if methode == "bords":
        mid = ((rayons["dep"] + rayons["arr"]) / 2).astype(int)
        garde = ~excl[mid[:, 0], mid[:, 1]]
        # Cohérence d'échelle avec le masque : un segment ≥ 3× plus fin que
        # l'objet où il se trouve est un détail (liseré, texture) ; ≥ 3× plus
        # large, il enjambe plusieurs fibres. Le masque ne MESURE pas, il
        # donne seulement l'ordre de grandeur local.
        T = epaisseur_locale(mask, dist, skel)
        t_loc = ndi.grey_dilation(T, size=5)[mid[:, 0], mid[:, 1]]
        fac = float(p.get("coherence_echelle", 3.0))
        L = rayons["larg"]
        garde &= (t_loc <= 0) | ((L >= t_loc / fac) & (L <= t_loc * fac))
        rayons = {k: v[garde] for k, v in rayons.items()}
        if rayons["larg"].size:
            r_ref = 0.5 * float(np.median(rayons["larg"]))
        diam_px, pos, _ = sites_mesure(rayons, 2.0 * r_ref)
        # Orientation de chaque rayon (axe de fibre), angle dans [0,180)
        ang = (np.degrees(np.arctan2(-rayons["axe"][:, 0],
                                     rayons["axe"][:, 1])) % 180.0)
        skel_fib = skel & ~excl
    else:
        if p.get("exclure_jonctions", True):
            K = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
            vois = cv2.filter2D(skel.astype(np.uint8), -1, K)
            excl = excl | ndi.binary_dilation(skel & (vois >= 3),
                                              iterations=2)
        skel_fib = skel & ~excl
        diam_px = 2.0 * dist[skel_fib]
        diam_px = diam_px[diam_px > 0]
        Axx, Axy, Ayy = structure_tensor(g.astype(float), sigma=2.0,
                                         order="rc")
        theta = 0.5 * np.arctan2(2 * Axy, (Ayy - Axx))
        ang = np.degrees(theta[skel_fib]) % 180.0
    diam_um = diam_px / ppm

    # --- Orientation / alignement (statistique circulaire sur 2θ) ---------
    if ang.size:
        C = float(np.mean(np.cos(2 * np.radians(ang))))
        S = float(np.mean(np.sin(2 * np.radians(ang))))
        align = float(np.hypot(C, S))            # 0 = aléatoire, 1 = aligné
        orient = float(np.degrees(0.5 * np.arctan2(S, C)) % 180.0)
    else:
        align, orient = 0.0, 0.0

    aire_um2 = (g.shape[0] * g.shape[1]) / (ppm ** 2)
    pore_d_um, porosite_fibre = _stats_pores(pores_mask, mask, ppm, p)
    return {
        "methode": methode,
        "diam_px": diam_px,
        "diam_um": diam_um,
        "diam_nm": diam_um * 1000.0,
        "diam_median_um": float(np.median(diam_um)) if diam_um.size else 0.0,
        "n_rayons": int(rayons["larg"].size) if rayons else int(diam_px.size),
        "rayons": rayons,
        "masque": mask,
        "squelette": skel,
        "skel_fib": skel_fib,
        "dist": dist,
        "bead_mask": bead_mask,
        "billes_d_um": billes_d_um,
        "n_billes": int(billes_d_um.size),
        "densite_billes_mm2": float(billes_d_um.size / aire_um2 * 1e6),
        "alignement": align,
        "orientation_deg": orient,
        "angles": ang,
        "porosite_pct": float(100 * (1 - mask.mean())),
        "couverture_pct": float(100 * mask.mean()),
        "pores_mask": pores_mask,
        "pore_d_um": pore_d_um,
        "n_pores": int(pore_d_um.size),
        "pore_d_median_um": (float(np.median(pore_d_um))
                             if pore_d_um.size else 0.0),
        "porosite_fibre_pct": porosite_fibre,
        "y_databar": y,
        "ppm": ppm,
    }


def _stats_pores(pores_mask, mask, ppm, p):
    """Diamètres équivalents des pores (µm) et porosité des fibres (%)."""
    if pores_mask is None or not pores_mask.any():
        return np.array([], float), 0.0
    pl, npr = ndi.label(pores_mask)
    ap = ndi.sum(np.ones_like(pl), pl, np.arange(1, npr + 1))
    ap = ap[ap >= int(p.get("pore_min_px", 8))]
    d = 2.0 * np.sqrt(ap / np.pi) / ppm if ap.size else np.array([], float)
    denom = float(mask.sum())
    return d, (float(100.0 * pores_mask.sum() / denom) if denom else 0.0)


def _resultat_vide(mask, skel, dist, y, ppm, pores_mask, msg):
    vide = np.array([], float)
    return {"erreur": msg, "methode": "", "masque": mask, "squelette": skel,
            "skel_fib": skel, "dist": dist, "rayons": None, "n_rayons": 0,
            "diam_px": vide, "diam_um": vide, "diam_nm": vide,
            "diam_median_um": 0.0, "y_databar": y, "ppm": ppm,
            "bead_mask": np.zeros_like(mask), "n_billes": 0,
            "billes_d_um": vide, "densite_billes_mm2": 0.0,
            "alignement": 0.0, "orientation_deg": 0.0, "angles": vide,
            "porosite_pct": float(100 * (1 - mask.mean())),
            "couverture_pct": float(100 * mask.mean()),
            "pores_mask": pores_mask, "pore_d_um": vide, "n_pores": 0,
            "pore_d_median_um": 0.0, "porosite_fibre_pct": 0.0}


def image_annotee_fibres(image_bgr, res, max_traits=2500):
    """Segments de mesure (de bord à bord) colorés par diamètre, billes
    encerclées en vert, pores en rouge. Avec la méthode « masque » : squelette
    coloré par diamètre."""
    y = res.get("y_databar", image_bgr.shape[0])
    g = (image_bgr[:y] if image_bgr.ndim == 2
         else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)[:y])
    vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    pores = res.get("pores_mask")
    if pores is not None and pores.any():
        ov = vis.copy()
        ov[pores] = (40, 40, 220)
        vis = cv2.addWeighted(ov, 0.45, vis, 0.55, 0)

    def couleurs(d):
        lo, hi = np.percentile(d, 3), np.percentile(d, 97)
        dn = np.clip((d - lo) / max(hi - lo, 1e-9), 0, 1)
        return cv2.applyColorMap((255 * dn).astype(np.uint8).reshape(-1, 1),
                                 cv2.COLORMAP_JET).reshape(-1, 3)

    ray = res.get("rayons")
    if ray is not None and ray["larg"].size:
        n = ray["larg"].size
        sel = (np.linspace(0, n - 1, max_traits).astype(int)
               if n > max_traits else np.arange(n))
        cols = couleurs(ray["larg"][sel])
        ep = 1 if g.shape[1] < 1600 else 2
        for i, c in zip(sel, cols):
            (y0, x0), (y1, x1) = ray["dep"][i], ray["arr"][i]
            cv2.line(vis, (int(x0), int(y0)), (int(x1), int(y1)),
                     (int(c[0]), int(c[1]), int(c[2])), ep, cv2.LINE_AA)
    else:
        skel, dist = res.get("skel_fib"), res.get("dist")
        if skel is not None and dist is not None and skel.any():
            vis[skel] = couleurs(2.0 * dist[skel])
    bead_mask = res.get("bead_mask")
    if bead_mask is not None and bead_mask.any():
        cnts, _ = cv2.findContours(bead_mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) > 20:
                (x, y0), r = cv2.minEnclosingCircle(c)
                cv2.circle(vis, (int(x), int(y0)), int(r) + 2,
                           (40, 200, 90), 2)
    return vis


if __name__ == "__main__":
    import sys
    f = sys.argv[1] if len(sys.argv) > 1 else "fibres_ex.png"
    img = cv2.imread(f)
    r = analyser_fibres(img, 800 / 208.0)        # calibration FW de l'exemple
    print(f"Median fibre Ø: {r['diam_median_um']*1000:.0f} nm  "
          f"(n={len(r['diam_um'])})")
    print(f"beads: {r['n_billes']}  density {r['densite_billes_mm2']:.0f}/mm²")
    print(f"alignment: {r['alignement']:.2f}  porosity: {r['porosite_pct']:.0f}%")
