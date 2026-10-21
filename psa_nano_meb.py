# -*- coding: utf-8 -*-
"""
=============================================================================
PSA NANO / MEB  –  Module d'extension du Particle Size Analyser v0.5
Centre de Recherche Materia Nova – Mons, Belgique – 2026
=============================================================================
Analyse de nanocarbones agglomérés sur images MEB (électrons secondaires).

Pourquoi un module séparé :
  Le pipeline optique (Otsu global + watershed sur carte de distance) suppose
  un fort contraste particule/fond et des objets séparables. En MEB :
    • effet de bord SE  → les contours sont brillants, pas les corps,
    • fond non uniforme → seuil global Otsu inutilisable,
    • agrégats fractals → pas de "vallée" entre particules primaires.

Quatre méthodes complémentaires (terminologie ISO 26824) :
  1. segmenter_hmax()     : Sauvola local + watershed à h-maxima
                            → taille des AGRÉGATS / AGGLOMÉRATS (ECD, nm)
  2. granulometrie_gris() : granulométrie morphologique SANS segmentation
                            → distribution des PARTICULES PRIMAIRES (pond. aire)
  3. primaires_hough()    : Hough circulaire
                            → primaires quasi-sphériques, distribution en nombre
  4. dimension_fractale() : box-counting → Df des agglomérats (~1.8 attendu)

Unités : ppnm = pixels par nanomètre.  ppnm = ppm / 1000 (ppm = px/µm v0.5).

Dépendances : opencv-python, numpy, scipy, scikit-image, matplotlib.
=============================================================================
"""

import math

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import measure
from skimage.filters import threshold_sauvola
from skimage.morphology import h_maxima
from skimage.segmentation import watershed


# =============================================================================
# PARAMÈTRES PAR DÉFAUT  (à fusionner avec PARAMS de v0.5 en "mode nano")
# =============================================================================
NANO_PARAMS = {
    # Prétraitement MEB
    "particules_claires":  True,    # résolu si polarite="auto"
    "polarite":            "auto",  # "auto" | "claires" (SE) | "sombres" (TE/STEM)
    "couper_databar":      True,    # recadrage auto de la barre d'info MEB
    "denoise_nlm_h":       9.0,     # force du débruitage non-local means (0 = off)
    "flatten":             True,    # aplatissement du fond (division gaussienne)
    "clahe_clip":          2.0,
    "clahe_grid":          8,
    # Segmentation (agrégats / agglomérats)
    "sauvola_window":      51,      # fenêtre du seuil local (px, impair)
    "sauvola_k":           0.10,    # sensibilité Sauvola (0.05–0.25)
    "seuil_plancher_rel":  0.55,    # plancher global = 0.55 × seuil Otsu
    "h_rel":               0.35,    # proéminence h-maxima, relative au P90 des distances
    "fermeture_iter":      1,       # itérations de fermeture morpho (1 en nano)
    "d_min_nm":            5.0,     # diamètre minimal physique (nm)
    "aire_min_px":         30,      # aire minimale en PIXELS (discrétisation)
    "circ_min":            0.30,
    "circ_aire_min_px":    80,      # circ. évaluée seulement si aire >= 80 px²
    # Particules primaires
    "hough_d_min_nm":      10.0,
    "hough_d_max_nm":      150.0,
    "hough_sensibilite":   0.80,    # 0 = strict … 1 = permissif
    "granulo_points":      26,      # nb d'échelles d'ouverture
    "granulo_d_max_nm":    None,    # None = auto (min(image)/4)
    "granulo_dim_max":     1024,    # sous-échantillonnage si image plus grande
}


def ppnm_depuis_ppm(ppm: float) -> float:
    """Convertit la calibration v0.5 (px/µm) en px/nm."""
    return float(ppm) / 1000.0


# =============================================================================
# PRÉTRAITEMENT MEB
# =============================================================================
def detecter_polarite(gray):
    """
    Les particules sont-elles CLAIRES (électrons secondaires : objets
    brillants sur fond sombre) ou SOMBRES (mode transmission/STEM « TE » :
    objets absorbants sur fond clair) ?

    Méthode robuste : le fond domine la scène, donc son niveau est le mode
    de l'histogramme. On compare ensuite la masse des pixels nettement plus
    sombres vs nettement plus clairs que ce fond. Retourne True si les
    particules sont claires (convention SE attendue par le seuillage).
    """
    g = gray if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    mode = int(np.bincount(g.ravel()).argmax())
    seuil = 25
    sombres = int((g < mode - seuil).sum())
    clairs = int((g > mode + seuil).sum())
    return clairs >= sombres


def detecter_databar(gray, frac=0.30, saut_min=18.0):
    """
    Détecte la barre d'information MEB (bandeau bas Zeiss/FEI/JEOL/Tescan).
    Cherche, dans le tiers inférieur, la plus forte discontinuité du profil
    de luminance moyenne par ligne. Retourne l'indice de ligne de coupe
    (= hauteur image si aucune barre détectée).
    """
    h = gray.shape[0]
    z0 = int(h * (1.0 - frac))
    profil = gray.mean(axis=1)
    grad = np.abs(np.diff(profil[z0:]))
    if grad.size and float(grad.max()) > saut_min:
        return z0 + int(grad.argmax()) + 1
    return h


def detecter_barre_echelle(gray, y_databar=None):
    """
    Longueur (en pixels) de la barre d'échelle graduée d'une databar MEB
    (Hitachi/Zeiss/FEI : peigne de graduations collé au bord droit, séparé
    du texte instrument par un large vide). Retourne le span en pixels, ou
    None si aucune barre fiable n'est trouvée. L'opérateur fournit ensuite
    la valeur physique (lue sous la barre, p.ex. « 500 nm »).
    """
    g = gray if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if y_databar is None:
        y_databar = detecter_databar(g)
    if y_databar >= g.shape[0]:
        return None
    bar = g[y_databar:]
    h = bar.shape[0]
    if h < 6:
        return None
    blanc = (bar[:max(1, h // 2)] > 160).astype(np.uint8)   # moitié haute
    col = blanc.sum(axis=0)
    ticks = np.nonzero(col > 0)[0]
    if len(ticks) < 5:
        return None
    gaps = np.diff(ticks)
    if len(gaps) and int(gaps.max()) > 15:        # coupe texte / barre
        ticks = ticks[int(np.argmax(gaps)) + 1:]
    if len(ticks) < 5:
        return None
    x0, x1 = int(ticks.min()), int(ticks.max())
    if x1 < bar.shape[1] * 0.80:                  # barre toujours à droite
        return None
    return x1 - x0


def pretraiter_meb(image, params=None):
    """
    Pipeline de prétraitement MEB :
      1. niveaux de gris
      2. recadrage de la databar
      3. débruitage non-local means (bruit MEB)
      4. aplatissement du fond (division par flou gaussien large)
      5. CLAHE
    Retourne (gray_pretraitee_uint8, y_coupe_databar).
    """
    p = dict(NANO_PARAMS)
    if params:
        p.update(params)

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    y = gray.shape[0]
    if p.get("couper_databar", True):
        y = detecter_databar(gray)
        gray = gray[:y]

    # Résolution de la polarité AVANT tout traitement, sur le gris brut
    # recadré (le plus fiable physiquement). En mode "auto", le résultat
    # est inscrit dans le dict de paramètres reçu — partagé avec l'appel
    # suivant à segmenter_hmax/granulometrie_gris, qui inverseront l'image
    # si les particules sont sombres (transmission/STEM).
    if str(p.get("polarite", "auto")) == "auto":
        claires = detecter_polarite(gray)
        p["particules_claires"] = claires
        if params is not None:
            params["particules_claires"] = claires
            params.setdefault("polarite", "auto")

    h_nlm = float(p.get("denoise_nlm_h", 9.0))
    if h_nlm > 0:
        gray = cv2.fastNlMeansDenoising(gray, None, h_nlm, 7, 21)

    if p.get("flatten", True):
        gf = gray.astype(np.float32)
        sigma = max(15.0, min(gray.shape) / 8.0)
        # Le fond est par définition basse fréquence : flou large calculé
        # sur une image réduite puis ré-agrandi (≈ exact, ~20× plus vite).
        k = max(1, int(sigma // 8))
        if k > 1:
            H0, W0 = gf.shape
            petit = cv2.resize(gf, (max(8, W0 // k), max(8, H0 // k)),
                               interpolation=cv2.INTER_AREA)
            fond = cv2.GaussianBlur(petit, (0, 0), sigmaX=sigma / k)
            fond = cv2.resize(fond, (W0, H0), interpolation=cv2.INTER_LINEAR)
        else:
            fond = cv2.GaussianBlur(gf, (0, 0), sigmaX=sigma)
        # Aplatissement ADAPTATIF. Critère : la CONCENTRATION de l'histo-
        # gramme au niveau du fond. Un fond uniforme (transmission/STEM à
        # fond clair) forme un pic net — soustraire un flou dominé par les
        # gros objets sombres y créerait halos et vignetage, gonflant le
        # masque. Un éclairement en gradient étale l'histogramme : on
        # corrige alors. (L'amplitude du flou est trompeuse : un gros objet
        # sombre suffit à la gonfler même sans gradient réel.)
        hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
        mode = int(hist.argmax())
        concentration = hist[max(0, mode - 8):mode + 9].sum() / gray.size
        if concentration < 0.30:                # fond étalé → gradient réel
            # Soustraction (et non division : la division amplifie le bruit
            # là où le fond est sombre → faux objets en périphérie).
            plat = gf - fond + float(fond.mean())
            gray = np.clip(plat, 0, 255).astype(np.uint8)

    clip = float(p.get("clahe_clip", 2.0))
    if clip > 0:
        clahe = cv2.createCLAHE(clipLimit=clip,
                                tileGridSize=(p["clahe_grid"], p["clahe_grid"]))
        gray = clahe.apply(gray)

    return gray, y


# =============================================================================
# 1. SEGMENTATION : SAUVOLA LOCAL + WATERSHED À H-MAXIMA
# =============================================================================
def _completer_et_nettoyer(lws, aire_min_px):
    """Pour les poudres sphériques : complète chaque région par sa coque
    convexe (referme le bas ombragé), rebouche les trous internes et absorbe
    les micro-régions ENCLAVÉES — ces deux derniers étant des artefacts de
    texture qui, sinon, encombrent le résultat sous forme de petits contours
    parasites impossibles à supprimer manuellement (un trou n'est pas une
    particule cliquable). Les vraies petites particules libres (satellites,
    au contact du fond) sont conservées."""
    out = lws.copy()
    # 1. Coque convexe de chaque région, peinte dans le FOND uniquement
    for pr in measure.regionprops(lws):
        try:
            ch = pr.image_convex
        except AttributeError:
            ch = pr.convex_image
        minr, minc, maxr, maxc = pr.bbox
        sub = out[minr:maxr, minc:maxc]
        sub[ch & (sub == 0)] = pr.label
    # 2. Absorber les micro-régions enclavées (ne touchant pas le fond
    #    principal) : artefacts internes, pas des satellites.
    fond_lbl, nf = ndi.label(out == 0)
    if nf:
        tailles = ndi.sum(np.ones_like(fond_lbl), fond_lbl,
                          np.arange(1, nf + 1))
        principal = fond_lbl == (int(np.argmax(tailles)) + 1)
        bord_fond = ndi.binary_dilation(principal, iterations=2)
        aires = {pr.label: pr.area for pr in measure.regionprops(out)}
        med = float(np.median(list(aires.values()))) if aires else 0.0
        seuil = max(aire_min_px * 2.0, 0.06 * med)
        for lab_i, a in aires.items():
            if a < seuil and not (bord_fond & (out == lab_i)).any():
                out[out == lab_i] = 0          # enclavée → deviendra un trou
    # 3. Reboucher tous les trous internes au label voisin le plus proche
    fg = out > 0
    trous = ndi.binary_fill_holes(fg) & ~fg
    if trous.any():
        _, ind = ndi.distance_transform_edt(~fg, return_distances=True,
                                            return_indices=True)
        out[trous] = out[tuple(ind)][trous]
    return out


def mesurer_regions(lws, masque, dist, d_aggl_nm, ppnm, p, aire_min_px,
                    combler=False, marge_bord=0):
    """Measures the labelled regions (ECD, circularity, aspect, solidity,
    centroid) with the filters of the SEM modes: regions touching the image
    border are excluded and counted, then minimum area / diameter and
    circularity. Shared by the h-maxima watershed and the learned
    segmentation (StarDist), so that both are measured identically."""
    H, W = lws.shape[:2]
    # --- Mesures par région -------------------------------------------------------
    d_min_nm = float(p.get("d_min_nm", 5.0))
    circ_min = float(p.get("circ_min", 0.30))
    circ_aire_min = int(p.get("circ_aire_min_px", 80))

    d_regs, circs, centres = [], [], []
    ids, aspects, solidites = [], [], []
    n_bord = 0
    # Poudres sphériques : on complète chaque région par sa coque convexe
    # (referme le bas ombragé), on rebouche les trous internes et on absorbe
    # les micro-régions enclavées — AVANT la mesure, pour que la liste de
    # particules soit directement propre (cf. _completer_et_nettoyer).
    if combler:
        lws = _completer_et_nettoyer(lws, aire_min_px)
    lws_out = lws
    for pr in measure.regionprops(lws):
        minr, minc, maxr, maxc = pr.bbox
        m_ = marge_bord
        if minr <= m_ or minc <= m_ or maxr >= H - m_ or maxc >= W - m_:
            n_bord += 1
            continue
        if pr.area < aire_min_px:
            continue
        if combler:
            try:                       # scikit-image ≥ 0.26
                aire = float(pr.area_convex)
                img_hull = pr.image_convex
            except AttributeError:     # versions antérieures
                aire = float(pr.convex_area)
                img_hull = pr.convex_image
            try:
                per = float(measure.perimeter(img_hull))
            except Exception:
                per = float(pr.perimeter)
            sol = 1.0
        else:
            aire = float(pr.area)
            per = float(pr.perimeter) if pr.perimeter > 0 else 0.0
            try:
                sol = float(pr.solidity)
            except Exception:
                sol = 0.0
        d_nm = 2.0 * math.sqrt(aire / math.pi) / ppnm
        if d_nm < d_min_nm:
            continue
        circ = min(1.0, 4.0 * math.pi * aire / per ** 2) if per > 0 else 0.0
        # Filtre circularité UNIQUEMENT au-dessus de l'aire de confiance :
        # en dessous, le périmètre pixelisé sous-estime systématiquement circ.
        if aire >= circ_aire_min and circ < circ_min:
            continue
        d_regs.append(d_nm)
        circs.append(circ)
        cy, cx = pr.centroid
        centres.append((int(cx), int(cy)))
        ids.append(int(pr.label))
        solidites.append(sol)
        try:
            mino = float(pr.axis_minor_length)
            majo = float(pr.axis_major_length)
        except AttributeError:        # scikit-image < 0.19
            mino = float(pr.minor_axis_length)
            majo = float(pr.major_axis_length)
        aspects.append(majo / mino if mino > 0 else 1.0)

    return {
        "masque":     masque,
        "labels":     lws_out,
        "dist":       dist,
        "d_aggl_nm":  np.array(d_aggl_nm),
        "d_regs_nm":  np.array(d_regs),
        "circs":      np.array(circs),
        "centres_px": centres,
        "ids":        ids,
        "aspects":    np.array(aspects),
        "solidites":  np.array(solidites),
        "n_bord":     n_bord,
    }


def segmenter_hmax(gray_pre, ppnm, params=None):
    """
    Segmentation adaptée MEB :
      • seuillage LOCAL Sauvola (robuste au fond non uniforme / charge),
      • remplissage des trous (effet de bord SE → coquilles brillantes),
      • marqueurs watershed par h-maxima de la carte de distance
        (le paramètre h = proéminence relative, bien plus stable qu'un
         seuil absolu sur les plateaux des agrégats).

    Retourne un dict :
      masque, labels, dist,
      d_aggl_nm  : ECD des agglomérats (composantes connexes du masque),
      d_regs_nm  : ECD des régions watershed retenues,
      circs, centres_px, n_bord
    """
    p = dict(NANO_PARAMS)
    if params:
        p.update(params)

    g = gray_pre if p.get("particules_claires", True) else cv2.bitwise_not(gray_pre)
    combler = bool(p.get("combler_convexe", False))

    # --- Seuillage local + plancher global -------------------------------------
    # Sauvola seul sature les grandes zones de fond (seuil local < bruit) :
    # on impose un PLANCHER global (fraction du seuil d'Otsu) en plus du
    # seuil local. Local → robuste au fond non uniforme ; plancher → empêche
    # le bruit de fond de passer en avant-plan.
    win = int(p.get("sauvola_window", 51))
    win = win if win % 2 == 1 else win + 1
    th_loc = threshold_sauvola(g, window_size=win,
                               k=float(p.get("sauvola_k", 0.10)))
    th_otsu, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plancher = float(p.get("seuil_plancher_rel", 0.55)) * th_otsu
    masque = (((g > th_loc) & (g >= plancher)).astype(np.uint8)) * 255

    # --- Nettoyage morphologique ----------------------------------------------
    noy = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    masque = cv2.morphologyEx(masque, cv2.MORPH_OPEN, noy, iterations=1)
    masque = cv2.morphologyEx(masque, cv2.MORPH_CLOSE, noy,
                              iterations=int(p.get("fermeture_iter", 1)))
    masque = (ndi.binary_fill_holes(masque > 0).astype(np.uint8)) * 255

    # Suppression des composantes sous l'aire minimale (discrétisation)
    aire_min_px = int(p.get("aire_min_px", 30))
    n_cc, lab_cc, stats, _ = cv2.connectedComponentsWithStats(masque, connectivity=8)
    for i in range(1, n_cc):
        if stats[i, cv2.CC_STAT_AREA] < aire_min_px:
            masque[lab_cc == i] = 0

    H, W = masque.shape

    # --- Agglomérats (avant séparation) ----------------------------------------
    lab_aggl, _ = ndi.label(masque > 0)
    d_aggl_nm = []
    for pr in measure.regionprops(lab_aggl):
        minr, minc, maxr, maxc = pr.bbox
        if minr <= 0 or minc <= 0 or maxr >= H or maxc >= W:
            continue
        d_aggl_nm.append(2.0 * math.sqrt(pr.area / math.pi) / ppnm)

    # --- Watershed h-maxima ------------------------------------------------------
    dist = cv2.distanceTransform(masque, cv2.DIST_L2, 5)
    if combler and dist.max() > 0:
        # Poudres sphériques : la surface alvéolée crée plusieurs pics de
        # distance par bille → sur-segmentation (une bille coupée en
        # morceaux). On lisse légèrement la carte de distance pour fusionner
        # ces pics parasites en un seul par bille. Le lissage reste modéré
        # pour ne PAS fusionner les pics de billes voisines (sous-segmentation).
        # Les creux de bord, eux, sont refermés ensuite par la coque convexe.
        ref0 = float(np.percentile(dist[dist > 0], 90))
        dist = cv2.GaussianBlur(dist, (0, 0), max(1.5, ref0 * 0.08))
    dmax = float(dist.max())
    if dmax <= 0:
        return {"masque": masque, "labels": np.zeros_like(lab_aggl),
                "dist": dist, "d_aggl_nm": np.array(d_aggl_nm),
                "d_regs_nm": np.array([]), "circs": np.array([]),
                "centres_px": [], "ids": [], "aspects": np.array([]),
                "solidites": np.array([]), "n_bord": 0}

    # h adaptatif : référencé au percentile 90 des distances (échelle TYPIQUE
    # des particules) et non au max global — sinon un seul gros agglomérat
    # fusionné impose un h démesuré et supprime tous les marqueurs des
    # petites particules.
    ref = float(np.percentile(dist[dist > 0], 90))
    h_abs = max(1.0, float(p.get("h_rel", 0.35)) * ref)
    pics = h_maxima(dist, h_abs)
    # Garantie : au moins un marqueur par composante connexe. Sur une
    # distribution LARGE (ex. poudre 5-200 µm), h calibré sur les grosses
    # particules peut dépasser le max de distance des petites : sans
    # marqueur, elles disparaîtraient du watershed. On marque alors le
    # maximum de distance de chaque composante orpheline.
    lab_cc, n_cc = ndi.label(masque > 0)
    if n_cc:
        a_marqueur = ndi.labeled_comprehension(
            pics, lab_cc, np.arange(1, n_cc + 1), np.any, bool, False)
        orphelines = np.nonzero(~a_marqueur)[0] + 1
        if len(orphelines):
            positions = ndi.maximum_position(dist, lab_cc, orphelines)
            for py, px in np.atleast_2d(positions):
                pics[int(py), int(px)] = True
    marqueurs, _ = ndi.label(pics)
    lws = watershed(-dist, marqueurs, mask=(masque > 0))

    return mesurer_regions(lws, masque, dist, d_aggl_nm, ppnm, p,
                           aire_min_px, combler)


# =============================================================================
# 2. GRANULOMÉTRIE MORPHOLOGIQUE EN NIVEAUX DE GRIS  (sans segmentation)
# =============================================================================
def granulometrie_gris(gray_pre, ppnm, params=None):
    """
    Granulométrie par ouvertures (Matheron/Serra) sur l'image en GRIS :
    aucune binarisation, aucune séparation nécessaire — idéal pour les
    nanocarbones agglomérés où les primaires se touchent.

    Principe : ouverture morphologique de rayon r croissant ; le "volume"
    (somme des intensités) retiré entre deux échelles successives mesure
    la quantité de structures de cette taille → spectre de motifs PS(d).
    La distribution est pondérée par l'AIRE projetée (≈ volumique en 2D).

    Retourne un dict :
      d_nm   : diamètres milieux de classe (nm)
      ps     : spectre de motifs normalisé (somme = 1)
      cum    : cumulée [0–1]
      dx10, dx50, dx90 : percentiles pondérés aire (nm)
    ou None si l'image est vide.
    """
    p = dict(NANO_PARAMS)
    if params:
        p.update(params)

    g = gray_pre if p.get("particules_claires", True) else cv2.bitwise_not(gray_pre)

    # Sous-échantillonnage de sécurité (ouvertures larges = coûteuses)
    sc = 1.0
    H, W = g.shape
    dim_max = int(p.get("granulo_dim_max", 1024))
    if max(H, W) > dim_max:
        sc = dim_max / max(H, W)
        g = cv2.resize(g, (int(W * sc), int(H * sc)), interpolation=cv2.INTER_AREA)
    ppnm_eff = ppnm * sc

    d_max_nm = p.get("granulo_d_max_nm")
    if d_max_nm:
        r_max = int(round(d_max_nm / 2.0 * ppnm_eff))
    else:
        r_max = min(g.shape) // 4
    r_max = max(3, min(r_max, min(g.shape) // 3))

    n_pts = int(p.get("granulo_points", 26))
    # Rayons LOG-espacés : résolution fine aux petites tailles (les rayons
    # linéaires donnaient une première classe de ~10 px qui écrasait tout
    # le bas du spectre), et cohérent avec des distributions log-normales.
    rayons = np.unique(np.round(np.geomspace(1, r_max, n_pts)).astype(int))

    gf = g.astype(np.float32)
    # Soustraction du piédestal de fond : sans elle, l'ouverture à grand
    # rayon érode le plancher de bruit du fond (des millions de pixels),
    # comptabilisé à tort comme du volume de structure — ce qui gonflait
    # artificiellement la queue du spectre. Volume mesuré = luminance
    # AU-DESSUS du fond.
    pied = float(np.percentile(gf, 25))
    gf = np.maximum(gf - pied, 0.0)
    # Pyramide multi-échelle : le spectre de motifs est covariant par
    # changement d'échelle — une ouverture de rayon r sur l'image pleine
    # équivaut à une ouverture de rayon r/k sur l'image réduite k fois,
    # le volume étant multiplié par k². Coût divisé par ~k⁴ : les grands
    # rayons (élément structurant 513×513 !) deviennent quasi gratuits.
    pyram = {1: gf}

    def niveau(k):
        if k not in pyram:
            H0, W0 = gf.shape
            pyram[k] = cv2.resize(gf, (max(8, W0 // k), max(8, H0 // k)),
                                  interpolation=cv2.INTER_AREA)
        return pyram[k]

    vols = []
    R_EXACT = 64          # ouvertures exactes jusqu'ici (coût maîtrisé)
    k_courant, cal, r_prec = 1, 1.0, None

    def _ouvert(img_k, r_eff):
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2 * r_eff + 1, 2 * r_eff + 1))
        return float(cv2.morphologyEx(img_k, cv2.MORPH_OPEN, se).sum())

    v0 = None
    for r in rayons:
        # Sortie anticipée : plus de structure au-delà de cette échelle
        # (le fond étant à zéro, V s'effondre une fois les structures
        # rasées) — inutile de payer les grandes ouvertures.
        if v0 is not None and vols and vols[-1] < 0.01 * v0:
            vols.append(vols[-1])
            continue
        if r <= R_EXACT:
            k = 1
        else:
            k = 2
            while (r / k > R_EXACT / 2
                   and min(gf.shape) // (2 * k) >= 96):
                k *= 2
        img_k = niveau(k)
        if k != k_courant and r_prec is not None:
            r_j = max(1, int(round(r_prec / k)))
            v_j = _ouvert(img_k, r_j) * k * k
            cal = vols[-1] / max(v_j, 1e-9)
            k_courant = k
        r_eff = max(1, int(round(r / k)))
        vols.append(_ouvert(img_k, r_eff) * k * k * cal)
        if v0 is None:
            v0 = vols[0]
        r_prec = r
    # Nettoyage résiduel : V(r) est théoriquement décroissante.
    vols = np.minimum.accumulate(np.asarray(vols))

    ps = np.maximum(0.0, -np.diff(vols))            # volume retiré par classe
    if ps.sum() <= 0:
        return None
    ps /= ps.sum()
    d_nm = 2.0 * rayons / ppnm_eff
    d_mid = 0.5 * (d_nm[1:] + d_nm[:-1])
    cum = np.cumsum(ps)

    def dx(q):
        return float(np.interp(q, cum, d_mid))

    return {"d_nm": d_mid, "ps": ps, "cum": cum,
            "dx10": dx(0.10), "dx50": dx(0.50), "dx90": dx(0.90),
            "rayons_px": rayons}


# =============================================================================
# 3. PARTICULES PRIMAIRES PAR TRANSFORMÉE DE HOUGH CIRCULAIRE
# =============================================================================
def primaires_hough(gray_pre, ppnm, params=None):
    """
    Détection des particules primaires quasi-sphériques (noir de carbone…)
    par transformée de Hough circulaire — fonctionne même si les primaires
    se chevauchent ou ne sont visibles que partiellement (effet de bord SE
    = arcs brillants, exactement ce que Hough exploite).

    Retourne (d_nm, cercles) :
      d_nm    : np.array des diamètres détectés (nm) — distribution EN NOMBRE
      cercles : np.array (N, 3) [x, y, r] en pixels pour l'affichage.
    """
    p = dict(NANO_PARAMS)
    if params:
        p.update(params)

    rmin = max(2, int(round(p["hough_d_min_nm"] / 2.0 * ppnm)))
    rmax = max(rmin + 1, int(round(p["hough_d_max_nm"] / 2.0 * ppnm)))

    img = cv2.medianBlur(gray_pre, 5)
    sens = float(np.clip(p.get("hough_sensibilite", 0.8), 0.0, 1.0))
    param2 = float(np.interp(sens, [0.0, 1.0], [55.0, 18.0]))   # bas = permissif

    cercles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT,
                               dp=1.2,
                               minDist=max(3.0, 1.1 * rmin),
                               param1=120, param2=param2,
                               minRadius=rmin, maxRadius=rmax)
    if cercles is None:
        return np.array([]), np.empty((0, 3))
    c = cercles[0]
    return 2.0 * c[:, 2] / ppnm, c


# =============================================================================
# 4. DIMENSION FRACTALE PAR BOX-COUNTING
# =============================================================================
def dimension_fractale(masque):
    """
    Dimension fractale 2D (box-counting) du masque des agglomérats.
    Pour des agrégats de nanocarbone (croissance DLCA), Df ≈ 1.7–1.9.
    Retourne (Df, r2) ou (None, None) si trop peu d'échelles disponibles.
    """
    fg = masque > 0
    if not fg.any():
        return None, None
    tailles, comptes = [], []
    s = 2
    while s <= min(fg.shape) // 4:
        h2 = (fg.shape[0] // s) * s
        w2 = (fg.shape[1] // s) * s
        bloc = fg[:h2, :w2].reshape(h2 // s, s, w2 // s, s)
        n_boites = int(bloc.any(axis=(1, 3)).sum())
        if n_boites > 0:
            tailles.append(s)
            comptes.append(n_boites)
        s *= 2
    if len(tailles) < 4:
        return None, None
    x = np.log(np.asarray(tailles, float))
    y = np.log(np.asarray(comptes, float))
    a, b = np.polyfit(x, y, 1)
    y_fit = a * x + b
    ss_res = float(((y - y_fit) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) or 1e-12
    return float(-a), 1.0 - ss_res / ss_tot


# =============================================================================
# CORDE MAXIMALE — version corrigée pour v0.5 (remplace corde_max)
# =============================================================================
def corde_max_v2(cnt):
    """
    Diamètre de Feret exact, O(n²) vectorisé sur l'enveloppe convexe.
    Corrige le bug des rotating calipers de v0.5 : la condition
    `abs(cross) > 0` ne s'annule que pour des arêtes parallèles, ce qui
    rend l'avancée de j incorrecte (voire non bornée) sur un hull générique.
    Avec n_hull <= 200, le O(n²) vectorisé est exact et instantané.
    """
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


# =============================================================================
# ORCHESTRATEUR
# =============================================================================
def analyser_meb(image_bgr, ppm, params=None, verbose=True):
    """
    Analyse complète d'une image MEB de nanocarbones.
      image_bgr : image OpenCV (BGR ou gris)
      ppm       : calibration v0.5 en px/µm  (sera convertie en px/nm)

    Retourne un dict :
      gray_pre, seg (cf. segmenter_hmax), granulo (cf. granulometrie_gris),
      d_hough_nm, cercles_hough, df, df_r2, ppnm, y_databar
    """
    p = dict(NANO_PARAMS)
    if params:
        p.update(params)
    ppnm = ppnm_depuis_ppm(ppm)

    if verbose:
        print(f"[NANO] Calibration: {ppnm:.4f} px/nm "
              f"({1.0/ppnm:.2f} nm/px)" if ppnm > 0 else "[NANO] ppnm is zero!")

    gray_pre, y_bar = pretraiter_meb(image_bgr, p)
    if verbose and y_bar < (image_bgr.shape[0]):
        print(f"[NANO] Databar detected → cropped at y={y_bar}")

    if verbose:
        print("[NANO] 1/4 Sauvola segmentation + h-maxima watershed…")
    seg = segmenter_hmax(gray_pre, ppnm, p)

    if verbose:
        print("[NANO] 2/4 Morphological granulometry (primaries)…")
    gran = granulometrie_gris(gray_pre, ppnm, p)

    if verbose:
        print("[NANO] 3/4 Circular Hough (spherical primaries)…")
    d_h, c_h = primaires_hough(gray_pre, ppnm, p)

    if verbose:
        print("[NANO] 4/4 Fractal dimension…")
    df, r2 = dimension_fractale(seg["masque"])

    if verbose:
        n_a = len(seg["d_aggl_nm"]); n_r = len(seg["d_regs_nm"])
        print(f"[NANO]   {n_a} agglomerate(s), {n_r} watershed region(s), "
              f"{len(d_h)} Hough circle(s)")
        if gran:
            print(f"[NANO]   Granulometry Dx10/50/90 = {gran['dx10']:.1f} / "
                  f"{gran['dx50']:.1f} / {gran['dx90']:.1f} nm (area weighted)")
        if df:
            print(f"[NANO]   Df (box counting) = {df:.2f}  (R²={r2:.3f})")

    return {"gray_pre": gray_pre, "seg": seg, "granulo": gran,
            "d_hough_nm": d_h, "cercles_hough": c_h,
            "df": df, "df_r2": r2, "ppnm": ppnm, "y_databar": y_bar}


# =============================================================================
# FIGURE DE SYNTHÈSE (matplotlib — backend laissé au programme appelant)
# =============================================================================
def figure_resultats(res, titre="PSA Nano / SEM, Materia Nova 2026"):
    """Figure 2×3 : prétraitée | watershed | Hough | histos | granulo | résumé."""
    import matplotlib.pyplot as plt

    gray = res["gray_pre"]
    seg = res["seg"]
    gran = res["granulo"]
    d_h = res["d_hough_nm"]
    ppnm = res["ppnm"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(titre, fontsize=12, fontweight="bold")
    (ax1, ax2, ax3), (ax4, ax5, ax6) = axes

    ax1.imshow(gray, cmap="gray")
    ax1.set_title("① Preprocessed image (databar cropped, NLM, flattened background)")
    ax1.axis("off")

    lws = seg["labels"]
    rng = np.random.default_rng(42)
    lrgb = np.zeros((*lws.shape, 3), np.uint8)
    for lbl in np.unique(lws):
        if lbl == 0:
            continue
        lrgb[lws == lbl] = rng.integers(60, 255, 3)
    ax2.imshow(lrgb)
    ax2.set_title(f"② h-maxima watershed  ({len(seg['d_regs_nm'])} regions, "
                  f"{len(seg['d_aggl_nm'])} agglomerates)")
    ax2.axis("off")

    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    for (x, y, r) in res["cercles_hough"]:
        cv2.circle(overlay, (int(x), int(y)), int(r), (255, 80, 80), 1)
    ax3.imshow(overlay)
    ax3.set_title(f"③ Circular Hough  ({len(d_h)} primaries)")
    ax3.axis("off")

    if len(seg["d_regs_nm"]):
        ax4.hist(seg["d_regs_nm"], bins=20, alpha=0.65,
                 label=f"Watershed (n={len(seg['d_regs_nm'])})")
    if len(d_h):
        ax4.hist(d_h, bins=20, alpha=0.65,
                 label=f"Hough (n={len(d_h)})")
    ax4.set_xlabel("Equivalent diameter (nm)")
    ax4.set_ylabel("Count")
    ax4.set_title("④ Number distributions")
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3, ls=":")

    if gran:
        ax5.bar(gran["d_nm"], gran["ps"],
                width=np.gradient(gran["d_nm"]) * 0.85,
                alpha=0.7, label="Pattern spectrum")
        ax5b = ax5.twinx()
        ax5b.plot(gran["d_nm"], gran["cum"] * 100, lw=2, color="tab:red")
        ax5b.set_ylabel("Cumulative (%)", color="tab:red")
        ax5b.set_ylim(0, 105)
        for q, v in [("Dx10", gran["dx10"]), ("Dx50", gran["dx50"]),
                     ("Dx90", gran["dx90"])]:
            ax5.axvline(v, ls="--", lw=1.2, color="tab:green")
            ax5.text(v, ax5.get_ylim()[1] * 0.9, f"{q}\n{v:.0f}",
                     fontsize=7, ha="center")
        ax5.set_xlabel("Diameter (nm)")
        ax5.set_ylabel("Fraction (area weighted)")
        ax5.set_title("⑤ Morphological granulometry (no segmentation)")
    else:
        ax5.text(0.5, 0.5, "Granulometry unavailable",
                 ha="center", va="center", transform=ax5.transAxes)
    ax5.grid(alpha=0.3, ls=":")

    ax6.axis("off")
    lignes = [("Resolution", f"{1.0/ppnm:.2f} nm/px")]
    if len(seg["d_aggl_nm"]):
        lignes.append(("Agglomerates ECD",
                       f"{np.mean(seg['d_aggl_nm']):.0f} ± "
                       f"{np.std(seg['d_aggl_nm']):.0f} nm "
                       f"(n={len(seg['d_aggl_nm'])})"))
    if len(seg["d_regs_nm"]):
        lignes.append(("WS regions ECD",
                       f"{np.mean(seg['d_regs_nm']):.0f} ± "
                       f"{np.std(seg['d_regs_nm']):.0f} nm"))
    if len(d_h):
        lignes.append(("Hough primaries",
                       f"{np.mean(d_h):.0f} ± {np.std(d_h):.0f} nm "
                       f"(n={len(d_h)})"))
    if gran:
        lignes.append(("Granulometry primaries Dx50", f"{gran['dx50']:.0f} nm"))
        lignes.append(("Granulo Dx10 / Dx90",
                       f"{gran['dx10']:.0f} / {gran['dx90']:.0f} nm"))
    if res["df"]:
        lignes.append(("Fractal dimension Df",
                       f"{res['df']:.2f}  (R²={res['df_r2']:.3f})"))
    y0 = 0.92
    ax6.text(0.02, 0.98, "⑥ Summary", fontsize=11, fontweight="bold",
             va="top", transform=ax6.transAxes)
    for lb, vl in lignes:
        ax6.text(0.04, y0, lb, fontsize=9, color="#555",
                 va="top", transform=ax6.transAxes)
        ax6.text(0.50, y0, vl, fontsize=9, fontweight="bold",
                 va="top", transform=ax6.transAxes)
        y0 -= 0.075
    ax6.text(0.02, 0.04,
             "Area weighting ≈ volume weighting in 2D projection.\n"
             "Hough = number distribution of the primaries.",
             fontsize=7.5, color="#777", va="bottom",
             transform=ax6.transAxes)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# =============================================================================
# DÉMO AUTONOME
# =============================================================================
def _image_synthetique(n_primaires=400, r_px=(6, 14), taille=900, seed=7):
    """Agglomérats synthétiques de disques brillants + bruit + gradient + databar."""
    rng = np.random.default_rng(seed)
    img = np.zeros((taille, taille), np.float32)
    # 6 agglomérats : marches aléatoires de disques, chaque primaire avec
    # une intensité propre + liseré sombre (mime le contraste SE réel)
    for _ in range(6):
        x, y = rng.uniform(120, taille - 120, 2)
        for _ in range(n_primaires // 6):
            r = rng.uniform(*r_px)
            lum = rng.uniform(120, 220)
            cv2.circle(img, (int(x), int(y)), int(r), 0.0, 2)       # liseré
            cv2.circle(img, (int(x), int(y)), max(1, int(r) - 1), lum, -1)
            ang = rng.uniform(0, 2 * math.pi)
            pas = r * rng.uniform(0.9, 1.7)
            x = np.clip(x + pas * math.cos(ang), 40, taille - 40)
            y = np.clip(y + pas * math.sin(ang), 40, taille - 40)
    # gradient d'éclairement + bruit MEB
    gx = np.linspace(0, 50, taille, dtype=np.float32)
    img += gx[None, :] + 20
    img += rng.normal(0, 14, img.shape).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    # databar
    img[-60:, :] = 8
    cv2.putText(img, "SEM 5kV x50k  WD 4.2mm  --- 500 nm ---",
                (30, taille - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
    return img


if __name__ == "__main__":
    import sys
    import matplotlib
    matplotlib.use("Agg")

    if len(sys.argv) > 2:
        chemin, ppm = sys.argv[1], float(sys.argv[2])
        image = cv2.imread(chemin)
    else:
        print("[DEMO] No image given → synthetic image "
              "(usage: python psa_nano_meb.py image.tif ppm_px_per_um)")
        image = _image_synthetique()
        ppm = 500.0          # 0.5 px/nm → disques 6–14 px ≈ 24–56 nm

    res = analyser_meb(image, ppm)
    fig = figure_resultats(res)
    fig.savefig("psa_nano_demo.png", dpi=130)
    print("[DEMO] Figure saved: psa_nano_demo.png")
