# -*- coding: utf-8 -*-
"""PSA, Default parameters and slider ⇄ expert parameter mapping."""
from psa.theme import *  # noqa: F401,F403

# =============================================================================
# PARAMÈTRES PAR DÉFAUT
# =============================================================================
PARAMS_OPTIQUE = {
    "moteur":               "stardist",   # "stardist" (default) | "watershed"
    "diametre_min_um":      8.0,
    "circularite_min":      0.4,
    "clahe_clip":           2.5,
    "clahe_grid":           8,
    "gamma":                0.4,
    "watershed_dist_pct":   0.3,
    "blur_ksize":           5,
    "rolling_ball_radius":  50,
    "rolling_ball_enabled": False,
    "inverser_couleurs":    False,
    "threshold_mode":       "otsu",
    "threshold_manual":     128,
    "otsu_offset":          0.0,     # décalage relatif du seuil Otsu [-0.25..0.25]
}

PARAMS_NANO = dict(nano.NANO_PARAMS) if NANO_OK else {}

# Mode « MEB micro » : mêmes algorithmes que le nano (Sauvola, h-maxima,
# databar, NLM) mais paramétré et rapporté en MICROMÈTRES — poudres
# métalliques 5-200 µm, granulats, etc.
PARAMS_MEB = {
    "moteur":              "stardist",    # "stardist" (default) | "watershed"
    "particules_claires":  True,
    "polarite":            "auto",
    "couper_databar":      True,
    "combler_convexe":     False,   # coque convexe (poudres sphériques)
    "denoise_nlm_h":       7.0,
    "flatten":             True,
    "clahe_clip":          2.0,
    "clahe_grid":          8,
    "sauvola_window":      101,
    "sauvola_k":           0.08,
    "seuil_plancher_rel":  0.55,
    "h_rel":               0.35,
    "fermeture_iter":      2,
    "d_min_um":            2.0,
    "aire_min_px":         50,
    "circ_min":            0.30,
    "circ_aire_min_px":    80,
} if NANO_OK else {}


def params_vers_nano(p):
    """Paramètres MEB micro (µm) → conventions du module nano (nm)."""
    q = dict(p)
    if "d_min_um" in q:
        q["d_min_nm"] = float(q.pop("d_min_um")) * 1000.0
    return q

# =============================================================================
# CURSEURS MÉTIER  ⇄  PARAMÈTRES EXPERTS
# =============================================================================
MOTEURS = {"Deep learning (StarDist)": "stardist",
           "Classic (threshold + watershed)": "watershed"}
MOTEURS_INV = {v: k for k, v in MOTEURS.items()}


def _appliquer_curseurs_classique(params, mode, sens, sep, taille_min,
                       actifs=(True, True, True)):
    """
    Traduit les 3 curseurs technicien (0-100) en paramètres experts.
    `actifs` indique quels curseurs ont réellement été déplacés : un
    curseur non touché ne réécrit PAS ses paramètres (le mapping n'est
    pas une inversion exacte — sans ce garde-fou, entrer dans l'étape
    Réglages modifierait silencieusement la recette, p.ex. le plancher
    Sauvola, et pouvait saturer le masque).
    """
    p = dict(params)
    a_sens, a_sep, a_taille = actifs
    if mode == "fibres":
        # Sensibilité : plus haut → masque plus généreux (couverture, billes,
        # pores, échelle locale) et bords plus faibles acceptés pour la
        # mesure par appariement. Le DIAMÈTRE lui-même ne dépend pas du seuil.
        if a_sens:
            p["seuil_facteur"] = float(np.interp(sens, [0, 100], [1.0, 0.45]))
            # bords plus faibles acceptés (fibres peu contrastées)
            p["bords_k_bruit"] = float(np.interp(sens, [0, 100], [5.0, 2.0]))
            p["sauvola_k"] = float(np.interp(sens, [0, 100], [0.20, 0.02]))
        return p
    if mode in ("nano", "meb"):
        if a_sens:
            p["sauvola_k"] = float(np.interp(sens, [0, 100],
                                             [0.22, 0.03]))
            p["seuil_plancher_rel"] = float(np.interp(sens, [0, 100],
                                                      [0.75, 0.40]))
        if a_sep:
            p["h_rel"] = float(np.interp(sep, [0, 100], [0.65, 0.15]))
        if a_taille:
            if mode == "nano":
                p["d_min_nm"] = float(taille_min)
            else:
                p["d_min_um"] = float(taille_min)
    else:
        if a_sens:
            off = float(np.interp(sens, [0, 100], [-0.25, 0.25]))
            if p.get("inverser_couleurs"):
                off = -off
            p["otsu_offset"] = off
        if a_sep:
            p["watershed_dist_pct"] = float(np.interp(sep, [0, 100],
                                                      [0.60, 0.10]))
        if a_taille:
            p["diametre_min_um"] = float(taille_min)
    return p


def _curseurs_depuis_params_classique(params, mode):
    """Positions initiales des curseurs à partir des paramètres experts."""
    if mode == "fibres":
        sens = float(np.interp(params.get("seuil_facteur", 0.65),
                               [0.45, 1.0], [100, 0]))
        return sens, 50.0, 0.0
    if mode in ("nano", "meb"):
        sens = float(np.interp(params.get("sauvola_k", 0.10),
                               [0.03, 0.22], [100, 0]))
        sep  = float(np.interp(params.get("h_rel", 0.35),
                               [0.15, 0.65], [100, 0]))
        tmin = (float(params.get("d_min_nm", 5.0)) if mode == "nano"
                else float(params.get("d_min_um", 2.0)))
    else:
        off = params.get("otsu_offset", 0.0)
        if params.get("inverser_couleurs"):
            off = -off
        sens = float(np.interp(off, [-0.25, 0.25], [0, 100]))
        sep  = float(np.interp(params.get("watershed_dist_pct", 0.3),
                               [0.10, 0.60], [100, 0]))
        tmin = float(params.get("diametre_min_um", 8.0))
    return sens, sep, tmin


# Gouttes & capsules en microfluidique (v1.7.5) : anneau sombre, intérieur
# aussi clair que le liquide (psa/droplets.py)
PARAMS_GOUTTES = {
    "convention": "ext",          # "ext" bord extérieur | "milieu" de l'anneau
    "hauteur_canal_um": 0.0,      # 0 = sphères ; sinon gouttes écrasées
    "k_anneau": 0.5,              # seuil de l'anneau (× étalement du passe-haut)
    "hough": True,                # rattrapage des anneaux interrompus
    "taille_max": 0.0,
}
K_ANNEAU = (1.0, 0.1)             # curseur Sensibilité 0 → 100


# Filtre de taille minimale désactivable (v1.7) : clé du paramètre selon le
# mode, et neutralisation (valeur 0) quand l'utilisateur coupe le filtre.
# La valeur du curseur est gardée dans "taille_min_memo" pour la restaurer.
CLE_TAILLE_MIN = {"nano": "d_min_nm", "meb": "d_min_um",
                  "optique": "diametre_min_um"}


def cle_taille_min(mode):
    return CLE_TAILLE_MIN.get(mode)


def appliquer_taille_min_off(params, mode):
    """Minimum-size filter switched off ("taille_min_off"): its value is 0,
    so no object is rejected for its physical size (the pixel floor of the
    mode, objects a few pixels across, still applies: their size cannot be
    measured)."""
    cle = cle_taille_min(mode)
    if cle and params.get("taille_min_off"):
        p = dict(params)
        p[cle] = 0.0
        return p
    return params


# StarDist: Sensitivity = object probability threshold, Separation = NMS
# overlap threshold (higher = touching neighbours kept apart more readily).
SD_PROB = (0.85, 0.15)      # sensitivity 0 → 100
SD_NMS = (0.05, 0.60)       # separation 0 → 100
# optimised thresholds of the bundled models (probability, overlap)
SD_SEUILS_MODELE = {"nano": (0.601, 0.40),        # nanocarbon_te
                    "defaut": (0.479, 0.30)}      # 2D_versatile_fluo


def appliquer_curseurs(params, mode, sens, sep, taille_min,
                       actifs=(True, True, True)):
    """Translates the 3 sliders (0-100) into expert parameters. With the
    StarDist engine, Sensitivity and Separation drive the network thresholds
    and leave the watershed parameters untouched."""
    if mode == "gouttes":            # seul le curseur Sensibilité sert
        p = dict(params)
        if actifs[0]:
            p["k_anneau"] = float(np.interp(sens, [0, 100], K_ANNEAU))
        return p
    if params.get("moteur", "stardist" if mode == "nano" else "") == "stardist" \
            and mode in ("meb", "optique", "nano"):
        a_sens, a_sep, a_taille = actifs
        p = _appliquer_curseurs_classique(params, mode, sens, sep, taille_min,
                                          actifs=(False, False, a_taille))
        if a_sens:
            p["stardist_prob"] = float(np.interp(sens, [0, 100], SD_PROB))
        if a_sep:
            p["stardist_nms"] = float(np.interp(sep, [0, 100], SD_NMS))
        return appliquer_taille_min_off(p, mode)
    return appliquer_taille_min_off(
        _appliquer_curseurs_classique(params, mode, sens, sep, taille_min,
                                      actifs), mode)


def curseurs_depuis_params(params, mode):
    if mode == "gouttes":
        k = float(params.get("k_anneau", PARAMS_GOUTTES["k_anneau"]))
        return float(np.interp(k, K_ANNEAU[::-1], [100, 0])), 50.0, 0.0
    sens, sep, tmin = _curseurs_depuis_params_classique(params, mode)
    if params.get("moteur", "stardist" if mode == "nano" else "") == "stardist" \
            and mode in ("meb", "optique", "nano"):
        pr, nm = params.get("stardist_prob"), params.get("stardist_nms")
        # default positions = the optimised thresholds of the model used
        # (Models/stardist/<model>/thresholds.json)
        pr0, nm0 = SD_SEUILS_MODELE.get(mode, SD_SEUILS_MODELE["defaut"])
        pr = pr0 if pr is None else pr
        nm = nm0 if nm is None else nm
        sens = float(np.interp(pr, SD_PROB[::-1], [100, 0]))
        sep = float(np.interp(nm, SD_NMS, [0, 100]))
    return sens, sep, tmin


__all__ = [_n for _n in dir() if not _n.startswith("__")]
