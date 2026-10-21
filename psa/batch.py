# -*- coding: utf-8 -*-
"""PSA, Batch analysis of a folder.

v1.7.7: the images of a batch are usually COMPARED (same magnification,
different process conditions). For every image the full single-image export
is written (report PDF, CSV, parameters), and a COMPARISON report sets the
images side by side: key results per image, overlaid distributions, box
plots, medians with their 95 % confidence interval, Kruskal-Wallis test and
pairwise Mann-Whitney tests (Holm correction), mode-specific indicators,
thumbnails. The pooled statistics are kept on one page, meaningful only
when the images are replicates of the same sample."""
# Imports explicites (v1.8.0) : ce module est vérifiable par pyflakes.
import datetime
import json
import os
from pathlib import Path

import cv2
import numpy as np
from matplotlib.figure import Figure

from psa.base import (APP_NAME, APP_VENDOR, APP_VERSION, EXT_IMAGES,
                      journal)
from psa.detection import calibration_metadonnees, charger_image, \
    filtrer_taille_max
from psa.modes import MODE_NOMS, mode_de
from psa.sphero import sphericite
from psa.stats import dx_volume
from psa.report import (A4, ENCRE, LIGNE_TABLE, NAVY, PDF_C, REFS_NORMES,
                        REFS_STARDIST, STYLE_SCI, _entete, _fmt_v, _legende,
                        _tableau, _val, avertissements_qualite,
                        axe_log_lisible, exporter_resultats,
                        image_annotee_rapport, lignes_methode, lignes_resume,
                        reglages_graphe, safe_name, tracer_psd, write_csv,
                        fmt)

import copy

# =============================================================================
# DÉTECTION D'UNE IMAGE DU LOT
# =============================================================================
def _detecter_image(img, sm):
    """Same detection as the interface, for the mode of the recipe."""
    return mode_de(sm).detecter(img, sm.ppm, dict(sm.params))


def _vignette(img, res, sm, cote=520):
    """Annotated thumbnail (same drawing as the single-image report)."""
    s = copy.copy(sm)
    s.image_bgr, s.resultats = img, res
    try:
        vis = image_annotee_rapport(s)
    except Exception:
        vis = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = vis.shape[:2]
    k = cote / max(h, w)
    return cv2.resize(vis, (max(1, int(w * k)), max(1, int(h * k))),
                      interpolation=cv2.INTER_AREA) if k < 1 else vis


_CLES_IMAGE = ("mode", "unite", "diametres", "n_particules", "n_bord",
               "n_trop_grands", "score_ws", "statut_ws", "msg_ws",
               "circularites", "aspects", "solidites", "moteur", "stardist",
               "taille_max", "convention", "hauteur_canal_um", "um_par_px",
               "gouttes_stats", "ressemble", "population")


def _alleger(res, sm):
    """Per-image summary kept for the batch (the masks are not kept)."""
    r = {k: res[k] for k in _CLES_IMAGE if k in res}
    r["diametres"] = np.asarray(res.get("diametres", []), float)
    if res.get("mode") == "gouttes":
        g = res["gouttes"]
        r["gouttes"] = {k: np.asarray(g[k]) for k in
                        ("d_milieu", "d_ext", "bord", "rms", "aspect", "cx")}
    if res.get("mode") == "fibres":
        fr = res.get("fibres") or {}
        r["fibres"] = {k: fr.get(k) for k in
                       ("n_billes", "densite_billes_mm2", "alignement",
                        "orientation_deg", "porosite_pct", "n_pores",
                        "porosite_fibre_pct", "pore_d_median_um",
                        "diam_median_um")}
        r["fibres"]["pores"] = fr.get("pores_mask") is not None
    if res.get("mode") == "nano":
        nr = res.get("nano") or {}
        r["nano"] = {"methode": nr.get("methode"),
                     "imagerie": nr.get("imagerie"),
                     "agregats": nr.get("agregats", [])}
    sph = sphericite(res, getattr(sm, "sphero", None))
    if sph:
        r["_sph_image"] = {"classes": sph["classes"],
                           "porteuses": sph["porteuses"],
                           "d": np.asarray(res["diametres"], float),
                           "criteres": sph["criteres"]}
    return r


# =============================================================================
# AGRÉGATION
# =============================================================================
def agreger_lot(entrees, sm):
    """Result of the whole batch in the common result format (diameters of
    all the images together), so that the standard summary rows, method
    table and graphs apply unchanged."""
    ok = [e["res"] for e in entrees if e["res"] is not None]
    mode = sm.mode
    u = ok[0]["unite"] if ok else sm.unite()

    def cat(k):
        a = [np.asarray(r.get(k, []), float) for r in ok]
        a = [x for x in a if x.size]
        return np.concatenate(a) if a else np.zeros(0)

    d = cat("diametres")
    n_img = [r["n_particules"] for r in ok]
    scores = [r["score_ws"] for r in ok]
    agg = {"mode": mode, "unite": u, "diametres": d,
           "n_particules": int(d.size),
           "n_bord": int(sum(r.get("n_bord", 0) for r in ok)),
           "n_trop_grands": int(sum(r.get("n_trop_grands", 0) for r in ok)),
           "taille_max": float(sm.params.get("taille_max", 0) or 0),
           "score_ws": float(np.average(scores, weights=np.maximum(n_img, 1)))
           if ok else 0.0,
           "statut_ws": "good", "msg_ws": "", "lot": True}
    agg["statut_ws"] = ("good" if agg["score_ws"] >= 70 else
                        "warn" if agg["score_ws"] >= 45 else "bad")
    for k in ("circularites", "aspects", "solidites"):
        v = cat(k)
        if v.size == d.size and v.size:
            agg[k] = v
    if ok:
        for k in ("moteur", "stardist", "convention", "hauteur_canal_um",
                  "population"):
            if k in ok[0]:
                agg[k] = ok[0][k]
    if ok:                        # indicateurs propres au mode
        mode_de(mode).agreger(agg, ok, sm)
    return agg


# =============================================================================
# COMPARAISON DES IMAGES
# =============================================================================
def _court(nom, n=22):
    import textwrap
    return textwrap.shorten(os.path.splitext(nom)[0], n, placeholder="…")


def mesures_image(e, sm):
    """Key results of one image (dict), common and mode-specific."""
    r = e["res"]
    d = np.asarray(r["diametres"], float) if r is not None else np.zeros(0)
    m = {"image": e["image"], "n": int(d.size), "ok": r is not None,
         "score": r["score_ws"] if r is not None else None}
    if d.size:
        m.update(mean=float(d.mean()), sd=float(d.std(ddof=1)) if d.size > 1 else 0.0,
                 d10=float(np.percentile(d, 10)), d50=float(np.median(d)),
                 d90=float(np.percentile(d, 90)),
                 cv=float(100 * d.std(ddof=1) / d.mean()) if d.size > 1 else 0.0)
        if sm.mode in ("optique", "meb", "nano"):
            m["dv50"] = float(dx_volume(d)[1])
        lo, hi = ic_mediane(d)
        m.update(d50_lo=lo, d50_hi=hi)
    if r is None:
        return m
    if r.get("_sph_image"):
        from psa.sphero import taux_sphero
        x = r["_sph_image"]
        t = taux_sphero(x["classes"], x["porteuses"], x["d"], x["criteres"])
        m.update(sph_n=t["taux_nombre_hors_sat"], sph_m=t["taux_masse_hors_sat"],
                 sat_n=t["satellites_nombre"])
    if sm.mode == "gouttes":
        g = r.get("gouttes_stats") or {}
        if g.get("n", 0) > 1:
            m.update(vol=g["v_moy"] * 1e-3,
                     gap=100 * abs(g["ecart_convention"]) / g["d_moy"])
    if sm.mode == "fibres":
        fr = r.get("fibres") or {}
        m.update(beads=fr.get("densite_billes_mm2"), align=fr.get("alignement"),
                 poro=fr.get("porosite_pct"))
    return m


def ic_mediane(d, conf=0.95):
    """Distribution-free confidence interval of the median (order
    statistics, binomial law)."""
    from scipy.stats import binom
    d = np.sort(np.asarray(d, float))
    n = d.size
    if n < 6:
        return (float("nan"), float("nan"))
    a = (1 - conf) / 2
    lo = int(binom.ppf(a, n, 0.5))
    hi = int(binom.ppf(1 - a, n, 0.5))
    return float(d[max(lo - 1, 0)]), float(d[min(hi, n - 1)])


def comparer_images(entrees, n_min=5):
    """Kruskal-Wallis over the images, then pairwise Mann-Whitney tests with
    the Holm correction. Returns None with fewer than 2 usable images."""
    from scipy.stats import kruskal, mannwhitneyu
    gr = [(e["image"], np.asarray(e["res"]["diametres"], float))
          for e in entrees if e["res"] is not None
          and len(e["res"]["diametres"]) >= n_min]
    if len(gr) < 2:
        return None
    H, p_kw = kruskal(*[g for _, g in gr])
    k = len(gr)
    paires, p_brut = [], []
    for i in range(k):
        for j in range(i + 1, k):
            p = mannwhitneyu(gr[i][1], gr[j][1], alternative="two-sided").pvalue
            paires.append((i, j))
            p_brut.append(float(p))
    # correction de Holm (contrôle du risque global sur toutes les paires)
    ordre = np.argsort(p_brut)
    m_ = len(p_brut)
    p_holm = np.empty(m_)
    cour = 0.0
    for r_, idx in enumerate(ordre):
        cour = max(cour, min(1.0, (m_ - r_) * p_brut[idx]))
        p_holm[idx] = cour
    P = np.ones((k, k))
    D = np.zeros((k, k))
    med = np.array([np.median(g) for _, g in gr])
    for (i, j), p in zip(paires, p_holm):
        P[i, j] = P[j, i] = p
    for i in range(k):
        for j in range(k):
            D[i, j] = 100 * (med[i] / med[j] - 1)
    return {"noms": [n for n, _ in gr], "H": float(H), "p_kw": float(p_kw),
            "P": P, "D": D, "medianes": med,
            "paires": [(gr[i][0], gr[j][0], p_brut[q], float(p_holm[q]),
                        float(D[i, j])) for q, (i, j) in enumerate(paires)]}


def _cmap_div():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "psa_div", ["#2c6fbb", "#f4f6f8", "#c8453b"])


def _cmp_tableau(c, n_avant):
    """Page(s) 1: key results per image."""
    s_lot = c.s_lot
    entrees = c.entrees
    dossier_lot = c.dossier_lot
    u = c.u
    mode = c.mode
    fib = c.fib
    gt = c.gt
    mes = c.mes
    _fv = c.fv
    pages = []
    f = Figure(figsize=A4)
    info = [["Batch folder", Path(dossier_lot).name],
            ["Images analysed", f"{len([e for e in entrees if e['res'] is not None])}"
             f" / {len(entrees)}"],
            ["Operator", s_lot.operateur or "–"],
            ["Date of analysis", f"{datetime.datetime.now():%Y-%m-%d %H:%M}"],
            ["Recipe", s_lot.recette or "–"],
            ["Analysis mode", MODE_NOMS.get(mode, mode)],
            ["Calibration", f"{fmt(s_lot.ppm, '.5g')} px/µm (same for every "
             "image)"],
            ["Individual reports", "one folder per image (images/)"]]
    h_i = LIGNE_TABLE * (len(info) + 1) + 0.026
    _tableau(f.add_axes([0.07, 0.885 - h_i, 0.55, h_i]), info,
             "Batch and analysis", ("Item", "Value"), (0.40, 0.60))
    col = ["Image", "n", f"Mean ({u})", f"D10 ({u})", f"D50 ({u})",
           f"D90 ({u})", "CV"]
    larg = [0.27, 0.07, 0.12, 0.11, 0.11, 0.11, 0.08]
    cles = ["n", "mean", "d10", "d50", "d90", "cv"]
    if mode in ("optique", "meb") and any("sph_n" in m for m in mes):
        col += ["Sph. %"]; larg += [0.08]; cles += ["sph_n"]
    if gt:
        col += ["Vol. (pL)"]; larg += [0.09]; cles += ["vol"]
    if fib:
        col += ["Beads/mm²", "Align."]; larg += [0.10, 0.07]; cles += ["beads", "align"]
    col += ["Score"]; larg += [0.08]; cles += ["score"]
    larg = list(np.array(larg) / sum(larg))
    rows = []
    for m in mes:
        if not m["ok"]:
            rows.append([_court(m["image"], 30)] + ["–"] * (len(col) - 2)
                        + ["not analysed"])
            continue
        ligne = [_court(m["image"], 30)]
        for k in cles:
            v = m.get(k)
            ligne.append(f"{v}" if k == "n" else
                         (f"{v:.1f} %" if (k == "cv" and v is not None) else
                          f"{v:.0f}/100" if (k == "score" and v is not None)
                          else _fv(v, ".3g" if k in ("align",) else ".4g")))
        rows.append(ligne)
    y = 0.885 - h_i - 0.05
    par = 28
    bloc, reste = rows[:par], rows[par:]
    h_t = LIGNE_TABLE * (len(bloc) + 1) + 0.026
    _tableau(f.add_axes([0.07, y - h_t, 0.86, h_t]), bloc,
             "Key results per image", col, larg, fs=7)
    _legende(f, 0.07, y - h_t - 0.012,
             "Table 1. Number-weighted statistics of each image (particles "
             "touching the border excluded). " +
             ("Sph. %: spheroidization rate by number, satellites "
              "excluded. " if "sph_n" in cles else "") +
             ("Vol.: mean droplet volume. " if gt else "") +
             "Full results of each image: its individual report.")
    pages.append((f, "1. Comparison of the images"))
    while reste:
        f = Figure(figsize=A4)
        bloc, reste = reste[:45], reste[45:]
        h_t = LIGNE_TABLE * (len(bloc) + 1) + 0.026
        _tableau(f.add_axes([0.07, 0.885 - h_t, 0.86, h_t]), bloc,
                 "Key results per image (continued)", col, larg, fs=7)
        pages.append((f, "1. Comparison of the images (continued)"))
    return pages


def _cmp_distributions(c, n_avant):
    """Page 2: overlaid cumulative curves, box plots."""
    u = c.u
    fib = c.fib
    gt = c.gt
    xsc = c.xsc
    ok = c.ok
    mes = c.mes
    nom_g = c.nom_g
    couleurs = c.couleurs
    pages = []
    f = Figure(figsize=A4)
    a = f.add_axes([0.11, 0.56, 0.74, 0.30])
    for i, e in enumerate(ok):
        v = np.sort(np.asarray(e["res"]["diametres"], float))
        if not fib and not gt:
            w = v ** 3
            yy = 100 * np.cumsum(w) / w.sum()
        else:
            yy = 100 * np.arange(1, v.size + 1) / v.size
        a.step(v, yy, where="post", lw=1.2, color=couleurs(i),
               label=_court(e["image"]))
    if xsc == "log":
        a.set_xscale("log")
        axe_log_lisible(a)
    a.set_xlabel(f"{nom_g} ({u})")
    a.set_ylabel("Cumulative " + ("volume" if not (fib or gt) else "number")
                 + " (%)")
    a.set_ylim(0, 100)
    a.grid(alpha=0.3)
    a.legend(fontsize=6.5, ncol=2 if len(ok) > 8 else 1, loc="lower right")
    _legende(f, 0.07, 0.515, "Figure 1. Cumulative "
             + ("volume-weighted (d³) " if not (fib or gt) else "number ")
             + f"{nom_g.lower()} distribution of each image. A curve "
             "shifted to the right = coarser.")
    a = f.add_axes([0.11, 0.17, 0.80, 0.27])
    donnees = [np.asarray(e["res"]["diametres"], float) for e in ok]
    bp = a.boxplot(donnees, widths=0.55, showfliers=False, patch_artist=True)
    for i, b_ in enumerate(bp["boxes"]):
        b_.set_facecolor(couleurs(i))
        b_.set_alpha(0.55)
    for i, m in enumerate([m for m in mes if m["ok"] and m["n"]], 1):
        if np.isfinite(m.get("d50_lo", np.nan)):
            a.plot([i, i], [m["d50_lo"], m["d50_hi"]], color=ENCRE, lw=2.2)
    a.set_xticks(range(1, len(ok) + 1))
    a.set_xticklabels([_court(e["image"], 18) for e in ok], rotation=60,
                      ha="right", fontsize=6.5)
    a.set_ylabel(f"{nom_g} ({u})")
    if xsc == "log":
        a.set_yscale("log")
    a.grid(axis="y", alpha=0.3)
    _legende(f, 0.07, 0.035 + 0.06,
             "Figure 2. Box plots per image (box: quartiles; line: median; "
             "whiskers: 1.5 × IQR, outliers not drawn). Black bar: 95 % "
             "confidence interval of the median: two images whose bars do "
             "not overlap have clearly different medians.")
    pages.append((f, "2. Distributions compared"))
    return pages


def _cmp_tests(c, n_avant):
    """Page 3: Kruskal-Wallis, pairwise matrix, mode indicators."""
    import textwrap
    fib = c.fib
    gt = c.gt
    mes = c.mes
    cmp_ = c.cmp_
    couleurs = c.couleurs
    pages = []
    f = Figure(figsize=A4)
    if cmp_:
        k = len(cmp_["noms"])
        txt = (f"Kruskal-Wallis test over the {k} images: H = "
               f"{cmp_['H']:.3g}, p = {cmp_['p_kw']:.2g}: "
               + ("the size distributions differ between the images."
                  if cmp_["p_kw"] < 0.05 else
                  "no significant difference between the images."))
        f.text(0.07, 0.875, "\n".join(textwrap.wrap(txt, 105)), fontsize=8.5,
               va="top", color=ENCRE)
        if k <= 25:
            a = f.add_axes([0.22, 0.38, 0.62, 0.44])
            lim = max(5.0, float(np.nanmax(np.abs(cmp_["D"]))))
            im = a.imshow(cmp_["D"], cmap=_cmap_div(), vmin=-lim, vmax=lim)
            for i in range(k):
                for j in range(k):
                    if i == j:
                        continue
                    p = cmp_["P"][i, j]
                    et = "***" if p < 0.001 else "**" if p < 0.01 else \
                        "*" if p < 0.05 else ""
                    a.text(j, i, f"{cmp_['D'][i, j]:+.0f}{et}", ha="center",
                           va="center", fontsize=6 if k > 10 else 7.5,
                           color=ENCRE)
            noms_c = [_court(n, 16) for n in cmp_["noms"]]
            a.set_xticks(range(k))
            a.set_yticks(range(k))
            a.set_xticklabels(noms_c, rotation=60, ha="right", fontsize=6.5)
            a.set_yticklabels(noms_c, fontsize=6.5)
            cb = f.colorbar(im, ax=a, fraction=0.04, pad=0.02)
            cb.set_label("Median of the row vs column (%)", fontsize=7.5)
            _legende(f, 0.07, 0.30,
                     "Figure 3. Difference between the median diameters of "
                     "each pair of images (row compared with column, %). "
                     "Stars: Mann-Whitney test with the Holm correction "
                     "for the number of pairs (* p < 0.05, ** p < 0.01, "
                     "*** p < 0.001). With many particles a few % can be "
                     "significant: judge the size of the difference as well."
                     + (" Fibre sites along the same fibre are not "
                        "independent: the p-values are optimistic."
                        if fib else ""))
        else:
            f.text(0.07, 0.80, f"{k} images: pairwise matrix not drawn "
                   "(see batch_pairwise.csv).", fontsize=8.5, color=ENCRE)
    else:
        f.text(0.07, 0.86, "Fewer than 2 images with enough objects: no "
               "statistical comparison.", fontsize=8.5, color=ENCRE)
    # indicateurs propres au mode
    ind = []
    if any("sph_n" in m for m in mes):
        ind = [("sph_n", "Spheroidization rate, number (%)"),
               ("sph_m", "Spheroidization rate, mass (%)")]
    elif gt:
        ind = [("cv", "Size CV (%)"), ("vol", "Mean droplet volume (pL)")]
    elif fib:
        ind = [("beads", "Bead density (/mm²)"), ("align", "Alignment (0-1)")]
    mm = [m for m in mes if m["ok"] and m["n"]]
    for q, (cle, titre) in enumerate(ind):
        a = f.add_axes([0.10 + q * 0.46, 0.08, 0.36, 0.15])
        v = [m.get(cle) if m.get(cle) is not None else np.nan for m in mm]
        a.bar(range(len(mm)), v, color=[couleurs(i) for i in range(len(mm))],
              alpha=0.8)
        a.set_xticks(range(len(mm)))
        a.set_xticklabels([_court(m["image"], 10) for m in mm], rotation=70,
                          ha="right", fontsize=5.5)
        a.set_title(titre, fontsize=8)
        a.grid(axis="y", alpha=0.3)
    pages.append((f, "3. Statistical comparison"))
    return pages


def _cmp_vignettes(c, n_avant):
    """Page(s) 4: annotated thumbnails."""
    entrees = c.entrees
    pages = []
    vign = [(e["image"], e["vignette"]) for e in entrees
            if e.get("vignette") is not None]
    for i0 in range(0, min(len(vign), 36), 12):
        f = Figure(figsize=A4)
        for j, (nom, im) in enumerate(vign[i0:i0 + 12]):
            c_, l_ = j % 3, j // 3
            a = f.add_axes([0.07 + c_ * 0.293, 0.70 - l_ * 0.195, 0.27, 0.17])
            a.imshow(im, aspect="equal")
            a.set_xticks([])
            a.set_yticks([])
            a.set_title(_court(nom, 34), fontsize=6.5, color=ENCRE, pad=2)
        if i0 == 0:
            _legende(f, 0.07, 0.085, "Figure 4. Detections on every image "
                     "(same drawing as the individual reports)."
                     + (" Only the first 36 images are shown."
                        if len(vign) > 36 else ""))
        pages.append((f, "4. Annotated images" if i0 == 0 else
                      "4. Annotated images (continued)"))
    return pages


def _cmp_annexe(c, n_avant):
    """Annex: all the images pooled (replicates only)."""
    import textwrap
    s_lot = c.s_lot
    res = c.res
    d = c.d
    u = c.u
    fib = c.fib
    gt = c.gt
    xsc = c.xsc
    nom_g = c.nom_g
    pages = []
    f = Figure(figsize=A4)
    f.text(0.07, 0.875, "\n".join(textwrap.wrap(
        "All the images together. Only meaningful when the images are "
        "replicates of the SAME sample (several fields of one powder, "
        "one condition): the pooled distribution then has a larger n and "
        "a better representativeness. Do not pool images of different "
        "process conditions.", 105)), fontsize=8.5, va="top",
        color=PDF_C["red"])
    lignes = lignes_resume(s_lot)
    sm_ = {k_: (v, un) for k_, _, v, un in lignes}
    lab = {k_: lb for k_, lb, _, _ in lignes}
    cl = [k_ for k_ in mode_de(res).cles_resultats(res)
          if k_ not in ("quality",)]
    resu = [[lab[k_], _fmt_v(*sm_[k_])] for k_ in cl if k_ in sm_]
    h_r = LIGNE_TABLE * (len(resu) + 1) + 0.026
    _tableau(f.add_axes([0.07, 0.80 - h_r, 0.45, h_r]), resu,
             "Pooled results", ("Quantity", "Value"), (0.6, 0.4))
    if d.size:
        a = f.add_axes([0.11, 0.12, 0.74, 0.28])
        tracer_psd(a, d, u, "nombre" if (fib or gt) else "volume", xsc,
                   nom_g, g={"xscale": xsc})
        _legende(f, 0.07, 0.075, f"Figure 5. Pooled "
                 f"{'number' if (fib or gt) else 'volume'}-weighted "
                 f"distribution (n = {d.size}).")
    pages.append((f, f"{n_avant + len(pages) + 1}. Annex: images pooled"))
    return pages


def _cmp_methode(c, n_avant):
    """Last page: method, quality, references."""
    import textwrap
    s_lot = c.s_lot
    entrees = c.entrees
    res = c.res
    d = c.d
    u = c.u
    fib = c.fib
    sd = c.sd
    pages = []
    f = Figure(figsize=A4)
    meth = lignes_methode(s_lot, res, d, u, fib, sd)
    meth.insert(0, ["Batch", f"{len(entrees)} images, same recipe and "
                    "calibration; one individual report per image"])
    meth.append(["Comparison", "Kruskal-Wallis, pairwise Mann-Whitney "
                 "(Holm), 95 % CI of the median (order statistics)"])
    h_m = LIGNE_TABLE * (len(meth) + 1) + 0.026
    _tableau(f.add_axes([0.07, 0.885 - h_m, 0.86, h_m]), meth, "Method",
             ("Item", "Setting"), (0.25, 0.75))
    y = 0.885 - h_m - 0.035
    f.text(0.07, y, "Quality assessment", fontsize=9.5, fontweight="bold",
           color=NAVY)
    q = []
    for e in entrees:
        if e["res"] is None:
            q.append(f"Warning: {e['image']} not analysed ({e['err']}).")
            continue
        if e.get("avert_calib"):
            q.append(f"Warning: {_court(e['image'], 40)}: {e['avert_calib']}")
        for av in avertissements_qualite(e["res"]):
            q.append(f"Warning: {_court(e['image'], 40)}: {av}")
    if not q:
        q = ["No warning on any image."]
    for t in q[:24]:
        for ln in textwrap.wrap(t, 110):
            y -= 0.017
            f.text(0.08, y, ln, fontsize=7.6, color=PDF_C["red"]
                   if t.startswith("Warning") else ENCRE)
    if len(q) > 24:
        y -= 0.017
        f.text(0.08, y, f"… and {len(q) - 24} more (see the individual "
               "reports).", fontsize=7.6, color=ENCRE)
    y -= 0.035
    f.text(0.07, y, "References", fontsize=9.5, fontweight="bold", color=NAVY)
    refs = (REFS_STARDIST if sd else []) + REFS_NORMES
    for i, r in enumerate(refs, 1):
        y -= 0.019
        f.text(0.075, y, f"[{i}]", fontsize=7.6, color=ENCRE)
        for j, ln in enumerate(textwrap.wrap(r, 104)):
            if j:
                y -= 0.015
            f.text(0.105, y, ln, fontsize=7.6, color=ENCRE)
    pages.append((f, f"{n_avant + len(pages) + 1}. Method, quality and references"))
    return pages


def _fv(v, spec=".4g"):
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) \
        else fmt(v, spec)


PAGES_COMPARAISON = (_cmp_tableau, _cmp_distributions, _cmp_tests,
                     _cmp_vignettes, _cmp_annexe, _cmp_methode)


def pages_rapport_comparaison(s_lot, entrees, dossier_lot):
    """A4 pages of the comparison report (same layout as the single-image
    report; one function per page, v1.8.0)."""
    import matplotlib.pyplot  # noqa: F401  (mathtext, contexte rc)
    from types import SimpleNamespace
    res = s_lot.resultats
    ok = [e for e in entrees if e["res"] is not None and len(e["res"]["diametres"])]
    c = SimpleNamespace(
        s_lot=s_lot, entrees=entrees, dossier_lot=dossier_lot, res=res,
        d=np.asarray(res["diametres"], float), u=res["unite"], mode=res["mode"],
        fib=res["mode"] == "fibres", gt=res["mode"] == "gouttes",
        sd=res.get("moteur") == "stardist", xsc=reglages_graphe(s_lot)["xscale"],
        ok=ok, mes=[mesures_image(e, s_lot) for e in entrees],
        cmp_=comparer_images(entrees), nom_g=mode_de(res).nom_diametre_de(res),
        couleurs=matplotlib.colormaps["tab20"].resampled(max(len(ok), 2)),
        fv=_fv)
    pages = []
    with matplotlib.rc_context(STYLE_SCI):
        for construire in PAGES_COMPARAISON:
            pages += construire(c, len(pages))
        figs = []
        n = len(pages)
        for i, (fg, titre) in enumerate(pages, 1):
            _entete(fg, s_lot, titre, i, n, titre_doc="Batch comparison report")
            figs.append(fg)
    return figs


def ecrire_rapport_lot(s_lot, entrees, sortie):
    """Comparison report PDF + CSV (per image, per object, pairwise tests,
    pooled summary) + parameters."""
    from matplotlib.backends.backend_pdf import PdfPages
    res = s_lot.resultats
    u = res["unite"]
    fichiers = []
    figs = pages_rapport_comparaison(s_lot, entrees, sortie)
    f = os.path.join(sortie, "batch_comparison_report.pdf")
    with PdfPages(f) as pdf:
        for fg in figs:
            pdf.savefig(fg)
        m = pdf.infodict()
        m["Title"] = f"Batch comparison report, {s_lot.nom_echantillon}"
        m["Author"] = s_lot.operateur or APP_VENDOR
        m["Creator"] = f"{APP_NAME} {APP_VERSION}"
    fichiers.append(f)
    head = [f"{APP_NAME} {APP_VERSION}", f"batch: {s_lot.nom_echantillon}",
            f"exported: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
            f"operator: {s_lot.operateur or '–'}",
            f"recipe: {s_lot.recette or '–'}",
            f"mode: {MODE_NOMS.get(res['mode'], res['mode'])}",
            f"calibration: {fmt(s_lot.ppm, '.6g')} px/µm"]
    mes = [mesures_image(e, s_lot) for e in entrees]
    cles = ["n", "mean", "sd", "d10", "d50", "d50_lo", "d50_hi", "d90", "cv",
            "dv50", "sph_n", "sph_m", "sat_n", "vol", "gap", "beads", "align",
            "poro", "score"]
    cles = [k for k in cles if any(k in m for m in mes)]
    noms_col = {"d50_lo": f"d50_ci95_low_{u}", "d50_hi": f"d50_ci95_high_{u}",
                "cv": "cv_pct", "sph_n": "sphero_number_pct",
                "sph_m": "sphero_mass_pct", "sat_n": "satellites_pct",
                "vol": "mean_volume_pL", "gap": "convention_gap_pct",
                "beads": "beads_per_mm2", "align": "alignment",
                "poro": "mat_porosity_pct"}
    f = os.path.join(sortie, "batch_images.csv")
    write_csv(f, head, ["image", "status"] + [noms_col.get(k, f"{k}_{u}" if k in
                                               ("mean", "sd", "d10", "d50",
                                                "d90", "dv50") else k)
                                              for k in cles],
              [[m["image"], "ok" if m["ok"] else
                next(e["err"] for e in entrees if e["image"] == m["image"])]
               + [fmt(m[k], ".6g") if m.get(k) is not None else "–" for k in cles]
               for m in mes])
    fichiers.append(f)
    f = os.path.join(sortie, "batch_objects.csv")
    write_csv(f, head, ["image", "object", f"diameter_{u}"],
              [[e["image"], i + 1, fmt(v, ".6g")] for e in entrees
               if e["res"] is not None
               for i, v in enumerate(e["res"]["diametres"])])
    fichiers.append(f)
    cmp_ = comparer_images(entrees)
    if cmp_:
        f = os.path.join(sortie, "batch_pairwise.csv")
        write_csv(f, head + [f"Kruskal-Wallis H = {cmp_['H']:.6g}, "
                             f"p = {cmp_['p_kw']:.6g}"],
                  ["image_A", "image_B", "median_diff_A_vs_B_pct",
                   "p_mann_whitney", "p_holm"],
                  [[a, b, fmt(dd, ".4g"), fmt(p, ".4g"), fmt(ph, ".4g")]
                   for a, b, p, ph, dd in cmp_["paires"]])
        fichiers.append(f)
    lignes = lignes_resume(s_lot)
    f = os.path.join(sortie, "batch_pooled_summary.csv")
    write_csv(f, head + ["all images pooled: meaningful only for replicates "
                         "of the same sample"],
              ["quantity", "value", "unit"],
              [[lb, _val(v, ".8g"), un] for _, lb, v, un in lignes])
    fichiers.append(f)
    f = os.path.join(sortie, "batch_parameters.json")
    with open(f, "w", encoding="utf-8") as fh:
        json.dump({"app": f"{APP_NAME} {APP_VERSION}", "mode": res["mode"],
                   "recipe": s_lot.recette, "calibration_px_per_um": s_lot.ppm,
                   "images": [e["image"] for e in entrees],
                   "params": {k: (v if isinstance(v, (int, float, str, bool,
                                                      type(None))) else str(v))
                              for k, v in s_lot.params.items()}},
                  fh, indent=2, ensure_ascii=False)
    fichiers.append(f)
    return fichiers


# =============================================================================
# MODE LOT  (dossier entier, même recette → distribution agrégée)
# =============================================================================
def _rapport_image(chemin, img, res, sm, dossier_images):
    """Full single-image export (report PDF, CSV, summary, parameters), in
    images/<image name>/, exactly as from the interface."""
    s = copy.copy(sm)
    s.chemin, s.image_bgr, s.resultats = str(chemin), img, res
    s.echantillon = ""                     # nom d'échantillon = nom d'image
    stem = safe_name(Path(chemin).stem)
    d = os.path.join(dossier_images, stem)
    try:
        exporter_resultats(d, stem, s, formats=("pdf",))
    except Exception as e:                 # un rapport raté n'arrête pas le lot
        journal("error", f"batch: report of {chemin.name} failed: {e}")


def analyser_lot(dossier, session_modele, progress=None,
                 rapports_individuels=True):
    """Applies the current recipe and calibration to every image of the
    folder: full individual report of each image (images/<name>/) and a
    comparison report. Returns (all diameters, rows per image, output
    folder)."""
    dossier = Path(dossier)
    images = sorted([f for f in dossier.iterdir()
                     if f.suffix.lower() in EXT_IMAGES])
    if not images:
        raise RuntimeError("No image in this folder.")
    sm = session_modele
    entrees, lignes = [], []
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sortie = dossier / f"PSA_batch_{ts}"
    dossier_images = sortie / "images"
    dossier_images.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(images):
        if progress:
            progress(100 * i / len(images), f"{f.name} ({i+1}/{len(images)})")
        img, _note = charger_image(f)
        if img is None:
            entrees.append({"image": f.name, "res": None, "err": "unreadable",
                            "vignette": None})
            lignes.append((f.name, 0, "unreadable"))
            continue
        # garde-fou : calibration lue dans les métadonnées de l'image
        ppm_m, _src = calibration_metadonnees(f)
        av_cal = None
        if ppm_m and sm.ppm > 0 and abs(ppm_m / sm.ppm - 1) > 0.02:
            av_cal = (f"its metadata give {ppm_m:.4g} px/µm, the batch uses "
                      f"{sm.ppm:.4g} px/µm: different magnification? Its sizes "
                      "are wrong by " f"{100 * (ppm_m / sm.ppm - 1):+.0f} %.")
            journal("warning", f"batch: {f.name}: {av_cal}")
        try:
            res = filtrer_taille_max(_detecter_image(img, sm), sm.params)
            if rapports_individuels:
                _rapport_image(f, img, res, sm, dossier_images)
            entrees.append({"image": f.name, "res": _alleger(res, sm),
                            "err": None, "vignette": _vignette(img, res, sm),
                            "avert_calib": av_cal})
            lignes.append((f.name, res["n_particules"],
                           f"{res['score_ws']:.0f}/100"))
        except Exception as e:
            journal("error", f"batch: {f.name} failed: {e}")
            entrees.append({"image": f.name, "res": None,
                            "err": f"error: {e}", "vignette": None})
            lignes.append((f.name, 0, f"error: {e}"))

    s_lot = copy.copy(sm)
    s_lot.resultats = agreger_lot(entrees, sm)
    s_lot.chemin = str(dossier)
    s_lot.echantillon = (sm.echantillon or "").strip() or dossier.name
    s_lot.image_bgr = None
    if progress:
        progress(95, "Comparison report…")
    if s_lot.resultats["n_particules"]:
        ecrire_rapport_lot(s_lot, entrees, str(sortie))
    if progress:
        progress(100, "Batch finished")
    return s_lot.resultats["diametres"], lignes, str(sortie)


__all__ = [_n for _n in dir() if not _n.startswith("__")]
