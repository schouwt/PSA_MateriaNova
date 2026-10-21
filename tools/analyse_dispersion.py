# -*- coding: utf-8 -*-
"""PSA, dispersion analysis of a filler in a resin / polymer (SEM section).

    python tools/analyse_dispersion.py IMAGE [--nm-per-px 4.96]
           [--engine stardist|threshold] [--labels MASK] [--n-mc 39]
           [--diam-px D] [--out DIR]

Segmentation:
  stardist   (default) particles as instances, PSA StarDist engine, dark or
             bright particles detected automatically (as in the PSA);
  threshold  filler / matrix only (Otsu on the PSA SEM preprocessing):
             each connected component is an AGGLOMERATE, the particle-level
             statistics are then computed on agglomerates;
  hybrid     StarDist instances clipped to the Otsu filler mask: particle
             identities from StarDist, borders from the threshold. On
             synthetic sections StarDist inflates small particles (about
             +1.5 px on the diameter, +13 % at 12 px, area fraction 19 %
             instead of 15 %); the clipping removes most of it when the
             filler / matrix contrast is clean (check the overlay);
  --labels   use an existing label image instead (e.g. Cellpose masks,
             napari annotation): no segmentation by the PSA.
Calibration: --nm-per-px, else the Hitachi .txt file next to the image
(PixelSize, nm), else the TIFF metadata read by the PSA, else 1 (px).

Outputs (DIR, default <image>_dispersion/): figure PNG, summary.txt,
summary.json, particles.csv. See psa/dispersion.py for the methods.
"""
import argparse
import csv
import json
import os
import re
import sys
import warnings

import numpy as np
import cv2

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ICI))
warnings.simplefilter("ignore")

from psa import dispersion as D            # noqa: E402


# =============================================================================
# CALIBRATION
# =============================================================================
def nm_par_px_hitachi(chemin_image):
    """PixelSize (nm) of the Hitachi SEM text file saved with the image
    (same name, .txt). None if absent or unreadable."""
    base = os.path.splitext(chemin_image)[0]
    for ext in (".txt", ".TXT"):
        f = base + ext
        if not os.path.exists(f):
            continue
        for enc in ("utf-8", "cp1252", "latin-1"):
            try:
                txt = open(f, encoding=enc).read()
                break
            except (UnicodeDecodeError, OSError):
                txt = ""
        m = re.search(r"^\s*PixelSize\s*=\s*([0-9.eE+-]+)", txt, re.M)
        if m:
            return float(m.group(1))
    return None


def calibration(chemin, nm_par_px=None):
    if nm_par_px:
        return float(nm_par_px), "command line"
    v = nm_par_px_hitachi(chemin)
    if v:
        return v, "Hitachi .txt (PixelSize)"
    try:
        import psa_app as A
        ppm, src = A.calibration_metadonnees(chemin)
        if ppm:
            return 1000.0 / ppm, src
    except Exception:
        pass
    return 1.0, "none (distances in px)"


# =============================================================================
# SEGMENTATION
# =============================================================================
def segmenter(image_bgr, moteur="stardist", diam_px=None, couper_databar=True):
    """Returns (labels, roi, info). roi = field above the SEM data bar."""
    import psa_nano_meb as nano
    p = dict(nano.NANO_PARAMS)
    p.update({"couper_databar": couper_databar, "polarite": "auto"})
    gray, y = nano.pretraiter_meb(image_bgr, p)
    claires = bool(p.get("particules_claires", True))
    H, W = image_bgr.shape[:2]
    roi = np.zeros((H, W), bool)
    roi[:gray.shape[0], :gray.shape[1]] = True
    brut = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)[:gray.shape[0]]
    info = {"particles_bright": claires, "engine": moteur}
    def masque_otsu():
        # seuillage sur l'image BRUTE légèrement lissée, pas sur l'image
        # prétraitée : le CLAHE et la mise à plat du fond égalisent
        # l'histogramme (Otsu tombe alors vers 50 % de surface quoi qu'il
        # arrive) et effacent les variations de composition à grande
        # échelle, qui sont précisément le signal d'une dispersion
        g = cv2.GaussianBlur(brut, (0, 0), 1.0)
        g = g if claires else cv2.bitwise_not(g)
        _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    if moteur in ("stardist", "hybrid"):
        from psa.learned import labels_stardist
        g = brut if claires else cv2.bitwise_not(brut)
        q = {"stardist_diametre_px": diam_px} if diam_px else {}
        lab, inf = labels_stardist(g, q)
        info.update({k: inf[k] for k in ("echelle", "diametre_typique_px",
                                         "modele")})
        if moteur == "hybrid":
            lab = np.where(masque_otsu() > 0, lab, 0)
            info["note"] = "StarDist instances clipped to the Otsu mask"
    else:
        m = masque_otsu()
        n, lab = cv2.connectedComponents(m, connectivity=8)
        aires = np.bincount(lab.ravel())
        lab[(aires < 9)[lab]] = 0
        info["note"] = "objects = agglomerates (connected filler)"
    plein = np.zeros((H, W), np.int32)
    plein[:lab.shape[0], :lab.shape[1]] = lab
    return plein, roi, info


# =============================================================================
# FIGURE
# =============================================================================
def figure(res, image_bgr, titre, chemin):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(15, 9), dpi=110)
    FigureCanvasAgg(fig)
    ax = fig.subplots(2, 3).ravel()
    o = res["observe"]
    u = "nm" if res["nm_par_px"] != 1.0 else "px"
    # (a) image et contours
    img = image_bgr.copy()
    lab = res["labels"]
    bord = (lab != cv2.erode(lab.astype(np.float32), np.ones((3, 3))).astype(
        np.int32)) & (lab > 0)
    img[bord] = (255, 0, 255)
    ax[0].imshow(img[..., ::-1])
    ax[0].set_title(f"{res['n_particules']} particles, area fraction "
                    f"{100 * res['phi']:.1f} %")
    ax[0].axis("off")

    def courbe(a, x, cle_obs, cle_env, lab_y, ref=None, logxy=False):
        if res.get(f"env_{cle_env}") is not None:
            lo, med, hi = res[f"env_{cle_env}"]
            a.fill_between(x, lo, hi, color="0.8", label="random, same particles")
            a.plot(x, med, color="0.4", lw=1, ls="--")
        a.plot(x, o[cle_obs], color="C3", lw=2, label="observed")
        if ref is not None:
            a.axhline(ref, color="0.3", lw=0.8)
        if logxy:
            a.set_xscale("log")
            a.set_yscale("log")
        a.set_ylabel(lab_y)
        a.legend(fontsize=8)
    courbe(ax[1], res["r_g_nm"], "g", "g", "pair correlation g(r)", ref=1.0)
    ax[1].set_xlabel(f"r ({u})")
    courbe(ax[2], res["r_g_nm"], "L", "L", "L(r) - r", ref=0.0)
    ax[2].set_xlabel(f"r ({u})")
    courbe(ax[3], res["tailles_q_nm"], "cv_local", "cv_local",
           "CV of local area fraction", logxy=True)
    ax[3].set_xlabel(f"window size ({u})")
    courbe(ax[4], res["r_c_nm"], "H", "H", "H(r), matrix within r of filler")
    ax[4].set_xlabel(f"distance to filler ({u})")
    # (f) tailles : sections et dépliage de Saltykov
    e = res["ecd_nm"]
    ax[5].hist(e, bins=25, density=True, color="C0", alpha=0.6,
               label="section ECD")
    b, nv = res["saltykov"]
    if b is not None:
        w = np.diff(b)
        ax[5].step(b[1:], nv / w, where="pre", color="C1", lw=2,
                   label="spheres (Saltykov)")
    ax[5].set_xlabel(f"diameter ({u})")
    ax[5].set_ylabel("number density")
    ax[5].legend(fontsize=8)
    v = res.get("verdict", "-").upper()
    fig.suptitle(f"{titre}   |   spatial structure: {v} "
                 f"(global L test p = {res.get('p_mad_L', np.nan):.3f}, "
                 f"{res['n_mc']} simulations)", fontsize=12)
    fig.tight_layout()
    fig.savefig(chemin)


def _json(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, tuple):
        return [_json(x) for x in v]
    return v


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("image")
    ap.add_argument("--nm-per-px", type=float)
    ap.add_argument("--engine", default="stardist",
                    choices=("stardist", "threshold", "hybrid"))
    ap.add_argument("--labels", help="existing label image (skips segmentation)")
    ap.add_argument("--diam-px", type=float, help="force StarDist diameter")
    ap.add_argument("--n-mc", type=int, default=39)
    ap.add_argument("--no-databar", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    import psa_app as A
    img, _ = A.charger_image(a.image)
    if img is None:
        sys.exit(f"Cannot read {a.image}")
    nm, src = calibration(a.image, a.nm_per_px)
    print(f"Calibration: {nm:g} nm/px ({src})")
    if a.labels:
        sys.path.insert(0, ICI)
        from eval_annotations import charger_verite
        lab = charger_verite(a.labels, img.shape[:2])
        roi = np.ones(lab.shape, bool)
        info = {"engine": "external labels", "file": a.labels}
    else:
        lab, roi, info = segmenter(img, a.engine, a.diam_px,
                                   not a.no_databar)
    res = D.analyser_dispersion(lab, nm, roi, n_mc=a.n_mc,
                                progress=lambda p, m: print(f"\r{m}   ", end=""))
    print()
    nom = os.path.splitext(os.path.basename(a.image))[0]
    out = a.out or os.path.splitext(a.image)[0] + "_dispersion"
    os.makedirs(out, exist_ok=True)
    txt = (f"{nom}   calibration {nm:g} nm/px ({src})   segmentation: "
           f"{info.get('engine')}" +
           (f" ({info['note']})" if info.get("note") else "") + "\n" +
           D.resume_texte(res))
    print(txt)
    open(os.path.join(out, f"{nom}_summary.txt"), "w",
         encoding="utf-8").write(txt + "\n")
    figure(res, img, nom, os.path.join(out, f"{nom}_dispersion.png"))
    cles = ("phi", "n_particules", "d50_nm", "d_sphere_moy_nm", "verdict",
            "p_mad_L", "r_ecart_max_nm", "libre_median_nm",
            "libre_median_nul_nm", "nul_clark_evans", "nul_voronoi_cv",
            "r_g_nm", "env_g", "env_L", "tailles_q_nm", "env_cv_local",
            "r_c_nm", "env_H", "nm_par_px", "n_mc", "nul_echecs_max")
    resume = {k: _json(res.get(k)) for k in cles}
    resume["observed"] = {k: _json(v) for k, v in res["observe"].items()
                          if k not in ("nnd", "ecd")}
    resume["segmentation"] = {k: _json(v) for k, v in info.items()}
    json.dump(resume, open(os.path.join(out, f"{nom}_summary.json"), "w"),
              indent=1)
    from scipy import ndimage as ndi
    labs = res["labels"]
    idx = np.arange(1, int(labs.max()) + 1)
    cyx = np.array(ndi.center_of_mass(np.ones_like(labs), labs, idx)).reshape(-1, 2)
    with open(os.path.join(out, f"{nom}_particles.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "x_px", "y_px", "ecd_nm"])
        for i, ((y, x), e) in enumerate(zip(cyx, res["ecd_nm"]), 1):
            w.writerow([i, f"{x:.1f}", f"{y:.1f}", f"{e:.2f}"])
    print(f"Outputs in {out}")


if __name__ == "__main__":
    main()
