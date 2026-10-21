# -*- coding: utf-8 -*-
"""v1.7.7: batch analysis = the full individual report of every image plus a
comparison report (images side by side, statistical tests), every mode."""
import csv
import os
import shutil
import sys
import tempfile
import warnings

import numpy as np
import cv2
import pytest

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ICI))
warnings.simplefilter("ignore")

import psa_app as A                                  # noqa: E402
from psa.droplets import gouttes_synthetiques        # noqa: E402


def _session(recette, ppm):
    s = A.Session()
    r = A.RECETTES_DEFAUT[recette]
    s.mode, s.recette, s.ppm, s.operateur = r["mode"], recette, ppm, "test"
    base = {"meb": A.PARAMS_MEB, "nano": A.PARAMS_NANO,
            "gouttes": A.PARAMS_GOUTTES}.get(s.mode, A.PARAMS_OPTIQUE)
    s.params = dict(base)
    s.params.update(r["params"])
    return s


def _lignes(chemin):
    return [l for l in csv.reader(open(chemin, encoding="utf-8"))
            if l and not l[0].startswith("#")]


def test_lot_gouttes_rapport_standard():
    with tempfile.TemporaryDirectory() as d:
        for k in range(3):                          # +3 % de diamètre / image
            img, _ = gouttes_synthetiques(seed=20 + k, d_moy=40 * (1 + 0.03 * k),
                                          H=360, W=480)
            cv2.imwrite(os.path.join(d, f"drops_{k}.png"), img)
        open(os.path.join(d, "broken.png"), "w").write("not an image")
        s = _session("Droplets & capsules · microfluidics", 0.5)
        dd, lignes, sortie = A.analyser_lot(d, s)
        assert len(lignes) == 4 and dd.size > 200
        assert [l for l in lignes if l[0] == "broken.png"][0][2] == "unreadable"
        noms = set(os.listdir(sortie))
        assert {"batch_comparison_report.pdf", "batch_images.csv",
                "batch_objects.csv", "batch_pairwise.csv",
                "batch_pooled_summary.csv", "batch_parameters.json",
                "images"} <= noms
        # rapport complet de chaque image lisible
        for k in range(3):
            ind = os.listdir(os.path.join(sortie, "images", f"drops_{k}"))
            assert f"drops_{k}_report.pdf" in ind
            assert f"drops_{k}_droplets.csv" in ind
        im = _lignes(os.path.join(sortie, "batch_images.csv"))
        assert len(im) == 1 + 4                       # en-tête + 4 images
        assert "mean_volume_pL" in im[0] and "d50_ci95_low_µm" in im[0]
        pw = _lignes(os.path.join(sortie, "batch_pairwise.csv"))
        extreme = [l for l in pw[1:] if {l[0], l[1]} == {"drops_0.png",
                                                          "drops_2.png"}][0]
        assert float(extreme[4]) < 0.05               # 6 % d'écart : détecté
        assert float(extreme[2]) < 0                  # drops_0 plus petite


def test_lot_poudre_sphero_agregee():
    """SEM powder: spheroidization rows computed on the whole batch."""
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(os.path.dirname(ICI), "images_test", "spheres_meb.png")
        for k in range(2):
            shutil.copy(src, os.path.join(d, f"spheres_{k}.png"))
        s = _session("Metal powder · SEM", 1.0)
        s.params.update(couper_databar=False, moteur="watershed")
        dd, lignes, sortie = A.analyser_lot(d, s)
        assert dd.size == sum(l[1] for l in lignes) > 0
        im = _lignes(os.path.join(sortie, "batch_images.csv"))
        assert "sphero_number_pct" in im[0]
        su = {l[0]: l[1] for l in _lignes(os.path.join(
            sortie, "batch_pooled_summary.csv"))}
        assert "Spheroidization rate (number)" in su
        # deux images identiques : aucune différence entre elles
        pw = _lignes(os.path.join(sortie, "batch_pairwise.csv"))
        assert float(pw[1][2]) == 0 and float(pw[1][4]) == 1.0


def test_lot_fibres_identique_a_l_analyse_seule():
    """Real electrospun fibre image: the batch gives exactly the result of
    the single-image analysis (the fibre recipes formerly went through the
    optical pipeline in batch mode)."""
    import psa_nano_meb as nano
    src = os.path.join(os.path.dirname(ICI), "images_test", "A67_i005.tif")
    g = cv2.imread(src, 0)
    ppm = nano.detecter_barre_echelle(g, nano.detecter_databar(g)) / 10.0
    rec = [r for r, v in A.RECETTES_DEFAUT.items() if v["mode"] == "fibres"]
    if not rec:
        pytest.skip("fibre module missing")
    img, _ = A.charger_image(src)
    seul = A.detecter_fibres(img, ppm, dict(A.RECETTES_DEFAUT[rec[0]]["params"]))
    with tempfile.TemporaryDirectory() as d:
        shutil.copy(src, d)
        s = _session(rec[0], ppm)
        dd, lignes, sortie = A.analyser_lot(d, s)
        assert lignes[0][1] == seul["n_particules"]
        assert np.median(dd) == pytest.approx(np.median(seul["diametres"]))
        assert os.path.exists(os.path.join(sortie, "images", "A67_i005",
                                           "A67_i005_report.pdf"))


def test_lot_mode_optique():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(os.path.dirname(ICI), "images_test", "spheres_meb.png")
        shutil.copy(src, os.path.join(d, "a.png"))
        rec = [r for r, v in A.RECETTES_DEFAUT.items() if v["mode"] == "optique"]
        s = _session(rec[0], 1.0)
        dd, lignes, sortie = A.analyser_lot(d, s)
        assert os.path.exists(os.path.join(sortie, "batch_comparison_report.pdf"))
        assert os.path.exists(os.path.join(sortie, "images", "a", "a_report.pdf"))
