# -*- coding: utf-8 -*-
"""v1.8.0: the PDF report and the LaTeX source build in EVERY mode (the
nano report used to stop on page 3)."""
import os
import sys
import tempfile
import warnings

import cv2
import pytest

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(ICI)
sys.path.insert(0, RACINE)
warnings.simplefilter("ignore")

import psa_app as A                                   # noqa: E402
import psa_nano_meb as nano                           # noqa: E402
from psa.droplets import gouttes_synthetiques         # noqa: E402
from psa.report_latex import source_latex             # noqa: E402


def _img(nom):
    return cv2.imread(os.path.join(RACINE, "images_test", nom))


def _ppm_fibres():
    g = cv2.imread(os.path.join(RACINE, "images_test", "A67_i005.tif"), 0)
    return nano.detecter_barre_echelle(g, nano.detecter_databar(g)) / 10.0


@pytest.mark.parametrize("mode, image, ppm, extra, pages_min", [
    ("optique", "spheres_meb.png", 1.0, {}, 4),
    ("meb", "spheres_meb.png", 1.0, {"couper_databar": False}, 5),
    ("nano", "86_sonic_i014.tif", 100.0, {}, 4),
    ("fibres", "A67_i005.tif", None, {}, 4),
    ("gouttes", None, 0.5, {}, 4),
])
def test_rapport_chaque_mode(mode, image, ppm, extra, pages_min):
    if mode == "fibres" and not A.FIB_OK:
        pytest.skip("fibre module missing")
    img = _img(image) if image else gouttes_synthetiques(seed=1)[0]
    ppm = ppm or _ppm_fibres()
    s = A.Session()
    s.chemin, s.image_bgr, s.mode, s.ppm = image or "drops.png", img, mode, ppm
    md = A.mode_de(mode)
    s.params = md.params_defaut()
    s.params.update(extra)
    s.resultats = A.filtrer_taille_max(md.detecter(img, ppm, s.params), s.params)
    assert s.resultats["n_particules"] > 0
    assert len(A.pages_rapport(s)) >= pages_min
    with tempfile.TemporaryDirectory() as d:
        tex = open(source_latex(s, d), encoding="utf-8").read()
        assert md.nom.split()[0] in tex or mode == "optique"
