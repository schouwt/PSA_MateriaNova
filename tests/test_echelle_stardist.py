# -*- coding: utf-8 -*-
"""v1.7.3: working scale of StarDist. In a dense field the particles form a
network; filling the holes of the quick mask turned the matrix pockets into
huge blobs, the typical size was overestimated 7 to 9 times, the image was
shrunk 5 times and NO particle was found (resin x20k i009, powder x100
SA125_i001). The estimate now falls back to the unfilled mask in that case;
the user can also impose the typical size."""
import os
import sys
import threading
import time
import warnings

import numpy as np
import cv2
import pytest
from scipy import ndimage as ndi

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ICI))
warnings.simplefilter("ignore")

import psa_app as A                                  # noqa: E402
from psa.learned import (estimer_diametre_px, _diametre_p90,   # noqa: E402
                         segmenter_stardist)


def _reseau(d=20, pas=19, manque=0.2, seed=0, H=600, W=800):
    """Bright discs packed into a network, a few missing: the dark matrix
    only remains as pockets enclosed by touching particles."""
    rng = np.random.default_rng(seed)
    img = np.full((H, W), 60, np.float32)
    for j, y in enumerate(np.arange(0, H + pas, pas * 0.866)):
        for x in np.arange((j % 2) * pas / 2, W + pas, pas):
            if rng.random() < manque:
                continue
            cv2.circle(img, (int(x + rng.normal(0, 1)), int(y + rng.normal(0, 1))),
                       int(d / 2 * rng.uniform(.85, 1.15)), 190, -1)
    img = cv2.GaussianBlur(img, (0, 0), 1.0) + rng.normal(0, 8, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def _remplissage_seul(g):
    """Former estimate (holes always filled), for the comparison."""
    gs = cv2.GaussianBlur(g, (0, 0), 2.0)
    _, m0 = cv2.threshold(gs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(ndi.binary_fill_holes(m0 > 0).astype(np.uint8) * 255,
                         cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return _diametre_p90(m)


@pytest.mark.parametrize("manque", [0.1, 0.2, 0.3])
def test_reseau_dense_taille_raisonnable(manque):
    g = _reseau(manque=manque)
    assert _remplissage_seul(g) > 10 * 20          # l'ancien comportement
    assert estimer_diametre_px(g) < 3 * 20         # corrigé


def test_spheres_ombrees_inchangees():
    """Shaded SEM spheres (dark centre): the filled mask is still used."""
    g = cv2.imread(os.path.join(os.path.dirname(ICI), "images_test",
                                "spheres_meb.png"), 0)
    assert estimer_diametre_px(g) == pytest.approx(_remplissage_seul(g))


def test_masque_degenere_valeur_finie():
    for g in (np.full((200, 200), 128, np.uint8),
              np.pad(np.full((180, 180), 220, np.uint8), 10)):
        v = estimer_diametre_px(g)
        assert np.isfinite(v) and v > 0


def test_taille_typique_imposee():
    g = _reseau()
    ppnm = 0.05                                    # 20 nm / px
    seg = segmenter_stardist(g, ppnm, {"stardist_diametre_nm": 500.0})
    assert seg["stardist"]["diametre_typique_px"] == pytest.approx(25.0)


def test_detection_reseau_dense():
    """End to end: the dense network is detected (it gave 0 particles
    when the scale was wrong)."""
    img = cv2.cvtColor(_reseau(), cv2.COLOR_GRAY2BGR)
    r = A.detecter_meb(img, 1.0, dict(A.PARAMS_MEB, moteur="stardist",
                                      couper_databar=False,
                                      taille_min_off=True))
    assert len(r["diametres"]) > 400


def test_gui_taille_typique():
    if not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
        pytest.skip("no display (run under xvfb-run)")
    import ttkbootstrap as tb
    n0 = threading.active_count()
    root = tb.Window(themename="darkly")
    app = A.AppPSA(root)

    def attendre(c, d=120):
        t0 = time.time()
        while not c():
            root.update()
            time.sleep(0.02)
            if time.time() - t0 > d:
                raise TimeoutError

    try:
        img = cv2.imread(os.path.join(os.path.dirname(ICI), "images_test",
                                      "spheres_meb.png"))
        app.session.chemin, app.session.image_bgr = "spheres", img
        app.var_recette.set("Metal powder · SEM")
        app.appliquer_recette()
        app.session.image_bgr, app.session.ppm = img, 1.0
        app.session.params["couper_databar"] = False
        app.faites = {0, 1}
        app.aller_etape(3)
        attendre(lambda: app._apercu is not None and
                 "automatic" in app.lbl_dtyp.cget("text"))
        app.var_dtyp.set("100")
        app._changer_dtyp()
        assert app.session.params["stardist_diametre_nm"] == pytest.approx(1e5)
        attendre(lambda: "imposed" in app.lbl_dtyp.cget("text"))
        app.var_dtyp.set("")
        app._changer_dtyp()
        assert "stardist_diametre_nm" not in app.session.params
        attendre(lambda: app._after_id is None and
                 "automatic" in app.lbl_dtyp.cget("text") and
                 threading.active_count() <= n0)
    finally:
        root.destroy()
