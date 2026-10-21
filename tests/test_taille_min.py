# -*- coding: utf-8 -*-
"""v1.7: the minimum-size filter can be switched off (all modes with a
minimum size: optical, SEM micro, nano), in the GUI and in batch use."""
import os
import sys
import time
import warnings

import numpy as np
import pytest

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ICI))
sys.path.insert(0, os.path.join(os.path.dirname(ICI), "tools"))
warnings.simplefilter("ignore")

import psa_app as A                       # noqa: E402


@pytest.mark.parametrize("mode, cle", [("meb", "d_min_um"),
                                       ("optique", "diametre_min_um"),
                                       ("nano", "d_min_nm")])
@pytest.mark.parametrize("moteur", ["stardist", "watershed"])
def test_curseurs_filtre_coupe(mode, cle, moteur):
    p = {cle: 2.0, "moteur": moteur, "taille_min_off": True}
    # même si le curseur de taille a bougé, le filtre coupé impose 0
    q = A.appliquer_curseurs(p, mode, 50, 50, 25.0, actifs=(True, True, True))
    assert q[cle] == 0.0
    p.pop("taille_min_off")
    q = A.appliquer_curseurs(p, mode, 50, 50, 25.0, actifs=(True, True, True))
    assert q[cle] == 25.0


def test_detection_sans_taille_min():
    """Batch path: the switch is honoured by the detector itself."""
    from eval_annotations import poudre_dense_synthetique
    img, _ = poudre_dense_synthetique(H=320, W=480, d_med=18, seed=4)
    base = dict(A.PARAMS_MEB, couper_databar=False, moteur="watershed",
                d_min_um=16.0)
    r1 = A.detecter_meb(img, 1.0, base)
    r2 = A.detecter_meb(img, 1.0, dict(base, taille_min_off=True))
    d1, d2 = np.asarray(r1["diametres"]), np.asarray(r2["diametres"])
    assert d1.min() >= 16.0 - 1e-6
    assert len(d2) > len(d1) and d2.min() < 16.0


def _app():
    if not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
        pytest.skip("no display (run under xvfb-run)")
    import ttkbootstrap as tb
    root = tb.Window(themename="darkly")
    return root, A.AppPSA(root)


def _attendre(root, cond, delai=180):
    t0 = time.time()
    while not cond():
        root.update()
        time.sleep(0.02)
        if time.time() - t0 > delai:
            raise TimeoutError("timeout")


def test_gui_case_taille_min_metal():
    """Metal powder recipe: the checkbox switches the filter off and back
    on, the slider is greyed out and its value restored."""
    import cv2
    import threading
    img = cv2.imread(os.path.join(os.path.dirname(ICI), "images_test",
                                  "spheres_meb.png"))
    n_threads = threading.active_count()
    root, app = _app()
    try:
        app.session.chemin = "spheres"
        app.session.image_bgr = img
        app.var_recette.set("Metal powder · SEM")
        app.appliquer_recette()
        app.session.image_bgr, app.session.ppm = img, 1.0
        app.session.params["couper_databar"] = False
        app.faites = {0, 1}
        app.aller_etape(3)
        _attendre(root, lambda: app._apercu is not None)
        assert app.var_tmin_on.get()
        app.var_tmin.set(5.0)
        app.var_tmin_on.set(False)
        app._basculer_taille_min()
        p = app.session.params
        assert p["taille_min_off"] and p["d_min_um"] == 0.0
        assert app.var_tmin._scale.instate(["disabled"])
        app.var_tmin_on.set(True)
        app._basculer_taille_min()
        assert "taille_min_off" not in app.session.params
        assert app.session.params["d_min_um"] == pytest.approx(5.0)
        assert not app.var_tmin._scale.instate(["disabled"])
        # laisser finir les aperçus lancés par les bascules : un calcul
        # ONNX encore en cours à la sortie de Python fait avorter le process
        _attendre(root, lambda: app._after_id is None
                  and threading.active_count() <= n_threads)
    finally:
        root.destroy()
