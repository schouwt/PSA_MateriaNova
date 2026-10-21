# -*- coding: utf-8 -*-
"""v1.7.2: the window fits small screens (13-inch laptops): it never
exceeds the screen, the navigation buttons stay visible, and every widget
of the side panels can be reached by scrolling. Run under xvfb-run (its
default screen is small, which is the point)."""
import os
import sys
import threading
import time
import warnings

import pytest

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ICI))
warnings.simplefilter("ignore")

import psa_app as A                       # noqa: E402


def _attendre(root, cond, delai=120):
    t0 = time.time()
    while not cond():
        root.update()
        time.sleep(0.02)
        if time.time() - t0 > delai:
            raise TimeoutError("timeout")


def test_fenetre_tient_dans_l_ecran():
    if not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
        pytest.skip("no display (run under xvfb-run)")
    import cv2
    import ttkbootstrap as tb
    n0 = threading.active_count()
    root = tb.Window(themename="darkly")
    app = A.AppPSA(root)
    try:
        img = cv2.imread(os.path.join(os.path.dirname(ICI), "images_test",
                                      "spheres_meb.png"))
        app.session.chemin = "spheres"
        app.session.image_bgr = img
        app.var_recette.set("Metal powder · SEM")
        app.appliquer_recette()
        app.session.image_bgr, app.session.ppm = img, 1.0
        app.session.params.update(couper_databar=False, moteur="watershed")
        app.faites = {0, 1}
        app.aller_etape(3)                  # écran le plus haut : Settings
        _attendre(root, lambda: app._apercu is not None)
        for _ in range(20):
            root.update()
            time.sleep(0.02)
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        assert root.winfo_width() <= sw and root.winfo_height() <= sh
        for b in (app.btn_action, app.btn_prec):      # navigation visible
            assert b.winfo_viewable()
            assert b.winfo_rooty() + b.winfo_height() <= sh
            assert b.winfo_rootx() + b.winfo_width() <= sw
        pan = app.panneau_droit
        pan.canvas.yview_moveto(1.0)
        root.update()
        dernier = app.droit.winfo_children()[-1]
        bas_canvas = pan.canvas.winfo_rooty() + pan.canvas.winfo_height()
        assert dernier.winfo_rooty() + dernier.winfo_height() <= bas_canvas + 1
        _attendre(root, lambda: app._after_id is None
                  and threading.active_count() <= n0)
    finally:
        root.destroy()
