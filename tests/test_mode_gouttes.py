# -*- coding: utf-8 -*-
"""v1.7.5: Droplets & capsules is a RECIPE (mode "gouttes"): same workflow,
same graphs and a specialised report, like the other modes."""
import os
import sys
import tempfile
import threading
import time
import warnings

import numpy as np
import pytest

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ICI))
warnings.simplefilter("ignore")

import psa_app as A                                  # noqa: E402
from psa.droplets import gouttes_synthetiques        # noqa: E402

RECETTE = "Droplets & capsules · microfluidics"


def test_detection_format_commun():
    img, vrai = gouttes_synthetiques(seed=2, d_moy=40)
    r = A.detecter_gouttes_mode(img, 0.5, dict(A.PARAMS_GOUTTES,
                                                convention="milieu"))
    n = r["n_particules"]
    for k in ("diametres", "circularites", "aspects", "solidites"):
        assert len(r[k]) == n
    assert len(r["contours"]) == len(r["centres_px"]) == len(r["ids"]) == n
    assert r["labels_ws"].max() == n
    assert np.mean(r["diametres"]) == pytest.approx(2.0 * vrai[:, 2].mean(),
                                                    rel=0.03)
    assert r["statut_ws"] in ("good", "warn") and r["ressemble"]
    # bord extérieur : diamètres plus grands que le milieu de l'anneau
    r2 = A.detecter_gouttes_mode(img, 0.5, dict(A.PARAMS_GOUTTES))
    assert np.mean(r2["diametres"]) > np.mean(r["diametres"])


def test_curseur_sensibilite():
    p = A.appliquer_curseurs(dict(A.PARAMS_GOUTTES), "gouttes", 100, 50, 0,
                             actifs=(True, False, False))
    assert p["k_anneau"] == pytest.approx(0.1)
    s, _, _ = A.curseurs_depuis_params(p, "gouttes")
    assert s == pytest.approx(100)


def test_gui_recette_gouttes_bout_en_bout():
    if not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
        pytest.skip("no display (run under xvfb-run)")
    import ttkbootstrap as tb
    n0 = threading.active_count()
    root = tb.Window(themename="darkly")
    app = A.AppPSA(root)

    def attendre(c, d=150):
        t0 = time.time()
        while not c():
            root.update()
            time.sleep(0.02)
            if time.time() - t0 > d:
                raise TimeoutError

    def textes(w):
        out = []
        for c in w.winfo_children():
            try:
                out.append(str(c.cget("text")))
            except Exception:
                pass
            out += textes(c)
        return out

    try:
        assert RECETTE in app.recettes
        assert "gouttes" not in app.stepper.boutons     # plus un outil
        img, vrai = gouttes_synthetiques(seed=6, d_moy=40)
        app.session.chemin, app.session.image_bgr = "drops", img
        app.var_recette.set(RECETTE)
        app.appliquer_recette()
        assert app.session.mode == "gouttes"
        app.session.image_bgr, app.session.ppm = img, 0.5
        app.faites = {0, 1}
        app.aller_etape(3)
        attendre(lambda: app._derniere_seg is not None)
        assert app._derniere_seg["n"] > 0.9 * len(vrai)
        app.var_g_h.set("60")
        app._changer_gouttes()
        assert app.session.params["hauteur_canal_um"] == 60.0
        app.var_autoarch.set(False)
        app.lancer_analyse()
        attendre(lambda: app.session.resultats is not None and app.etape == 4)
        res = app.session.resultats
        assert res["mode"] == "gouttes" and res["n_particules"] > 0.9 * len(vrai)
        assert any("Droplets & capsules" in t for t in textes(app.droit))
        for vue in ("pipeline", "stats", "distribution", "image"):
            app._changer_vue(vue)                    # aucune vue ne plante
            root.update()
        with tempfile.TemporaryDirectory() as d:
            fs = A.exporter_resultats(d, "drops", app.session, formats=("pdf",))
            noms = [os.path.basename(f) for f in fs]
            assert "drops_droplets.csv" in noms and "drops_report.pdf" in noms
        attendre(lambda: app._after_id is None and
                 threading.active_count() <= n0)
    finally:
        root.destroy()
