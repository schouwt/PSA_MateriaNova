# -*- coding: utf-8 -*-
"""v1.7.8 (audit, lot A): faster StarDist NMS, serialized background tasks,
safe closing, Hitachi calibration read by the PSA, batch calibration guard."""
import importlib.util
import os
import sys
import tempfile
import threading
import time
import warnings

import numpy as np
import cv2
import pytest

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ICI))
sys.path.insert(0, os.path.join(os.path.dirname(ICI), "tools"))
sys.path.insert(0, ICI)
warnings.simplefilter("ignore")

import psa_app as A                                  # noqa: E402
import psa.learned as L                              # noqa: E402


def _x_reseau():
    from test_echelle_stardist import _reseau
    g = _reseau(seed=4)
    return L._normaliser(cv2.resize(g, None, fx=1.4, fy=1.4))


def _raster(polys, shape):
    lab = np.zeros(shape, np.int32)
    for k, c in enumerate(polys[::-1], 1):
        q = np.round(np.stack([c[1], c[0]], 1)).astype(np.int32)
        cv2.fillPoly(lab, [q], k)
    return lab


def test_nms_rapide_identique_a_l_exhaustive():
    from eval_annotations import scorer, _reetiqueter
    x = _x_reseau()
    pts, p, d = L.candidats_stardist(x, prob_thresh=0.5, tta=False)
    assert len(p) > 20 * 300                    # beaucoup de candidats
    pa, ca, _ = L.nms_candidats(pts, p, d, x.shape, 0.3, exact=True)
    pb, cb, _ = L.nms_candidats(pts, p, d, x.shape, 0.3)
    assert abs(len(pa) - len(pb)) <= max(2, 0.005 * len(pa))
    f1 = scorer(_reetiqueter(_raster(ca, x.shape)),
                _reetiqueter(_raster(cb, x.shape)))["iou50"]["f1"]
    assert f1 > 0.995


@pytest.mark.skipif(importlib.util.find_spec("stardist") is None,
                    reason="official StarDist not installed")
def test_nms_rapide_contre_stardist_officiel():
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    from stardist.nms import non_maximum_suppression
    x = _x_reseau()
    prob, dist = L._predire_onnx(x)
    po, _, _ = non_maximum_suppression(dist, prob, grid=(2, 2),
                                       prob_thresh=0.5, nms_thresh=0.3,
                                       verbose=False)
    pts, p, d = L.candidats_stardist(x, prob_thresh=0.5, tta=False)
    pb, _, _ = L.nms_candidats(pts, p, d, x.shape, 0.3)
    assert abs(len(pb) - len(po)) <= max(3, 0.015 * len(po))


def _hitachi(dossier, nom, px_nm):
    img = np.full((300, 400), 120, np.uint8)
    cv2.circle(img, (200, 150), 40, 220, -1)
    cv2.imwrite(os.path.join(dossier, nom + ".tif"), img)
    with open(os.path.join(dossier, nom + ".txt"), "w") as f:
        f.write(f"[SemImageFile]\nInstructName=SU8000\nPixelSize={px_nm}\n")


def test_calibration_hitachi_lue_par_le_psa():
    with tempfile.TemporaryDirectory() as d:
        _hitachi(d, "img", 992.1875)
        ppm, src = A.calibration_metadonnees(os.path.join(d, "img.tif"))
        assert ppm == pytest.approx(1000 / 992.1875) and "Hitachi" in src
        # un .txt quelconque n'est pas pris pour un fichier Hitachi
        cv2.imwrite(os.path.join(d, "autre.png"), np.zeros((10, 10), np.uint8))
        open(os.path.join(d, "autre.txt"), "w").write("PixelSize=5\n")
        assert A.calibration_metadonnees(os.path.join(d, "autre.png"))[0] is None


def test_lot_avertit_si_grossissement_different(monkeypatch):
    import psa.batch as B
    messages = []
    monkeypatch.setattr(B, "journal", lambda niv, m: messages.append(m))
    with tempfile.TemporaryDirectory() as d:
        _hitachi(d, "a_x500", 198.4375)
        _hitachi(d, "b_x100", 992.1875)              # autre grossissement
        s = A.Session()
        s.mode, s.recette, s.ppm = "meb", "Metal powder · SEM", 1000 / 198.4375
        s.params = dict(A.PARAMS_MEB, moteur="watershed", couper_databar=False)
        A.analyser_lot(d, s, rapports_individuels=False)
    alertes = [m for m in messages if "different magnification" in m]
    assert len(alertes) == 1 and "b_x100" in alertes[0]


def _app():
    if not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
        pytest.skip("no display (run under xvfb-run)")
    import ttkbootstrap as tb
    root = tb.Window(themename="darkly")
    return root, A.AppPSA(root)


def test_taches_d_un_canal_serialisees():
    """Three requests in a row on one channel: never two at a time, only
    the LAST result is applied; a cancelled channel applies nothing."""
    root, app = _app()
    try:
        actives, max_actives, recus = [0], [0], []

        def tache(v, duree=0.3):
            def fn(_p):
                actives[0] += 1
                max_actives[0] = max(max_actives[0], actives[0])
                time.sleep(duree)
                actives[0] -= 1
                return v
            return fn
        for v in (1, 2, 3):
            app.lancer_tache(tache(v), recus.append, "t", canal="test")
        t0 = time.time()
        while time.time() - t0 < 5 and not (recus and not app.taches_actives()):
            root.update()
            time.sleep(0.02)
        assert recus == [3] and max_actives[0] == 1
        app.lancer_tache(tache(4), recus.append, "t", canal="test")
        app.annuler_canal("test")
        t0 = time.time()
        while time.time() - t0 < 3:
            root.update()
            time.sleep(0.02)
        assert recus == [3]
    finally:
        root.destroy()


def test_fermeture_attend_le_calcul():
    root, app = _app()
    fini = threading.Event()
    au_moment = []
    detruire = root.destroy

    def espion():
        au_moment.append(fini.is_set())
        detruire()
    root.destroy = espion

    def long(_p):
        time.sleep(0.8)
        fini.set()
        return 0
    app.lancer_tache(long, lambda _v: None, "t")
    app.fermer()
    t0 = time.time()
    while not au_moment and time.time() - t0 < 5:
        root.update()
        time.sleep(0.02)
    assert au_moment == [True]          # détruite APRÈS la fin du calcul
