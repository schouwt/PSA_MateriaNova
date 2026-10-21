# -*- coding: utf-8 -*-
"""Tests of tools/eval_annotations.py (PSA vs hand annotation)."""
import os
import sys
import tempfile
import importlib.util

import numpy as np
import cv2
import pytest

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ICI))
sys.path.insert(0, os.path.join(os.path.dirname(ICI), "tools"))

import eval_annotations as E      # noqa: E402


def _disques(n=4, taille=200):
    lab = np.zeros((taille, taille), np.int32)
    for k in range(n):
        cv2.circle(lab, (40 + 40 * k, 100), 15, k + 1, -1)
    return lab


def test_score_parfait_et_erreurs():
    lt = _disques()
    s = E.scorer(lt, lt.copy())["centre"]
    assert s["f1"] == 1.0 and s["count_err_pct"] == 0.0
    assert abs(s["size_bias_pct"]) < 1e-9
    # un objet manqué, un objet coupé en deux
    lp = lt.copy()
    lp[lp == 1] = 0
    m = (lp == 2) & (np.arange(200)[None, :] > 80)
    lp[m] = 9
    s = E.scorer(lt, E._reetiqueter(lp))["centre"]
    assert s["n_truth"] == 4 and s["n_pred"] == 4
    assert s["fn_missed"] == 1
    assert s["fp_split_or_satellite"] == 1


def test_bord_de_zone_ignore():
    """An object cut by the zone border is scored neither as truth nor as
    false positive, even if the prediction stops short of the border."""
    lt = _disques()
    cv2.circle(lt, (5, 30), 15, 7, -1)              # touche le bord
    lp = lt.copy()
    lp[lp == 7] = 0
    cv2.circle(lp, (17, 30), 12, 7, -1)             # ne touche pas le bord
    s = E.scorer(E._reetiqueter(lt), E._reetiqueter(lp))["centre"]
    assert s["n_truth"] == 4 and s["fp"] == 0 and s["f1"] == 1.0


def test_lecture_formats():
    lt = _disques() * 3                               # étiquettes non consécutives
    with tempfile.TemporaryDirectory() as d:
        import tifffile
        tifffile.imwrite(os.path.join(d, "t.tif"), lt.astype(np.uint16))
        cv2.imwrite(os.path.join(d, "t.png"), lt.astype(np.uint16))
        np.save(os.path.join(d, "t.npy"), lt)
        np.save(os.path.join(d, "t_seg.npy"), {"masks": lt}, allow_pickle=True)
        for f in ("t.tif", "t.png", "t.npy", "t_seg.npy"):
            v = E.charger_verite(os.path.join(d, f))
            assert v.max() == 4 and ((v > 0) == (lt > 0)).all(), f
        import roifile
        rois = []
        for k in range(1, 5):
            cnt, _ = cv2.findContours((lt == 3 * k).astype(np.uint8),
                                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            rois.append(roifile.ImagejRoi.frompoints(cnt[0][:, 0, :]))
        chemin = os.path.join(d, "t.zip")
        roifile.roiwrite(chemin, rois)
        v = E.charger_verite(chemin, lt.shape)
        assert v.max() == 4
        assert ((v > 0) & (lt > 0)).sum() / (lt > 0).sum() > 0.95


def test_verite_partielle_placee():
    lt = _disques(taille=100)
    plein, zone = E.placer_verite(lt, (300, 400), (50, 20, 150, 120))
    assert zone == (50, 20, 150, 120)
    assert (plein[20:120, 50:150] == lt).all() and plein.sum() == lt.sum()


@pytest.mark.skipif(importlib.util.find_spec("onnxruntime") is None,
                    reason="onnxruntime missing")
def test_psa_sur_poudre_dense_synthetique():
    """End to end on a small dense synthetic powder: the PSA (StarDist)
    must find most particles; guards the harness, not a validation."""
    img, lab = E.poudre_dense_synthetique(H=320, W=480, d_med=24, seed=1)
    lignes = E.evaluer(img, lab, zone=(0, 0, 480, 320), ppm=1.0,
                       moteurs=("stardist_auto",),
                       params_extra={"couper_databar": False})
    assert lignes[0]["centre_f1"] > 0.85
    assert lignes[0]["iou50_f1"] > 0.8
