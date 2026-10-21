# -*- coding: utf-8 -*-
"""Tests of the filler dispersion analysis (psa/dispersion.py) on synthetic
sections of KNOWN spatial structure, and of tools/analyse_dispersion.py."""
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

from psa import dispersion as D                                 # noqa: E402
from psa.synth_dispersion import (dispersion_synthetique,      # noqa: E402
                                  sections_de_spheres)

_RES = {}


def _analyse(structure):
    if structure not in _RES:
        _, lab = dispersion_synthetique(structure, seed=3)
        _RES[structure] = (lab, D.analyser_dispersion(lab, 5.0, n_mc=19,
                                                      seed=0))
    return _RES[structure]


def test_fraction_surfacique_exacte():
    lab, r = _analyse("random")
    assert abs(r["phi"] - (lab > 0).mean()) < 1e-9
    assert r["n_particules"] == len(np.unique(lab)) - 1


@pytest.mark.parametrize("structure", ["random", "clustered", "regular"])
def test_verdict_structure_connue(structure):
    """The global L test recovers the generated structure."""
    _, r = _analyse(structure)
    assert r["verdict"] == structure


def test_indicateurs_ordonnes():
    """Clark-Evans R and Voronoi CV rank the three structures, and the
    observed random pattern lies within the null band."""
    R = {s: _analyse(s)[1]["observe"]["clark_evans"] for s in
         ("random", "clustered", "regular")}
    cv = {s: _analyse(s)[1]["observe"]["voronoi_cv"] for s in R}
    assert R["clustered"] < R["random"] < R["regular"]
    assert cv["regular"] < cv["random"] < cv["clustered"]
    lo, _, hi = _analyse("random")[1]["nul_clark_evans"]
    assert lo - 0.02 <= R["random"] <= hi + 0.02
    # agglomérats : plus de particules en contact dans le cas en amas
    fa = {s: _analyse(s)[1]["agglomerats"]["frac_part_agglomerees"] for s in R}
    assert fa["clustered"] > 2 * fa["random"] and fa["regular"] < 0.05


def test_modele_nul_conserve_les_particules():
    _, lab = dispersion_synthetique("random", seed=4)
    lab_s, echecs = D.placer_au_hasard(lab, np.ones(lab.shape, bool),
                                       np.random.default_rng(0))
    assert echecs == 0
    assert lab_s.max() == lab.max()
    # coupe par le bord du champ : la fraction ne baisse que très peu
    assert abs((lab_s > 0).mean() - (lab > 0).mean()) < 0.01


def test_saltykov_diametre_moyen():
    """Mean sphere diameter recovered within 5 % from random sections."""
    rng = np.random.default_rng(1)
    Dv = 50 * np.exp(rng.normal(0, 0.25, 20000))
    bords, nv = D.saltykov(sections_de_spheres(Dv, seed=2), n_classes=15)
    assert abs((nv * bords[1:]).sum() / Dv.mean() - 1) < 0.05


def test_calibration_hitachi():
    import analyse_dispersion as T
    with tempfile.TemporaryDirectory() as d:
        img = os.path.join(d, "resine_i001.tif")
        open(img, "wb").close()
        with open(os.path.join(d, "resine_i001.txt"), "w") as f:
            f.write("[SemImageFile]\nMagnification=20000\nPixelSize=4.961\n")
        assert T.nm_par_px_hitachi(img) == pytest.approx(4.961)


@pytest.mark.skipif(importlib.util.find_spec("onnxruntime") is None,
                    reason="onnxruntime missing")
def test_outil_bout_en_bout():
    """Command line on a small synthetic image, hybrid engine: outputs
    written, area fraction close to the truth."""
    import analyse_dispersion as T
    img, lab = dispersion_synthetique("clustered", H=300, W=400, seed=5)
    with tempfile.TemporaryDirectory() as d:
        ch = os.path.join(d, "synth.png")
        cv2.imwrite(ch, img)
        T.main([ch, "--nm-per-px", "5", "--engine", "hybrid",
                "--no-databar", "--n-mc", "5", "--out", d])
        for suf in ("_dispersion.png", "_summary.txt", "_summary.json",
                    "_particles.csv"):
            assert os.path.exists(os.path.join(d, "synth" + suf))
        import json
        s = json.load(open(os.path.join(d, "synth_summary.json")))
        assert abs(s["phi"] - (lab > 0).mean()) < 0.02


def test_homogeneite_sans_seuil():
    """Threshold-free grey-level variance scaling: slope close to -2 for a
    random dispersion, clearly shallower with clusters, and invariant to
    brightness / contrast settings."""
    pentes = {}
    for s in ("random", "clustered"):
        img, _ = dispersion_synthetique(s, seed=2, H=900, W=1200)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, r, p = D.homogeneite_intensite(g)
        _, r2, _ = D.homogeneite_intensite(0.5 * g.astype(float) + 40)
        assert np.allclose(r, r2)
        pentes[s] = p
    assert -2.3 < pentes["random"] < -1.7
    assert pentes["clustered"] > pentes["random"] + 0.4
