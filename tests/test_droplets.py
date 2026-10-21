# -*- coding: utf-8 -*-
"""Tests of the microfluidic droplet detection (psa/droplets.py) on
synthetic bright-field images with KNOWN diameters."""
import os
import sys

import numpy as np
import pytest
from scipy.spatial import cKDTree

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ICI))

from psa import droplets as G                     # noqa: E402


def _comparer(img, vrai):
    r = G.detecter_gouttes(img)
    ok = ~r["bord"]
    p = np.c_[r["cx"][ok], r["cy"][ok]]
    d, i = cKDTree(p).query(vrai[:, :2])
    m = d < 0.25 * vrai[:, 2]
    biais = float((r["d_milieu"][ok][i[m]] / vrai[m, 2] - 1).mean())
    return r, m.mean(), int(ok.sum() - m.sum()), biais


@pytest.mark.parametrize("kw, rappel_min, biais_max", [
    (dict(tassees=True, d_moy=40), 0.97, 0.025),          # émulsion tassée
    (dict(tassees=True, d_moy=60, cv=0.06), 0.9, 0.02),   # polydisperse
    (dict(tassees=False, d_moy=36, coeur_clair=True, epaisseur=5),
     0.99, 0.02),                                          # gouttes isolées
])
def test_detection_et_diametre(kw, rappel_min, biais_max):
    img, vrai = G.gouttes_synthetiques(seed=1, **kw)
    r, rappel, faux, biais = _comparer(img, vrai)
    assert rappel >= rappel_min
    assert faux <= 1
    assert abs(biais) <= biais_max
    s = G.statistiques(r)
    cv_vrai = 100 * vrai[:, 2].std(ddof=1) / vrai[:, 2].mean()
    assert abs(s["cv_pct"] - cv_vrai) < 1.0


def test_reconnaissance_type_image():
    """Droplet images are recognised; filled particles are not."""
    img, _ = G.gouttes_synthetiques(seed=2)
    assert G.ressemble_a_des_gouttes(img)[0]
    from psa.synth_dispersion import dispersion_synthetique
    img2, _ = dispersion_synthetique("random", seed=2)
    assert not G.ressemble_a_des_gouttes(img2)[0]


def test_volume_goutte_ecrasee():
    """Sphere below the channel height, squeezed disc above, continuous at
    D = h, cylinder-like for D >> h."""
    h = 50.0
    assert G.volume_goutte(30.0, h) == pytest.approx(np.pi * 30 ** 3 / 6)
    assert G.volume_goutte(h + 1e-9, h) == pytest.approx(np.pi * h ** 3 / 6,
                                                         rel=1e-6)
    D = 2000.0
    assert G.volume_goutte(D, h) / (np.pi * D ** 2 * h / 4) == pytest.approx(
        1.0, rel=0.03)
