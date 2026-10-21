# -*- coding: utf-8 -*-
"""Annotation kit (tools/kit_annotation.py) and tile preparation for the
training (tools/train_meb.py): no TensorFlow needed for these tests."""
import csv
import os
import sys
import tempfile

import numpy as np
import cv2
import pytest

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ICI))
sys.path.insert(0, os.path.join(os.path.dirname(ICI), "tools"))

import kit_annotation as K                 # noqa: E402


@pytest.mark.parametrize("signal, mag, fam", [
    ("LM(L)", 100, "lowmag_powder"), ("SE(L)", 500, "highmag_powder"),
    ("SE(L)", 15000, "se_nano_crystals"), ("TE", 100000, "te_section")])
def test_familles(signal, mag, fam):
    assert K.famille({"signal": signal, "magnification": mag}) == fam


def _hitachi(dossier, nom, mag, sig, px):
    img = np.full((960, 1280), 120, np.uint8)
    rng = np.random.default_rng(mag)
    for _ in range(40):
        cv2.circle(img, (int(rng.uniform(40, 1240)), int(rng.uniform(40, 840))),
                   int(rng.uniform(10, 30)), 220, -1)
    img[890:] = 0                                    # barre d'info noire
    cv2.imwrite(os.path.join(dossier, nom + ".tif"), img)
    with open(os.path.join(dossier, nom + ".txt"), "w") as f:
        f.write(f"[SemImageFile]\nPixelSize={px}\nSignalName={sig}\n"
                f"AcceleratingVoltage=5000 Volt\nMagnification={mag}\n")


def test_kit_bout_en_bout():
    with tempfile.TemporaryDirectory() as d:
        raw, out = os.path.join(d, "raw"), os.path.join(d, "kit")
        os.makedirs(raw)
        _hitachi(raw, "poudre A_i001", 100, "LM(L)", 992.1875)
        _hitachi(raw, "poudre A_i002", 100, "LM(L)", 992.1875)
        _hitachi(raw, "poudre A_i003", 500, "SE(L)", 198.4375)
        K.main([raw, "--out", out, "--no-proposal"])
        lignes = list(csv.DictReader(open(os.path.join(out, "manifest.csv"))))
        low = [l for l in lignes if l["family"] == "lowmag_powder"]
        assert len(low) == 6                        # 3 tuiles par image
        assert {l["split"] for l in low} == {"train", "val"}
        # la dernière image de chaque famille sert à la validation
        assert all(l["split"] == "val" for l in low
                   if l["source"] == "poudre A_i002.tif")
        assert float(low[0]["nm_per_px"]) == pytest.approx(992.1875)
        for l in lignes:                            # tuiles hors barre d'info
            assert int(l["y0"]) + int(l["h"]) <= 890
            assert os.path.exists(os.path.join(out, l["family"],
                                               l["tile"] + ".tif"))
        assert os.path.exists(os.path.join(out, "ANNOTATION_GUIDE.md"))
        assert os.path.exists(os.path.join(out, "annotate_napari.py"))


def test_preparation_tuile_comme_le_psa():
    """Small tile: padded with the background value (never mirrored),
    labels kept aligned with the image after the PSA rescaling; dark
    particles inverted as in the PSA."""
    import train_meb as T
    img = np.full((120, 150), 200, np.uint8)
    lab = np.zeros((120, 150), np.int32)
    for k, (x, y) in enumerate([(40, 40), (100, 70), (60, 95)], 1):
        cv2.circle(img, (x, y), 12, 40, -1)          # particules SOMBRES
        cv2.circle(lab, (x, y), 12, k, -1)
    x, y = T.preparer({"image": img, "labels": lab}, 1.0, patch=256)
    assert x.shape == y.shape and min(x.shape) >= 256
    assert len(np.unique(y)) - 1 == 3
    # inversion : l'intérieur des particules est clair après préparation
    assert x[y > 0].mean() > x[(y == 0)].mean()
