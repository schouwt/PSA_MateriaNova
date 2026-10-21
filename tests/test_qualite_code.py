# -*- coding: utf-8 -*-
"""v1.8.0: static checks. No undefined name anywhere (the star imports used
to hide them: the PDF report of the nano mode crashed on one), and the
modules with explicit imports stay clean for pyflakes."""
import importlib.util
import os
import subprocess
import sys

import pytest

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(ICI)
sys.path.insert(0, RACINE)
sys.path.insert(0, os.path.join(RACINE, "tools"))

pytestmark = pytest.mark.skipif(importlib.util.find_spec("pyflakes") is None,
                                reason="pyflakes not installed")

EXPLICITES = ["psa/modes.py", "psa/batch.py", "psa/droplets.py",
              "psa/dispersion.py", "psa/synth_dispersion.py"]


def test_aucun_nom_indefini():
    import warnings
    warnings.simplefilter("ignore")
    from verifier_noms import verifier, fichiers_du_paquet
    assert verifier(fichiers_du_paquet()) == []


def test_modules_explicites_propres():
    out = subprocess.run([sys.executable, "-m", "pyflakes"]
                         + [os.path.join(RACINE, f) for f in EXPLICITES],
                         capture_output=True, text=True).stdout
    assert out.strip() == "", out
