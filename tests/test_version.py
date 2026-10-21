# -*- coding: utf-8 -*-
"""v1.8.1: one version number everywhere (psa/base.py is the reference)."""
import os
import re
import sys
import zipfile

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(ICI)
sys.path.insert(0, RACINE)

from psa.base import APP_VERSION                     # noqa: E402

V = APP_VERSION.lstrip("v")


def test_readme():
    premiere = open(os.path.join(RACINE, "README.txt"), encoding="utf-8").readline()
    assert premiere.strip().endswith(APP_VERSION)


def test_changelog_commence_par_la_version():
    txt = open(os.path.join(RACINE, "CHANGELOG.md"), encoding="utf-8").read()
    assert re.findall(r"^## (v[\d.]+)", txt, re.M)[0] == APP_VERSION


def test_guide():
    with zipfile.ZipFile(os.path.join(RACINE, "docs", "guide_psa_sources.zip")) as z:
        tex = z.read("guide_psa.tex").decode("utf-8")
    assert f"User guide v{V}" in tex and f"version {V}" in tex
    assert "1.6.4" not in tex.replace("since v1.6.4", "")
