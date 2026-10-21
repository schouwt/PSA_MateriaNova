# -*- coding: utf-8 -*-
"""
=============================================================================
PARTICLE SIZE ANALYSER  –  Materia Nova 2026  –  v0.6  « Workbench »
Centre de Recherche Materia Nova – Mons, Belgique
=============================================================================
Refonte ergonomique pour déploiement technicien :

  • FENÊTRE UNIQUE persistante — navigation par étapes (stepper cliquable),
    un seul canvas matplotlib central qui change de rôle selon l'étape.
  • RECETTES : presets matériau/microscope en JSON (~/.psa_materia/),
    le technicien choisit une recette, charge l'image, clique Analyser.
  • CURSEURS MÉTIER : Sensibilité / Séparation / Taille min — les paramètres
    experts (CLAHE, Sauvola, h_rel…) restent derrière « Mode avancé… ».
  • APERÇU SUPERPOSÉ : détections dessinées sur l'image avec opacité réglable,
    pastille qualité permanente (vert/orange/rouge).
  • ANALYSE EN THREAD : l'interface ne gèle jamais ; barre de progression
    dans la fenêtre, plus aucune dépendance à la console.
  • MODE LOT : un dossier entier avec la même recette → distribution agrégée
    (atteint n ≥ 30 exigé par la représentativité).
  • TRAÇABILITÉ : archivage auto (CSV + PDF + paramètres JSON + opérateur
    + horodatage) à côté de l'image. Garde-fous à l'export (n<30, score bas).
  • MODE NANO/MEB : via le module psa_nano_meb (Sauvola + watershed h-maxima,
    granulométrie morphologique, Hough, dimension fractale).

Dépendances : opencv-python, numpy, matplotlib, scipy, scikit-image
Optionnel   : ttkbootstrap (thème moderne), psa_nano_meb.py (mode nano)
=============================================================================
"""

import csv
import datetime
import json
import math
import os
import re
import queue
import sqlite3
import sys

from scipy import stats as _sps
import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import matplotlib
from matplotlib.image import imread as plt_imread
matplotlib.use("TkAgg")
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, find_boundaries

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:
    print("ERROR: tkinter is not available.")
    sys.exit(1)

# Thème moderne optionnel ----------------------------------------------------
try:
    import ttkbootstrap as tb
    TB = True
except ImportError:
    tb = None
    TB = False

# Métadonnées TIFF optionnelles (pip install tifffile) -------------------------
try:
    import tifffile
    TIF_OK = True
except ImportError:
    tifffile = None
    TIF_OK = False

# Interop ROI ImageJ optionnelle (pip install roifile) -------------------------
try:
    import roifile
    ROI_OK = True
except ImportError:
    roifile = None
    ROI_OK = False

# Module nano/MEB optionnel ---------------------------------------------------
# On garantit que le dossier de CE fichier est sur le chemin de recherche,
# afin que psa_nano_meb.py soit trouvé même si l'EDI (PyCharm…) lance le
# script depuis un autre répertoire de travail. Seule exigence : que
# psa_nano_meb.py se trouve dans le MÊME dossier que psa_app.py.
# Application folder = parent of the psa package (psa_app.py lives there)
_DOSSIER_APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DOSSIER_APP not in sys.path:
    sys.path.insert(0, _DOSSIER_APP)
try:
    import psa_nano_meb as nano
    NANO_OK = True
except ImportError as _err_nano:
    nano = None
    NANO_OK = False
    print("[PSA] nano/SEM mode unavailable:", _err_nano)
    print("[PSA] → Place 'psa_nano_meb.py' in the SAME folder as "
          "'psa_app.py'.")
    print("[PSA]   Folder of psa_app.py:", _DOSSIER_APP)

# Module fibres électrospinnées optionnel (même dossier que psa_app.py).
try:
    import psa_fibres as fibres
    FIB_OK = True
except ImportError:
    fibres = None
    FIB_OK = False

APP_NAME    = "Particle Size Analyser"
APP_VENDOR  = "Materia Nova"
APP_VERSION = "v1.8.1"
APP_TITLE   = f"{APP_NAME}  ·  SEM & optical granulometry  ·  {APP_VERSION}"

DOSSIER_CONF = Path.home() / ".psa_materia"
FICHIER_RECETTES = DOSSIER_CONF / "recettes.json"
FICHIER_AIDE = "guide_psa.pdf"
FICHIER_DB = DOSSIER_CONF / "historique.sqlite"
FICHIER_JOURNAL = DOSSIER_CONF / "journal.log"


def init_journal():
    """Journal de bord (~/.psa_materia/journal.log) : versions au démarrage,
    exceptions des analyses — à joindre en cas de problème sur image réelle."""
    import logging
    DOSSIER_CONF.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=str(FICHIER_JOURNAL), level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s %(message)s")
    try:
        import scipy
        import skimage
        logging.info("=== %s %s, start ===", APP_TITLE, APP_VERSION)
        logging.info("Python %s | OpenCV %s | numpy %s | scipy %s | "
                     "skimage %s | nano=%s roifile=%s tifffile=%s",
                     sys.version.split()[0], cv2.__version__,
                     np.__version__, scipy.__version__, skimage.__version__,
                     NANO_OK, ROI_OK, TIF_OK)
    except Exception:
        pass


def journal(niveau, msg):
    import logging
    try:
        getattr(logging, niveau)(msg)
    except Exception:
        pass
SEUIL_SOLIDITE_SATELLITE = 0.95   # en dessous : particule satellitée/irrégulière


def chemins_aide():
    """Emplacements où chercher le guide PDF (F1 / bouton Aide)."""
    base = Path(_DOSSIER_APP)
    chemins = [base / FICHIER_AIDE, DOSSIER_CONF / FICHIER_AIDE]
    if getattr(sys, "frozen", False):          # exécutable PyInstaller
        chemins.insert(0, Path(sys.executable).resolve().parent / FICHIER_AIDE)
    return chemins
EXT_IMAGES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


__all__ = [_n for _n in dir() if not _n.startswith("__")]
