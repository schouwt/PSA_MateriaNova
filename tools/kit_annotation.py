# -*- coding: utf-8 -*-
"""PSA, annotation kit to train the StarDist model on OUR SEM images.

    python tools/kit_annotation.py RAW_FOLDER [--out kit_annotation]

For every Hitachi image (TIFF + .txt) of RAW_FOLDER:
  - the image FAMILY is deduced from the acquisition (signal, magnification):
      lowmag_powder     LM / SE, magnification <= 200  (dense powders x100)
      highmag_powder    SE, 200 < magnification < 5000 (powders x500, lumps)
      se_nano_crystals  SE, magnification >= 5000      (e.g. TCP aggregates)
      te_section        TE (transmitted electrons, thin sections, e.g. resin)
  - tiles are cut (size suited to the family, data bar excluded);
  - a PROPOSAL is computed with the current PSA (StarDist) and saved as a
    label image: the annotator CORRECTS it instead of drawing everything;
  - one image per family is set aside for VALIDATION: annotate those tiles
    WITHOUT the proposal (from scratch), so that the validation truth is not
    anchored on the model being evaluated.
Outputs: one folder per family with <tile>.tif (raw grey), <tile>_proposal.tif
(labels, uint16), <tile>_overlay.png, plus manifest.csv, ANNOTATION_GUIDE.md
and annotate_napari.py (opens the tiles one by one in napari).
"""
import argparse
import csv
import os
import re
import sys
import warnings

import numpy as np
import cv2

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ICI))
warnings.simplefilter("ignore")


# =============================================================================
# MÉTADONNÉES ET FAMILLES
# =============================================================================
def lire_hitachi(chemin_tif):
    base = os.path.splitext(chemin_tif)[0]
    for ext in (".txt", ".TXT"):
        if os.path.exists(base + ext):
            s = open(base + ext, encoding="latin-1").read()
            g = lambda k: (re.search(rf"^\s*{k}\s*=\s*(.*?)\s*$", s, re.M) or
                           [None, ""])[1]
            try:
                mag = float(g("Magnification") or 0)
            except ValueError:
                mag = 0.0
            try:
                px = float(g("PixelSize") or 0)
            except ValueError:
                px = 0.0
            kv = re.sub(r"[^0-9.]", "", g("AcceleratingVoltage"))
            return {"magnification": mag, "nm_per_px": px,
                    "signal": g("SignalName"),
                    "kv": float(kv) / 1000 if kv else 0.0}
    return {"magnification": 0.0, "nm_per_px": 0.0, "signal": "", "kv": 0.0}


def famille(meta):
    sig = meta["signal"].upper()
    mag = meta["magnification"]
    if sig.startswith("TE"):
        return "te_section"
    if mag and mag <= 200:
        return "lowmag_powder"
    if mag and mag < 5000:
        return "highmag_powder"
    return "se_nano_crystals"


# taille de tuile (px) et nombre de tuiles par image ; None = champ entier
TUILES = {"lowmag_powder": (256, 3), "highmag_powder": (None, 1),
          "se_nano_crystals": (None, 1), "te_section": (384, 1)}


# =============================================================================
# DÉCOUPE ET PROPOSITION
# =============================================================================
def hauteur_utile(img):
    """Height above the SEM data bar (PSA preprocessing)."""
    import psa_nano_meb as nano
    p = dict(nano.NANO_PARAMS, couper_databar=True)
    gray, _ = nano.pretraiter_meb(img, p)
    return gray.shape[0]


def proposition(img, fam, databar=True):
    """Labels of the current PSA (StarDist engine, SEM micro mode), ALL
    objects (none removed for touching the border or for its size)."""
    import psa_app as A
    p = dict(A.PARAMS_MEB, moteur="stardist", polarite="auto",
             taille_min_off=True, couper_databar=databar)
    if fam == "highmag_powder":
        p["combler_convexe"] = True
    r = A.detecter_meb(img, 1.0, p)
    lab = np.asarray(r["labels_ws"], np.int32)
    plein = np.zeros(img.shape[:2], np.int32)
    plein[:lab.shape[0], :lab.shape[1]] = lab
    return plein


def positions(H, W, t, n, rng):
    """n well spread tile origins (deterministic for a given seed)."""
    if t is None:
        return [(0, 0, W, H)]
    xs = np.linspace(0, W - t, n + 2)[1:-1] if n > 1 else [(W - t) / 2]
    ys = rng.uniform(0, H - t, n)
    return [(int(x), int(y), t, t) for x, y in zip(xs, ys)]


def reetiqueter(lab):
    vals = np.unique(lab)
    vals = vals[vals != 0]
    lut = np.zeros(int(lab.max()) + 1, np.int32)
    lut[vals] = np.arange(1, len(vals) + 1)
    return lut[lab]


def superposition(gris, lab, chemin):
    o = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)
    er = cv2.erode(lab.astype(np.float32), np.ones((3, 3))).astype(np.int32)
    o[(lab != er) & (lab > 0)] = (255, 0, 255)
    cv2.imwrite(chemin, o)


# =============================================================================
# GUIDE ET SCRIPT NAPARI
# =============================================================================
GUIDE = """# Annotation guide, PSA StarDist training (Materia Nova, CHIPS)

## The principle
Every tile `<tile>.tif` comes with `<tile>_proposal.tif`, the labels of the
CURRENT model. Correct them (add, delete, split, merge, redraw), then save
as `<tile>_label.tif` (one integer per particle, 0 = background).
VALIDATION tiles (`split = val` in manifest.csv) are annotated FROM SCRATCH,
without the proposal: they measure the model, they must not be anchored on
it. The script `annotate_napari.py` does this for you.

## General rules (all families)
1. Annotate at the FINEST level: a satellite is its own particle, two
   touching particles are two labels. Merging for a counting rule is easy
   afterwards; splitting is not.
2. Annotate the VISIBLE outline. A partly hidden particle gets its visible
   part, as long as at least about half of it is visible; otherwise leave it.
3. Particles cut by the tile edge ARE annotated (their visible part): the
   training needs them, the scoring removes them.
4. Every particle of the tile, not a selection: an unannotated particle
   teaches the network that it is background.
5. When in doubt (blur, shadow), decide once and apply the same rule
   everywhere; note it at the end of this file.

## Family rules
- lowmag_powder (x100, dense powder): every particle, top layer AND the
  visible particles underneath (rule 2). Satellites separately.
- highmag_powder (x500, large particles, lumps): spheres, satellites
  (separately), and LUMPS (irregular agglomerates) as ONE object each when
  no clear boundary separates their parts.
- se_nano_crystals (e.g. TCP aggregates): the CRYSTALLITES inside the
  aggregate, one label per grain where a facet edge or a groove separates
  it from its neighbours. The aggregate outline itself needs no network
  (threshold).
- te_section (thin section, transmitted electrons, dark particles):
  every dark particle; overlapping particles are separated where a neck is
  visible, otherwise one object.

## How much
At least 150 to 300 particles per family for training (several tiles), and
one validation image per family. With the proposal, count about 10 to 20
minutes per tile; micro-SAM (napari plugin) speeds up the from-scratch
validation tiles a lot (one click per particle).

## Decisions taken (fill in)
-
"""

NAPARI = '''# -*- coding: utf-8 -*-
"""Annotate the kit tiles in napari, one after the other.

    pip install "napari[all]"        (once)
    python annotate_napari.py [family]

Keys:  S = save <tile>_label.tif and go to the next tile,
       N = next tile without saving.
Training tiles open with the PSA proposal to correct; validation tiles
open EMPTY (annotate from scratch). Tiles already saved are skipped.
"""
import csv, os, sys
import numpy as np
import napari
import tifffile

ICI = os.path.dirname(os.path.abspath(__file__))
fam = sys.argv[1] if len(sys.argv) > 1 else None
lignes = [l for l in csv.DictReader(open(os.path.join(ICI, "manifest.csv")))
          if (fam is None or l["family"] == fam)]
a_faire = [l for l in lignes if not os.path.exists(
    os.path.join(ICI, l["family"], l["tile"] + "_label.tif"))]
print(f"{len(a_faire)} tile(s) to annotate")
etat = {"i": -1}
viewer = napari.Viewer()


def ouvrir():
    etat["i"] += 1
    viewer.layers.clear()
    if etat["i"] >= len(a_faire):
        print("All tiles done.")
        return
    l = a_faire[etat["i"]]
    d = os.path.join(ICI, l["family"])
    img = tifffile.imread(os.path.join(d, l["tile"] + ".tif"))
    viewer.add_image(img, name=l["tile"])
    if l["split"] == "val":
        lab = np.zeros(img.shape[:2], np.int32)
    else:
        lab = tifffile.imread(os.path.join(d, l["tile"] + "_proposal.tif")).astype(np.int32)
    viewer.add_labels(lab, name="labels")
    viewer.title = f"{l['tile']}  ({l['family']}, {l['split']})  {etat['i'] + 1}/{len(a_faire)}"


@viewer.bind_key("s")
def sauver(v):
    l = a_faire[etat["i"]]
    lab = np.asarray(v.layers["labels"].data, np.uint16)
    tifffile.imwrite(os.path.join(ICI, l["family"], l["tile"] + "_label.tif"), lab)
    print("saved", l["tile"], int(len(np.unique(lab)) - 1), "objects")
    ouvrir()


@viewer.bind_key("n")
def suivant(v):
    ouvrir()


ouvrir()
napari.run()
'''


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("raw")
    ap.add_argument("--out", default="kit_annotation")
    ap.add_argument("--no-proposal", action="store_true")
    a = ap.parse_args(argv)
    import tifffile
    fichiers = sorted(f for f in os.listdir(a.raw) if f.lower().endswith((".tif", ".tiff")))
    par_fam = {}
    for f in fichiers:
        meta = lire_hitachi(os.path.join(a.raw, f))
        par_fam.setdefault(famille(meta), []).append((f, meta))
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)
    lignes = []
    for fam, liste in sorted(par_fam.items()):
        d = os.path.join(a.out, fam)
        os.makedirs(d, exist_ok=True)
        val = liste[-1][0]                       # une image tenue à l'écart
        for f, meta in liste:
            img = cv2.imread(os.path.join(a.raw, f), cv2.IMREAD_COLOR)
            gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            Hu = hauteur_utile(img)
            W = gris.shape[1]
            t, n = TUILES[fam]
            lab_plein = None
            if t is None and not a.no_proposal:
                lab_plein = proposition(img, fam)
            for k, (x, y, w, h) in enumerate(positions(Hu, W, t, n, rng)):
                h = min(h, Hu - y)
                nom = f"{os.path.splitext(f)[0].replace(' ', '_')}_t{k + 1}"
                tuile = gris[y:y + h, x:x + w]
                tifffile.imwrite(os.path.join(d, nom + ".tif"), tuile)
                n_prop = 0
                lab = lab_plein
                if t is not None and not a.no_proposal:
                    # proposition calculée sur la tuile + une marge, pas sur
                    # tout le champ : plus rapide
                    m = 48
                    ya, yb = max(0, y - m), min(Hu, y + h + m)
                    xa, xb = max(0, x - m), min(W, x + w + m)
                    lab = np.zeros(gris.shape, np.int32)
                    lab[ya:yb, xa:xb] = proposition(
                        np.ascontiguousarray(img[ya:yb, xa:xb]), fam, False)
                if lab is not None:
                    lt = reetiqueter(lab[y:y + h, x:x + w])
                    n_prop = int(lt.max())
                    tifffile.imwrite(os.path.join(d, nom + "_proposal.tif"),
                                     lt.astype(np.uint16))
                    superposition(tuile, lt, os.path.join(d, nom + "_overlay.png"))
                lignes.append({"tile": nom, "family": fam, "source": f,
                               "x0": x, "y0": y, "w": w, "h": h,
                               "nm_per_px": meta["nm_per_px"],
                               "magnification": meta["magnification"],
                               "signal": meta["signal"], "kv": meta["kv"],
                               "split": "val" if f == val else "train",
                               "n_proposal": n_prop})
            print(f"{fam:18s} {f}: {n} tile(s)", flush=True)
    with open(os.path.join(a.out, "manifest.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, list(lignes[0]))
        w.writeheader()
        w.writerows(lignes)
    open(os.path.join(a.out, "ANNOTATION_GUIDE.md"), "w", encoding="utf-8").write(GUIDE)
    open(os.path.join(a.out, "annotate_napari.py"), "w", encoding="utf-8").write(NAPARI)
    print(f"Kit written to {a.out}: {len(lignes)} tiles")


if __name__ == "__main__":
    main()
