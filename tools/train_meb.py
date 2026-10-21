# -*- coding: utf-8 -*-
"""PSA, train the StarDist model on OUR annotated SEM tiles (annotation kit
of tools/kit_annotation.py), by transfer learning from the pretrained model,
then optimise the thresholds, export to ONNX for the PSA and compare the new
model with the current one on the VALIDATION tiles.

Requirements (training PC only):  pip install -r requirements-dl.txt
    python tools/train_meb.py KIT [--name meb_mano] [--families a,b]
           [--epochs 100] [--steps 100] [--eval-only]

The tiles are prepared EXACTLY as the PSA will see them at analysis time:
raw grey image, inverted when the particles are dark (same polarity test as
the PSA), rescaled with the PSA's own typical-size estimate so that the
particles are about 45 px, normalised (percentiles 1 - 99.8). A copy at
0.8x and 1.25x is added so that the model tolerates an imperfect scale
estimate. Model written to Models/stardist/<name>/ (model.onnx,
thresholds.json, config.json, weights_best.h5); select it in a recipe with
the parameter  "stardist_modele": "<name>".
GPU strongly advised (minutes instead of hours); CPU works.
"""
import argparse
import csv
import json
import os
import shutil
import sys
import warnings

import numpy as np
import cv2

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(ICI)
sys.path.insert(0, RACINE)
sys.path.insert(0, ICI)
warnings.simplefilter("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# TensorFlow >= 2.16 ships Keras 3, which StarDist / CSBDeep do not support:
# they run on the Keras 2 package "tf-keras" (requirements-dl.txt)
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")


def charger_kit(kit, familles=None):
    """Annotated tiles: list of dicts (image, labels, family, split, name)."""
    import tifffile
    out = []
    for l in csv.DictReader(open(os.path.join(kit, "manifest.csv"))):
        if familles and l["family"] not in familles:
            continue
        d = os.path.join(kit, l["family"])
        ch = os.path.join(d, l["tile"] + "_label.tif")
        if not os.path.exists(ch):
            continue
        img = tifffile.imread(os.path.join(d, l["tile"] + ".tif"))
        lab = tifffile.imread(ch).astype(np.int32)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        out.append({"image": img.astype(np.uint8), "labels": lab,
                    "family": l["family"], "split": l["split"],
                    "tile": l["tile"]})
    return out


def comme_le_psa(gris):
    """(grey image with BRIGHT particles, scale) as in the PSA inference."""
    import psa_nano_meb as nano
    from psa.learned import estimer_diametre_px, STARDIST_CIBLE_PX
    claires = nano.detecter_polarite(gris)
    g = gris if claires else cv2.bitwise_not(gris)
    ech = float(np.clip(STARDIST_CIBLE_PX / max(estimer_diametre_px(g), 1.0),
                        0.05, 2.0))
    return g, ech


def preparer(t, ech_extra=1.0, patch=256):
    from psa.learned import _normaliser
    g, ech = comme_le_psa(t["image"])
    ech *= ech_extra
    H, W = g.shape
    taille = (max(16, int(round(W * ech))), max(16, int(round(H * ech))))
    x = cv2.resize(g, taille, interpolation=cv2.INTER_AREA if ech < 1
                   else cv2.INTER_LINEAR)
    y = cv2.resize(t["labels"].astype(np.float32), taille,
                   interpolation=cv2.INTER_NEAREST).astype(np.int32)
    x = _normaliser(x)
    # tuiles plus petites que le patch : complétées par une valeur de fond
    # (pas de miroir : il recopierait des particules marquées « fond »)
    ph, pw = max(0, patch - x.shape[0]), max(0, patch - x.shape[1])
    if ph or pw:
        x = np.pad(x, ((0, ph), (0, pw)), constant_values=float(np.median(x)))
        y = np.pad(y, ((0, ph), (0, pw)))
    return x.astype(np.float32), y


def augmenter(x, y):
    k = np.random.randint(4)
    x, y = np.rot90(x, k), np.rot90(y, k)
    if np.random.rand() < .5:
        x, y = x[:, ::-1], y[:, ::-1]
    x = x * np.random.uniform(0.8, 1.2) + np.random.uniform(-0.1, 0.1) + \
        np.random.normal(0, 0.02, x.shape)
    return np.ascontiguousarray(x, np.float32), np.ascontiguousarray(y)


def evaluer(tuiles, noms_modeles):
    """Scores of each model on the validation tiles, per family, with the
    PSA inference itself (labels_stardist) and the scoring of
    eval_annotations (centre matching, border objects excluded)."""
    from psa.learned import labels_stardist
    from eval_annotations import scorer, _reetiqueter
    lignes = []
    for nom in noms_modeles:
        for t in tuiles:
            g, _ = comme_le_psa(t["image"])
            lab, _ = labels_stardist(g, {"stardist_modele": nom})
            s = scorer(_reetiqueter(t["labels"]), _reetiqueter(lab))
            c, i = s["centre"], s["iou50"]
            lignes.append({"model": nom, "family": t["family"],
                           "tile": t["tile"], "n_truth": c["n_truth"],
                           "n_pred": c["n_pred"], "f1": c["f1"],
                           "count_err_pct": c["count_err_pct"],
                           "size_bias_pct": c["size_bias_pct"],
                           "f1_iou50": i["f1"]})
    return lignes


def resume(lignes):
    import collections
    agg = collections.defaultdict(list)
    for l in lignes:
        agg[(l["family"], l["model"])].append(l)
    txt = [f"{'family':18s} {'model':20s} {'truth':>6s} {'F1':>6s} "
           f"{'count':>8s} {'size':>7s} {'F1 IoU':>7s}"]
    for (fam, nom), ls in sorted(agg.items()):
        nt = sum(l["n_truth"] for l in ls)
        npred = sum(l["n_pred"] for l in ls)
        f1 = np.mean([l["f1"] for l in ls])
        sb = np.nanmean([l["size_bias_pct"] for l in ls])
        fi = np.mean([l["f1_iou50"] for l in ls])
        txt.append(f"{fam:18s} {nom:20s} {nt:6d} {f1:6.3f} "
                   f"{100 * (npred - nt) / max(nt, 1):+7.1f}% {sb:+6.1f}% {fi:7.3f}")
    return "\n".join(txt)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("kit")
    ap.add_argument("--name", default="meb_mano")
    ap.add_argument("--base", default="2D_versatile_fluo")
    ap.add_argument("--families", default="")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--eval-only", action="store_true")
    a = ap.parse_args(argv)
    fams = [f for f in a.families.split(",") if f] or None
    tuiles = charger_kit(a.kit, fams)
    tr = [t for t in tuiles if t["split"] == "train"]
    va = [t for t in tuiles if t["split"] == "val"]
    print(f"{len(tr)} training tile(s), {len(va)} validation tile(s)")
    if not va:
        sys.exit("No annotated VALIDATION tile: annotate at least one per "
                 "family (split = val in manifest.csv).")
    base = os.path.join(RACINE, "Models", "stardist")
    dossier = os.path.join(base, a.name)
    if not a.eval_only:
        if not tr:
            sys.exit("No annotated training tile.")
        from stardist.models import StarDist2D
        X, Y = [], []
        for t in tr:
            for e in (1.0, 0.8, 1.25):
                x, y = preparer(t, e, a.patch)
                X.append(x)
                Y.append(y)
        Xv, Yv = zip(*[preparer(t, 1.0, a.patch) for t in va])
        if not os.path.exists(os.path.join(dossier, "weights_best.h5")):
            os.makedirs(dossier, exist_ok=True)
            for f in ("config.json", "weights_best.h5", "thresholds.json"):
                shutil.copy(os.path.join(base, a.base, f), dossier)
        c = json.load(open(os.path.join(dossier, "config.json")))
        c.update(train_patch_size=[a.patch, a.patch], train_batch_size=4,
                 train_learning_rate=3e-4, train_tensorboard=False,
                 train_reduce_lr={"factor": 0.5, "patience": 8, "min_delta": 0})
        json.dump(c, open(os.path.join(dossier, "config.json"), "w"))
        m = StarDist2D(None, name=a.name, basedir=base)
        m.train(X, Y, validation_data=(list(Xv), list(Yv)),
                augmenter=augmenter, epochs=a.epochs, steps_per_epoch=a.steps)
        m.load_weights("weights_best.h5")
        m.optimize_thresholds(list(Xv), list(Yv),
                              nms_threshs=[0.3, 0.4, 0.5, 0.6])
        import tensorflow as tf
        import tf2onnx
        tf2onnx.convert.from_keras(
            m.keras_model, input_signature=(tf.TensorSpec(
                (None, None, None, 1), tf.float32, name="input"),),
            opset=17, output_path=os.path.join(dossier, "model.onnx"))
        print("Model exported:", os.path.join(dossier, "model.onnx"), flush=True)
        # session ONNX éventuellement en cache : on repart du fichier neuf
        import psa.learned as L
        L._SESSIONS.pop(a.name, None)
    lignes = evaluer(va, [a.base, a.name])
    txt = resume(lignes)
    print("\nValidation tiles (annotated from scratch):\n" + txt)
    os.makedirs(dossier, exist_ok=True)
    open(os.path.join(dossier, "validation.txt"), "w").write(txt + "\n")
    with open(os.path.join(dossier, "validation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(lignes[0]))
        w.writeheader()
        w.writerows(lignes)


if __name__ == "__main__":
    main()
