# -*- coding: utf-8 -*-
"""Train (or re-train) the StarDist model for nanocarbon primary particles in
TE images, on synthetic aggregates of known primaries (psa/synth_nano.py),
by transfer learning from the pretrained StarDist model. Then optimise the
probability / overlap thresholds and export the network to ONNX for the PSA
(no TensorFlow needed to USE the model, only to train it).

Requirements (training PC only):  pip install -r requirements-dl.txt
Usage:  python tools/train_nanocarbon.py [--images 400] [--epochs 60]
        [--steps 100] [--name nanocarbon_te]
The model is written to Models/stardist/<name>/ (model.onnx, thresholds.json,
config.json, weights_best.h5). Validation (unseen synthetic images): F1 of
the primaries and error on the geometric mean diameter are printed.
Uses every CPU core (or the GPU when TensorFlow sees one).
"""
import argparse, json, os, shutil, sys
import numpy as np

ICI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ICI)
from psa.synth_nano import image_te          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--images", type=int, default=400)
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--steps", type=int, default=100)
ap.add_argument("--name", default="nanocarbon_te")
a = ap.parse_args()
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")   # StarDist needs Keras 2 (tf-keras)
from stardist.models import StarDist2D       # noqa: E402
from csbdeep.utils import normalize          # noqa: E402

base = os.path.join(ICI, "Models", "stardist")
dossier = os.path.join(base, a.name)
print(f"Generating {a.images} synthetic TE images…", flush=True)
X, Y = [], []
for i in range(a.images):
    im, lab, _ = image_te(seed=1000 + i)
    X.append(normalize(255 - im, 1, 99.8).astype(np.float32))
    Y.append(lab.astype(np.int32))
nv = max(8, a.images // 8)
Xt, Yt, Xv, Yv = X[:-nv], Y[:-nv], X[-nv:], Y[-nv:]
if not os.path.exists(os.path.join(dossier, "weights_best.h5")):
    os.makedirs(dossier, exist_ok=True)
    src = os.path.join(base, "2D_versatile_fluo")
    for f in ("config.json", "weights_best.h5", "thresholds.json"):
        shutil.copy(os.path.join(src, f), dossier)
c = json.load(open(os.path.join(dossier, "config.json")))
c.update(train_patch_size=[256, 256], train_batch_size=4,
         train_learning_rate=3e-4, train_tensorboard=False,
         train_reduce_lr={"factor": 0.5, "patience": 8, "min_delta": 0})
json.dump(c, open(os.path.join(dossier, "config.json"), "w"))
m = StarDist2D(None, name=a.name, basedir=base)


def aug(x, y):
    k = np.random.randint(4)
    x, y = np.rot90(x, k), np.rot90(y, k)
    if np.random.rand() < .5:
        x, y = x[:, ::-1], y[:, ::-1]
    x = x * np.random.uniform(0.8, 1.2) + np.random.uniform(-0.1, 0.1) + \
        np.random.normal(0, 0.02, x.shape)
    return np.ascontiguousarray(x, np.float32), np.ascontiguousarray(y)


m.train(Xt, Yt, validation_data=(Xv, Yv), augmenter=aug, epochs=a.epochs,
        steps_per_epoch=a.steps)
m.load_weights("weights_best.h5")
m.optimize_thresholds(Xv[:16], Yv[:16], nms_threshs=[0.3, 0.4, 0.5, 0.6])
import tensorflow as tf, tf2onnx              # noqa: E402,E401
tf2onnx.convert.from_keras(
    m.keras_model, input_signature=(tf.TensorSpec((None, None, None, 1),
                                                  tf.float32, name="input"),),
    opset=17, output_path=os.path.join(dossier, "model.onnx"))
print("Model exported:", os.path.join(dossier, "model.onnx"), flush=True)
