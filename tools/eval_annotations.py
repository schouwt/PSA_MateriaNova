# -*- coding: utf-8 -*-
"""PSA, evaluation against a hand-annotated ground truth (SEM powders).

Runs the PSA SEM-micron analysis on an image and compares the particles it
reports with an instance annotation of a zone of that image.

    python tools/eval_annotations.py IMAGE TRUTH [--zone x0 y0 x1 y1]
           [--ppm PX_PER_UM] [--out DIR] [--sweep 20,30,45]

TRUTH: label image (TIFF/PNG, 0 = background, e.g. a napari Labels layer),
.npy label array, Cellpose *_seg.npy, or ImageJ ROI set (.zip / .roi).
If TRUTH is smaller than IMAGE, it is placed at (x0, y0) of --zone.

Variants evaluated (one row each):
  stardist_auto     PSA default: working scale from the estimated diameter
  stardist_forced   diameter forced to the TRUE median ECD (scale diagnosis)
  stardist_dXX      optional --sweep values
  watershed         classic engine, for reference

Scoring follows the colleague's protocol (meb_grains report, 11/10/2026):
objects touching the zone border are not scored, matching by centre, count
error and size bias of matched objects. A stricter IoU >= 0.5 matching is
given as well, plus a breakdown of the errors (missed, merged, split) and
the recall per size class, to see WHERE the counts differ.
"""
import argparse
import csv
import json
import os
import sys
import warnings

import numpy as np
import cv2
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ICI))
warnings.simplefilter("ignore")


# =============================================================================
# LECTURE DE LA VÉRITÉ
# =============================================================================
def _reetiqueter(lab):
    """Étiquettes consécutives 1..N (0 = fond), int32."""
    lab = np.asarray(lab)
    if lab.ndim == 3:                      # RGB d'un masque mal exporté
        lab = lab[..., 0]
    vals = np.unique(lab)
    vals = vals[vals != 0]
    lut = np.zeros(int(lab.max()) + 1, np.int32)
    lut[vals.astype(np.int64)] = np.arange(1, len(vals) + 1, dtype=np.int32)
    return lut[lab.astype(np.int64)]


def _depuis_rois(chemin, forme):
    import roifile
    rois = roifile.roiread(chemin)
    if not isinstance(rois, list):
        rois = [rois]
    if forme is None:
        raise ValueError("ROI sets need the image size (pass the image)")
    lab = np.zeros(forme, np.int32)
    for i, r in enumerate(rois, 1):
        pts = np.round(r.coordinates()).astype(np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(lab, [pts], i)
    return lab


def charger_verite(chemin, forme_image=None):
    """Label image of the annotation (int32, 0 = background)."""
    ext = os.path.splitext(chemin)[1].lower()
    if ext in (".zip", ".roi"):
        return _reetiqueter(_depuis_rois(chemin, forme_image))
    if ext == ".npy":
        a = np.load(chemin, allow_pickle=True)
        if a.dtype == object:              # Cellpose *_seg.npy
            a = a.item()["masks"]
        return _reetiqueter(a)
    if ext in (".tif", ".tiff"):
        import tifffile
        return _reetiqueter(np.squeeze(tifffile.imread(chemin)))
    a = cv2.imread(chemin, cv2.IMREAD_UNCHANGED)
    if a is None:
        raise ValueError(f"Cannot read {chemin}")
    return _reetiqueter(a)


# =============================================================================
# APPARIEMENT ET SCORES
# =============================================================================
def _objets(lab):
    """Per object: area, centroid (y, x), ECD (px), touches the border."""
    n = int(lab.max())
    if n == 0:
        z = np.zeros(0)
        return {"aire": z, "cy": z, "cx": z, "ecd": z, "bord": z.astype(bool)}
    idx = np.arange(1, n + 1)
    aire = ndi.sum(np.ones_like(lab), lab, idx)
    cyx = np.array(ndi.center_of_mass(np.ones_like(lab), lab, idx)).reshape(-1, 2)
    bords = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    bord = np.isin(idx, bords)
    return {"aire": aire, "cy": cyx[:, 0], "cx": cyx[:, 1],
            "ecd": 2.0 * np.sqrt(aire / np.pi), "bord": bord}


def _iou(lt, lp, nt, np_):
    """IoU matrix (nt x np) from the joint histogram of the two labellings."""
    m = (lt > 0) | (lp > 0)
    inter = np.zeros((nt + 1, np_ + 1))
    np.add.at(inter, (lt[m], lp[m]), 1)
    at = np.bincount(lt.ravel(), minlength=nt + 1).astype(float)
    ap = np.bincount(lp.ravel(), minlength=np_ + 1).astype(float)
    union = at[:, None] + ap[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou[1:, 1:], inter[1:, 1:]


def _apparier_centres(ot, op, lt, lp):
    """Centre matching: the pair is valid when each centroid falls inside
    the other object (or within 0.5 ECD of the true centre for concave
    shapes); optimal assignment on the centre distance."""
    nt, np_ = len(ot["ecd"]), len(op["ecd"])
    if nt == 0 or np_ == 0:
        return []
    d = np.hypot(ot["cy"][:, None] - op["cy"][None, :],
                 ot["cx"][:, None] - op["cx"][None, :])
    H, W = lt.shape
    pyi = np.clip(np.round(op["cy"]).astype(int), 0, H - 1)
    pxi = np.clip(np.round(op["cx"]).astype(int), 0, W - 1)
    tyi = np.clip(np.round(ot["cy"]).astype(int), 0, H - 1)
    txi = np.clip(np.round(ot["cx"]).astype(int), 0, W - 1)
    p_dans_t = lt[pyi, pxi]                 # objet vrai sous le centre prédit
    t_dans_p = lp[tyi, txi]
    ok = (d <= 0.5 * ot["ecd"][:, None])
    ok |= (p_dans_t[None, :] == np.arange(1, nt + 1)[:, None]) & \
          (t_dans_p[:, None] == np.arange(1, np_ + 1)[None, :])
    cout = np.where(ok, d, 1e9)
    r, c = linear_sum_assignment(cout)
    return [(i, j) for i, j in zip(r, c) if ok[i, j]]


def _apparier_iou(iou, seuil=0.5):
    if iou.size == 0:
        return []
    r, c = linear_sum_assignment(-iou)
    return [(i, j) for i, j in zip(r, c) if iou[i, j] >= seuil]


def _d_quantiles(d):
    if len(d) == 0:
        return (np.nan, np.nan, np.nan)
    return tuple(float(np.percentile(d, q)) for q in (10, 50, 90))


def scorer(lt, lp, classes_px=None):
    """Scores of the prediction lp against the truth lt (same shape, zone
    crop). Objects touching the zone border are left out of the score: a
    border truth object is not counted, a prediction matched to one is
    neither counted as true nor as false positive."""
    ot, op = _objets(lt), _objets(lp)
    nt, np_ = len(ot["ecd"]), len(op["ecd"])
    iou, inter = _iou(lt, lp, nt, np_)
    res = {}
    for nom, paires in (("centre", _apparier_centres(ot, op, lt, lp)),
                        ("iou50", _apparier_iou(iou))):
        t_app = {i: j for i, j in paires}
        p_app = {j: i for i, j in paires}
        t_ok = ~ot["bord"]
        # une prédiction compte si elle ne touche pas le bord et n'est pas
        # appariée à un objet vrai de bord
        p_ok = np.array([(not op["bord"][j]) and
                         not (j in p_app and ot["bord"][p_app[j]])
                         for j in range(np_)], bool)
        tp = sum(1 for i, j in paires if t_ok[i] and p_ok[j])
        n_t, n_p = int(t_ok.sum()), int(p_ok.sum())
        fn, fp = n_t - tp, n_p - tp
        prec = tp / n_p if n_p else np.nan
        rap = tp / n_t if n_t else np.nan
        f1 = 2 * tp / (n_t + n_p) if (n_t + n_p) else np.nan
        dt = np.array([ot["ecd"][i] for i, j in paires if t_ok[i] and p_ok[j]])
        dp = np.array([op["ecd"][j] for i, j in paires if t_ok[i] and p_ok[j]])
        r = {"n_truth": n_t, "n_pred": n_p, "tp": tp, "fp": fp, "fn": fn,
             "precision": prec, "recall": rap, "f1": f1,
             "count_err_pct": 100.0 * (n_p - n_t) / n_t if n_t else np.nan,
             "size_bias_pct": 100.0 * (dp.mean() / dt.mean() - 1)
             if len(dt) else np.nan,
             "size_bias_median_pct": 100.0 * float(np.median(dp / dt - 1))
             if len(dt) else np.nan}
        if nom == "centre":
            # Nature des erreurs (sur les objets vrais hors bord non appariés)
            manques = fusions = 0
            for i in np.where(t_ok)[0]:
                if i in t_app and p_ok[t_app[i]]:
                    continue
                j = int(np.argmax(inter[i])) if np_ else -1
                if j < 0 or inter[i, j] < 0.2 * ot["aire"][i]:
                    manques += 1                # aucune prédiction dessus
                else:
                    fusions += 1                # couvert par une prédiction
                    #                             appariée à un autre objet
            coupes = fantomes = 0
            for j in np.where(p_ok)[0]:
                if j in p_app:
                    continue
                i = int(np.argmax(inter[:, j])) if nt else -1
                if i >= 0 and inter[i, j] >= 0.5 * op["aire"][j]:
                    coupes += 1                 # morceau / satellite d'un vrai
                else:
                    fantomes += 1               # sur le fond
            r.update({"fn_missed": manques, "fn_merged": fusions,
                      "fp_split_or_satellite": coupes,
                      "fp_background": fantomes})
            # Rappel par classe de taille (ECD vrai, px)
            bornes = classes_px or [0, 15, 25, 40, 1e9]
            rap_cl = []
            for a, b in zip(bornes[:-1], bornes[1:]):
                sel = [i for i in np.where(t_ok)[0] if a <= ot["ecd"][i] < b]
                n_ok = sum(1 for i in sel if i in t_app and p_ok[t_app[i]])
                rap_cl.append((a, b, len(sel), n_ok))
            r["recall_by_size"] = rap_cl
            r["truth_d10_d50_d90_px"] = _d_quantiles(ot["ecd"][t_ok])
            r["pred_d10_d50_d90_px"] = _d_quantiles(op["ecd"][p_ok])
            r["_paires"], r["_t_ok"], r["_p_ok"] = paires, t_ok, p_ok
        res[nom] = r
    return res


# =============================================================================
# POUDRE DENSE SYNTHÉTIQUE AVEC VÉRITÉ (auto-test, ordre de grandeur ×100)
# =============================================================================
def poudre_dense_synthetique(H=640, W=960, d_med=24.0, sigma=0.45,
                             couverture=0.55, seed=0):
    """Dense SE-like powder of touching shaded spheres, log-normal ECD,
    with its instance labels (visible part of each sphere). Used by the
    self-test; NOT a substitute for the real annotation."""
    rng = np.random.default_rng(seed)
    img = np.full((H, W), 38.0, np.float32)
    lab = np.zeros((H, W), np.int32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    k, essais = 0, 0
    while (lab > 0).mean() < couverture and essais < 20000:
        essais += 1
        d = float(np.clip(d_med * np.exp(rng.normal(0, sigma)), 8, 90))
        r = d / 2
        cx, cy = rng.uniform(-r / 2, W + r / 2), rng.uniform(-r / 2, H + r / 2)
        x0, x1 = int(max(0, cx - r - 1)), int(min(W, cx + r + 2))
        y0, y1 = int(max(0, cy - r - 1)), int(min(H, cy + r + 2))
        if x1 <= x0 or y1 <= y0:
            continue
        rho2 = (xx[y0:y1, x0:x1] - cx) ** 2 + (yy[y0:y1, x0:x1] - cy) ** 2
        m = rho2 <= r * r
        # chevauchement limité : contacts, pas d'empilement profond
        if (lab[y0:y1, x0:x1][m] > 0).mean() > 0.12:
            continue
        k += 1
        z = np.sqrt(np.clip(1 - rho2 / (r * r), 0, 1))
        val = 80 + 120 * z ** 0.6 + 35 * (1 - z) ** 4
        img[y0:y1, x0:x1] = np.where(m, val, img[y0:y1, x0:x1])
        lab[y0:y1, x0:x1] = np.where(m, k, lab[y0:y1, x0:x1])
    img = cv2.GaussianBlur(img, (0, 0), 0.8) + rng.normal(0, 6, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    # morceaux visibles trop petits (presque recouverts) : hors vérité
    lab = _reetiqueter(lab)
    aires = np.bincount(lab.ravel())
    lab[(aires < 25)[lab]] = 0
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), _reetiqueter(lab)


# =============================================================================
# ANALYSE PSA
# =============================================================================
def labels_psa(image_bgr, ppm, params):
    """Runs the PSA SEM-micron analysis. Returns the label image of the
    particles the PSA REPORTS (after its own filters), at full image size,
    and the result dict."""
    import psa_app as A
    r = A.detecter_meb(image_bgr, ppm, params)
    lab = np.asarray(r["labels_ws"])
    H, W = image_bgr.shape[:2]
    garde = np.zeros(int(lab.max()) + 1, bool)
    garde[np.asarray(r["ids"], int)] = True
    lab = np.where(garde[lab], lab, 0)
    plein = np.zeros((H, W), np.int32)           # la databar a été coupée
    plein[:lab.shape[0], :lab.shape[1]] = lab
    return _reetiqueter(plein), r


def placer_verite(verite, forme, zone):
    """Truth in image coordinates and the zone (x0, y0, x1, y1)."""
    H, W = forme
    h, w = verite.shape
    if (h, w) == (H, W):
        if zone is None:                         # boîte des objets annotés
            ys, xs = np.nonzero(verite)
            zone = (int(xs.min()), int(ys.min()),
                    int(xs.max()) + 1, int(ys.max()) + 1)
        return verite, tuple(zone)
    x0, y0 = (zone[0], zone[1]) if zone else (0, 0)
    if zone is None:
        print("WARNING: truth smaller than the image and no --zone: "
              "assumed at the top-left corner")
    plein = np.zeros((H, W), np.int32)
    plein[y0:y0 + h, x0:x0 + w] = verite[:H - y0, :W - x0]
    return plein, (x0, y0, x0 + w, y0 + h)


def superposition(gris, lt, lp, paires, t_ok, p_ok, chemin):
    """White: truth, magenta: PSA, red dot: missed truth, yellow dot:
    extra PSA object."""
    img = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)
    for lab, coul in ((lt, (255, 255, 255)), (lp, (255, 0, 255))):
        b = (lab != ndi.grey_erosion(lab, 3)) & (lab > 0)
        img[b] = coul
    ot, op = _objets(lt), _objets(lp)
    t_m = {i for i, j in paires if t_ok[i] and p_ok[j]}
    p_m = {j for i, j in paires if t_ok[i] and p_ok[j]}
    for i in np.where(t_ok)[0]:
        if i not in t_m:
            cv2.circle(img, (int(ot["cx"][i]), int(ot["cy"][i])), 4,
                       (0, 0, 255), -1)
    for j in np.where(p_ok)[0]:
        if j not in p_m:
            cv2.circle(img, (int(op["cx"][j]), int(op["cy"][j])), 4,
                       (0, 255, 255), -1)
    cv2.imwrite(chemin, img)


def evaluer(image_bgr, verite, zone=None, ppm=None, sweep=(), out=None,
            moteurs=("stardist_auto", "stardist_forced", "watershed"),
            params_extra=None):
    """Evaluates the PSA variants. Returns a list of result rows."""
    import psa_app as A
    H, W = image_bgr.shape[:2]
    vt, zone = placer_verite(verite, (H, W), zone)
    x0, y0, x1, y1 = zone
    lt = _reetiqueter(vt[y0:y1, x0:x1])
    ot = _objets(lt)
    d_med_vrai = float(np.median(ot["ecd"][~ot["bord"]]))
    ppm = float(ppm or 1.0)
    base = dict(A.PARAMS_MEB)
    base.update(params_extra or {})
    variantes = []
    for m in moteurs:
        if m == "stardist_auto":
            variantes.append((m, dict(base, moteur="stardist")))
        elif m == "stardist_forced":
            variantes.append((m, dict(base, moteur="stardist",
                                      stardist_diametre_px=d_med_vrai)))
        elif m == "watershed":
            variantes.append((m, dict(base, moteur="watershed")))
    for d in sweep:
        variantes.append((f"stardist_d{d:g}",
                          dict(base, moteur="stardist",
                               stardist_diametre_px=float(d))))
    lignes = []
    for nom, p in variantes:
        lp_plein, r = labels_psa(image_bgr, ppm, p)
        lp = _reetiqueter(lp_plein[y0:y1, x0:x1])
        s = scorer(lt, lp)
        info = r.get("stardist") or {}
        ligne = {"variant": nom, "engine": r.get("moteur"),
                 "diam_used_px": info.get("diametre_typique_px"),
                 "scale": info.get("echelle"),
                 "true_median_ecd_px": d_med_vrai}
        for k in ("centre", "iou50"):
            for c, v in s[k].items():
                if not c.startswith("_"):
                    ligne[f"{k}_{c}"] = v
        lignes.append(ligne)
        if out:
            gris = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]
            sc = s["centre"]
            superposition(gris, lt, lp, sc["_paires"], sc["_t_ok"],
                          sc["_p_ok"], os.path.join(out, f"overlay_{nom}.png"))
    return lignes


def tableau(lignes):
    """Console table, colleague's indicators first."""
    t = ["", f"{'Variant':<18}{'diam px':>8}{'Objects':>9}{'F1':>7}"
             f"{'Count':>9}{'Size':>8}{'F1 IoU':>8}  missed/merged | split/bg",
         "-" * 92]
    for l in lignes:
        d = l["diam_used_px"]
        t.append(f"{l['variant']:<18}{(f'{d:.1f}' if d else '-'):>8}"
                 f"{l['centre_n_pred']:>9}{l['centre_f1']:>7.3f}"
                 f"{l['centre_count_err_pct']:>+8.1f}%"
                 f"{l['centre_size_bias_pct']:>+7.1f}%"
                 f"{l['iou50_f1']:>8.3f}  "
                 f"{l['centre_fn_missed']}/{l['centre_fn_merged']} | "
                 f"{l['centre_fp_split_or_satellite']}/"
                 f"{l['centre_fp_background']}")
    l0 = lignes[0]
    t.append("-" * 92)
    t.append(f"Truth: {l0['centre_n_truth']} objects off the zone border, "
             f"median ECD {l0['true_median_ecd_px']:.1f} px")
    for l in lignes:
        cl = ", ".join(f"{a:g}-{('' if b > 1e8 else f'{b:g}')} px: "
                       f"{k}/{n}" for a, b, n, k in l["centre_recall_by_size"])
        t.append(f"  recall by true size, {l['variant']}: {cl}")
    return "\n".join(t)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("image")
    ap.add_argument("truth")
    ap.add_argument("--zone", nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--ppm", type=float, help="calibration, px/µm "
                    "(default: TIFF metadata, else 1)")
    ap.add_argument("--sweep", default="", help="forced diameters, px")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-databar", action="store_true",
                    help="image without SEM data bar")
    a = ap.parse_args(argv)
    import psa_app as A
    img = A.charger_image(a.image)
    if isinstance(img, tuple):
        img = img[0]
    ppm = a.ppm
    if ppm is None:
        ppm, src = A.calibration_metadonnees(a.image)
        print(f"Calibration: {ppm} px/µm ({src})" if ppm else
              "Calibration: none found, 1 px/µm assumed (sizes in px)")
    vt = charger_verite(a.truth, img.shape[:2])
    out = a.out or os.path.splitext(a.truth)[0] + "_psa_eval"
    os.makedirs(out, exist_ok=True)
    sweep = [float(x) for x in a.sweep.split(",") if x.strip()]
    extra = {"couper_databar": False} if a.no_databar else None
    lignes = evaluer(img, vt, a.zone, ppm, sweep, out, params_extra=extra)
    txt = tableau(lignes)
    print(txt)
    with open(os.path.join(out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    cles = [k for k in lignes[0]]
    with open(os.path.join(out, "scores.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, cles)
        w.writeheader()
        for l in lignes:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, tuple))
                            else v) for k, v in l.items()})
    print(f"\nOutputs in {out}")


if __name__ == "__main__":
    main()
