# -*- coding: utf-8 -*-
"""PSA, synthetic sections of a filled resin with a KNOWN spatial structure,
to validate the dispersion analysis (psa/dispersion.py).

Dark discs (filler, as in the SE images of the resin) on a brighter, noisy
matrix, log-normal diameters, three structures:
  "random"     random sequential addition, no overlap (hard-core random)
  "clustered"  Matern-like: particles packed around random cluster centres
  "regular"    perturbed hexagonal lattice
Returns the image (BGR uint8) and the true instance labels.
"""
import math

import numpy as np
import cv2


def _disque(lab, occupe, cx, cy, r, k, ecart=1):
    H, W = lab.shape
    x0, x1 = int(max(0, cx - r - ecart)), int(min(W, cx + r + ecart + 1))
    y0, y1 = int(max(0, cy - r - ecart)), int(min(H, cy + r + ecart + 1))
    if x1 <= x0 or y1 <= y0:
        return False
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    m_ex = d2 <= (r + ecart) ** 2               # zone d'exclusion
    if (occupe[y0:y1, x0:x1] & m_ex).any():
        return False
    m = d2 <= r * r
    lab[y0:y1, x0:x1][m] = k
    occupe[y0:y1, x0:x1] |= m
    return True


def dispersion_synthetique(structure="random", H=600, W=800, d_med=12.0,
                           sigma=0.25, phi=0.15, seed=0, taille_amas=6.0,
                           n_amas=None):
    rng = np.random.default_rng(seed)
    lab = np.zeros((H, W), np.int32)
    occupe = np.zeros((H, W), bool)
    k = 0
    cible = phi * H * W

    def diametre():
        return float(np.clip(d_med * math.exp(rng.normal(0, sigma)), 4, 4 * d_med))

    if structure == "regular":
        a_moy = math.pi * (d_med / 2) ** 2 * math.exp(2 * sigma ** 2)
        n = cible / a_moy
        pas = math.sqrt(2 * H * W / (math.sqrt(3) * n))
        for j, y in enumerate(np.arange(pas / 2, H, pas * math.sqrt(3) / 2)):
            for x in np.arange(pas / 2 + (j % 2) * pas / 2, W, pas):
                r = diametre() / 2
                jit = 0.08 * pas
                cx, cy = x + rng.normal(0, jit), y + rng.normal(0, jit)
                if _disque(lab, occupe, cx, cy, r, k + 1):
                    k += 1
    elif structure == "clustered":
        a_moy = math.pi * (d_med / 2) ** 2 * math.exp(2 * sigma ** 2)
        n_tot = cible / a_moy
        n_amas = n_amas or max(3, int(n_tot / taille_amas))
        R_amas = 0.5 * d_med * math.sqrt(taille_amas / 0.45) * 1.3
        centres = np.stack([rng.uniform(0, W, n_amas), rng.uniform(0, H, n_amas)], 1)
        essais = 0
        while occupe.sum() < cible and essais < 200000:
            essais += 1
            cx0, cy0 = centres[rng.integers(n_amas)]
            rr = R_amas * math.sqrt(rng.random())
            a = rng.uniform(0, 2 * math.pi)
            r = diametre() / 2
            # contacts permis : les amas forment des agglomérats connexes
            if _disque(lab, occupe, cx0 + rr * math.cos(a), cy0 + rr * math.sin(a),
                       r, k + 1, ecart=0):
                k += 1
    else:                                       # random
        essais = 0
        while occupe.sum() < cible and essais < 200000:
            essais += 1
            r = diametre() / 2
            # contacts permis, comme dans le modèle nul de l'analyse
            if _disque(lab, occupe, rng.uniform(0, W), rng.uniform(0, H), r,
                       k + 1, ecart=0):
                k += 1
    # image type SE : matrice claire bruitée, charge sombre à bord adouci
    img = np.full((H, W), 150.0, np.float32)
    img += cv2.GaussianBlur(rng.normal(0, 12, (H, W)).astype(np.float32), (0, 0), 6)
    img[lab > 0] = 60.0
    img = cv2.GaussianBlur(img, (0, 0), 1.0) + rng.normal(0, 6, (H, W))
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), lab


def sections_de_spheres(diametres, seed=0):
    """Section diameters of spheres cut by a random plane (the plane hits a
    sphere with a probability proportional to its diameter)."""
    rng = np.random.default_rng(seed)
    D = np.asarray(diametres, float)
    p = D / D.sum()
    tir = rng.choice(len(D), size=len(D), p=p)
    R = D[tir] / 2
    z = rng.uniform(0, 1, len(D)) * R
    return 2 * np.sqrt(R ** 2 - z ** 2)
