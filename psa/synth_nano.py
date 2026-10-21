# -*- coding: utf-8 -*-
"""PSA, synthetic TE (transmission) images of carbon aggregates with known
primary particles, to train and validate the primary-particle model.

Physics: every primary particle is a sphere (log-normal diameters); an
aggregate grows by attaching spheres to random members with a small overlap
(sintering necks), giving branched fractal-like aggregates. The image is the
transmitted intensity I = I0 exp(-mu t), t being the carbon thickness crossed
(sum of the sphere chords), so overlaps look darker, as in TE images. Then
blur (probe / defocus), background gradient, noise, and distractors (angular
platelets, not primaries). Labels: each primary's projected disc; where two
discs overlap, pixels go to the sphere of smallest power distance (radical
axis), which gives convex, star-convex pieces suited to StarDist.
"""
import numpy as np
import cv2


def _agregat(rng, n, d_med, sg, recouvrement=(0.0, 0.25)):
    """3-D sphere centres and radii of one aggregate. Growth mostly at the
    tips of chains (open, branched aggregates, Df ~ 1.6-2) with some random
    attachment (denser parts)."""
    r = 0.5 * d_med * np.exp(rng.normal(0.0, np.log(sg), n))
    r = np.clip(r, 0.35 * d_med, 1.8 * d_med)
    c = np.zeros((n, 3))
    p_chaine = rng.uniform(0.5, 0.9)
    for i in range(1, n):
        for _ in range(60):
            j = (i - 1 - rng.integers(0, min(i, 3))) if rng.random() < p_chaine \
                else rng.integers(0, i)
            v = rng.normal(size=3)
            v /= np.linalg.norm(v) + 1e-12
            ov = rng.uniform(*recouvrement)
            p = c[j] + v * (r[i] + r[j]) * (1.0 - ov)
            dd = np.linalg.norm(c[:i] - p, axis=1)
            if np.all(dd >= (r[:i] + r[i]) * (1.0 - recouvrement[1] - 0.05)):
                c[i] = p
                break
        else:
            c[i] = c[rng.integers(0, i)] + rng.normal(size=3) * r[i]
    return c, r


def image_te(H=512, W=512, seed=0, d_med=None, sg=None, n_agg=None,
             ppnm=None):
    """Returns (8-bit image with DARK particles on a bright background,
    instance labels, true primary diameters in px).
    d_med: median primary diameter in px (random 16-55 px if None)."""
    rng = np.random.default_rng(seed)
    d_med = d_med or rng.uniform(16, 55)
    sg = sg or rng.uniform(1.12, 1.40)
    n_agg = n_agg or rng.integers(3, 9)
    centres, rayons = [], []
    for _ in range(n_agg):
        n = int(rng.choice([1, 2, 4, 8, 15, 30, 60, 100],
                           p=[.08, .08, .12, .15, .2, .17, .12, .08]))
        c, r = _agregat(rng, n, d_med, sg)
        # random rotation, then position in the field
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        c = c @ q
        c[:, 0] += rng.uniform(0.1 * W, 0.9 * W) - c[:, 0].mean()
        c[:, 1] += rng.uniform(0.1 * H, 0.9 * H) - c[:, 1].mean()
        centres.append(c)
        rayons.append(r)
    C = np.concatenate(centres)
    R = np.concatenate(rayons)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    epaisseur = np.zeros((H, W), np.float32)
    puissance = np.full((H, W), np.inf, np.float32)
    lab = np.zeros((H, W), np.int32)
    vrais = []
    for k, ((x, y, _z), r) in enumerate(zip(C, R), start=1):
        x0, x1 = int(max(0, x - r - 1)), int(min(W, x + r + 2))
        y0, y1 = int(max(0, y - r - 1)), int(min(H, y + r + 2))
        if x1 <= x0 or y1 <= y0:
            continue
        rho2 = (xx[y0:y1, x0:x1] - x) ** 2 + (yy[y0:y1, x0:x1] - y) ** 2
        dedans = rho2 <= r * r
        epaisseur[y0:y1, x0:x1] += np.where(dedans,
                                            2 * np.sqrt(np.maximum(r * r - rho2, 0)), 0)
        pw = rho2 - r * r                       # power distance (< 0 inside)
        sous = puissance[y0:y1, x0:x1]
        prend = dedans & (pw < sous)
        sous[prend] = pw[prend]
        lab[y0:y1, x0:x1][prend] = k
        if dedans.any() and 0 <= x < W and 0 <= y < H:
            vrais.append(2 * r)
    # relabel consecutively (drop instances fully outside / hidden)
    ids = np.unique(lab[lab > 0])
    table = np.zeros(lab.max() + 1, np.int32)
    table[ids] = np.arange(1, ids.size + 1)
    lab = table[lab]
    # transmitted intensity: a single primary transmits 55-85 % at its
    # centre, multiple overlaps get darker (Beer-Lambert)
    mu = rng.uniform(0.16, 0.60) / d_med
    texture = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32),
                               (0, 0), max(1.5, 0.2 * d_med))
    texture = 1.0 + rng.uniform(0.0, 0.25) * texture / (texture.std() + 1e-6)
    I = np.exp(-mu * epaisseur * np.clip(texture, 0.3, 1.8))
    # distractors: dark angular platelets (not primaries, background label)
    for _ in range(rng.integers(0, 3)):
        k = rng.integers(3, 6)
        cx, cy = rng.uniform(0, W), rng.uniform(0, H)
        rr = rng.uniform(0.8, 2.0) * d_med
        a0 = rng.uniform(0, 2 * np.pi)
        pts = np.array([[cx + rr * np.cos(a0 + 2 * np.pi * i / k + rng.normal(0, .2)),
                         cy + rr * np.sin(a0 + 2 * np.pi * i / k + rng.normal(0, .2))]
                        for i in range(k)], np.int32)
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [pts], 1)
        I = np.where(m > 0, I * rng.uniform(0.55, 0.85), I)
    # smooth elastic distortion (image and labels): irregular primaries
    amp = rng.uniform(0.0, 0.10) * d_med
    if amp > 0.3:
        dx = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 0.5 * d_med)
        dy = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 0.5 * d_med)
        dx *= amp / (np.abs(dx).max() + 1e-6)
        dy *= amp / (np.abs(dy).max() + 1e-6)
        mx, my = xx + dx, yy + dy
        I = cv2.remap(I.astype(np.float32), mx, my, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REFLECT)
        lab = cv2.remap(lab.astype(np.float32), mx, my, cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT).astype(np.int32)
    bg = rng.uniform(170, 225) + rng.uniform(-15, 15) * (xx / W - .5) + \
        rng.uniform(-15, 15) * (yy / H - .5)
    img = bg * (I * rng.uniform(0.9, 1.0) + 0.0)
    img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), rng.uniform(1.2, 3.5))
    img += rng.normal(0, rng.uniform(2, 7), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, lab, np.asarray(vrais, float)


def jeu(n, H=512, W=512, graine=0):
    """n synthetic (image, labels) pairs."""
    X, Y = [], []
    for i in range(n):
        im, lab, _ = image_te(H, W, seed=graine + i)
        X.append(im)
        Y.append(lab)
    return X, Y
