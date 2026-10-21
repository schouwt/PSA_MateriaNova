# -*- coding: utf-8 -*-
"""PSA, synthetic SEM powder images with a KNOWN spheroidization rate, to
validate the spherical / irregular / satellite classification.

Particles: shaded spheres (SE-like), irregular particles (angular polygons,
elongated ellipses, sintered doublets) and satellites (small spheres touching
a large one). Returns the image and the true classes, diameters and rates.
"""
import numpy as np
import cv2


def image_poudre(H=900, W=1200, seed=0, frac_sph=0.7, n=70, d_med=70.0,
                 frac_sat=0.15):
    rng = np.random.default_rng(seed)
    img = np.full((H, W), 28.0, np.float32)
    occupe = np.zeros((H, W), bool)
    vrais = []                      # (class, ECD)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    def libre(m):
        return not (m & occupe).any()

    def sphere(cx, cy, r):
        rho2 = (xx - cx) ** 2 + (yy - cy) ** 2
        m = rho2 <= r * r
        z = np.sqrt(np.clip(1 - rho2 / (r * r), 0, 1))
        val = 70 + 140 * z ** 0.6 + 30 * (1 - z) ** 4       # SE shading + edge
        return m, val

    essais = 0
    while len([v for v in vrais if v[0] != "satellite"]) < n and essais < 4000:
        essais += 1
        d = float(np.clip(d_med * np.exp(rng.normal(0, 0.3)), 25, 180))
        r = d / 2
        cx, cy = rng.uniform(r + 5, W - r - 5), rng.uniform(r + 5, H - r - 5)
        if rng.random() < frac_sph:
            m, val = sphere(cx, cy, r)
            cl = "spherique"
        else:
            kind = rng.integers(3)
            m = np.zeros((H, W), np.uint8)
            if kind == 0:               # angular polygon
                k = rng.integers(4, 8)
                ang = np.sort(rng.uniform(0, 2 * np.pi, k))
                rr = r * rng.uniform(0.65, 1.25, k)
                pts = np.stack([cx + rr * np.cos(ang), cy + rr * np.sin(ang)], 1)
                cv2.fillPoly(m, [pts.astype(np.int32)], 1)
            elif kind == 1:             # elongated ellipse
                ar = rng.uniform(1.45, 2.6)
                cv2.ellipse(m, (int(cx), int(cy)), (int(r * np.sqrt(ar)),
                                                   int(r / np.sqrt(ar))),
                            float(rng.uniform(0, 180)), 0, 360, 1, -1)
            else:                       # sintered doublet
                a = rng.uniform(0, 2 * np.pi)
                for s in (-1, 1):
                    cv2.circle(m, (int(cx + s * 0.42 * r * np.cos(a)),
                                   int(cy + s * 0.42 * r * np.sin(a))),
                               int(0.62 * r), 1, -1)
            m = m.astype(bool)
            val = 120 + 60 * rng.random() + 25 * rng.normal(0, 1, (H, W))
            cl = "irreguliere"
        dil = cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        if not libre(dil):
            continue
        img = np.where(m, val, img)
        occupe |= m
        vrais.append((cl, 2 * np.sqrt(m.sum() / np.pi)))
        # satellites on some large spheres
        if cl == "spherique" and r > 0.8 * d_med / 2 and rng.random() < frac_sat * 3:
            for _ in range(rng.integers(1, 3)):
                rs = r * rng.uniform(0.12, 0.28)
                a = rng.uniform(0, 2 * np.pi)
                sx, sy = cx + (r + rs * 0.75) * np.cos(a), cy + (r + rs * 0.75) * np.sin(a)
                ms, vs = sphere(sx, sy, rs)
                ms &= ~m
                if ms.sum() < 10 or (ms & occupe).any():
                    continue
                img = np.where(ms, vs, img)
                occupe |= ms
                vrais.append(("satellite", 2 * np.sqrt(ms.sum() / np.pi)))
    img = cv2.GaussianBlur(img, (0, 0), 1.2)
    img += rng.normal(0, 5, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    cl = np.array([v[0] for v in vrais], object)
    d = np.array([v[1] for v in vrais], float)
    v3 = d ** 3
    sat = cl == "satellite"
    sp = cl == "spherique"
    verite = {"taux_nombre_hors_sat": 100 * sp.sum() / max((~sat).sum(), 1),
              "taux_masse_hors_sat": 100 * v3[sp].sum() / max(v3[~sat].sum(), 1e-30),
              "n_satellites": int(sat.sum()), "classes": cl, "d": d}
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), verite
