# -*- coding: utf-8 -*-
"""PSA, spheroidization rate of powders (plasma spheroidization, atomised
powders): every particle is classified SPHERICAL, IRREGULAR or SATELLITE,
and the rate is given by number and by mass (volume, d^3, same density).

Criteria (adjustable, written in the report):
  roundness R = 4 A / (pi F_max^2)  (1 for a disc; drops for elongated AND
      for angular shapes, even convex ones: 0.64 for a square). Preferred to
      the circularity 4 pi A / P^2, biased by the pixel perimeter (a perfect
      sphere saturates near 0.90), and to the solidity, close to 1 by
      construction for the star-convex StarDist contours;
  aspect ratio AR = major / minor axis;
  spherical: R >= R_min and AR <= AR_max;
  satellite: a particle touching a particle at least 1 / ratio times larger
      (default ratio 0.35); counted apart, and the particles bearing them
      are reported.
"""
from psa.database import *  # noqa: F401,F403

SPHERO_DEFAUT = {"rondeur_min": 0.80, "aspect_max": 1.20,
                 "satellite_ratio": 0.35, "contact_px": 2}


def _feret_max(c):
    pts = c.reshape(-1, 2).astype(float)
    if len(pts) < 2:
        return 0.0
    h = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)
    d = np.sqrt(((h[:, None, :] - h[None, :, :]) ** 2).sum(-1))
    return float(d.max())


def classer_sphericite(res, params=None):
    """Classification of every particle of a particle result (optical or SEM
    powder modes). Returns a dict: classes (array of 'spherique' /
    'irreguliere' / 'satellite'), rondeur, aspect, porteuses (particles
    bearing satellites), the rates and the criteria used."""
    p = dict(SPHERO_DEFAUT)
    p.update(params or {})
    d = np.asarray(res.get("diametres", []), float)
    n = d.size
    vide = {"n": 0, "classes": np.array([], object), "criteres": p}
    if n == 0 or res.get("mode") not in ("optique", "meb"):
        return vide
    cs = contours_du_resultat(res)
    if len(cs) != n:
        return vide
    rond = np.zeros(n)
    for i, c in enumerate(cs):
        a = float(cv2.contourArea(c))
        f = _feret_max(c)
        rond[i] = min(1.0, 4 * a / (np.pi * f * f)) if f > 0 else 0.0
    asp = np.asarray(res.get("aspects", np.ones(n)), float)
    if asp.size != n:
        asp = np.ones(n)
    # contacts: particle masks drawn with their index, dilated by contact_px
    H = max(int(max(c[:, 0, 1].max() for c in cs)) + 5, 10)
    W = max(int(max(c[:, 0, 0].max() for c in cs)) + 5, 10)
    lab = np.zeros((H, W), np.int32)
    for i, c in enumerate(cs):
        cv2.drawContours(lab, [c], -1, i + 1, -1)
    k = 2 * int(p["contact_px"]) + 1
    ker = np.ones((k, k), np.uint8)
    satellite = np.zeros(n, bool)
    porteuse = np.zeros(n, bool)
    for i, c in enumerate(cs):
        x, y, w, h = cv2.boundingRect(c)
        x0, y0 = max(0, x - k), max(0, y - k)
        x1, y1 = min(W, x + w + k), min(H, y + h + k)
        m = (lab[y0:y1, x0:x1] == i + 1).astype(np.uint8)
        anneau = cv2.dilate(m, ker) > 0
        voisins = np.unique(lab[y0:y1, x0:x1][anneau])
        voisins = voisins[(voisins > 0) & (voisins != i + 1)] - 1
        if voisins.size:
            grand = voisins[np.argmax(d[voisins])]
            if d[i] <= p["satellite_ratio"] * d[grand]:
                satellite[i] = True
                porteuse[grand] = True
    spher = (rond >= p["rondeur_min"]) & (asp <= p["aspect_max"])
    classes = np.where(satellite, "satellite",
                       np.where(spher, "spherique", "irreguliere")).astype(object)
    return taux_sphero(classes, porteuse, d, p, rond, asp)


def taux_sphero(classes, porteuse, d, criteres, rond=None, asp=None):
    """Class fractions by number and by mass (d³) from the per-particle
    classes; shared by one image and by a batch (classes concatenated)."""
    classes = np.asarray(classes, object)
    porteuse = np.asarray(porteuse, bool)
    d = np.asarray(d, float)
    n = len(classes)
    p = criteres
    v = d ** 3
    pc = lambda m: float(100.0 * m.sum() / n)                       # noqa: E731
    pv = lambda m: float(100.0 * v[m].sum() / max(v.sum(), 1e-30))  # noqa: E731
    sp, ir, sa = classes == "spherique", classes == "irreguliere", \
        classes == "satellite"
    return {
        "n": n, "classes": classes, "rondeur": rond, "aspect": asp,
        "d": d,
        "porteuses": porteuse, "criteres": p,
        "taux_nombre": pc(sp), "taux_masse": pv(sp),
        "irreg_nombre": pc(ir), "irreg_masse": pv(ir),
        "satellites_nombre": pc(sa), "satellites_masse": pv(sa),
        "porteuses_pct": float(100.0 * porteuse.sum() / n),
        # spheroidization rate among the main particles (satellites excluded),
        # the usual definition for plasma spheroidization yield
        "taux_nombre_hors_sat": float(100.0 * sp.sum() / max((~sa).sum(), 1)),
        "taux_masse_hors_sat": float(100.0 * v[sp].sum() / max(v[~sa].sum(), 1e-30)),
    }



def sphericite(res, params=None):
    """Cached classification, recomputed when the particle list changes
    (manual editing, size filter) or the criteria change."""
    if res.get("mode") not in ("optique", "meb"):
        return None
    cle = (int(res.get("n_particules", 0)), str(res.get("editions", {})),
           str(sorted((params or {}).items())))
    c = res.get("_sphero")
    if c is None or c.get("_cle") != cle:
        c = classer_sphericite(res, params)
        c["_cle"] = cle
        res["_sphero"] = c
    return c if c.get("n") else None


__all__ = [_n for _n in dir() if not _n.startswith("__")]
