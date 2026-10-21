# -*- coding: utf-8 -*-
"""PSA, Shape descriptors and annotation of detections."""
from psa.params import *  # noqa: F401,F403

def forme_contour(cnt, aire=None):
    """(rapport d'aspect par ellipse ajustée, solidité aire/aire convexe)."""
    if aire is None:
        aire = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    ah = cv2.contourArea(hull)
    sol = float(aire / ah) if ah > 0 else 0.0
    if len(cnt) >= 5:
        (_, _), (a1, a2), _ = cv2.fitEllipse(cnt)
        asp = float(max(a1, a2) / max(1e-6, min(a1, a2)))
    else:
        asp = 1.0
    return asp, min(1.0, sol)


def stats_forme(res, seuil=SEUIL_SOLIDITE_SATELLITE):
    """Synthèse de forme : aspect, solidité, % de particules satellitées
    (solidité < seuil — défaut classique des poudres atomisées gaz)."""
    out = {}
    sol = np.asarray(res.get("solidites", []), float)
    asp = np.asarray(res.get("aspects", []), float)
    if len(asp):
        out["aspect_moy"] = float(np.mean(asp))
        out["aspect_std"] = float(np.std(asp))
        out["pct_allongees"] = float((asp > 1.5).mean() * 100)
    if len(sol):
        out["solidite_moy"] = float(np.mean(sol))
        out["solidite_std"] = float(np.std(sol))
        out["pct_satellites"] = float((sol < seuil).mean() * 100)
    return out


def rendre_annotation(image_orig, contours, centres, cordes, zones=None):
    """Image annotée (contours/croix/cordes) — re-rendue après édition."""
    image_base = image_orig.copy()
    for k, cnt in enumerate(contours):
        cv2.drawContours(image_base, [cnt], -1, (251, 126, 49), 2)
        if k < len(centres):
            cv2.drawMarker(image_base, tuple(centres[k]), (0, 220, 100),
                           cv2.MARKER_CROSS, 10, 2)
        if k < len(cordes) and cordes[k][0] is not None:
            p1, p2 = cordes[k][0], cordes[k][1]
            cv2.line(image_base, tuple(int(v) for v in p1),
                     tuple(int(v) for v in p2), (0, 180, 255), 1)
    if zones:
        ov = image_base.copy()
        for z in zones:
            cv2.fillPoly(ov, [z], (90, 92, 255))
        image_base = cv2.addWeighted(image_base, 0.7, ov, 0.3, 0)
    return image_base


__all__ = [_n for _n in dir() if not _n.startswith("__")]
