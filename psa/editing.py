# -*- coding: utf-8 -*-
"""PSA, Shape descriptors, annotation, manual editing."""
from psa.detection import *  # noqa: F401,F403

# =============================================================================
# DESCRIPTEURS DE FORME, ANNOTATION, ÉDITION MANUELLE
# =============================================================================






def _ed_compteur(res, cle):
    ed = res.setdefault("editions",
                        {"suppr": 0, "ajout": 0, "scind": 0, "fusion": 0})
    ed[cle] += 1


def _ed_listes(res):
    """Les listes parallèles à maintenir lors des éditions."""
    cles = ["circularites", "aspects", "solidites"]
    return [c for c in cles if c in res]


def edition_index_a(res, x, y):
    """Particule sous le point (x, y) — None sinon."""
    xi, yi = int(round(x)), int(round(y))
    if "ids" in res and res.get("labels_ws") is not None:
        lab = res["labels_ws"]
        if not (0 <= yi < lab.shape[0] and 0 <= xi < lab.shape[1]):
            return None
        v = int(lab[yi, xi])
        return res["ids"].index(v) if v and v in res["ids"] else None
    for k, cnt in enumerate(res.get("contours", [])):
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bx <= xi <= bx + bw and by <= yi <= by + bh:
            if cv2.pointPolygonTest(cnt, (float(x), float(y)), False) >= 0:
                return k
    return None


def edition_supprimer(res, idx):
    if "ids" in res and res.get("labels_ws") is not None:
        res["labels_ws"][res["labels_ws"] == res["ids"][idx]] = 0
        del res["ids"][idx]
    for cle in ("contours", "cordes", "centres_px"):
        if cle in res and idx < len(res[cle]):
            del res[cle][idx]
    res["diametres"] = np.delete(res["diametres"], idx)
    for cle in _ed_listes(res):
        res[cle] = np.delete(res[cle], idx)
    res["n_particules"] = len(res["diametres"])
    _ed_compteur(res, "suppr")


def _edition_ajouter_contour(res, cnt, ppm, compteur="ajout"):
    """Ajoute une particule depuis un contour (unité du résultat)."""
    aire = cv2.contourArea(cnt)
    if aire < 5:
        return False
    peri = cv2.arcLength(cnt, True)
    circ = min(1.0, 4 * math.pi * aire / peri ** 2) if peri > 0 else 0.0
    asp, sol = forme_contour(cnt, aire)
    fac = 1000.0 if res["unite"] == "nm" else 1.0
    d = 2.0 * math.sqrt(aire / math.pi) / ppm * fac
    M = cv2.moments(cnt)
    cx = int(M["m10"] / M["m00"]) if M["m00"] else int(cnt[:, 0, 0].mean())
    cy = int(M["m01"] / M["m00"]) if M["m00"] else int(cnt[:, 0, 1].mean())
    res["diametres"] = np.append(res["diametres"], d)
    ajouts = {"circularites": circ, "aspects": asp, "solidites": sol}
    for cle in _ed_listes(res):
        res[cle] = np.append(res[cle], ajouts[cle])
    if "centres_px" in res:
        res["centres_px"].append((cx, cy))
    if "contours" in res:
        res["contours"].append(cnt)
    if "cordes" in res:
        res["cordes"].append(corde_max_v2(cnt))
    if "ids" in res and res.get("labels_ws") is not None:
        nouveau = int(res["labels_ws"].max()) + 1
        cv2.fillPoly(res["labels_ws"], [cnt], nouveau)
        res["ids"].append(nouveau)
    res["n_particules"] = len(res["diametres"])
    if compteur:
        _ed_compteur(res, compteur)
    return True


def edition_ajouter(res, points, ppm):
    cnt = np.array(points, np.int32).reshape(-1, 1, 2)
    return _edition_ajouter_contour(res, cnt, ppm, "ajout")


def _masque_particule(res, idx, shape):
    m = np.zeros(shape, np.uint8)
    if "ids" in res and res.get("labels_ws") is not None:
        m[res["labels_ws"] == res["ids"][idx]] = 255
    else:
        cv2.fillPoly(m, [res["contours"][idx]], 255)
    return m


def edition_scinder(res, idx, p1, p2, ppm):
    """Coupe la particule idx le long du trait p1→p2 (sur-fusion)."""
    shape = (res["labels_ws"].shape if res.get("labels_ws") is not None
             else res["masque"].shape)
    m = _masque_particule(res, idx, shape)
    cv2.line(m, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), 0, 3)
    n_cc, lab = cv2.connectedComponents(m, connectivity=8)
    if n_cc - 1 < 2:
        return 0
    edition_supprimer(res, idx)
    res["editions"]["suppr"] -= 1        # opération comptée comme scission
    n_ok = 0
    for v in range(1, n_cc):
        comp = np.uint8(lab == v) * 255
        if comp.sum() < 20 * 255:
            continue
        cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        if cs and _edition_ajouter_contour(res, max(cs, key=cv2.contourArea),
                                           ppm, None):
            n_ok += 1
    if n_ok >= 2:
        _ed_compteur(res, "scind")
    return n_ok


def edition_fusionner(res, i, j, ppm):
    """Fusionne deux particules sous-segmentées en une seule."""
    if i == j:
        return False
    shape = (res["labels_ws"].shape if res.get("labels_ws") is not None
             else res["masque"].shape)
    m = cv2.bitwise_or(_masque_particule(res, i, shape),
                       _masque_particule(res, j, shape))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                   (5, 5)))
    for idx in sorted((i, j), reverse=True):
        edition_supprimer(res, idx)
        res["editions"]["suppr"] -= 1    # opération comptée comme fusion
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return False
    if len(cs) == 1:
        cnt = cs[0]
    else:
        # Particules NON jointives (séparées par un creux/un interstice) :
        # la simple union laisse des morceaux disjoints et seul le plus gros
        # était conservé — l'autre disparaissait (« la fusion bugue »). On
        # prend l'enveloppe convexe de l'ensemble pour ne former qu'une seule
        # particule couvrant les deux.
        cnt = cv2.convexHull(np.vstack([c.reshape(-1, 2) for c in cs]))
    ok = _edition_ajouter_contour(res, cnt, ppm, None)
    if ok:
        _ed_compteur(res, "fusion")
    return ok


__all__ = [_n for _n in dir() if not _n.startswith("__")]
