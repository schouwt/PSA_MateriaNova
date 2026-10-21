# -*- coding: utf-8 -*-
"""PSA, SQLite history, statistical comparison, ImageJ ROIs."""
from psa.stats import *  # noqa: F401,F403

# =============================================================================
# HISTORIQUE SQLITE  +  COMPARAISON STATISTIQUE  (contrôle qualité)
# =============================================================================
def db_connecter():
    DOSSIER_CONF.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(FICHIER_DB)
    con.execute("""CREATE TABLE IF NOT EXISTS analyses(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT, image TEXT, operateur TEXT, recette TEXT, mode TEXT,
        unite TEXT, ppm REAL, n INTEGER, dn50 REAL, dv50 REAL, d32 REAL,
        d43 REAL, span REAL, sigma_g REAL, score REAL, editions TEXT,
        diametres TEXT)""")
    return con


def db_enregistrer(session):
    res = session.resultats
    d = res["diametres"]
    if not len(d):
        return None
    st = stats_distribution(d)
    con = db_connecter()
    cur = con.execute(
        "INSERT INTO analyses(date,image,operateur,recette,mode,unite,ppm,"
        "n,dn50,dv50,d32,d43,span,sigma_g,score,editions,diametres) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (datetime.datetime.now().isoformat(timespec="seconds"),
         session.nom_image, session.operateur, session.recette,
         session.mode, res["unite"], session.ppm, int(st["n"]),
         st["dn50"], st["dv50"], st["D32"], st["D43"], st["span"],
         st["lognorm"][1], float(res["score_ws"]),
         json.dumps(res.get("editions", {})),
         json.dumps([round(float(x), 4) for x in d])))
    con.commit()
    ident = cur.lastrowid
    con.close()
    return ident


def db_lister(recette=None, limite=300):
    if not FICHIER_DB.exists():
        return []
    con = db_connecter()
    if recette:
        rows = con.execute(
            "SELECT id,date,image,recette,unite,n,dn50,score FROM analyses "
            "WHERE recette=? ORDER BY date DESC LIMIT ?",
            (recette, limite)).fetchall()
    else:
        rows = con.execute(
            "SELECT id,date,image,recette,unite,n,dn50,score FROM analyses "
            "ORDER BY date DESC LIMIT ?", (limite,)).fetchall()
    con.close()
    return rows


def db_recettes():
    if not FICHIER_DB.exists():
        return []
    con = db_connecter()
    r = [x[0] for x in con.execute(
        "SELECT DISTINCT recette FROM analyses ORDER BY recette").fetchall()]
    con.close()
    return r


def db_diametres(ident):
    con = db_connecter()
    row = con.execute("SELECT diametres,unite,image,date FROM analyses "
                      "WHERE id=?", (ident,)).fetchone()
    con.close()
    if row is None:
        return None, None, None, None
    return np.array(json.loads(row[0])), row[1], row[2], row[3]


def db_tendance(recette):
    con = db_connecter()
    rows = con.execute("SELECT date,dn50,dv50 FROM analyses WHERE recette=? "
                       "ORDER BY date", (recette,)).fetchall()
    con.close()
    return rows


def comparer_distributions(d_ref, d_test):
    """
    Le lot testé est-il conforme à la référence ?
    Kolmogorov-Smirnov (forme globale) + Mann-Whitney (décalage de médiane).
    """
    sps = _sps
    d_ref = np.asarray(d_ref, float)
    d_test = np.asarray(d_test, float)
    ks = sps.ks_2samp(d_ref, d_test)
    try:
        mw = sps.mannwhitneyu(d_ref, d_test, alternative="two-sided")
        p_mw = float(mw.pvalue)
    except ValueError:
        p_mw = 1.0
    delta = float(np.median(d_test) - np.median(d_ref))
    delta_rel = delta / float(np.median(d_ref)) * 100 if np.median(d_ref) else 0
    if ks.pvalue >= 0.05 and p_mw >= 0.05:
        verdict, msg = "good", (
            "Distributions statistically compatible with the reference "
            f"(KS p={ks.pvalue:.3f}, MW p={p_mw:.3f}).")
    else:
        verdict, msg = "warn", (
            f"Distributions DIFFERENT from the reference (KS p={ks.pvalue:.3g}"
            f", MW p={p_mw:.3g}), median shift "
            f"{delta:+.2f} ({delta_rel:+.1f} %).")
    return {"ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue),
            "mw_p": p_mw, "delta_med": delta, "delta_rel": delta_rel,
            "verdict": verdict, "message": msg}


# =============================================================================
# INTEROPÉRABILITÉ ROI IMAGEJ  (pip install roifile)
# =============================================================================
def contours_du_resultat(res):
    """Contours de toutes les particules, quel que soit le mode."""
    if "contours" in res:
        return list(res["contours"])
    cnts = []
    lab = res.get("labels_ws")
    if lab is None:
        return cnts
    # Recadrage par boîte englobante : findContours sur l'image entière
    # pour chaque label coûtait O(n_labels × pixels).
    from scipy import ndimage as _ndi
    boites = _ndi.find_objects(lab)
    for v in res.get("ids", []):
        tr = boites[v - 1] if 0 < v <= len(boites) else None
        if tr is None:
            continue
        m = np.uint8(lab[tr] == v) * 255
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        if cs:
            c = max(cs, key=cv2.contourArea)
            c[:, 0, 0] += tr[1].start
            c[:, 0, 1] += tr[0].start
            cnts.append(c)
    return cnts


def exporter_rois_imagej(res, chemin_zip):
    if not ROI_OK:
        raise RuntimeError("The « roifile » package is required "
                           "(pip install roifile).")
    rois = []
    for k, cnt in enumerate(contours_du_resultat(res)):
        r = roifile.ImagejRoi.frompoints(
            cnt.reshape(-1, 2).astype(np.float32))
        r.name = f"P{k+1:04d}"
        rois.append(r)
    if not rois:
        raise RuntimeError("No particle to export.")
    Path(chemin_zip).unlink(missing_ok=True)
    roifile.roiwrite(chemin_zip, rois)
    return len(rois)


def importer_rois_imagej(chemin):
    if not ROI_OK:
        raise RuntimeError("The « roifile » package is required "
                           "(pip install roifile).")
    lus = roifile.roiread(chemin)
    if not isinstance(lus, list):
        lus = [lus]
    zones = []
    for r in lus:
        pts = np.asarray(r.coordinates(), float)
        if len(pts) >= 3:
            zones.append(np.round(pts).astype(np.int32).reshape(-1, 1, 2))
    return zones


__all__ = [_n for _n in dir() if not _n.startswith("__")]
