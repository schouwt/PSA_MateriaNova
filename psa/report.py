# -*- coding: utf-8 -*-
"""PSA, Statistics, report figure, exports, archiving."""
from psa.session import *  # noqa: F401,F403

# =============================================================================
# EXPORTS  (CSV / PDF, thème clair imprimable, garde-fous inclus)
# =============================================================================
def avertissements_qualite(res):
    av = []
    rep = evaluer_repr(res["diametres"])
    if rep["n"] < 30:
        av.append(f"n = {rep['n']} < 30: distribution not representative "
                  f"(aim for n ≥ {max(30, rep['n_min_10pct'])}, "
                  "use the Batch mode).")
    if res["score_ws"] < 45:
        av.append(f"Segmentation score {res['score_ws']:.0f}/100: "
                  "check the results visually.")
    if res["mode"] == "optique" and rep["n"] < 100:
        av.append("Volume-weighted Dx are sensitive to large isolated "
                  "particles (d³ weighting).")
    d = res["diametres"]
    if len(d):
        dn50 = float(np.percentile(d, 50))
        if res["unite"] == "nm" and dn50 > 2000:
            av.append(f"Dn50 ≈ {dn50/1000:.0f} µm: micron sizes, "
                      "use the « SEM, micron powders (µm) » mode "
                      "for readable statistics.")
        if res["mode"] == "meb" and dn50 < 1.0:
            av.append(f"Dn50 ≈ {dn50*1000:.0f} nm: nanometric sizes, "
                      "use the « SEM, nano (nm) » mode.")
    if res["mode"] == "gouttes":
        sg = res.get("gouttes_stats") or {}
        if sg.get("n", 0) > 1 and sg.get("d_moy"):
            gap = 100 * abs(sg["ecart_convention"]) / sg["d_moy"]
            if gap > 8:
                av.append(f"Outer edge and ring middle differ by {gap:.0f} % "
                          "of the diameter (volume about "
                          f"{(1 + gap / 100) ** 3:.2f}×): the diameter "
                          "convention dominates the uncertainty, check it "
                          "against V = Q / f.")
        if not res.get("ressemble", True):
            av.append("The image does not look like droplets seen as dark "
                      "rings: check the detections visually.")
    return av


# =============================================================================
# REPORT AND EXPORT  (Materia Nova template, as in Grating Analyzer:
# multi-panel report PNG / PDF, CSV tables with header lines, text summary,
# parameters JSON, all in a <name>_analysis folder)
# =============================================================================
CSV_MODES = {"International ( , separator, . decimal)": "intl",
             "Excel FR/BE ( ; separator, , decimal)": "excel"}
FIG_STYLES = {"Publication (white background)": "light", "Current theme": "theme"}
CSV_STYLE = {"mode": "intl"}
_NUM_RE = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?")


def fmt(x, spec=".4g"):
    if x is None:
        return "–"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(xf):
        return "–"
    if math.isinf(xf):
        return "∞"
    return format(xf, spec)


def safe_name(s):
    return re.sub(r"[^\w.-]+", "_", str(s).strip()) or "x"


def _csv_cell(x):
    s = str(x)
    if s == "–":
        return ""
    if s == "∞":
        return "inf"
    if CSV_STYLE["mode"] == "excel" and _NUM_RE.fullmatch(s):
        s = s.replace(".", ",")
    return s


def write_csv(path, header_lines, columns, rows):
    """CSV with '# ' header lines; flavour from CSV_STYLE (template)."""
    enc = "utf-8-sig" if CSV_STYLE["mode"] == "excel" else "utf-8"
    with open(path, "w", newline="", encoding=enc) as f:
        for line in header_lines:
            f.write(f"# {line}\n")
        w = csv.writer(f, delimiter=";" if CSV_STYLE["mode"] == "excel" else ",")
        w.writerow(columns)
        for r in rows:
            w.writerow([_csv_cell(c) for c in r])


from psa.modes import MODES, MODE_NOMS, mode_de  # noqa: E402,F401


def classe_monodispersite(cv):
    """Usual microfluidics wording for the size CV (%)."""
    return ("monodisperse" if cv < 3 else "narrow" if cv < 5
            else "polydisperse")


def lignes_resume(session):
    """Summary rows (key, label, value, unit) shared by the report table, the
    summary CSV and the text summary."""
    res = session.resultats
    d = np.asarray(res["diametres"], float)
    u = res["unite"]
    fib = res["mode"] == "fibres"
    nom = "fibre diameter" if fib else "diameter"
    gt = res["mode"] == "gouttes"
    R = [("mode", "Analysis mode", MODE_NOMS.get(res["mode"], res["mode"]), ""),
         ("n", "Measurement sites" if fib else ("Droplets" if gt else "Particles"),
          int(res["n_particules"]), "")]
    if not fib:
        R.append(("n_edge", "Excluded (touching the border)", int(res.get("n_bord", 0)), ""))
        if float(res.get("taille_max", 0) or 0) > 0:
            R.append(("n_too_large",
                      f"Excluded (above the maximum size {fmt(res['taille_max'], '.4g')} {u})",
                      int(res.get("n_trop_grands", 0)), ""))
    if d.size:
        st = stats_distribution(d)
        dv10, dv50, dv90 = dx_volume(d)
        R += [("mean", f"Mean {nom}", float(np.mean(d)), u),
              ("std", "Standard deviation", float(np.std(d)), u),
              ("min", "Minimum", float(np.min(d)), u),
              ("max", "Maximum", float(np.max(d)), u),
              ("dn10", "Dn10 (number)", st["dn10"], u),
              ("dn50", "Dn50 (number)", st["dn50"], u),
              ("dn90", "Dn90 (number)", st["dn90"], u)]
        if not fib:
            R += [("dv10", "Dv10 (volume)", dv10, u),
                  ("dv50", "Dv50 (volume)", dv50, u),
                  ("dv90", "Dv90 (volume)", dv90, u),
                  ("d32", "D[3,2] Sauter", st["D32"], u),
                  ("d43", "D[4,3] De Brouckere", st["D43"], u),
                  ("span", "Span (Dv90 − Dv10) / Dv50", st["span"], "")]
        if st.get("lognorm") and st["lognorm"][1] > 1.0:
            R.append(("sigma_g", "Geometric std σg (log-normal)", st["lognorm"][1], ""))
        if st.get("meilleur_fit"):
            R.append(("best_fit", "Best fit (AIC)", NOMS_FIT.get(st["meilleur_fit"], st["meilleur_fit"]), ""))
    c = np.asarray(res.get("circularites", []), float)
    if c.size and not fib:
        R.append(("circ", "Mean circularity", float(np.mean(c)), ""))
        R.append(("circ_ms", "Mean circularity ± std",
                  f"{np.mean(c):.3f} ± {np.std(c, ddof=1) if c.size > 1 else 0.0:.3f}",
                  ""))
        sf = stats_forme(res)
        if sf.get("solidite_moy") is not None:
            R += [("aspect", "Mean aspect ratio", sf.get("aspect_moy", 0.0), ""),
                  ("solidity", "Mean solidity", sf.get("solidite_moy", 0.0), ""),
                  ("satellites", f"Satellite particles (solidity < {SEUIL_SOLIDITE_SATELLITE})",
                   sf.get("pct_satellites", 0.0), "%")]
    if fib:
        fr = res["fibres"]
        R += [("beads", "Beads", int(fr["n_billes"]), ""),
              ("bead_density", "Bead density", fr["densite_billes_mm2"], "/mm²"),
              ("alignment", "Alignment (0 random, 1 aligned)", fr["alignement"], ""),
              ("orientation", "Main orientation", fr["orientation_deg"], "°"),
              ("mat_porosity", "Mat porosity", fr["porosite_pct"], "%")]
        if fr.get("n_pores"):
            R += [("fibre_porosity", "Fibre porosity", fr["porosite_fibre_pct"], "%"),
                  ("pores", "Pores", int(fr["n_pores"]), ""),
                  ("pore_d50", "Median pore diameter", fr["pore_d_median_um"] * 1000, "nm")]
    if gt:
        sg = res.get("gouttes_stats") or {}
        h = res.get("hauteur_canal_um")
        R.append(("drop_conv", "Diameter measured at",
                  "outer edge" if res.get("convention") == "ext"
                  else "ring middle", ""))
        if sg.get("n", 0) > 1:
            R += [("drop_cv", "Size CV", sg["cv_pct"], "%"),
                  ("drop_class", "Size distribution",
                   classe_monodispersite(sg["cv_pct"]), ""),
                  ("drop_gap", "Outer edge − ring middle",
                   abs(sg["ecart_convention"]), u),
                  ("drop_gap_pct", "Convention gap",
                   100 * abs(sg["ecart_convention"]) / sg["d_moy"], "%"),
                  ("drop_vol", "Mean droplet volume", sg["v_moy"] * 1e-3, "pL")]
        R.append(("drop_vmodel", "Volume model",
                  f"squeezed disc, channel height {h:g} µm" if h else "spheres",
                  ""))
        a_ = np.asarray(res.get("aspects", []), float)
        if a_.size:
            R.append(("drop_aspect", "Mean aspect ratio (ellipse fit)",
                      float(np.mean(a_)), ""))
    if res["mode"] == "nano":
        nr = res["nano"]
        if nr.get("methode") == "stardist_deux_etapes":
            from scipy.stats import gmean
            R.append(("imaging", "Imaging", "TE (transmission)" if nr["imagerie"] == "TE"
                      else "SE (experimental)", ""))
            if d.size:
                R.append(("dp_gm", "Primary dp, geometric mean", float(gmean(d)), "nm"))
                R.append(("dp_sg", "Primary dp, geometric std σg",
                          float(np.exp(np.std(np.log(d)))), ""))
            agg = [a for a in nr.get("agregats", []) if not a["bord"]]
            R.append(("n_agg", "Complete aggregates", len(agg), ""))
            if agg:
                R.append(("np_moy", "Primaries per aggregate (mean)",
                          float(np.mean([a["np_aire"] for a in agg])), ""))
                R.append(("rg_moy", "Radius of gyration Rg (mean)",
                          float(np.mean([a["rg_nm"] for a in agg])), "nm"))
            if nr.get("df"):
                R.append(("kf", "Fractal prefactor kf", nr["kf"], ""))
                R.append(("df_r2", "Fractal fit R²", nr["df_r2"], ""))
            else:
                R.append(("df", "Fractal dimension Df",
                          "n/a (too few aggregates)", ""))
        if nr.get("df"):
            R.append(("df", "Fractal dimension Df", nr["df"], ""))
    if d.size:
        rep = evaluer_repr(d)
        R += [("quality", "Segmentation quality score", res["score_ws"], "/100"),
              ("cv", "Coefficient of variation", rep["cv"], "%"),
              ("representativeness", "Representativeness",
               {"good": "good", "warn": "limited", "bad": "insufficient"}.get(
                   rep["verdict"], rep["verdict"]), "")]
    if session.bet_m2g and d.size:
        cb = comparer_bet(stats_distribution(d)["D32"], u, session.rho_gcm3, session.bet_m2g)
        R += [("ssa_geom", "Geometric SSA 6/(ρ·D[3,2])", cb["ssa_geom"], "m²/g"),
              ("ssa_bet", "Measured BET SSA", session.bet_m2g, "m²/g"),
              ("bet_factor", "BET / geometric factor", cb["facteur"], "")]
    ed = res.get("editions", {})
    if any(ed.values()):
        R.append(("edits", "Manual edits",
                  f"{ed['suppr']} deleted, {ed['ajout']} added, {ed['scind']} split, "
                  f"{ed['fusion']} merged", ""))
    if res["mode"] in ("meb", "optique"):
        sd = res.get("moteur") == "stardist"
        R.append(("engine", "Segmentation engine",
                  f"StarDist ({res.get('stardist', {}).get('modele', STARDIST_MODELE)})"
                  if sd else "threshold + watershed", ""))
    sph = sphericite(res, getattr(session, "sphero", None))
    if sph:
        R += [("sph_n", "Spheroidization rate (number)", sph["taux_nombre_hors_sat"], "%"),
              ("sph_m", "Spheroidization rate (mass)", sph["taux_masse_hors_sat"], "%"),
              ("sat_n", "Satellites (number)", sph["satellites_nombre"], "%"),
              ("porteuses", "Particles bearing satellites", sph["porteuses_pct"], "%")]
    R.append(("calibration", "Calibration", session.ppm, "px/µm"))
    uc = incertitude_calibration(session)
    R.append(("u_cal", "Calibration uncertainty (k = 2)",
              100 * uc["U_rel"] if uc["U_rel"] > 0 else "not evaluated", "%"
              if uc["U_rel"] > 0 else ""))
    return R


def _val(v, spec=".4g"):
    return v if isinstance(v, str) else fmt(v, spec)


def image_annotee_rapport(session):
    """RGB image with the detections drawn (particles or fibre segments)."""
    res = session.resultats
    img = session.image_bgr
    if res["mode"] == "fibres" and FIB_OK:
        vis = fibres.image_annotee_fibres(img, res["fibres"])
    else:
        g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        ep = max(1, int(round(max(vis.shape[:2]) / 900)))
        cv2.drawContours(vis, contours_du_resultat(res), -1, (62, 168, 224),
                         max(2, ep + 1) if res["mode"] == "nano" else ep)
        nr = res.get("nano") or {}
        if nr.get("methode") == "stardist_deux_etapes":
            for (x, y, r) in nr.get("cercles_hough", []):
                cv2.circle(vis, (int(x), int(y)), max(1, int(r)), (40, 190, 60),
                           max(2, ep + 1), cv2.LINE_AA)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def exporter_resultats(dossier, stem, session, pal=None, dpi=300,
                       formats=("png", "pdf")):
    """Report figure, per-object CSV, summary CSV and TXT, parameters JSON
    (template file set). Returns the list of written files."""
    os.makedirs(dossier, exist_ok=True)
    pal = pal or LIGHT_PALETTE
    res = session.resultats
    u = res["unite"]
    fichiers = []
    head = [f"{APP_NAME} {APP_VERSION}", f"sample: {session.nom_echantillon}",
            f"image: {session.nom_image}",
            f"exported: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
            f"operator: {session.operateur or '–'}",
            f"recipe: {session.recette or '–'}",
            f"mode: {MODE_NOMS.get(res['mode'], res['mode'])}",
            f"calibration: {fmt(session.ppm, '.6g')} px/µm"]
    fichiers += ecrire_rapport(session, dossier, stem, formats, dpi)
    # Optional LaTeX version (Materia Nova template), off by default: when
    # chosen and compiled, it replaces the built-in PDF; sources exported
    session.moteur_rapport_msg = ""
    if "pdf" in formats and getattr(session, "rapport_latex", False):
        try:
            from psa.report_latex import rapport_latex      # later layer
            pdf_tex, zip_tex, msg = rapport_latex(session, dossier, stem)
            session.moteur_rapport_msg = msg
            if zip_tex:
                fichiers.append(zip_tex)
        except Exception as e:                     # never block the export
            journal("error", f"LaTeX report: {e}")
            session.moteur_rapport_msg = f"LaTeX report failed ({e}): built-in report used"
    d = res["diametres"]
    if res["mode"] == "fibres":
        fr = res["fibres"]
        ray = fr.get("rayons")
        f = os.path.join(dossier, f"{stem}_fibre_sites.csv")
        write_csv(f, head, ["site", f"diameter_{u}"],
                  [[i + 1, fmt(v, ".6g")] for i, v in enumerate(d)])
        fichiers.append(f)
        if ray is not None and len(ray["larg"]):
            f = os.path.join(dossier, f"{stem}_fibre_segments.csv")
            fac = 1000.0 if u == "nm" else 1.0
            write_csv(f, head, ["segment", "x0_px", "y0_px", "x1_px", "y1_px",
                                f"width_{u}"],
                      [[i + 1, fmt(a[1], ".6g"), fmt(a[0], ".6g"), fmt(b[1], ".6g"),
                        fmt(b[0], ".6g"), fmt(L / session.ppm * fac, ".6g")]
                       for i, (a, b, L) in enumerate(zip(ray["dep"], ray["arr"],
                                                         ray["larg"]))])
            fichiers.append(f)
    elif res["mode"] == "gouttes":
        g = res["gouttes"]
        ump = res.get("um_par_px", 1.0)
        from psa.droplets import volume_goutte
        h_ = res.get("hauteur_canal_um")
        ok = ~g["bord"]
        rows, k = [], 0
        for i in range(len(g["cx"])):
            dm, de = g["d_milieu"][i] * ump, g["d_ext"][i] * ump
            mes = d[k] if ok[i] else None
            if ok[i]:
                k += 1
            rows.append([i + 1, fmt(g["cx"][i], ".6g"), fmt(g["cy"][i], ".6g"),
                         fmt(dm, ".6g"), fmt(de, ".6g") if np.isfinite(de) else "–",
                         fmt(mes, ".6g") if mes is not None else "–",
                         fmt(float(volume_goutte(mes, h_)) * 1e-3, ".6g")
                         if mes is not None else "–",
                         fmt(g["aspect"][i], ".4g"), fmt(g["rms"][i], ".3g"),
                         "yes" if g["bord"][i] else "no"])
        f = os.path.join(dossier, f"{stem}_droplets.csv")
        write_csv(f, head + [f"diameter convention: "
                             f"{'outer edge' if res.get('convention') == 'ext' else 'ring middle'}"],
                  ["droplet", "x_px", "y_px", f"d_ring_middle_{u}",
                   f"d_outer_edge_{u}", f"d_measured_{u}", "volume_pL",
                   "aspect_ratio", "fit_rms_px", "cut_by_edge"], rows)
        fichiers.append(f)
    else:
        c, asp = res.get("circularites", []), res.get("aspects", [])
        sol, cordes = res.get("solidites", []), res.get("cordes", [])
        fac = 1000.0 if u == "nm" else 1.0

        def _g(a, i, spec):
            return fmt(a[i], spec) if i < len(a) else "–"
        rows = [[i + 1, fmt(d[i], ".6g"),
                 fmt(cordes[i][2] / session.ppm * fac, ".6g")
                 if i < len(cordes) and cordes[i][2] else "–",
                 _g(c, i, ".4f"), _g(asp, i, ".4g"), _g(sol, i, ".4f")]
                for i in range(len(d))]
        f = os.path.join(dossier, f"{stem}_particles.csv")
        write_csv(f, head, ["particle", f"diameter_{u}", f"max_chord_{u}",
                            "circularity", "aspect_ratio", "solidity"], rows)
        fichiers.append(f)
    lignes = lignes_resume(session)
    f = os.path.join(dossier, f"{stem}_summary.csv")
    write_csv(f, head, ["quantity", "value", "unit"],
              [[lb, _val(v, ".8g"), un] for _, lb, v, un in lignes])
    fichiers.append(f)
    f = os.path.join(dossier, f"{stem}_summary.txt")
    with open(f, "w", encoding="utf-8") as fh:
        fh.write("\n".join(head) + "\n" + "=" * 64 + "\n")
        for _, lb, v, un in lignes:
            fh.write(f"{lb:<44s} {_val(v)} {un}\n")
        for a in avertissements_qualite(res):
            fh.write(f"WARNING: {a}\n")
    fichiers.append(f)
    f = os.path.join(dossier, f"{stem}_parameters.json")
    with open(f, "w", encoding="utf-8") as fh:
        json.dump({"app": f"{APP_NAME} {APP_VERSION}", "mode": res["mode"],
                   "recipe": session.recette, "calibration_px_per_um": session.ppm,
                   "params": {k: (v if isinstance(v, (int, float, str, bool,
                                                      type(None))) else str(v))
                              for k, v in session.params.items()}},
                  fh, indent=2, ensure_ascii=False)
    fichiers.append(f)
    return fichiers


def archiver_analyse(session):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(session.chemin).parent / "PSA_analyses" / \
           f"{ts}_{Path(session.chemin).stem}"
    base.mkdir(parents=True, exist_ok=True)
    exporter_resultats(str(base), Path(session.chemin).stem, session,
                       formats=("pdf",), dpi=200)
    meta = {
        "application": APP_TITLE,
        "date":       datetime.datetime.now().isoformat(timespec="seconds"),
        "image":      session.chemin,
        "operateur":  session.operateur,
        "recette":    session.recette,
        "mode":       session.mode,
        "ppm_px_par_um": session.ppm,
        "rho_gcm3":   session.rho_gcm3,
        "bet_m2g":    session.bet_m2g,
        "mesures_epaisseur_um": [round(m["um"], 4)
                                 for m in session.mesures],
        "zones_exclusion": [z.reshape(-1, 2).tolist() for z in session.zones],
        "parametres": session.params,
        "n_particules": session.resultats["n_particules"],
        "score_segmentation": session.resultats["score_ws"],
        "avertissements": avertissements_qualite(session.resultats),
    }
    meta["editions_manuelles"] = session.resultats.get("editions", {})
    (base / "archive.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        db_enregistrer(session)
    except Exception as e:
        journal("warning", f"SQLite archiving failed: {e}")
    return str(base)


# =============================================================================
# SCIENTIFIC MULTI-PAGE REPORT (A4 portrait, Materia Nova / UMONS / CHIPS)
# =============================================================================
A4 = (8.27, 11.69)
LIGNE_TABLE = 0.0172       # height of a table row, as a fraction of the page
NAVY, ENCRE, GRIS, CLAIR = MN_DARK, "#1a1a1a", "#7a7a7a", "#e6eaf0"
STYLE_SCI = {
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.linewidth": 0.8, "axes.edgecolor": ENCRE, "axes.labelcolor": ENCRE,
    "axes.labelsize": 9, "axes.titlesize": 9.5, "axes.facecolor": "white",
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 4, "ytick.major.size": 4,
    "xtick.minor.size": 2, "ytick.minor.size": 2,
    "xtick.color": ENCRE, "ytick.color": ENCRE,
    "legend.frameon": False, "legend.fontsize": 8,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    # independent of the screen theme (ManoDark sets light text colours)
    "text.color": ENCRE, "axes.titlecolor": ENCRE, "legend.labelcolor": ENCRE,
    "figure.edgecolor": "white", "savefig.edgecolor": "white",
    "axes.prop_cycle": matplotlib.cycler(color=[ENCRE, GRIS, MN_LIGHT]),
    "grid.color": "#d5dbe3", "axes.grid": False,
}
REFS_STARDIST = [
    "U. Schmidt, M. Weigert, C. Broaddus, G. Myers. Cell Detection with "
    "Star-Convex Polygons. MICCAI 2018, 21st International Conference, "
    "Granada, Spain, Proceedings Part II, pp. 265-273, 2018. "
    "doi:10.1007/978-3-030-00934-2_30",
    "M. Weigert, U. Schmidt, R. Haase, K. Sugawara, G. Myers. Star-convex "
    "Polyhedra for 3D Object Detection and Segmentation in Microscopy. "
    "IEEE Winter Conference on Applications of Computer Vision (WACV), "
    "March 2020. doi:10.1109/WACV45572.2020.9093435",
    "M. Weigert, U. Schmidt. Nuclei Instance Segmentation and Classification "
    "in Histopathology Images with StarDist. IEEE International Symposium on "
    "Biomedical Imaging Challenges (ISBIC), 2022. "
    "doi:10.1109/ISBIC56247.2022.9854534",
]
REFS_NANO = [
    "T. A. Sipkens, M. Frei, A. Baldelli, P. Kirchen, F. E. Kruis, S. N. Rogak. "
    "Characterizing soot in TEM images using a convolutional neural network. "
    "Powder Technology 387, 313-324, 2021.",
    "U. O. Köylü, G. M. Faeth, T. L. Farias, M. G. Carvalho. Fractal and "
    "projected structure properties of soot aggregates. Combustion and "
    "Flame 100, 621-633, 1995.",
]
REFS_NORMES = [
    "ISO 13322-1:2014. Particle size analysis, Image analysis methods, "
    "Part 1: Static image analysis methods.",
    "ISO 9276-1:1998. Representation of results of particle size analysis, "
    "Part 1: Graphical representation.",
    "ISO 9276-2:2014. Representation of results of particle size analysis, "
    "Part 2: Calculation of average particle sizes/diameters and moments "
    "from particle size distributions.",
]


import matplotlib.lines  # noqa: E402  (Line2D in the page header)
import matplotlib.transforms  # noqa: E402


def _logos_pil():
    from PIL import Image, ImageChops
    out = []
    for fn, key, _ in (("MANO_logo.png", "MANO", ""), ("UMONS_logo.png", "UMONS", ""),
                       ("CHIPS_logo.png", "CHIPS", "")):
        try:
            path = find_asset(fn)
            src = path if path else _io.BytesIO(base64.b64decode(EMBEDDED_LOGOS[key]))
            im = Image.open(src).convert("RGBA")
            rgb = im.convert("RGB")
            bb = ImageChops.difference(rgb, Image.new("RGB", rgb.size,
                                                      (255, 255, 255))).getbbox()
            im = im.crop(bb) if bb else im
            if im.height < 400:
                # Embedded logos are 64 px high (title bar). For print, they
                # are enlarged with a high-quality filter (no blocky pixels);
                # real sharpness needs the original files in Assets/.
                from PIL import ImageFilter
                f = 400.0 / im.height
                im = im.resize((int(im.width * f), 400), Image.LANCZOS)
                im = im.filter(ImageFilter.UnsharpMask(radius=2, percent=60,
                                                       threshold=2))
            out.append(im)
        except Exception as e:
            journal("warning", f"report: partner logo not loaded: {e}")
    return out


def _entete(fig, session, titre_page, n_page, n_pages,
            titre_doc="Particle size analysis report"):
    """Header (partner logos, report title, sample) and footer of a page."""
    x = 0.07
    for im in _logos_pil():
        w, h = im.size
        haut = 0.034
        larg = haut * (w / h) * A4[1] / A4[0]
        ax = fig.add_axes([x, 0.945, larg, haut])
        ax.imshow(np.asarray(im), interpolation="lanczos")
        ax.set_axis_off()
        x += larg + 0.018
    fig.text(0.93, 0.968, titre_doc, ha="right",
             va="center", fontsize=12.5, fontweight="bold", color=NAVY)
    fig.text(0.93, 0.950, f"Sample: {session.nom_echantillon}", ha="right",
             va="center", fontsize=9, color=ENCRE)
    fig.add_artist(matplotlib.lines.Line2D([0.07, 0.93], [0.934, 0.934],
                                           color=NAVY, lw=1.2))
    fig.text(0.07, 0.915, titre_page, ha="left", va="center", fontsize=11,
             fontweight="bold", color=NAVY)
    fig.add_artist(matplotlib.lines.Line2D([0.07, 0.93], [0.040, 0.040],
                                           color=GRIS, lw=0.5))
    fig.text(0.07, 0.027, "Materia Nova - CHIPS - Optical Particle Size "
                          f"Analyser - {datetime.datetime.now():%Y-%m-%d %H:%M}",
             ha="left", va="center", fontsize=7, color=GRIS)
    fig.text(0.93, 0.027, f"Page {n_page} / {n_pages}", ha="right",
             va="center", fontsize=7, color=GRIS)


def _tableau(ax, lignes, titre=None, col=("Quantity", "Value"), larg=(0.6, 0.4),
             fs=8):
    ax.set_axis_off()
    if titre:
        ax.set_title(titre, loc="left", fontsize=9.5, fontweight="bold",
                     color=NAVY, pad=4)
    t = ax.table(cellText=lignes, colLabels=list(col), colWidths=list(larg),
                 loc="upper left", cellLoc="left", colLoc="left")
    t.auto_set_font_size(False)
    t.set_fontsize(fs)
    t.scale(1, 1.32)
    for (r, _c), ce in t.get_celld().items():
        ce.set_edgecolor("#b9c2ce")
        ce.set_linewidth(0.5)
        if r == 0:
            ce.set_facecolor(CLAIR)
            ce.get_text().set_fontweight("bold")
            ce.get_text().set_color(NAVY)
        else:
            ce.set_facecolor("white")
    return t


def _legende(fig, x, y, texte, largeur=0.86):
    import textwrap
    fig.text(x, y, "\n".join(textwrap.wrap(texte, int(118 * largeur))),
             ha="left", va="top", fontsize=7.8, color=ENCRE, style="italic")


def _fmt_v(v, un=""):
    if isinstance(v, str):
        return v
    if isinstance(v, (int, np.integer)):
        return f"{v} {un}".strip()
    spec = ".0f" if un == "/100" else (".1f" if un in ("%", "°") else ".4g")
    return f"{fmt(v, spec)} {un}".strip()


_trapeze = getattr(np, "trapezoid", None) or getattr(np, "trapz")


GRAPHE_DEFAUT = {
    "xscale": "log", "xmin": None, "xmax": None, "ymax": None, "bins": None,
    "barres": True, "courbe": True, "cumul": True, "reperes": True,
    "lognormale": False, "normale": False, "titre": "", "xlabel": "", "ylabel": "",
    "police": 9.0, "trait": 1.5, "rapport": True,
}


def reglages_graphe(session):
    """Graph settings of the session, completed with the defaults."""
    g = dict(GRAPHE_DEFAUT)
    g.update(getattr(session, "graphe", None) or {})
    return g


def tracer_psd(ax, d, unite, ponderation="nombre", xscale="log",
               nom_grandeur="Equivalent diameter", g=None):
    """Scientific particle-size distribution plot: density per size class
    (bars, light grey) with its smoothed curve (black), cumulative on the
    right axis (grey dashed), D10 / D50 / D90 as dotted lines, optional
    log-normal fit. Number or volume (d³) weighting, logarithmic or linear
    size axis. g: graph settings (GRAPHE_DEFAUT keys) from the dialog."""
    g = dict(GRAPHE_DEFAUT, **(g or {}))
    g["xscale"] = xscale or g["xscale"]
    fs, lw = float(g["police"]), float(g["trait"])
    d = np.asarray(d, float)
    d = d[np.isfinite(d) & (d > 0)]
    vol = ponderation == "volume"
    w = d ** 3 if vol else np.ones_like(d)
    lo = float(g["xmin"]) if g.get("xmin") else d.min() / 1.25
    hi = float(g["xmax"]) if g.get("xmax") else d.max() * 1.25
    if g["xscale"] == "log":
        lo = max(lo, 1e-9)
    if hi <= lo:
        hi = lo * 10
    nb = int(g["bins"]) if g.get("bins") else int(np.clip(2 * np.sqrt(d.size), 12, 40))
    if g["xscale"] == "log":
        bords = np.logspace(np.log10(lo), np.log10(hi), nb + 1)
    else:
        bords = np.linspace(max(0.0, lo), hi, nb + 1)
    h, _ = np.histogram(d, bords, weights=w)
    dens = 100.0 * h / max(w.sum(), 1e-30)
    if g["barres"]:
        ax.bar(bords[:-1], dens, width=np.diff(bords), align="edge",
               color="#d7dce3", edgecolor="white", linewidth=0.4, zorder=1)
    pas = np.mean(np.diff(np.log10(bords))) if g["xscale"] == "log" \
        else np.mean(np.diff(bords))
    xs = np.logspace(np.log10(lo), np.log10(hi), 400) if g["xscale"] == "log" \
        else np.linspace(max(0.0, lo), hi, 400)
    ymax_c = 0.0
    if g["courbe"] and d.size >= 5:
        try:
            from scipy.stats import gaussian_kde
            if g["xscale"] == "log":
                y = gaussian_kde(np.log10(d), weights=w)(np.log10(xs)) * pas * 100
            else:
                y = gaussian_kde(d, weights=w)(xs) * pas * 100
            ax.plot(xs, y, color=ENCRE, lw=lw, zorder=3,
                    label="smoothed density")
            ymax_c = float(y.max())
        except Exception as e:
            journal("warning", f"report: smoothed density not drawn: {e}")
    if (g["lognormale"] or g["normale"]) and d.size >= 5:
        st = stats_distribution(d)
        fits = []
        med, sg = st.get("lognorm", (0, 0))
        if g["lognormale"] and med and sg > 1.0:
            z = (np.log(xs) - np.log(med)) / np.log(sg)
            f_x = np.exp(-0.5 * z ** 2) / (np.log(sg) * np.sqrt(2 * np.pi) * xs)
            fits.append((f_x, f"log-normal fit (σg = {sg:.2f})", MN_LIGHT, "-."))
        mu_n, sd_n = st.get("norm", (0, 0))
        if g["normale"] and sd_n > 0:
            f_x = np.exp(-0.5 * ((xs - mu_n) / sd_n) ** 2) / (sd_n * np.sqrt(2 * np.pi))
            fits.append((f_x, f"normal fit ({mu_n:.3g} ± {sd_n:.2g})",
                         PDF_C["red"], (0, (5, 2))))
        for f_x, lab, coul, style in fits:
            if vol:                          # d³ weighting, renormalised
                f_x = f_x * xs ** 3
                aire = _trapeze(f_x, xs)
                f_x = f_x / aire if aire > 0 else f_x
            # density per unit size -> % per size class of the histogram
            y = f_x * (xs * np.log(10) if g["xscale"] == "log" else 1.0) * pas * 100
            ax.plot(xs, y, color=coul, lw=lw * 0.9, ls=style, zorder=4,
                    label=lab)
            ymax_c = max(ymax_c, float(np.nanmax(y)))
        if fits:
            ax.legend(loc="upper left", fontsize=fs * 0.85)
    ax.set_xscale(g["xscale"])
    ax.set_xlim(lo, hi)
    ymax = float(g["ymax"]) if g.get("ymax") else \
        max(dens.max() if g["barres"] else 0, ymax_c, 1e-6) * 1.18
    ax.set_ylim(0, ymax)
    ax.set_xlabel(g["xlabel"] or f"{nom_grandeur} ({unite})", fontsize=fs)
    ax.set_ylabel(g["ylabel"] or ("Volume density (%)" if vol else
                                  "Number density (%)"), fontsize=fs)
    ax.tick_params(labelsize=fs * 0.9)
    if g["titre"]:
        ax.set_title(g["titre"], fontsize=fs * 1.1, pad=18)
    o = np.argsort(d)
    cum = 100 * np.cumsum(w[o]) / w.sum()
    ax2 = None
    if g["cumul"]:
        ax2 = ax.twinx()
        ax2.plot(d[o], cum, color=GRIS, lw=lw * 0.85, ls="--", zorder=2)
        ax2.set_ylim(0, 105)
        ax2.set_ylabel("Cumulative volume (%)" if vol else
                       "Cumulative number (%)", color=GRIS, fontsize=fs)
        ax2.tick_params(axis="y", colors=GRIS, direction="in",
                        labelsize=fs * 0.9)
        ax2.spines["right"].set_color(ENCRE)
    axe_log_lisible(ax)
    if g["reperes"]:
        trans = matplotlib.transforms.blended_transform_factory(ax.transData,
                                                                ax.transAxes)
        dernier = -1.0
        for q in (10, 50, 90):
            dq = float(np.interp(q, cum, d[o]))
            if not lo <= dq <= hi:
                continue
            ax.axvline(dq, color=GRIS, lw=0.7, ls=":", zorder=2)
            xf = (np.log10(dq / lo) / np.log10(hi / lo)) if g["xscale"] == "log" \
                else (dq - lo) / (hi - lo)
            haut = 1.075 if xf - dernier < 0.12 else 1.012
            dernier = xf if haut == 1.012 else dernier
            ax.text(dq, haut, f"D{'v' if vol else 'n'}{q} = {fmt(dq, '.3g')}",
                    transform=trans, fontsize=fs * 0.78, color=ENCRE,
                    va="bottom", ha="center")
    return ax2


def axe_log_lisible(ax, axe="x"):
    """Plain numbers on a logarithmic axis (30, 40, 60, 100 rather than
    3×10¹); minor labels only when the axis spans less than ~1.5 decades."""
    from matplotlib import ticker
    a = ax.xaxis if axe == "x" else ax.yaxis
    if (ax.get_xscale() if axe == "x" else ax.get_yscale()) != "log":
        return
    lo, hi = ax.get_xlim() if axe == "x" else ax.get_ylim()
    simple = ticker.FuncFormatter(lambda v, _p: f"{v:g}")
    a.set_major_formatter(simple)
    if np.log10(hi / lo) < 1.6:
        a.set_minor_locator(ticker.LogLocator(subs=(2, 3, 5)))
        a.set_minor_formatter(simple)
    else:
        a.set_minor_formatter(ticker.NullFormatter())


def _hist_sci(ax, v, xlab, bins=25, rng=None):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size:
        _, b, _ = ax.hist(v, bins=bins, range=rng, color="#d7dce3",
                          edgecolor=ENCRE, linewidth=0.5)
        med = float(np.median(v))
        ax.axvline(med, color=ENCRE, lw=1.0, ls="--")
        droite = (med - b[0]) / max(b[-1] - b[0], 1e-12) < 0.5
        ax.text(0.97 if droite else 0.03, 0.95, f"median {fmt(med, '.3g')}\n"
                f"n = {v.size}", transform=ax.transAxes,
                ha="right" if droite else "left", va="top", fontsize=7.5)
    else:
        ax.text(0.5, 0.5, "none detected", transform=ax.transAxes,
                ha="center", va="center", color=GRIS)
    ax.set_xlabel(xlab)
    ax.set_ylabel("Count")


def lignes_methode(session, res, d, u, fib, sd):
    """Rows (item, setting) of the Method table, shared by both reports."""
    p = session.params
    meth = []
    if res["mode"] in ("meb", "optique"):
        if sd:
            info_sd = res.get("stardist", {}) or {}
            meth.append(["Segmentation", f"StarDist star-convex polygons [1-3], "
                         f"model {info_sd.get('modele', STARDIST_MODELE)}"])
            meth.append(["Inference", f"{info_sd.get('backend', '–')}, test-time "
                         f"augmentation {'on' if info_sd.get('tta') else 'off'}"])
            meth.append(["Thresholds", f"probability {fmt(info_sd.get('prob_thresh'), '.3g')}, "
                         f"overlap {fmt(info_sd.get('nms_thresh'), '.3g')}"])
            meth.append(["Working scale", f"× {fmt(info_sd.get('echelle'), '.3g')} "
                         f"(typical size {fmt(info_sd.get('diametre_typique_px'), '.3g')} px)"])
        else:
            meth.append(["Segmentation", "local threshold + h-maxima watershed"])
    nr = res.get("nano") or {}
    nano2 = res["mode"] == "nano" and nr.get("methode") == "stardist_deux_etapes"
    if nano2:
        sdi = nr.get("stardist", {})
        meth.append(["Method", "two steps: aggregate segmentation, then "
                     "primary particles [Sipkens 2021]"])
        meth.append(["Primaries", f"StarDist [1-3], model {sdi.get('modele')} "
                     f"(trained on synthetic TE aggregates), ×{fmt(sdi.get('echelle'), '.3g')}"])
        meth.append(["Primary diameter", "2 × 90th percentile of the polygon "
                     "rays (free part of the contour)"])
        meth.append(["Aggregates", "Np = 1.15 (Aa/Ap)^1.09 [Köylü 1995]; "
                     "ensemble fit Np = kf (2Rg/dp)^Df"])
    if res["mode"] == "gouttes":
        sg = res.get("gouttes_stats") or {}
        h = res.get("hauteur_canal_um")
        meth.append(["Detection", "dark rings, enclosed round regions"
                     + (", Hough recovery" if p.get("hough", True) else "")
                     + f" (k = {fmt(p.get('k_anneau', 0.5), '.2g')})"])
        autre = "middle" if res.get("convention") == "ext" else "outer edge"
        meth.append(["Diameter", "circle fit on 72 radial profiles, sub-pixel"])
        meth.append(["Convention", ("OUTER EDGE" if res.get("convention") == "ext"
                                    else "RING MIDDLE") + " of the dark ring"
                     + (f" ({autre}: {fmt(abs(sg.get('ecart_convention', 0)), '.3g')} "
                        f"{u} away)" if sg.get("n", 0) > 1 else "")])
        meth.append(["Volume", f"disc h = {h:g} µm, rounded rim (D > h); sphere "
                     "otherwise" if h else "spheres (π D³ / 6)"])
    meth.append(["Size measure", "fibre diameter edge to edge" if fib else
                 ("primary particle diameter (nm)" if nano2 else
                  "fitted ring diameter" if res["mode"] == "gouttes" else
                  "equivalent circle diameter (ECD)")])
    if not fib:
        meth.append(["Border objects", f"excluded ({res.get('n_bord', 0)})"])
        dmax = float(res.get("taille_max", 0) or 0)
        dmin = p.get('diametre_min_um', p.get('d_min_um', p.get('d_min_nm', 0)))
        txt_min = ("no minimum" if (p.get("taille_min_off") or
                                    res["mode"] == "gouttes")
                   else f"minimum {fmt(dmin, '.3g')} {u}")
        meth.append(["Size filter", f"{txt_min}, " +
                     (f"maximum {fmt(dmax, '.4g')} {u} "
                      f"({res.get('n_trop_grands', 0)} excluded)"
                      if dmax > 0 else "no maximum")])
    uc = incertitude_calibration(session)
    src_txt = {"manual": "manual line on the scale bar",
               "auto": "automatic scale-bar detection",
               "metadata": "pixel size from the image metadata",
               "recipe": "recipe value"}.get(uc["source"], uc["source"])
    bud = f"{fmt(session.ppm, '.5g')} px/µm, {src_txt}"
    if uc["L_px"]:
        bud += (f" ({uc['L_px']:.0f} px, ±{U_EXTREMITE_PX.get(uc['source'], 0)} px "
                f"per end): reading {100 * uc['u_lecture']:.2f} %")
    bud += (f"; magnification {100 * uc['u_instr']:.2f} %" if uc["instr_inclus"]
            else "; magnification accuracy not included")
    bud += (f"; U = {100 * uc['U_rel']:.2f} % (k = 2) on every size"
            if uc["U_rel"] > 0 else "")
    meth.append(["Calibration", bud])
    ed = res.get("editions", {})
    if any(ed.values()):
        meth.append(["Manual edits", f"{ed['suppr']} deleted, {ed['ajout']} added, "
                     f"{ed['scind']} split, {ed['fusion']} merged"])
    return meth


def _panneaux_forme(ax, session, res, d, u, fib, xsc, sd):
    """Shape / aggregate / fibre-network panels (a)-(d) on 4 axes; returns
    the caption. Shared by the built-in and the LaTeX reports."""
    nr = res.get("nano") or {}
    if res["mode"] == "gouttes":
        g = res["gouttes"]
        ok = ~g["bord"]
        ump = res.get("um_par_px", 1.0)
        _hist_sci(ax[0], res.get("aspects", []), "Aspect ratio (ellipse fit)")
        xs_ = np.asarray(g["cx"])[ok] * ump
        if xs_.size and d.size == xs_.size:
            ax[1].scatter(xs_, d, s=10, color=ENCRE, alpha=0.7, lw=0)
            if xs_.size > 3:
                a_, b_ = np.polyfit(xs_, d, 1)
                xx_ = np.array([xs_.min(), xs_.max()])
                ax[1].plot(xx_, a_ * xx_ + b_, color=MN_LIGHT, lw=1.2,
                           label=f"slope {a_ * 1000:+.3g} {u}/mm")
                ax[1].legend(loc="best", fontsize=7.5)
        ax[1].set_xlabel("Position x in the image (µm)")
        ax[1].set_ylabel(f"Diameter ({u})")
        dm = np.asarray(g["d_milieu"])[ok] * ump
        de = np.asarray(g["d_ext"])[ok] * ump
        f_ = np.isfinite(de)
        if f_.any():
            ax[2].scatter(dm[f_], de[f_], s=10, color=ENCRE, alpha=0.7, lw=0)
            lo_, hi_ = float(min(dm[f_].min(), de[f_].min())), \
                float(max(dm[f_].max(), de[f_].max()))
            ax[2].plot([lo_, hi_], [lo_, hi_], ls="--", lw=0.8, color=PDF_C["red"])
        ax[2].set_xlabel(f"Diameter, ring middle ({u})")
        ax[2].set_ylabel(f"Diameter, outer edge ({u})")
        rel = 100 * np.asarray(g["rms"])[ok] / np.maximum(
            np.asarray(g["d_milieu"])[ok] / 2, 1e-6)
        _hist_sci(ax[3], rel, "Ring-fit residual (% of the radius)")
        return ("Figure 5. Droplets measured (not cut by the edge): (a) "
                "deformation (aspect ratio of the ellipse fitted on the ring; 1 = "
                "round), (b) diameter across the field (a slope reveals a drift "
                "of the production or of the focus), (c) diameter at the ring "
                "middle versus outer edge (dashed: equality; the gap is the "
                "convention uncertainty), (d) quality of the circle fit.")
    if res["mode"] == "nano" and nr.get("methode") == "stardist_deux_etapes":
        agg = [a for a in nr.get("agregats", []) if not a["bord"]]
        _hist_sci(ax[0], [a["ecd_nm"] for a in agg],
                  "Aggregate equivalent diameter (nm)")
        _hist_sci(ax[1], [a["np_aire"] for a in agg],
                  "Primaries per aggregate (from area)")
        dpm = float(np.mean(d)) if d.size else 1.0
        xs_ = np.array([2 * a["rg_nm"] / dpm for a in agg])
        ys_ = np.array([a["np_aire"] for a in agg])
        if xs_.size:
            ax[2].scatter(xs_, ys_, s=12, color=ENCRE, alpha=0.7, lw=0)
            ax[2].set_xscale("log")
            ax[2].set_yscale("log")
            if nr.get("df"):
                xx_ = np.geomspace(max(xs_.min(), 1e-3), xs_.max(), 50)
                ax[2].plot(xx_, nr["kf"] * xx_ ** nr["df"], color=MN_LIGHT,
                           lw=1.2, label=f"Df = {nr['df']:.2f}, "
                           f"kf = {nr['kf']:.2f} (R² = {nr['df_r2']:.2f})")
                ax[2].legend(loc="upper left", fontsize=7.5)
            axe_log_lisible(ax[2])
            axe_log_lisible(ax[2], "y")
        ax[2].set_xlabel("2 Rg / dp")
        ax[2].set_ylabel("Np")
        _hist_sci(ax[3], nr.get("primaires", {}).get("prob", []),
                  "Primary detection confidence", rng=(0, 1))
        cap = ("Figure 5. Aggregates (complete, not touching the border): "
               "(a) equivalent diameter, (b) number of primaries from the "
               "projected area [Köylü 1995], (c) fractal law Np = kf "
               "(2 Rg / dp)^Df, ensemble fit, (d) network confidence of "
               "the detected primaries.")
    elif fib:
        fr = res["fibres"]
        _hist_sci(ax[0], fr.get("angles", []), "Fibre orientation (°)",
                  bins=36, rng=(0, 180))
        ray = fr.get("rayons")
        Lr = (np.asarray(ray["larg"], float) / session.ppm *
              (1000 if u == "nm" else 1)) if ray is not None and \
            len(ray["larg"]) else []
        _hist_sci(ax[1], Lr, f"Edge-to-edge segment width ({u})")
        _hist_sci(ax[2], np.asarray(fr.get("pore_d_um", []), float) * 1000,
                  "Pore diameter (nm)")
        _hist_sci(ax[3], fr.get("billes_d_um", []), "Bead diameter (µm)")
        cap = ("Figure 4. Fibre network: (a) orientation, (b) all edge-to-"
               "edge segments, (c) pores (porous fibres), (d) beads.")
    else:
        c = np.asarray(res.get("circularites", []), float)
        _hist_sci(ax[0], c, "Circularity 4πA/P²",
                  rng=(max(0.0, float(np.nanmin(c)) - 0.02) if c.size else 0, 1))
        _hist_sci(ax[1], res.get("aspects", []), "Aspect ratio (major/minor)")
        sol = np.asarray(res.get("solidites", []), float)
        _hist_sci(ax[2], sol, "Solidity (area/convex area)",
                  rng=(max(0.0, float(np.nanmin(sol)) - 0.02) if sol.size else 0, 1))
        if d.size and c.size == d.size:
            ax[3].scatter(d, c, s=7, color=ENCRE, alpha=0.6, lw=0)
            ax[3].set_xscale(xsc)
            ax[3].set_ylim(max(0.0, float(c.min()) - 0.05), 1.0)
            axe_log_lisible(ax[3])
        ax[3].set_xlabel(f"Equivalent diameter ({u})")
        ax[3].set_ylabel("Circularity")
        cap = ("Figure 5. Shape descriptors: (a) circularity, (b) aspect ratio, "
               "(c) solidity (low values: satellites, aggregates), (d) "
               "circularity versus size.")
    return cap


def _image_classes(session, res, sph):
    """RGB image with the particles coloured by class (green spherical,
    orange irregular, red satellite)."""
    img = session.image_bgr
    g_ = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vis = cv2.cvtColor(g_, cv2.COLOR_GRAY2RGB)
    ep = max(2, int(round(max(vis.shape[:2]) / 600)))
    coul = {"spherique": (46, 165, 90), "irreguliere": (230, 140, 30),
            "satellite": (210, 50, 50)}
    for c, k in zip(contours_du_resultat(res), sph["classes"]):
        cv2.drawContours(vis, [c], -1, coul[k], ep, cv2.LINE_AA)
    return vis


def _barres_sphero(ab, sph):
    """Class fractions by number and by mass."""
    cats = ["Spherical", "Irregular", "Satellites"]
    vn = [sph["taux_nombre"], sph["irreg_nombre"], sph["satellites_nombre"]]
    vm = [sph["taux_masse"], sph["irreg_masse"], sph["satellites_masse"]]
    xb = np.arange(3)
    ab.bar(xb - 0.19, vn, 0.36, color="#d7dce3", edgecolor=ENCRE, lw=0.6,
           label="by number")
    ab.bar(xb + 0.19, vm, 0.36, color="#7d8794", edgecolor=ENCRE, lw=0.6,
           label="by mass (d³)")
    for x_, v1, v2 in zip(xb, vn, vm):
        ab.text(x_ - 0.19, v1 + 1.5, f"{v1:.0f}", ha="center", fontsize=7.5)
        ab.text(x_ + 0.19, v2 + 1.5, f"{v2:.0f}", ha="center", fontsize=7.5)
    ab.set_xticks(xb)
    ab.set_xticklabels(cats)
    ab.set_ylabel("Fraction (%)")
    ab.set_ylim(0, 110)
    ab.legend(loc="upper right")


def _page_resume(c, n_avant):
    """Page 1: tables, images, main distribution."""
    (session, res, d, u, fib, gt, md, sd, xsc, g_rap, lignes, sm, nr,
     nano2) = (c.session, c.res, c.d, c.u, c.fib, c.gt, c.md, c.sd, c.xsc,
               c.g_rap, c.lignes, c.sm, c.nr, c.nano2)
    f = Figure(figsize=A4)
    info = [["Sample", session.nom_echantillon],
            ["Image file", session.nom_image],
            ["Operator", session.operateur or "–"],
            ["Date of analysis", f"{datetime.datetime.now():%Y-%m-%d %H:%M}"],
            ["Recipe", session.recette or "–"],
            ["Analysis mode", MODE_NOMS.get(res["mode"], res["mode"])]]
    info += md.lignes_info(res)
    info.append(["Calibration", f"{fmt(session.ppm, '.5g')} px/µm"])
    cles = md.cles_resultats(res)
    lab = {k: lb for k, lb, _, _ in lignes}
    U_cal = incertitude_calibration(session)["U_rel"]
    TAILLES = {"mean", "std", "dn10", "dn50", "dn90", "dv10", "dv50", "dv90",
               "d32", "d43", "dp_gm", "min", "max"}
    cles = cles + ["u_cal"]

    def _cellule(k):
        v, un = sm[k]
        if k in TAILLES and U_cal > 0 and not isinstance(v, str):
            return f"{fmt(v, '.4g')} ± {fmt(v * U_cal, '.2g')} {un}"
        return _fmt_v(v, un).replace(" /100", "/100")
    resu = [[lab[k], _cellule(k)] for k in cles if k in sm]
    haut = 0.885
    h_i = LIGNE_TABLE * (len(info) + 1) + 0.026
    h_r = LIGNE_TABLE * (len(resu) + 1) + 0.026
    _tableau(f.add_axes([0.07, haut - h_i, 0.40, h_i]), info,
             "Sample and analysis", ("Item", "Value"), (0.42, 0.58))
    _tableau(f.add_axes([0.53, haut - h_r, 0.40, h_r]), resu, "Key results")
    bas = haut - max(h_i, h_r) - 0.03
    img = session.image_bgr
    hi_, wi_ = img.shape[:2]
    larg = 0.415
    hf = larg * (A4[0] / A4[1]) * (hi_ / wi_)
    if hf > bas - 0.10:                      # tall image: keep it on the page
        hf = bas - 0.10
        larg = hf / ((A4[0] / A4[1]) * (hi_ / wi_))
    y0 = bas - hf
    a1 = f.add_axes([0.07, y0, larg, hf])
    a1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img,
              cmap="gray", aspect="auto")
    a2 = f.add_axes([0.93 - larg, y0, larg, hf])
    a2.imshow(image_annotee_rapport(session), aspect="auto")
    for a, t in ((a1, "(a)"), (a2, "(b)")):
        a.set_xticks([])
        a.set_yticks([])
        a.text(0.015, 0.975, t, transform=a.transAxes, ha="left", va="top",
               fontsize=9, fontweight="bold", color="white",
               bbox=dict(facecolor="black", alpha=0.55, lw=0, pad=1.5))
    _legende(f, 0.07, y0 - 0.010, "Figure 1. (a) Original image. (b) "
             + ("Edge-to-edge measurement segments (colour = diameter)."
                if fib else "Droplets measured (green) and cut by the "
                "image edge (red, excluded from the statistics)." if gt
                else "Detected particles; objects touching the "
                "image border are excluded from the statistics."))
    h_dist = min(0.27, y0 - 0.07 - 0.045 - 0.085)   # caption above footer
    if d.size and h_dist > 0.12:
        pond0 = md.ponderation_figure(res)
        a = f.add_axes([0.11, y0 - 0.07 - h_dist, 0.74, h_dist])
        tracer_psd(a, d, u, pond0, xsc, md.nom_diametre_de(res), g=g_rap)
        _legende(f, 0.07, y0 - 0.07 - h_dist - 0.045,
                 "Figure 2. " + ("Length-weighted fibre diameter distribution."
                                 if fib else "Number-weighted droplet diameter "
                                 "distribution (details on page 2)." if gt
                                 else "Volume-weighted size distribution "
                                 "(summary; details on page 2)." if
                                 pond0 == "volume" else "Number-weighted "
                                 "size distribution (details on page 2)."))
    return f, "1. Summary"


def _page_distribution(c, n_avant):
    """Page 2: number and volume distributions."""
    (session, res, d, u, fib, gt, md, sd, xsc, g_rap, lignes, sm, nr,
     nano2) = (c.session, c.res, c.d, c.u, c.fib, c.gt, c.md, c.sd, c.xsc,
               c.g_rap, c.lignes, c.sm, c.nr, c.nano2)
    f = Figure(figsize=A4)
    nom_g = md.nom_diametre_de(res)
    if d.size:
        a = f.add_axes([0.11, 0.56, 0.74, 0.30])
        tracer_psd(a, d, u, "nombre", xsc, nom_g, g=g_rap)
        _legende(f, 0.07, 0.515, f"Figure 3. Number-weighted {nom_g.lower()} "
                 f"distribution (n = {d.size}). Bars: density per class; "
                 "black curve: smoothed density; dashed: cumulative; "
                 "dotted lines: D10, D50, D90." + (" Length-weighted along "
                                                  "the fibres." if fib else ""))
        if gt:
            from psa.droplets import volume_goutte
            vols = volume_goutte(d, res.get("hauteur_canal_um")) * 1e-3
            a = f.add_axes([0.11, 0.15, 0.74, 0.30])
            _hist_sci(a, vols, "Droplet volume (pL)")
            h_ = res.get("hauteur_canal_um")
            _legende(f, 0.07, 0.105, "Figure 4. Droplet volume distribution ("
                     + (f"squeezed discs, channel height {h_:g} µm" if h_
                        else "spheres") + "). The volume follows D³: a "
                     "size CV of x % gives a volume CV of about 3x %.")
        elif not fib:
            a = f.add_axes([0.11, 0.15, 0.74, 0.30])
            tracer_psd(a, d, u, "volume", xsc, nom_g, g=g_rap)
            _legende(f, 0.07, 0.105, "Figure 4. Volume-weighted distribution "
                     "(d³ weighting, spheres assumed, ISO 9276-2). Comparable "
                     "to laser diffraction results.")
    return f, "2. Size distribution"


def _page_forme(c, n_avant):
    """Page 3: shape panels of the mode."""
    (session, res, d, u, fib, gt, md, sd, xsc, g_rap, lignes, sm, nr,
     nano2) = (c.session, c.res, c.d, c.u, c.fib, c.gt, c.md, c.sd, c.xsc,
               c.g_rap, c.lignes, c.sm, c.nr, c.nano2)
    f = Figure(figsize=A4)
    pos = [[0.10, 0.56, 0.36, 0.28], [0.57, 0.56, 0.36, 0.28],
           [0.10, 0.17, 0.36, 0.28], [0.57, 0.17, 0.36, 0.28]]
    ax = [f.add_axes(p) for p in pos]
    cap = _panneaux_forme(ax, session, res, d, u, fib, xsc, sd)
    for a, t in zip(ax, "abcd"):
        a.text(-0.16, 1.04, f"({t})", transform=a.transAxes, fontsize=9,
               fontweight="bold")
    _legende(f, 0.07, 0.115, cap)
    titre = ("3. Aggregate morphology" if nano2 else
             "3. Droplet shape and measurement" if gt else
             "3. Fibre network" if fib else "3. Particle shape")
    return f, titre


def _page_sphero(c, n_avant):
    """Spheroidization page (powder modes), or None."""
    (session, res, d, u, fib, gt, md, sd, xsc, g_rap, lignes, sm, nr,
     nano2) = (c.session, c.res, c.d, c.u, c.fib, c.gt, c.md, c.sd, c.xsc,
               c.g_rap, c.lignes, c.sm, c.nr, c.nano2)
    sph = sphericite(res, getattr(session, "sphero", None))
    if sph:
        f = Figure(figsize=A4)
        vis = _image_classes(session, res, sph)
        hi_, wi_ = vis.shape[:2]
        larg = 0.86
        hf = min(larg * (A4[0] / A4[1]) * (hi_ / wi_), 0.40)
        larg = hf / ((A4[0] / A4[1]) * (hi_ / wi_))
        a = f.add_axes([0.07 + (0.86 - larg) / 2, 0.885 - hf, larg, hf])
        a.imshow(vis, aspect="auto")
        a.set_xticks([])
        a.set_yticks([])
        _legende(f, 0.07, 0.885 - hf - 0.010,
                 "Figure 6. Particle classes: green spherical, orange "
                 "irregular, red satellite (small particle attached to a "
                 "larger one).")
        yb = 0.885 - hf - 0.075
        ab = f.add_axes([0.10, yb - 0.25, 0.40, 0.25])
        _barres_sphero(ab, sph)
        cr = sph["criteres"]
        tab = [["Spheroidization rate, number", f"{sph['taux_nombre_hors_sat']:.1f} %"],
               ["Spheroidization rate, mass", f"{sph['taux_masse_hors_sat']:.1f} %"],
               ["Satellites (number)", f"{sph['satellites_nombre']:.1f} %"],
               ["Particles bearing satellites", f"{sph['porteuses_pct']:.1f} %"],
               ["Spherical: roundness 4A/(πF²max)", f"≥ {cr['rondeur_min']:.2f}"],
               ["Spherical: aspect ratio", f"≤ {cr['aspect_max']:.2f}"],
               ["Satellite: size ratio", f"≤ {cr['satellite_ratio']:.2f}"],
               ["Particles classified", f"{sph['n']}"]]
        _tableau(f.add_axes([0.56, yb - 0.25, 0.37, 0.25]), tab,
                 None, ("Quantity", "Value"), (0.66, 0.34), fs=7.5)
        _legende(f, 0.07, yb - 0.27,
                 "Figure 7. Class fractions by number and by mass. The "
                 "spheroidization rate excludes the satellites (main "
                 "particles only). Validation on synthetic powders of known "
                 "composition: within ±5 points by number and ±6 by mass "
                 "with the deep learning engine"
                 + ("." if sd else "; the classic engine merges "
                    "satellites with their carrier and is NOT reliable "
                    "for this rate."))
        return f, f"{n_avant + 1}. Spheroidization"


def _page_methode(c, n_avant):
    """Last page: method, quality, references."""
    (session, res, d, u, fib, gt, md, sd, xsc, g_rap, lignes, sm, nr,
     nano2) = (c.session, c.res, c.d, c.u, c.fib, c.gt, c.md, c.sd, c.xsc,
               c.g_rap, c.lignes, c.sm, c.nr, c.nano2)
    f = Figure(figsize=A4)
    meth = lignes_methode(session, res, d, u, fib, sd)
    h_m = LIGNE_TABLE * (len(meth) + 1) + 0.026
    _tableau(f.add_axes([0.07, 0.885 - h_m, 0.86, h_m]), meth, "Method",
             ("Item", "Setting"), (0.25, 0.75))
    avert = avertissements_qualite(res)
    y = 0.885 - h_m - 0.035
    f.text(0.07, y, "Quality assessment", fontsize=9.5, fontweight="bold",
           color=NAVY)
    q = [f"Segmentation quality score: {fmt(res.get('score_ws', 0), '.0f')}/100."]
    if "representativeness" in sm:
        q.append(f"Representativeness: {sm['representativeness'][0]} "
                 f"(n = {res['n_particules']}, CV = {fmt(sm.get('cv', (0,))[0], '.1f')} %).")
    q += [f"Warning: {a}" for a in avert]
    import textwrap
    for t in q:
        for ln in textwrap.wrap(t, 110):
            y -= 0.018
            f.text(0.08, y, ln, fontsize=8, color=PDF_C["red"]
                   if t.startswith("Warning") else ENCRE)
    y -= 0.04
    f.text(0.07, y, "References", fontsize=9.5, fontweight="bold", color=NAVY)
    refs = (REFS_STARDIST if sd else []) + (REFS_NANO if nano2 else []) + \
        REFS_NORMES
    for i, r in enumerate(refs, 1):
        lignes_r = textwrap.wrap(r, 104)
        y -= 0.020
        f.text(0.075, y, f"[{i}]", fontsize=7.8, color=ENCRE)
        for j, ln in enumerate(lignes_r):
            if j:
                y -= 0.0155
            f.text(0.105, y, ln, fontsize=7.8, color=ENCRE)
    return f, f"{n_avant + 1}. Method, quality and references"


def _contexte_rapport(session):
    """Everything the report pages share, computed once."""
    from types import SimpleNamespace
    res = session.resultats
    gr = reglages_graphe(session)
    lignes = lignes_resume(session)
    nr = res.get("nano") or {}
    return SimpleNamespace(
        session=session, res=res, d=np.asarray(res["diametres"], float),
        u=res["unite"], fib=res["mode"] == "fibres",
        gt=res["mode"] == "gouttes", md=mode_de(res),
        sd=res.get("moteur") == "stardist", xsc=gr["xscale"],
        # même réglage de graphe qu'à l'écran si demandé (titres exclus :
        # le rapport utilise des légendes numérotées)
        g_rap={k: v for k, v in gr.items() if k not in ("titre", "ymax")}
        if gr.get("rapport") else {"xscale": gr["xscale"]},
        lignes=lignes, sm={k: (v, un) for k, _, v, un in lignes}, nr=nr,
        nano2=res["mode"] == "nano" and nr.get("methode") == "stardist_deux_etapes")


PAGES_RAPPORT = (_page_resume, _page_distribution, _page_forme, _page_sphero,
                 _page_methode)


def pages_rapport(session):
    """Builds the report pages; returns a list of matplotlib Figures (one
    function per page, v1.8.0)."""
    import matplotlib.pyplot  # noqa: F401  (mathtext / rc context)
    c = _contexte_rapport(session)
    pages = []
    with matplotlib.rc_context(STYLE_SCI):
        for construire in PAGES_RAPPORT:
            page = construire(c, len(pages))
            if page is not None:
                pages.append(page)
        n = len(pages)
        figs = []
        for i, (fg, titre) in enumerate(pages, 1):
            _entete(fg, session, titre, i, n)
            figs.append(fg)
    return figs

def ecrire_rapport(session, dossier, stem, formats=("pdf", "png"), dpi=200):
    """Writes <sample>_report.pdf (all pages) and <sample>_report_p<i>.png."""
    from matplotlib.backends.backend_pdf import PdfPages
    figs = pages_rapport(session)
    sorties = []
    if "pdf" in formats:
        f = os.path.join(dossier, f"{stem}_report.pdf")
        with PdfPages(f) as pdf:
            for fg in figs:
                pdf.savefig(fg)
            m = pdf.infodict()
            m["Title"] = f"Particle size analysis report, {session.nom_echantillon}"
            m["Author"] = session.operateur or APP_VENDOR
            m["Creator"] = f"{APP_NAME} {APP_VERSION}"
        sorties.append(f)
    if "png" in formats:
        for i, fg in enumerate(figs, 1):
            f = os.path.join(dossier, f"{stem}_report_p{i}.png")
            fg.savefig(f, dpi=dpi)
            sorties.append(f)
    return sorties


__all__ = [_n for _n in dir() if not _n.startswith("__")]
