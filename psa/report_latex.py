# -*- coding: utf-8 -*-
"""PSA, report written in LaTeX with the official Materia Nova template
(Template_Materia_Nova: article A4 12 pt, 2.5 cm margins, Materia Nova logo
in the header, CHIPS department, MN colours, title block with the logos,
1.5 line spacing, contact block).

The figures are produced by the same functions as the built-in report
(vector PDF plots), the tables by the same summary rows. The document is
compiled with pdflatex when a TeX distribution is installed (MiKTeX, TeX
Live); the sources are always exported as an Overleaf-ready zip.
"""
from psa.report import *  # noqa: F401,F403

import shutil
import subprocess
import tempfile
import zipfile

_TEX = [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
        ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
        ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]
_UNI = {"µ": r"\textmu{}", "±": r"$\pm$", "σ": r"$\sigma$", "≥": r"$\geq$",
        "≤": r"$\leq$", "²": r"\textsuperscript{2}", "³": r"\textsuperscript{3}",
        "Ø": r"\O{}", "×": r"$\times$", "π": r"$\pi$", "–": "--", "—": "---",
        "°": r"\textdegree{}", "·": r"\textperiodcentered{}", "⚠": "",
        "≈": r"$\approx$", "−": r"$-$", "→": r"$\rightarrow$", "α": r"$\alpha$",
        "γ": r"$\gamma$", "λ": r"$\lambda$", "ρ": r"$\rho$", "Δ": r"$\Delta$",
        "’": "'", "‘": "`", "“": "``", "”": "''", "…": r"\ldots{}",
        "₂": r"\textsubscript{2}", "Å": r"\AA{}", "é": r"\'e", "è": r"\`e",
        "à": r"\`a", "ö": r'\"o', "ü": r'\"u'}


def tex(x):
    """Plain text -> LaTeX (special characters and symbols)."""
    s = str(x).replace("\\", "\x00")
    for a, b in _TEX[1:]:
        s = s.replace(a, b)
    s = s.replace("\x00", r"\textbackslash{}")
    return "".join(_UNI.get(c, c) for c in s)


def trouver_pdflatex():
    """pdflatex of MiKTeX or TeX Live (PATH first, then usual folders)."""
    exe = shutil.which("pdflatex")
    if exe:
        return exe
    cands = []
    for base in (os.environ.get("ProgramFiles", r"C:\Program Files"),
                 os.environ.get("LOCALAPPDATA", ""),
                 os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs")):
        if base:
            cands.append(os.path.join(base, "MiKTeX", "miktex", "bin", "x64",
                                      "pdflatex.exe"))
    for racine in (r"C:\texlive", "/usr/local/texlive"):
        if os.path.isdir(racine):
            for an in sorted(os.listdir(racine), reverse=True):
                cands += [os.path.join(racine, an, "bin", d, "pdflatex.exe")
                          for d in ("windows", "win32", "win64")]
                cands.append(os.path.join(racine, an, "bin", "x86_64-linux",
                                          "pdflatex"))
    return next((c for c in cands if os.path.exists(c)), None)


def _logos_template(dossier):
    """Template logos (high resolution) copied next to the .tex. Returns the
    file names available."""
    noms = {}
    for fn, cible in (("MANO_logo.png", "Logo_MateriaNova.png"),
                      ("CHIPS_logo.png", "logo_CHIPS.png"),
                      ("UMONS_logo.png", "logo_UMONS.png")):
        src = find_asset(fn)
        if src:
            from PIL import Image
            try:
                h = Image.open(src).size[1]
            except Exception:
                h = 0
            if h >= 140:                      # only sharp enough logos
                shutil.copy(src, os.path.join(dossier, cible))
                noms[cible] = True
    return noms


def _fig(dossier, nom, figsize, dessin):
    with matplotlib.rc_context(STYLE_SCI):
        f = Figure(figsize=figsize)
        dessin(f)
        try:
            f.set_layout_engine("constrained")
        except Exception:
            pass
        f.savefig(os.path.join(dossier, "figures", nom))
    return f"figures/{nom}"


def _tableau_tex(lignes, entetes=("Quantity", "Value"), larg=(0.62, 0.38)):
    """booktabs table, single spacing, small font."""
    ent = " & ".join(rf"\textcolor{{MNbluedark}}{{\textbf{{{tex(e)}}}}}" for e in entetes)
    corps = "\n".join(" & ".join(tex(c) for c in ln) + r" \\" for ln in lignes)
    cols = "".join(rf">{{\raggedright\arraybackslash}}p{{{w:.3f}\linewidth}}"
                   for w in larg)
    return (r"{\singlespacing\footnotesize\renewcommand{\arraystretch}{1.12}"
            rf"\begin{{tabular}}{{{cols}}}" "\n\\toprule\n" + ent + r" \\" +
            "\n\\midrule\n" + corps + "\n\\bottomrule\n\\end{tabular}}")


def _tableau_double(lignes, entetes=("Quantity", "Value")):
    """Two (quantity, value) pairs per row: half the height for long lists."""
    m = (len(lignes) + 1) // 2
    g, d = lignes[:m], lignes[m:] + [["", ""]] * (m - len(lignes[m:]))
    return _tableau_tex([a + b for a, b in zip(g, d)], entetes * 2,
                        (0.29, 0.20, 0.29, 0.20))


def source_latex(session, dossier):
    """Writes report.tex and its figures in `dossier`; returns the path."""
    os.makedirs(os.path.join(dossier, "figures"), exist_ok=True)
    res = session.resultats
    d = np.asarray(res["diametres"], float)
    u = res["unite"]
    fib = res["mode"] == "fibres"
    nr = res.get("nano") or {}
    nano2 = res["mode"] == "nano" and nr.get("methode") == "stardist_deux_etapes"
    sd = res.get("moteur") == "stardist"
    gr = reglages_graphe(session)
    xsc = gr["xscale"]
    g_rap = {k: v for k, v in gr.items() if k not in ("titre", "ymax")} \
        if gr.get("rapport") else {"xscale": xsc}
    lignes = lignes_resume(session)
    sm = {k: (v, un) for k, _, v, un in lignes}
    lab = {k: lb for k, lb, _, _ in lignes}
    logos = _logos_template(dossier)
    U_cal = incertitude_calibration(session)["U_rel"]
    TAILLES = {"mean", "std", "dn10", "dn50", "dn90", "dv10", "dv50", "dv90",
               "d32", "d43", "dp_gm", "min", "max"}

    def cell(k):
        v, un = sm[k]
        if k in TAILLES and U_cal > 0 and not isinstance(v, str):
            return f"{fmt(v, '.4g')} ± {fmt(v * U_cal, '.2g')} {un}"
        if isinstance(v, str):
            return v
        if isinstance(v, (int, np.integer)):
            return f"{v} {un}".strip()
        spec = ".0f" if un == "/100" else (".1f" if un in ("%", "°") else ".4g")
        return f"{fmt(v, spec)}{'' if un == '/100' else ' '}{un}".strip()

    # ---- tables
    info = [["Sample", session.nom_echantillon], ["Image file", session.nom_image],
            ["Operator", session.operateur or "–"],
            ["Date of analysis", f"{datetime.datetime.now():%Y-%m-%d %H:%M}"],
            ["Recipe", session.recette or "–"],
            ["Analysis mode", MODE_NOMS.get(res["mode"], res["mode"])]]
    md = mode_de(res)
    gt = res["mode"] == "gouttes"
    info += md.lignes_info(res)
    info.append(["Calibration", f"{fmt(session.ppm, '.5g')} px/µm"])
    cles = md.cles_resultats(res)
    resu = [[lab[k], cell(k)] for k in cles + ["u_cal"] if k in sm]
    meth = lignes_methode(session, res, d, u, fib, sd)
    avert = avertissements_qualite(res)
    # ---- figures
    img = session.image_bgr
    cv2.imwrite(os.path.join(dossier, "figures", "image_original.png"), img)
    cv2.imwrite(os.path.join(dossier, "figures", "image_detections.png"),
                cv2.cvtColor(image_annotee_rapport(session), cv2.COLOR_RGB2BGR))
    nom_g = md.nom_diametre_de(res)
    figs_dist = []
    if d.size:
        for pond in (("nombre",) if (fib or nano2 or gt) else ("nombre", "volume")):
            fn = _fig(dossier, f"distribution_{pond}.pdf", (6.3, 3.3),
                      lambda f, pond=pond: tracer_psd(
                          f.add_subplot(111), d, u, pond, xsc, nom_g, g=g_rap))
            figs_dist.append((pond, fn))
        if gt:                         # volume des gouttes (sphères ou disques)
            from psa.droplets import volume_goutte
            vols = volume_goutte(d, res.get("hauteur_canal_um")) * 1e-3
            fn = _fig(dossier, "distribution_volume_gouttes.pdf", (6.3, 3.3),
                      lambda f: _hist_sci(f.add_subplot(111), vols,
                                          "Droplet volume (pL)"))
            figs_dist.append(("volume_gouttes", fn))
    cap_forme = {}

    def dessin_forme(f):
        ax = [f.add_subplot(2, 2, i + 1) for i in range(4)]
        cap_forme["t"] = _panneaux_forme(ax, session, res, d, u, fib, xsc, sd)
        for a, t in zip(ax, "abcd"):
            a.text(-0.17, 1.05, f"({t})", transform=a.transAxes, fontsize=9,
                   fontweight="bold")
    fig_forme = _fig(dossier, "shape.pdf", (6.3, 5.4), dessin_forme)
    cap_forme = cap_forme["t"].split(". ", 1)[1] if ". " in cap_forme["t"] else cap_forme["t"]
    sph = sphericite(res, getattr(session, "sphero", None))
    if sph:
        cv2.imwrite(os.path.join(dossier, "figures", "classes.png"),
                    cv2.cvtColor(_image_classes(session, res, sph), cv2.COLOR_RGB2BGR))
        fig_sph = _fig(dossier, "spheroidization.pdf", (4.6, 3.0),
                       lambda f: _barres_sphero(f.add_subplot(111), sph))
    # ---- references
    refs = (REFS_STARDIST if sd else []) + (REFS_NANO if nano2 else []) + REFS_NORMES
    # ---- document
    date = f"{datetime.datetime.now():%Y-%m-%d %H:%M}"
    titre_logos = "\n".join(
        rf"\includegraphics[height=1.1cm]{{{f}}}" + (r"\hspace{1cm}" if i < len(logos) - 1 else "")
        for i, f in enumerate(f for f in ("Logo_MateriaNova.png", "logo_UMONS.png",
                                          "logo_CHIPS.png") if f in logos))
    entete_logo = (r"\includegraphics[height=1.1cm]{Logo_MateriaNova.png}"
                   if "Logo_MateriaNova.png" in logos else
                   r"\textbf{\textcolor{MNbluedark}{Materia Nova}}")
    resume = (f"The sample {session.nom_echantillon} was analysed from the image "
              f"{session.nom_image} ({MODE_NOMS.get(res['mode'], res['mode'])}, "
              + ("sub-pixel ring fit). " if gt else
                 f"{'StarDist deep learning' if sd else 'classic'} segmentation). ")
              + f"{res['n_particules']} "
              + ("primary particles" if nano2 else md.objets)
              + " were measured.")
    if "dv50" in sm and not nano2:
        resume += f" Volume median diameter Dv50 = {cell('dv50')}."
    elif "dp_gm" in sm:
        resume += f" Geometric mean primary diameter = {cell('dp_gm')}."
    if sph:
        resume += (f" Spheroidization rate: {sph['taux_nombre_hors_sat']:.1f} % by "
                   f"number, {sph['taux_masse_hors_sat']:.1f} % by mass.")
    L = []
    w = L.append
    w(r"""\documentclass[a4paper,12pt]{article}

% --- Encoding & language ---
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\IfFileExists{lmodern.sty}{\usepackage{lmodern}}{}  % vector fonts when installed
\usepackage[english]{babel}

% --- General packages (Materia Nova template) ---
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{geometry}
\usepackage{fancyhdr}
\usepackage{setspace}
\usepackage{hyperref}
\usepackage{sectsty}
% --- Report additions ---
\usepackage{booktabs}
\usepackage{array}
\usepackage{float}
\usepackage[font=small,labelfont={bf,color=MNbluedark}]{caption}
\usepackage{lastpage}
\usepackage[section]{placeins}  % figures stay in their section
\usepackage{needspace}

\geometry{left=2.5cm, right=2.5cm, top=2.5cm, bottom=2.5cm}

% --- Materia Nova colours ---
\definecolor{MNbluedark}{RGB}{16,44,84}
\definecolor{MNbluelight}{RGB}{54,128,183}
\definecolor{MNgray}{RGB}{80,80,80}

% --- Header and footer ---
\pagestyle{fancy}
\fancyhf{}
\lhead{""" + entete_logo + r"""}
\rhead{\small\textcolor{MNgray}{Materia Nova — Département CHIPS \\ Chimie des Interactions Plasma-Surface}}
\lfoot{\footnotesize\textcolor{MNgray}{Materia Nova - CHIPS - Optical Particle Size Analyser - """ + tex(date) + r"""}}
\rfoot{\footnotesize\textcolor{MNgray}{\thepage\,/\,\pageref{LastPage}}}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0pt}
\setlength{\headsep}{40pt}
\setlength{\headheight}{36pt}

\hypersetup{colorlinks=true, linkcolor=MNbluedark, citecolor=MNbluedark,
    urlcolor=MNbluelight, pdftitle={Particle size analysis report, """ + tex(session.nom_echantillon) + r"""},
    pdfauthor={""" + tex(session.operateur or "Materia Nova") + r"""}}

\sectionfont{\color{MNbluedark}\bfseries\large}
\subsectionfont{\color{MNbluelight}\bfseries\normalsize}

\hyphenpenalty=10000
\exhyphenpenalty=10000

\title{\vspace{-1cm}\Huge\bfseries\color{MNbluedark} Particle size analysis report\\[4pt]
\Large\color{MNbluelight} Sample: """ + tex(session.nom_echantillon) + r"""}
\author{\large\color{MNgray} Département CHIPS — Materia Nova\\
\small\color{MNgray} Chimie des Interactions Plasma-Surface}
\date{""" + tex(f"{datetime.datetime.now():%d %B %Y}") + r"""}

\begin{document}

\maketitle
\thispagestyle{fancy}
\vspace{-0.5cm}

\begin{center}
""" + titre_logos + r"""
\end{center}

\onehalfspacing
""")
    w(r"\section*{Summary}" + "\n" + tex(resume) + "\n")
    w(r"\subsection*{Sample and analysis}" + "\n\\noindent " +
      _tableau_double(info, ("Item", "Value")) + "\n")
    w(r"\subsection*{Key results}" + "\n\\noindent " + _tableau_double(resu) + "\n")
    w(r"""\begin{figure}[htbp]\centering
\includegraphics[width=0.48\linewidth]{figures/image_original.png}\hfill
\includegraphics[width=0.48\linewidth]{figures/image_detections.png}
\caption{(a) Original image. (b) """ + tex(
        "Edge-to-edge measurement segments (colour = diameter)." if fib else
        ("Aggregates (blue contours) and detected primary particles (green)."
         if nano2 else "Detected particles; objects touching the image border "
         "are excluded from the statistics.")) + r"""}
\end{figure}
""")
    w(r"\Needspace{0.40\textheight}" + "\n" + r"\section*{Size distribution}" + "\n")
    for pond, fn in figs_dist:
        cap = ("Number-weighted " + nom_g.lower() + f" distribution (n = {d.size}). "
               "Bars: density per size class; black curve: smoothed density; "
               "dashed: cumulative; dotted lines: D10, D50, D90.") if pond == "nombre" else (
               "Droplet volume distribution ("
               + (f"squeezed discs, channel height {res['hauteur_canal_um']:g} µm"
                  if res.get("hauteur_canal_um") else "spheres")
               + "). The volume follows D³: a size CV of x % gives a volume "
               "CV of about 3x %.") if pond == "volume_gouttes" else (
               "Volume-weighted distribution (d³ weighting, spheres assumed, "
               "ISO 9276-2 [7]), comparable to laser diffraction results.")
        w(rf"\begin{{figure}}[htbp]\centering\includegraphics[width=0.95\linewidth]{{{fn}}}"
          + rf"\caption{{{tex(cap)}}}\end{{figure}}" + "\n")
    titre_forme = ("Aggregate morphology" if nano2 else
                   "Droplet shape and measurement" if gt else
                   ("Fibre network" if fib else "Particle shape"))
    # heading and figure glued in one unbreakable block (no orphan heading)
    w(r"\FloatBarrier" "\n" r"\noindent\begin{minipage}{\linewidth}" + "\n" +
      rf"\section*{{{titre_forme}}}" + "\n" + r"\centering" +
      rf"\includegraphics[width=0.95\linewidth]{{{fig_forme}}}" +
      rf"\captionof{{figure}}{{{tex(cap_forme)}}}" + "\n" + r"\end{minipage}" + "\n\n")
    if sph:
        cr = sph["criteres"]
        tab = [["Spheroidization rate, number", f"{sph['taux_nombre_hors_sat']:.1f} %"],
               ["Spheroidization rate, mass", f"{sph['taux_masse_hors_sat']:.1f} %"],
               ["Satellites (number)", f"{sph['satellites_nombre']:.1f} %"],
               ["Particles bearing satellites", f"{sph['porteuses_pct']:.1f} %"],
               ["Spherical: roundness 4A/(πF²max)", f"≥ {cr['rondeur_min']:.2f}"],
               ["Spherical: aspect ratio", f"≤ {cr['aspect_max']:.2f}"],
               ["Satellite: size ratio", f"≤ {cr['satellite_ratio']:.2f}"],
               ["Particles classified", f"{sph['n']}"]]
        w(r"\FloatBarrier" "\n" r"\noindent\begin{minipage}{\linewidth}" + "\n" +
          r"\section*{Spheroidization}" + "\n" + tex(
            "Every particle is classified spherical, irregular or satellite. The "
            "roundness 4A/(πF²max) equals 1 for a disc and decreases for elongated "
            "and for angular particles; a satellite is a small particle touching "
            "a particle at least about three times larger. The spheroidization "
            "rate is computed on the main particles (satellites excluded), by "
            "number and by mass (d³, same density).") + "\n\n" + r"""\centering
\includegraphics[width=0.85\linewidth]{figures/classes.png}
\captionof{figure}{Particle classes: green spherical, orange irregular, red satellite.}
\end{minipage}

""")
        w(r"""
\begin{figure}[htbp]\centering
\begin{minipage}{0.52\linewidth}\centering\includegraphics[width=\linewidth]{""" + fig_sph + r"""}\end{minipage}\hfill
\begin{minipage}{0.45\linewidth}\small """ + _tableau_tex(tab, ("Quantity", "Value"), (0.66, 0.30)) + r"""\end{minipage}
\caption{""" + tex("Class fractions by number and by mass. Validation on synthetic "
                   "powders of known composition: within ±5 points by number and "
                   "±6 by mass with the deep learning engine"
                   + ("." if sd else "; the classic engine merges satellites with "
                      "their carrier and is NOT reliable for this rate.")) + r"""}
\end{figure}
""")
    w(r"\FloatBarrier" "\n" r"\noindent\begin{minipage}{\linewidth}" + "\n" +
      r"\section*{Method}" + "\n" + _tableau_tex(meth, ("Item", "Setting"), (0.24, 0.72))
      + "\n" + r"\end{minipage}" + "\n\n")
    w(r"\section*{Quality assessment}" + "\n" + r"\begin{itemize}\setlength\itemsep{0pt}" + "\n")
    w(rf"\item Segmentation quality score: {fmt(res.get('score_ws', 0), '.0f')}/100." + "\n")
    if "representativeness" in sm:
        w(r"\item " + tex(f"Representativeness: {sm['representativeness'][0]} "
                          f"(n = {res['n_particules']}, CV = {fmt(sm.get('cv', (0,))[0], '.1f')} %).") + "\n")
    for a in avert:
        w(r"\item \textcolor{red!70!black}{" + tex("Warning: " + a) + "}\n")
    w(r"\end{itemize}" + "\n")
    w(r"\begin{thebibliography}{9}\small" + "\n")
    for i, r in enumerate(refs, 1):
        w(rf"\bibitem{{r{i}}} {tex(r)}" + "\n")
    w(r"\end{thebibliography}" + "\n")
    w(r"""\vfill
\noindent\textbf{Contact:} Département CHIPS — Materia Nova\\
\textit{Chimie des Interactions Plasma-Surface}\\[3pt]
\textcolor{MNgray}{\small Internal document — generated by """ + tex(f"{APP_NAME} {APP_VERSION}")
      + r""" on """ + tex(date) + r"""}

\end{document}
""")
    chemin = os.path.join(dossier, "report.tex")
    with open(chemin, "w", encoding="utf-8") as f:
        f.write("".join(L))
    return chemin


def rapport_latex(session, dossier_sortie, stem, compiler=True):
    """Builds the LaTeX report: <stem>_report_latex.zip (sources, always)
    and, when pdflatex is available and the compilation succeeds,
    <stem>_report.pdf. Returns (pdf or None, zip, message)."""
    tmp = tempfile.mkdtemp(prefix="psa_tex_")
    try:
        source_latex(session, tmp)
        zip_path = os.path.join(dossier_sortie, f"{stem}_report_latex.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for racine, _d, fichiers in os.walk(tmp):
                for fn in fichiers:
                    p = os.path.join(racine, fn)
                    z.write(p, os.path.relpath(p, tmp))
        exe = trouver_pdflatex() if compiler else None
        if not exe:
            return None, zip_path, ("LaTeX not found (install MiKTeX or TeX Live): "
                                    "built-in report used; LaTeX sources exported")
        for _passe in range(2):                       # 2 passes: page "x / N"
            pr = subprocess.run([exe, "-interaction=nonstopmode", "-halt-on-error",
                                 "report.tex"], cwd=tmp, capture_output=True,
                                timeout=240)
        pdf = os.path.join(tmp, "report.pdf")
        if pr.returncode != 0 or not os.path.exists(pdf):
            log = pr.stdout.decode("utf-8", "replace")[-1500:]
            journal("error", "LaTeX report failed:\n" + log)
            return None, zip_path, "LaTeX compilation failed (see journal.log): " \
                "built-in report used"
        sortie = os.path.join(dossier_sortie, f"{stem}_report.pdf")
        shutil.copy(pdf, sortie)
        return sortie, zip_path, "LaTeX report (Materia Nova template)"
    except subprocess.TimeoutExpired:
        return None, None, "LaTeX compilation timed out: built-in report used"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


__all__ = [_n for _n in dir() if not _n.startswith("__")]
