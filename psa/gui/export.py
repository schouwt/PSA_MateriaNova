# -*- coding: utf-8 -*-
"""PSA GUI, Report & export screen, theme, statistics view."""
from psa.widgets import *  # noqa: F401,F403

import subprocess


class MixinExport:
    """Methods of AppPSA: report & export screen, theme, statistics view."""

    # =========================================================================
    # REPORT & EXPORT SCREEN  (template: same options as Grating Analyzer)
    # =========================================================================
    def _vars_export(self):
        if hasattr(self, "var_a_cote"):
            return
        self.var_a_cote = tk.BooleanVar(value=True)
        self.var_dossier = tk.StringVar(value=str(Path.home() / "PSA_data"))
        self.var_style_fig = tk.StringVar(value=list(FIG_STYLES)[0])
        self.var_dpi = tk.StringVar(value="300")
        self.var_png = tk.BooleanVar(value=True)
        self.var_pdf = tk.BooleanVar(value=True)
        self._dernier_export = None

    def palette_export(self):
        if FIG_STYLES.get(self.var_style_fig.get(), "light") == "light":
            return LIGHT_PALETTE
        return dict(bg=T("BG"), fg=T("TEXT"), grid=T("GRID"),
                    panel=T("SURFACE"), primary=T("BLUE"), muted=T("TEXT_DIM"))

    def dossier_sortie(self):
        self.session.echantillon = self.var_echantillon.get() \
            if hasattr(self, "var_echantillon") else self.session.echantillon
        stem = safe_name(self.session.nom_echantillon)
        if self.var_a_cote.get() and self.session.chemin and \
                os.path.isdir(os.path.dirname(os.path.abspath(self.session.chemin))):
            base = os.path.dirname(os.path.abspath(self.session.chemin))
        else:
            base = self.var_dossier.get() or str(Path.home())
        return os.path.join(base, f"{stem}_analysis"), stem

    def ouvrir_export(self):
        self._vars_export()
        self._outil = "export"
        self._mode_canvas = None
        self.stepper.maj(self.etape, self.faites, "export")
        self.vider_droit()
        self.titre_droit("Report & export")
        ttk.Label(self.droit, text="Multi-page scientific report (PDF / PNG), "
                                   "CSV tables, text summary and parameters.",
                  foreground=COULEUR["dim"], font=(FONT_UI, 8),
                  wraplength=self.LARGEUR_DROIT - 24,
                  justify="left").pack(anchor="w", pady=(0, 8))
        tog = {"bootstyle": "round-toggle"} if TB else {}
        out = ttk.Labelframe(self.droit, text=" Output ", padding=8)
        out.pack(fill="x", pady=(0, 8))
        (tb if TB else ttk).Checkbutton(
            out, text="Save next to the image\n(<name>_analysis folder)",
            variable=self.var_a_cote, **tog).pack(anchor="w", pady=2)
        fr = ttk.Frame(out)
        fr.pack(fill="x", pady=(4, 2))
        ttk.Label(fr, text="Otherwise in").pack(side="left")
        ttk.Entry(fr, textvariable=self.var_dossier, width=14).pack(
            side="left", fill="x", expand=True, padx=4)
        bouton(fr, "…", self._choisir_dossier_export).pack(side="left")
        fm = ttk.Labelframe(self.droit, text=" Formats ", padding=8)
        fm.pack(fill="x", pady=(0, 8))
        ttk.Label(fm, text="CSV format").pack(anchor="w")
        cb = ttk.Combobox(fm, values=list(CSV_MODES), state="readonly")
        cb.set(next(k for k, v in CSV_MODES.items() if v == CSV_STYLE["mode"]))
        cb.pack(fill="x", pady=(0, 6))
        cb.bind("<<ComboboxSelected>>",
                lambda _e: CSV_STYLE.update(mode=CSV_MODES[cb.get()]))
        ttk.Label(fm, text="Preview").pack(anchor="w")
        self.cb_page = ttk.Combobox(fm, values=["Page 1"], state="readonly")
        self.cb_page.set(f"Page {getattr(self, '_page_apercu', 0) + 1}")
        self.cb_page.pack(fill="x", pady=(0, 6))
        self.cb_page.bind("<<ComboboxSelected>>", lambda _e: (
            setattr(self, "_page_apercu", int(self.cb_page.get().split()[1]) - 1),
            self._apercu_rapport()))
        f2 = ttk.Frame(fm)
        f2.pack(fill="x")
        ttk.Entry(f2, textvariable=self.var_dpi, width=5).pack(side="left")
        ttk.Label(f2, text="dpi").pack(side="left", padx=(4, 10))
        ttk.Checkbutton(f2, text="PNG", variable=self.var_png).pack(side="left")
        ttk.Checkbutton(f2, text="PDF", variable=self.var_pdf).pack(side="left",
                                                                   padx=6)
        fl = ttk.Labelframe(self.droit, text=" Report ", padding=8)
        fl.pack(fill="x", pady=(0, 8))
        if not hasattr(self, "var_latex"):
            self.var_latex = tk.BooleanVar(value=False)   # built-in report by default
        exe_tex = trouver_pdflatex()
        (tb if TB else ttk).Checkbutton(
            fl, text="Also a LaTeX version (Materia Nova template)",
            variable=self.var_latex,
            **({"bootstyle": "round-toggle"} if TB else {})).pack(anchor="w")
        ttk.Label(fl, text=(f"pdflatex found: {os.path.basename(os.path.dirname(exe_tex)) or exe_tex}"
                            if exe_tex else "LaTeX not found (MiKTeX or TeX Live): the "
                            "built-in report is used, the LaTeX sources are still "
                            "exported (Overleaf)."),
                  foreground=COULEUR["dim"], font=(FONT_UI, 8),
                  wraplength=self.LARGEUR_DROIT - 30, justify="left").pack(
            anchor="w", pady=(4, 0))
        bouton(self.droit, "💾  Export now  (Ctrl+S)", self.exporter_maintenant,
               kind="primary").pack(fill="x", pady=(4, 4), ipady=3)
        bouton(self.droit, "Open output folder",
               self._ouvrir_dossier_export).pack(fill="x")
        bouton(self.droit, "🗄  Archive with history (SQLite)",
               self.archiver_gui).pack(fill="x", pady=(6, 0))
        ttk.Label(self.droit, text="Files, named after the sample: "
                                   "_report.pdf (multi-page) and _report_p<i>.png, "
                                   "_particles.csv (or _fibre_sites.csv, "
                                   "_fibre_segments.csv), _summary.csv/.txt, "
                                   "_parameters.json.",
                  foreground=COULEUR["dim"], font=(FONT_UI, 8),
                  wraplength=self.LARGEUR_DROIT - 24,
                  justify="left").pack(anchor="w", pady=(10, 0))
        self.root.bind("<Control-s>", lambda _e: self.exporter_maintenant())
        self._apercu_rapport()

    def _apercu_rapport(self):
        self.fig.clf()
        if self.session.resultats is None:
            self.fig.set_facecolor(T("BG"))
            ax = self.fig.add_subplot(111)
            ax.set_axis_off()
            ax.text(0.5, 0.5, "The report preview appears after the analysis",
                    ha="center", va="center", color=T("TEXT_DIM"), fontsize=11)
            self.canvas.draw_idle()
            return
        # Preview of the multi-page report (page chosen in the panel)
        try:
            pages = pages_rapport(self.session)
        except Exception as ex:
            journal("error", f"report preview: {ex}")
            self.statut(f"Report preview failed: {ex}", "bad")
            return
        self._n_pages_rapport = len(pages)
        if hasattr(self, "cb_page"):
            self.cb_page.configure(values=[f"Page {i}" for i in
                                           range(1, len(pages) + 1)])
        k = min(max(0, getattr(self, "_page_apercu", 0)), len(pages) - 1)
        buf = _io.BytesIO()
        pages[k].savefig(buf, format="png", dpi=70)
        buf.seek(0)
        self.fig.set_facecolor(T("BG"))
        ax = self.fig.add_axes([0, 0, 1, 1])
        ax.imshow(plt_imread(buf))
        ax.set_axis_off()
        self.canvas.draw_idle()

    def _choisir_dossier_export(self):
        d = filedialog.askdirectory(parent=self.root, title="Export folder",
                                    initialdir=self.var_dossier.get() or str(Path.home()))
        if d:
            self.var_dossier.set(d)
            self.var_a_cote.set(False)

    def _ouvrir_dossier_export(self):
        d = getattr(self, "_dernier_export", None)
        if not d or not os.path.isdir(d):
            self.statut("Nothing exported yet.", "warn")
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(d)                       # noqa: F401
            elif sys.platform == "darwin":
                subprocess.Popen(["open", d])            # no shell: safe paths
            else:
                subprocess.Popen(["xdg-open", d])
        except Exception as e:
            self.statut(f"Cannot open the folder: {e}", "bad")

    def exporter_maintenant(self):
        self._vars_export()
        if self.session.resultats is None:
            self.statut("Run an analysis first.", "warn")
            return
        if not self._verifier_export():
            return
        self.session.operateur = self.var_oper.get()
        self.session.rapport_latex = bool(getattr(self, "var_latex", None) is not None
                                          and self.var_latex.get())
        formats = tuple(f for f, v in (("png", self.var_png), ("pdf", self.var_pdf))
                        if v.get())
        try:
            dpi = max(72, min(1200, int(float(self.var_dpi.get()))))
        except ValueError:
            dpi = 300
        dossier, stem = self.dossier_sortie()
        pal = self.palette_export()

        def travail(progress):
            # background task: the LaTeX compilation can take a while
            progress(20, "Writing the report…")
            return exporter_resultats(dossier, stem, self.session, pal, dpi,
                                      formats)

        def fin(fichiers):
            if isinstance(fichiers, Exception) or fichiers is None:
                journal("error", f"export failed: {fichiers}")
                messagebox.showerror("Export", str(fichiers), parent=self.root)
                return
            self._fin_export(dossier, fichiers)
        self.lancer_tache(travail, fin, "Exporting the report…")

    def _fin_export(self, dossier, fichiers):
        self._dernier_export = dossier
        msg = getattr(self.session, "moteur_rapport_msg", "")
        self.statut(f"{len(fichiers)} files exported to {dossier}"
                    + (f"  ·  {msg}" if msg else ""),
                    "good" if (not msg or msg.startswith("LaTeX report")) else "warn")

    def archiver_gui(self):
        self.session.operateur = self.var_oper.get()
        try:
            dossier = archiver_analyse(self.session)
            self.statut(f"Analysis archived: {dossier}", "good")
            messagebox.showinfo("Archiving",
                                f"Analysis archived in:\n{dossier}",
                                parent=self.root)
        except Exception as e:
            messagebox.showerror("Archiving", str(e), parent=self.root)

    # ── Mode lot ─────────────────────────────────────────────────────────────
    def ouvrir_lot(self):
        if self.session.ppm <= 0:
            self.statut("Calibrate first (the same calibration will be "
                        "applied to the whole batch).", "warn")
            return
        dossier = filedialog.askdirectory(
            parent=self.root,
            title="Image folder (same magnification as the calibrated "
                  "image!)")
        if not dossier:
            return
        if not messagebox.askyesno(
                "Batch analysis",
                f"All the images of:\n{dossier}\n\nwill be analysed "
                f"with the current recipe and calibration "
                f"({self.session.ppm:.3f} px/µm).\n\n⚠ All the images "
                "must have the same magnification.\n\nContinue?",
                parent=self.root):
            return
        self.session.operateur = self.var_oper.get()

        def travail(progress):
            return analyser_lot(dossier, self.session, progress)

        def fini(resultat):
            d, lignes, sortie = resultat
            rep = evaluer_repr(d)
            self.statut(f"Batch finished: {len(lignes)} images, "
                        f"n = {len(d)} particles → {sortie}",
                        rep["verdict"])
            messagebox.showinfo(
                "Batch finished",
                f"{len(lignes)} image(s) analysed\n"
                f"total n = {len(d)} particles\n"
                f"\nOne full report per image (images/) and a comparison "
                f"report (batch_comparison_report.pdf) in:\n{sortie}",
                parent=self.root)

        self.lancer_tache(travail, fini, "Analysing the batch…")


    # =========================================================================
    # THEME (ManoDark / ManoBlue / ManoWhite)
    # =========================================================================
    def changer_theme(self, _evt=None):
        nom = self.theme_cb.get() if hasattr(self, "theme_cb") else _THEME[0]
        if nom not in MANO_THEMES:
            return
        _THEME[0] = nom
        maj_couleurs()
        appliquer_theme_mpl()
        if TB:
            try:
                self.root.style.theme_use(nom)
            except Exception as e:
                journal("warning", f"theme not applied: {e}")
        else:
            appliquer_theme_tk(self.root)
        self.fig.set_facecolor(T("BG"))
        for pan in (getattr(self, "panneau_droit", None),
                    getattr(self, "panneau_gauche", None)):
            if pan is not None:
                pan.maj_fond()
        try:
            self.toolbar.config(background=T("BG"))
            for w in self.toolbar.winfo_children():
                w.config(background=T("BG"))
        except Exception:
            pass
        self.stepper.maj(self.etape, self.faites, self._outil)
        if self._mode_canvas == "mesure":
            self.ouvrir_mesures()
        elif self._outil == "export":
            self.ouvrir_export()
        else:
            self.aller_etape(self.etape)
        sauver_recettes(self.recettes, self.var_oper.get(), _THEME[0])
        self.statut(f"Theme {nom} (remembered for the next session).")

    # =========================================================================
    # STATISTICS VIEW (full sheet on the canvas)
    # =========================================================================
    def _dessiner_stats(self, res):
        d, u = res["diametres"], res["unite"]
        st = stats_distribution(d) if len(d) else None
        rep = evaluer_repr(d)
        s = self.session
        wc = {"good": T("GREEN"), "warn": T("YELLOW"), "bad": T("RED")}
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(T("BG"))
        ax.axis("off")
        yg, yd = [0.96], [0.96]

        def sec(t, dr=False):
            y = yd if dr else yg
            ax.text(0.52 if dr else 0.02, y[0], t, fontsize=10.5,
                    fontweight="bold", color=T("BLUE"),
                    transform=ax.transAxes, va="top")
            y[0] -= 0.045

        def kv(lb, vl, col=None, dr=False):
            y = yd if dr else yg
            x = 0.54 if dr else 0.04
            ax.text(x, y[0], lb, fontsize=8.5, color=T("TEXT_DIM"),
                    transform=ax.transAxes, va="top")
            ax.text(x + 0.24, y[0], str(vl), fontsize=8.5,
                    fontweight="bold", color=col or T("TEXT"),
                    transform=ax.transAxes, va="top")
            y[0] -= 0.032

        sec("Descriptive statistics")
        kv("Particles kept", res["n_particules"])
        kv("Excluded (border)", res.get("n_bord", 0))
        if float(res.get("taille_max", 0) or 0) > 0:
            kv(f"Excluded (> {fmt(res['taille_max'], '.4g')} {res['unite']})",
               res.get("n_trop_grands", 0))
        kv("Exclusion zones", len(s.zones))
        if st:
            kv(f"Mean Ø ({u})",
               f"{st['norm'][0]:.2f} ± {st['norm'][1]:.2f}")
            kv(f"Min / Max ({u})",
               f"{float(np.min(d)):.1f} / {float(np.max(d)):.1f}")
            kv(f"Dn10/50/90 ({u})",
               f"{st['dn10']:.1f} / {st['dn50']:.1f} / {st['dn90']:.1f}")
            kv(f"Dv10/50/90 ({u})",
               f"{st['dv10']:.1f} / {st['dv50']:.1f} / {st['dv90']:.1f}")
            kv(f"D[3,2] / D[4,3] ({u})",
               f"{st['D32']:.1f} / {st['D43']:.1f}")
            kv("Span", f"{st['span']:.2f}")
            if st["lognorm"][1] > 1.0:
                kv("Log-normale", f"geom. median {st['lognorm'][0]:.1f} {u}, "
                                  f"σg={st['lognorm'][1]:.2f}")
            if st.get("meilleur_fit"):
                kv("Best fit (AIC)", NOMS_FIT.get(st["meilleur_fit"], st["meilleur_fit"]))
            if len(res.get("circularites", [])):
                c = res["circularites"]
                kv("Mean circularity",
                   f"{float(np.mean(c)):.3f} ± {float(np.std(c)):.3f}")
        sf = stats_forme(res) if res["mode"] != "gouttes" else {}
        if sf.get("solidite_moy") is not None:
            kv("Mean aspect / solidity",
               f"{sf.get('aspect_moy', 0):.2f} / {sf['solidite_moy']:.3f}")
            kv("% satellites",
               f"{sf['pct_satellites']:.1f} % "
               f"(solidity < {SEUIL_SOLIDITE_SATELLITE})")
        ed = res.get("editions", {})
        if any(ed.values()):
            kv("Manual edits",
               f"{ed['suppr']} deleted · {ed['ajout']} added · "
               f"{ed['scind']} split · {ed['fusion']} merged", T("YELLOW"))
        sec("Segmentation quality")
        kv("Score", f"{res['score_ws']:.0f}/100",
           wc.get(res["statut_ws"]))
        kv("Message", res["msg_ws"], wc.get(res["statut_ws"]))
        sec("Representativeness")
        kv("Verdict", rep["message"], wc.get(rep["verdict"]))
        kv("CI95 on the mean",
           f"±{rep['ic95']:.2f} {u} (±{rep['precision_rel']:.1f}%)")
        kv("CV", f"{rep['cv']:.1f}% – {rep['cv_msg']}")
        kv("Recommended n (10%)", rep["n_min_10pct"])
        if res["mode"] == "nano":
            nr = res["nano"]
            sec("Nano scales (ISO 26824)")
            if len(nr["seg"]["d_aggl_nm"]):
                da = nr["seg"]["d_aggl_nm"]
                kv("Agglomerates ECD (nm)",
                   f"{float(np.mean(da)):.0f} ± {float(np.std(da)):.0f} "
                   f"(n={len(da)})")
            if nr["granulo"]:
                g = nr["granulo"]
                kv("Granulo Dx10/50/90 (nm)",
                   f"{g['dx10']:.0f} / {g['dx50']:.0f} / {g['dx90']:.0f}")
            if len(nr["d_hough_nm"]):
                kv("Hough primaries (nm)",
                   f"{float(np.mean(nr['d_hough_nm'])):.0f} ± "
                   f"{float(np.std(nr['d_hough_nm'])):.0f} "
                   f"(n={len(nr['d_hough_nm'])})")
            if nr["df"]:
                kv("Fractal dimension Df",
                   f"{nr['df']:.2f} (R²={nr['df_r2']:.3f})")
        if s.mesures:
            arr = np.array([m["um"] for m in s.mesures])
            sec("Thickness measurements 📏")
            kv("n measurements", len(arr))
            kv("Mean", fmt_long(float(np.mean(arr))))
            if len(arr) > 1:
                kv("Standard deviation", fmt_long(float(np.std(arr, ddof=1))))

        sec("Parameters used", dr=True)
        for k in sorted(s.params):
            kv(self.LIBELLES_AVANCE.get(k, k), s.params[k], dr=True)
        sec("Context", dr=True)
        kv("Image", s.nom_image, dr=True)
        kv("Calibration", f"{s.ppm:.4f} px/µm", dr=True)
        kv("Recipe", s.recette or "-", dr=True)
        kv("Operator", s.operateur or "-", dr=True)
        if s.bet_m2g and st:
            cb = comparer_bet(st["D32"], u, s.rho_gcm3, s.bet_m2g)
            sec("Validation BET", dr=True)
            kv("Geometric SSA / BET",
               f"{cb['ssa_geom']:.1f} / {s.bet_m2g:g} m²/g", dr=True)
            kv(f"d_BET ({u}) / F",
               f"{cb['d_bet']:.1f} / {cb['facteur']:.2f}",
               wc.get(cb["verdict"]), dr=True)
