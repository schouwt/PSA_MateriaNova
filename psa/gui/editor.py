# -*- coding: utf-8 -*-
"""PSA GUI, Help and manual editing of the detections."""
from psa.widgets import *  # noqa: F401,F403


class MixinHelpEditor:
    """Methods of AppPSA: help and manual editing of the detections."""

    # =========================================================================
    # AIDE  (F1)  —  ouvre le guide PDF, repli sur un résumé intégré
    # =========================================================================
    def ouvrir_aide(self):
        import subprocess
        import webbrowser
        for p in chemins_aide():
            if p.exists():
                try:
                    if sys.platform.startswith("win"):
                        os.startfile(str(p))            # noqa: F401  (attribut Windows)
                    elif sys.platform == "darwin":
                        subprocess.Popen(["open", str(p)])
                    else:
                        subprocess.Popen(["xdg-open", str(p)])
                except Exception:
                    try:
                        webbrowser.open(p.as_uri())
                    except Exception:
                        break
                self.statut(f"Guide opened: {p}")
                return
        self._aide_integree()

    AIDE_TEXTE = """\
QUICK START
  1. Choose a recipe (material / microscope preset) and enter the operator
     name, then open the image (Ctrl+O).
  2. Calibration: draw a line along the scale bar and enter its real
     length. This is the most critical step: a 5 % error on the line is a
     5 % error on every diameter of the report.
  3. Zones (optional): mask the scale bar, captions, artefacts.
  4. Settings: three sliders. Sensitivity (amount detected),
     Separation (touching particles), Minimum size. The overlaid preview
     updates by itself; the coloured badge gives the quality.
     Click Analyse.
  5. Results: Image / Statistics / Distribution / Pipeline views.
     Report & export (left panel): multi-panel report PNG / PDF, CSV
     tables, text summary, parameters; automatic time-stamped archiving.

THE SLIDERS, ONE SENTENCE EACH
  Sensitivity too high: noise becomes particles; too low: faint particles
  disappear. Separation too high: particles cut into pieces; too low:
  touching particles counted as one. Watch the orange contours of the
  preview.

KEY NOTIONS
  ECD: diameter of the disc with the same area as the particle.
  Dn50 / Dv50: medians by NUMBER / by VOLUME (d³ weighting). Never compare
  an imaging Dn50 with a laser diffraction Dv50.
  D[3,2] (Sauter): links the image to the specific surface area;
  D[4,3]: comparable to the laser granulometer. Span: width of the
  distribution. Log-normal: the natural shape of powder distributions
  (sigma_g ~ 1.3-1.5 for a carbon black).
  Representativeness: n >= 30 at least for the mean; use the Batch mode
  to aggregate several images.

SEM: AUTOMATIC POLARITY
  The software detects whether the particles are BRIGHT (secondary
  electrons, SE: bright objects on a dark background) or DARK
  (transmission / STEM mode, often noted « TE » in the databar: absorbing
  objects on a bright background, e.g. dispersed / sonicated nanocarbons).
  It can be forced in the advanced mode (« polarite » parameter:
  auto / claires / sombres) if the detection is wrong.

SEM
  Two SEM modes: « micron powders (µm) », same algorithms (Sauvola,
  h-maxima watershed) for 5-200 µm metal powders, granules…, and
  « nano (nm) » for nanometric agglomerates.
  Three scales (ISO 26824): primary particle, aggregate (fused),
  agglomerate (separable), always reported separately. The nano mode
  combines h-maxima watershed, morphological granulometry (no
  segmentation) and Hough (spherical primaries); Df ~ 1.7-1.9 expected.
  BET (« Carbon material » box): F = SSA_BET / SSA_geometric;
  F ~ 1 confirms the sizes, F > 1.25 porosity or under-segmentation,
  F < 0.8 sintering or over-segmentation.

FIBRES
  The fibre diameter is measured edge to edge, without threshold, by
  pairing the two edges of each fibre (coloured segments on the image).

THICKNESS MEASUREMENTS (left panel)
  Draw a line ACROSS the layer (PVD coating, shell); both interfaces are
  detected on the grey-level profile. Repeat along the deposit, or use
  the automatic sweep / bright ribbon tools: the CV = uniformity. An
  oblique line overestimates the thickness (1/cos).

MANUAL EDITING & QUALITY CONTROL
  ✏ Manual editing (results): Delete (click), Add (polygon, right click
  to close), Split (line across a merged particle), Merge (two clicks).
  Every edit is traced in the report.
  ⚖ History / Compare: every archived analysis enters the database;
  compare the batch of the day with a reference (Kolmogorov-Smirnov test)
  or follow the Dn50 trend of a material (control chart).
  🟡 ImageJ ROIs: export of the particles as a .zip readable by the
  ImageJ ROI Manager; import of ImageJ ROIs as exclusion zones.

CLASSIC PITFALLS
  Sloppy calibration; mixed weightings; n < 30; score looked at without
  checking the image; particles < ~10 px (increase the magnification);
  parameters changed within a series (use the recipes).

IN CASE OF PROBLEM
  A log is kept in ~/.psa_materia/journal.log (versions, analyses,
  detailed errors): attach it to any report. If an analysis keeps 0
  particle, the status bar gives the most likely cause (empty / saturated
  mask, minimum size, calibration). Self-test: psa_app.py --autotest.

FULL GUIDE
  Place guide_psa.pdf next to psa_app.py (or in ~/.psa_materia) so that
  this button opens the full PDF (theory of thresholding, watershed,
  statistics, SEM, BET, short bibliography). LaTeX source: guide_psa.tex.
"""

    def _aide_integree(self):
        top = tk.Toplevel(self.root)
        top.title(f"Help, {APP_TITLE}")
        ajuster_a_l_ecran(top, 760, 640)
        top.configure(bg=T("BG"))
        fr = ttk.Frame(top, padding=10)
        fr.pack(fill="both", expand=True)
        ttk.Label(fr, text="Quick guide (full PDF not found)",
                  font=(FONT_UI, 11, "bold")).pack(anchor="w")
        ttk.Label(fr, foreground=COULEUR["dim"], font=(FONT_UI, 8),
                  text=f"For the full guide, place {FICHIER_AIDE} "
                       f"next to psa_app.py.").pack(anchor="w", pady=(0, 6))
        txt = tk.Text(fr, wrap="word", bg=T("SURFACE"), fg=T("TEXT"),
                      insertbackground=T("TEXT"), relief="flat",
                      font=(FONT_UI, 9), padx=10, pady=8)
        sb = ttk.Scrollbar(fr, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        txt.pack(fill="both", expand=True)
        txt.insert("1.0", self.AIDE_TEXTE)
        txt.config(state="disabled")
        bouton(top, "Close", top.destroy).pack(pady=8)
        top.bind("<Escape>", lambda _e: top.destroy())




    # =========================================================================
    # ÉDITION MANUELLE DES DÉTECTIONS  (point central — façon ROI Manager)
    # =========================================================================
    def ouvrir_edition(self):
        res = self.session.resultats
        if res is None:
            return
        self._mode_canvas = "edition"
        self._ed_pile = []                  # pile d'annulation (8 niveaux)
        self._ed_poly, self._ed_poly_art = [], []
        self._ed_scind_p0, self._ed_tmp = None, None
        self._ed_fusion_i, self._ed_fusion_art = None, None
        self.vider_droit()
        self.titre_droit("✏ · Manual editing")
        ttk.Label(self.droit, wraplength=self.LARGEUR_DROIT - 24,
                  justify="left", foreground=COULEUR["dim"],
                  text="Correct the detection; all the statistics "
                       "are recomputed and the edit is traced in the "
                       "report.").pack(anchor="w")
        fr = ttk.Labelframe(self.droit, text=" Tool ", padding=6)
        fr.pack(fill="x", pady=8)
        self.var_outil_ed = tk.StringVar(value="suppr")
        OUTILS = [
            ("Delete (click)", "suppr",
             "Click a false detection to remove it."),
            ("Add (polygon)", "ajout",
             "Left click = vertex around the missed particle, "
             "right click = close."),
            ("Split (line)", "scind",
             "Draw a line ACROSS an over-merged particle: "
             "it is cut along the line."),
            ("Merge (2 clicks)", "fusion",
             "Click the two pieces of an over-segmented particle."),
        ]
        for txt, val, tip in OUTILS:
            rb = ttk.Radiobutton(fr, text=txt, variable=self.var_outil_ed,
                                 value=val, command=self._ed_outil_change)
            rb.pack(anchor="w")
            infobulle(rb, tip)
        self.lbl_ed = ttk.Label(self.droit, font=(FONT_UI, 9, "bold"),
                                wraplength=self.LARGEUR_DROIT - 24,
                                justify="left")
        self.lbl_ed.pack(anchor="w", pady=6)
        bouton(self.droit, "↶  Undo the last one",
               self._ed_undo).pack(fill="x", pady=2)
        bouton(self.droit, "✓  Finish editing", self._ed_terminer,
               kind="primary").pack(fill="x", pady=(10, 0))
        self._ed_redessiner()
        self._ed_maj_compteur()
        self.statut("Editing mode: the changes are traced in "
                    "the report.")

    def _ed_outil_change(self):
        self._ed_poly_clear()
        self._ed_fusion_reset()

    def _ed_redessiner(self):
        res = self.session.resultats
        if "contours" in res:               # optique : ré-annoter
            res["image_base"] = rendre_annotation(
                self.session.image_bgr, res["contours"],
                res["centres_px"], res.get("cordes", []),
                self.session.zones)
        self.fig.clf()
        self.fig.set_facecolor(T("BG"))
        self._dessiner_image_annotee(res)
        self._ax_ed = self.fig.axes[0]
        self._ax_ed.set_title(
            f"EDITING, {res['n_particules']} particles  ·  tool: "
            f"{NOMS_OUTILS_ED.get(self.var_outil_ed.get(), self.var_outil_ed.get())}", fontsize=9, color=T("YELLOW"))
        self.canvas.draw_idle()

    def _ed_maj_compteur(self):
        res = self.session.resultats
        ed = res.get("editions", {"suppr": 0, "ajout": 0,
                                  "scind": 0, "fusion": 0})
        self.lbl_ed.config(
            text=f"n = {res['n_particules']}\n"
                 f"{ed['suppr']} deleted · {ed['ajout']} added · "
                 f"{ed['scind']} split · {ed['fusion']} merged")

    # ── pile d'annulation ──
    def _ed_snapshot(self):
        res = self.session.resultats
        s = {"diametres": res["diametres"].copy(),
             "n_particules": res["n_particules"],
             "editions": dict(res.get("editions", {"suppr": 0, "ajout": 0,
                                                   "scind": 0, "fusion": 0}))}
        for cle in ("circularites", "aspects", "solidites"):
            if cle in res:
                s[cle] = res[cle].copy()
        for cle in ("contours", "cordes", "centres_px", "ids"):
            if cle in res:
                s[cle] = list(res[cle])
        if res.get("labels_ws") is not None:
            s["labels_ws"] = res["labels_ws"].copy()
        self._ed_pile.append(s)
        # Garde-fou mémoire : sur une image 4096², chaque instantané de
        # labels pèse ~67 Mo — profondeur d'annulation réduite.
        lab = self.session.resultats.get("labels_ws")
        prof = 8 if lab is None or lab.nbytes < 25e6 else 3
        if len(self._ed_pile) > prof:
            self._ed_pile.pop(0)

    def _ed_undo(self):
        if not self._ed_pile:
            self.statut("Nothing to undo.", "warn")
            return
        self.session.resultats.update(self._ed_pile.pop())
        self._ed_redessiner()
        self._ed_maj_compteur()

    # ── interactions canvas ──
    def _ed_press(self, e):
        if e.inaxes != self._ax_ed or e.xdata is None or self.toolbar.mode:
            return
        outil = self.var_outil_ed.get()
        res = self.session.resultats
        if outil == "suppr" and e.button == 1:
            idx = edition_index_a(res, e.xdata, e.ydata)
            if idx is None:
                self.statut("No particle under the click.", "warn")
                return
            self._ed_snapshot()
            edition_supprimer(res, idx)
            self._ed_redessiner()
            self._ed_maj_compteur()
        elif outil == "ajout":
            if e.button == 1:
                self._ed_poly.append((e.xdata, e.ydata))
                if len(self._ed_poly) > 1:
                    ln, = self._ax_ed.plot(
                        [self._ed_poly[-2][0], e.xdata],
                        [self._ed_poly[-2][1], e.ydata],
                        color=T("GREEN"), lw=1.6, ls="--")
                    self._ed_poly_art.append(ln)
                    self.canvas.draw_idle()
            elif e.button == 3 and len(self._ed_poly) >= 3:
                self._ed_snapshot()
                if edition_ajouter(res, self._ed_poly, self.session.ppm):
                    self._ed_poly_clear()
                    self._ed_redessiner()
                    self._ed_maj_compteur()
                else:
                    self._ed_pile.pop()
                    self.statut("Polygon too small.", "warn")
        elif outil == "scind" and e.button == 1:
            self._ed_scind_p0 = (e.xdata, e.ydata)
        elif outil == "fusion" and e.button == 1:
            idx = edition_index_a(res, e.xdata, e.ydata)
            if idx is None:
                return
            if self._ed_fusion_i is None:
                self._ed_fusion_i = idx
                cnts = contours_du_resultat(res)
                if idx < len(cnts):
                    xs = cnts[idx][:, 0, 0]
                    ys = cnts[idx][:, 0, 1]
                    self._ed_fusion_art, = self._ax_ed.plot(
                        np.append(xs, xs[0]), np.append(ys, ys[0]),
                        color=T("YELLOW"), lw=2.5)
                    self.canvas.draw_idle()
                self.statut("First particle selected: click "
                            "the second one.")
            else:
                self._ed_snapshot()
                if edition_fusionner(res, self._ed_fusion_i, idx,
                                     self.session.ppm):
                    self.statut("Merge done.")
                else:
                    self._ed_pile.pop()
                    self.statut("Fusion impossible.", "warn")
                self._ed_fusion_reset()
                self._ed_redessiner()
                self._ed_maj_compteur()
        elif outil == "fusion" and e.button == 3:
            self._ed_fusion_reset()
            self.canvas.draw_idle()

    def _ed_motion(self, e):
        if self.var_outil_ed.get() != "scind" or self._ed_scind_p0 is None \
                or e.inaxes != self._ax_ed or e.xdata is None:
            return
        if self._ed_tmp:
            self._ed_tmp.remove()
        x0, y0 = self._ed_scind_p0
        self._ed_tmp, = self._ax_ed.plot([x0, e.xdata], [y0, e.ydata],
                                         color=T("RED"), lw=1.8, ls="--")
        self.canvas.draw_idle()

    def _ed_release(self, e):
        if self.var_outil_ed.get() != "scind" or self._ed_scind_p0 is None:
            return
        p0 = self._ed_scind_p0
        self._ed_scind_p0 = None
        if self._ed_tmp:
            self._ed_tmp.remove()
            self._ed_tmp = None
        if e.inaxes != self._ax_ed or e.xdata is None:
            self.canvas.draw_idle()
            return
        res = self.session.resultats
        milieu = ((p0[0] + e.xdata) / 2, (p0[1] + e.ydata) / 2)
        idx = edition_index_a(res, *milieu)
        if idx is None:
            idx = edition_index_a(res, *p0)
        if idx is None:
            self.statut("The line must cross a particle.", "warn")
            self.canvas.draw_idle()
            return
        self._ed_snapshot()
        n_ok = edition_scinder(res, idx, p0, (e.xdata, e.ydata),
                               self.session.ppm)
        if n_ok < 2:
            self.session.resultats.update(self._ed_pile.pop())
            self.statut("Split impossible: the line must cut the "
                        "particle right through.", "warn")
        else:
            self.statut(f"Particle split into {n_ok}.")
        self._ed_redessiner()
        self._ed_maj_compteur()

    def _ed_poly_clear(self):
        for a in self._ed_poly_art:
            try:
                a.remove()
            except Exception:
                pass
        self._ed_poly_art.clear()
        self._ed_poly.clear()
        self.canvas.draw_idle()

    def _ed_fusion_reset(self):
        self._ed_fusion_i = None
        if self._ed_fusion_art is not None:
            try:
                self._ed_fusion_art.remove()
            except Exception:
                pass
            self._ed_fusion_art = None

    def _ed_terminer(self):
        self._ed_poly_clear()
        self._ed_fusion_reset()
        ed = self.session.resultats.get("editions", {})
        if any(ed.values()) and self.var_autoarch.get():
            try:
                archiver_analyse(self.session)
            except Exception:
                pass
        self.aller_etape(4)
