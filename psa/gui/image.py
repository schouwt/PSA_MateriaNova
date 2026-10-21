# -*- coding: utf-8 -*-
"""PSA GUI, Steps 1-2: image loading and calibration."""
from psa.widgets import *  # noqa: F401,F403


class MixinImage:
    """Methods of AppPSA: steps 1-2: image loading and calibration."""

    # =========================================================================
    # ÉTAPE 0 — IMAGE
    # =========================================================================
    def construire_etape0(self):
        self.vider_droit()
        self.titre_droit("1 · Image")
        b = bouton(self.droit, "📂  Open an image…", self.ouvrir_image,
                   kind="primary")
        b.pack(fill="x", pady=(0, 10))
        infobulle(b, "Formats : PNG, JPG, TIFF, BMP (Ctrl+O)")

        # The RECIPE (left panel) is the single choice of the analysis type:
        # it sets the mode and its settings. Here, a read-only summary.
        fr = ttk.Labelframe(self.droit, text=" Current recipe ", padding=8)
        fr.pack(fill="x", pady=(0, 10))
        ttk.Label(fr, text=self.session.recette or "–",
                  font=(FONT_UI, 10, "bold"),
                  wraplength=self.LARGEUR_DROIT - 40,
                  justify="left").pack(anchor="w")
        ttk.Label(fr, text=self._resume_recette(), font=(FONT_UI, 8),
                  foreground=COULEUR["dim"], wraplength=self.LARGEUR_DROIT - 40,
                  justify="left").pack(anchor="w", pady=(4, 0))
        ttk.Label(fr, text="Change it with Recipe in the left panel.",
                  font=(FONT_UI, 8, "italic"), foreground=COULEUR["dim"],
                  wraplength=self.LARGEUR_DROIT - 40,
                  justify="left").pack(anchor="w", pady=(6, 0))
        manque = ((self.session.mode in ("meb", "nano") and not NANO_OK)
                  or (self.session.mode == "fibres" and not FIB_OK))
        if manque:
            ttk.Label(fr, text="Module missing for this mode (psa_nano_meb / "
                               "psa_fibres): see README.", font=(FONT_UI, 8),
                      foreground=PDF_C["red"], wraplength=self.LARGEUR_DROIT - 40,
                      justify="left").pack(anchor="w", pady=(6, 0))

        self.lbl_infos_img = ttk.Label(self.droit, text="No image.",
                                       foreground=COULEUR["dim"],
                                       wraplength=self.LARGEUR_DROIT - 24,
                                       justify="left")
        self.lbl_infos_img.pack(anchor="w", pady=(4, 10))
        bouton(self.droit, "💾  Save as recipe…",
               self.sauver_recette_courante).pack(fill="x", side="bottom")

        if self.session.image_bgr is not None:
            self._afficher_image_chargee()
        else:
            self.fig.clf()
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.55, "Welcome",
                    ha="center", fontsize=11, color=T("TEXT_DIM"))
            ax.text(0.5, 0.45, "Open an image to start "
                               "(button on the right or Ctrl+O)",
                    ha="center", fontsize=10, color=T("TEXT_DIM"))
            ax.axis("off")
            self.canvas.draw_idle()

    def _resume_recette(self):
        """What the current recipe measures (mode, unit, engine)."""
        m = self.session.mode
        p = self.session.params
        moteur = ("deep learning (StarDist)" if moteur_effectif(p) == "stardist"
                  else "classic (threshold + watershed)")
        txt = {
            "optique": f"Optical microscopy, sizes in µm. Engine: {moteur}. "
                       "Measures equivalent diameters and shape of separable "
                       "particles.",
            "meb": f"SEM, micron powders, sizes in µm. Engine: {moteur}. "
                   "Equivalent diameter, shape, satellites"
                   + (", convex completion of spheres" if p.get("combler_convexe")
                      else "") + ".",
            "nano": "SEM nanocarbon, sizes in nm. Two steps: aggregates (TE "
                    "transmission), then primary particles (StarDist model "
                    "trained for carbon aggregates); fractal morphology.",
            "gouttes": "Microfluidic droplets and capsules in transmitted "
                       "light (dark ring, inside as bright as the liquid), "
                       "sizes in µm: diameter with sub-pixel ring fit, CV, "
                       "volume (spheres or squeezed droplets).",
            "fibres": "SEM electrospun fibres, sizes in µm. Fibre diameter "
                      "edge to edge, beads, alignment, mat porosity"
                      + (", fibre pores" if p.get("combler_pores") else "") + ".",
        }
        return txt.get(m, m)


    def ouvrir_image(self):
        chemin = filedialog.askopenfilename(
            parent=self.root, title="Select an image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
                       ("All", "*.*")])
        if not chemin:
            return
        img, note = charger_image(chemin)
        if img is None:
            messagebox.showerror("Error",
                                 f"Cannot read:\n{chemin}",
                                 parent=self.root)
            return
        if note:
            self.statut(f"Image converted: {note}")
            journal("info", f"{Path(chemin).name} — {note}")
        self._calib_meta = calibration_metadonnees(chemin)
        if self._calib_meta[0]:
            journal("info", f"{Path(chemin).name}, metadata calibration "
                            f"{self._calib_meta[0]:.4f} px/µm "
                            f"({self._calib_meta[1]})")
        self.session.chemin = chemin
        # sample name proposed from the image name (kept if typed by the user)
        if not self.var_echantillon.get().strip() or \
                self.var_echantillon.get() == getattr(self, "_echantillon_auto", None):
            self._echantillon_auto = Path(self.session.chemin).stem
            self.var_echantillon.set(self._echantillon_auto)
        self.session.image_bgr = img
        self.session.zones = []
        self.session.resultats = None
        self.faites = {0} | ({1} if self.session.ppm > 0 else set())
        self._img_pretraitee = None
        self.stepper.maj(self.etape, self.faites)
        self._afficher_image_chargee()
        self.statut(f"{Path(chemin).name} loaded, "
                    "go on to the calibration.", "good")

    def _afficher_image_chargee(self):
        img = self.session.image_bgr
        h, w = img.shape[:2]
        self.lbl_infos_img.config(
            text=f"{self.session.nom_image}\n{w} × {h} px"
                 + (f"\nCalibration : {self.session.ppm:.3f} px/µm"
                    if self.session.ppm > 0 else
                    "\nCalibration: to do (step 2)"))
        self.dessiner_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                            self.session.nom_image)

    # =========================================================================
    # ÉTAPE 1 — CALIBRATION
    # =========================================================================
    def _appliquer_calib_meta(self):
        ppm, source = self._calib_meta
        self.session.ppm = float(ppm)
        self.session.calib_source, self.session.calib_l_px = "metadata", 0.0
        self.faites.add(1)
        self.stepper.maj(self.etape, self.faites)
        self.statut(f"Calibration applied: {ppm:.4f} px/µm ({source}). "
                    "Check it once against the scale bar.", "good")
        journal("info", f"metadata calibration applied ({source})")
        self.aller_etape(1)

    def construire_etape1(self):
        self.vider_droit()
        self.titre_droit("2 · Scale calibration")
        ttk.Label(self.droit, wraplength=self.LARGEUR_DROIT - 24,
                  justify="left", foreground=COULEUR["dim"],
                  text="Click and drag along the scale bar of the "
                       "image, then enter its real length.\n\n"
                       "Tip: zoom in first with the magnifier of the "
                       "toolbar to aim precisely.").pack(anchor="w")
        if getattr(self, "_calib_meta", (None, None))[0]:
            ppm_m, src_m = self._calib_meta
            fr_m = ttk.Labelframe(self.droit,
                                  text=" Image metadata ", padding=6)
            fr_m.pack(fill="x", pady=(8, 2))
            ttk.Label(fr_m, wraplength=self.LARGEUR_DROIT - 40,
                      justify="left",
                      text=f"Pixel size found in the file:\n"
                           f"{ppm_m:.4f} px/µm "
                           f"({1000.0/ppm_m:.1f} nm/px)\n"
                           f"Source: {src_m}").pack(anchor="w")
            b_m = bouton(fr_m, "✓  Use this calibration",
                         self._appliquer_calib_meta, kind="primary")
            b_m.pack(fill="x", pady=(6, 0))
            infobulle(b_m, "Best-effort reading of the vendor metadata "
                           "(FEI/Thermo, Zeiss, TIFF tags). "
                           "Check it at least once against the scale "
                           "bar to validate your microscope.")
        # Détection automatique de la barre (reporte la longueur en pixels)
        if NANO_OK and self.session.image_bgr is not None:
            b_bar = bouton(self.droit, "🔍  Detect the scale bar",
                           self.detecter_barre_gui)
            b_bar.pack(fill="x", pady=(6, 4))
            infobulle(b_bar, "Automatically measures the graduated scale bar "
                             "of the SEM databar; the length in "
                             "pixels is reported below.")
        # Saisie inline (plus de fenêtre séparée) : longueur mesurée en
        # pixels (par le trait ou la détection) + longueur réelle → px/µm.
        fr_cal = ttk.Labelframe(self.droit, text=" Calibration ", padding=8)
        fr_cal.pack(fill="x", pady=(4, 2))
        self.lbl_calib_px = ttk.Label(fr_cal, text="Measured length: –",
                                      font=(FONT_UI, 9, "bold"))
        self.lbl_calib_px.pack(anchor="w")
        row = ttk.Frame(fr_cal)
        row.pack(fill="x", pady=(6, 0))
        ttk.Label(row, text="Real:").pack(side="left")
        self.var_calib_val = tk.StringVar()
        self._calib_entry = ttk.Entry(row, textvariable=self.var_calib_val,
                                      width=7)
        self._calib_entry.pack(side="left", padx=(4, 4))
        self.var_calib_unite = tk.StringVar(
            value="nm" if self.session.mode == "nano" else "µm")
        ttk.Combobox(row, textvariable=self.var_calib_unite,
                     values=["µm", "nm"], state="readonly",
                     width=5).pack(side="left")
        self._calib_entry.bind(
            "<Return>", lambda _e: self._appliquer_calibration_inline())
        b_val = bouton(fr_cal, "✓  Validate the calibration",
                       self._appliquer_calibration_inline, kind="primary")
        b_val.pack(fill="x", pady=(8, 0))
        self.lbl_calib = ttk.Label(fr_cal, font=(FONT_UI, 9, "bold"),
                                   wraplength=self.LARGEUR_DROIT - 30,
                                   justify="left")
        self.lbl_calib.pack(anchor="w", pady=(8, 0))
        fr_u = ttk.Frame(fr_cal)
        fr_u.pack(fill="x", pady=(8, 0))
        ttk.Label(fr_u, text="Magnification accuracy (%)",
                  font=(FONT_UI, 8)).pack(side="left")
        self.var_u_instr = tk.StringVar(
            value=f"{self.session.u_instr_pct:g}" if self.session.u_instr_pct else "")
        e_u = ttk.Entry(fr_u, textvariable=self.var_u_instr, width=6)
        e_u.pack(side="right")
        infobulle(e_u, "Standard uncertainty of the microscope magnification, "
                       "in %. Typically 1 to 3 % for an SEM, less after "
                       "calibration with a reference grid. Leave empty if "
                       "unknown: only the scale-bar reading is then included "
                       "in the uncertainty, which is stated in the report.")

        def maj_u(*_a):
            try:
                v = float(self.var_u_instr.get().replace(",", ".") or 0)
                self.session.u_instr_pct = max(0.0, v)
            except ValueError:
                return
            self._maj_lbl_calib()
        self.var_u_instr.trace_add("write", maj_u)
        self._maj_lbl_calib()
        bouton(self.droit, "↺  Measure again",
               self.construire_etape1).pack(fill="x", pady=(8, 0))

        img = self.session.image_bgr
        self._ax_calib = self.dessiner_image(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            "Calibration: draw a line along the scale bar")
        self._mode_canvas = "calib"
        self._calib_p0 = None
        self._calib_ligne = None
        self._calib_l_px = None        # longueur du dernier trait/barre (px)

    def _maj_lbl_calib(self):
        if self.session.ppm > 0:
            nmpx = 1000.0 / self.session.ppm
            u = incertitude_calibration(self.session)
            txt_u = (f"\n    scale uncertainty ±{100 * u['U_rel']:.2f} % (k = 2)"
                     + ("" if u["instr_inclus"] else ", reading only")
                     if u["U_rel"] > 0 else "\n    uncertainty: enter the "
                     "instrument accuracy below")
            self.lbl_calib.config(
                text=f"✓  {self.session.ppm:.4f} px/µm\n"
                     f"    ({nmpx:.2f} nm/px)" + txt_u,
                foreground=COULEUR["good"])
        else:
            self.lbl_calib.config(text="Not calibrated yet.",
                                  foreground=COULEUR["warn"])

    def _calib_press(self, e):
        if e.inaxes != self._ax_calib or e.button != 1 \
           or self.toolbar.mode:
            return
        self._calib_p0 = (e.xdata, e.ydata)

    def _calib_motion(self, e):
        if self._calib_p0 is None or e.inaxes != self._ax_calib \
           or e.xdata is None:
            return
        if self._calib_ligne:
            self._calib_ligne.remove()
        x0, y0 = self._calib_p0
        self._calib_ligne, = self._ax_calib.plot(
            [x0, e.xdata], [y0, e.ydata],
            color=COULEUR["accent"], lw=2.5, ls="--")
        self.canvas.draw_idle()

    def _calib_release(self, e):
        if self._calib_p0 is None:
            return
        x0, y0 = self._calib_p0
        self._calib_p0 = None
        if e.inaxes != self._ax_calib or e.xdata is None:
            return
        l_px = math.hypot(e.xdata - x0, e.ydata - y0)
        if l_px < 5:
            self.statut("Line too short, try again.", "warn")
            return
        # Report inline : la longueur en pixels s'affiche, l'opérateur saisit
        # la longueur réelle dans le panneau et valide (sans fenêtre séparée).
        self._calib_l_px = l_px
        self._calib_src = "manual"
        self.lbl_calib_px.config(text=f"Measured length: {l_px:.0f} px")
        self._calib_entry.focus_set()
        self.statut(f"Line of {l_px:.0f} px measured: enter its real length "
                    "in the panel, then Validate.", "good")

    def detecter_barre_gui(self):
        g = cv2.cvtColor(self.session.image_bgr, cv2.COLOR_BGR2GRAY)
        span = nano.detecter_barre_echelle(g)
        if not span:
            messagebox.showinfo(
                "Scale bar",
                "No graduated scale bar detected automatically.\n"
                "Draw it manually along the bar.",
                parent=self.root)
            return
        # Surligne la barre détectée pour que l'opérateur la vérifie
        y = nano.detecter_databar(g)
        self._ax_calib.axhline(y + 6, color=COULEUR["accent"], lw=1.0,
                               alpha=0.4)
        self.canvas.draw_idle()
        self._calib_l_px = span
        self._calib_src = "auto"
        self.lbl_calib_px.config(
            text=f"Measured length: {span} px  (bar detected)")
        self._calib_entry.focus_set()
        self.statut(f"Bar detected: {span} px. Enter its value (read "
                    "under the bar, e.g. 500 nm), then Validate.", "good")

    def _appliquer_calibration_inline(self):
        """Calcule px/µm à partir de la longueur mesurée et de la longueur
        réelle saisie dans le panneau. Remplace l'ancienne fenêtre modale."""
        l_px = getattr(self, "_calib_l_px", None)
        if not l_px:
            self.statut("Draw a line along the scale bar first "
                        "(or click « Detect the scale bar »).",
                        "warn")
            return
        try:
            val = float(self.var_calib_val.get().replace(",", "."))
            if val <= 0:
                raise ValueError
        except ValueError:
            self.statut("Invalid real length: enter a positive "
                        "number.", "warn")
            self._calib_entry.focus_set()
            return
        unite = self.var_calib_unite.get()
        val_um = val / 1000.0 if unite == "nm" else val
        self.session.ppm = l_px / val_um
        self.session.calib_l_px = float(l_px)
        self.session.calib_source = getattr(self, "_calib_src", "manual")
        self.faites.add(1)
        self.stepper.maj(self.etape, self.faites)
        self._maj_lbl_calib()
        self._ax_calib.set_title(
            f"✓  {self.session.ppm:.3f} px/µm  ({l_px:.0f} px = {val:g} "
            f"{unite}), go on to the settings", fontsize=10,
            color=COULEUR["good"])
        self.canvas.draw_idle()
        journal("info", f"calibration : {l_px:.1f}px = {val}{unite} → "
                        f"{self.session.ppm:.4f} px/µm")
        self.statut("Calibration saved. Remember to save the recipe if "
                    "this magnification is a usual one.", "good")
