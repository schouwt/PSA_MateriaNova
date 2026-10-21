# -*- coding: utf-8 -*-
"""PSA GUI, Steps 3-4: exclusion zones, settings and preview."""
from psa.widgets import *  # noqa: F401,F403


class MixinSettings:
    """Methods of AppPSA: steps 3-4: exclusion zones, settings and preview."""

    # =========================================================================
    # ÉTAPE 2 — ZONES D'EXCLUSION
    # =========================================================================
    def construire_etape2(self):
        self.vider_droit()
        self.titre_droit("3 · Exclusion zones (optional)")
        ttk.Label(self.droit, wraplength=self.LARGEUR_DROIT - 24,
                  justify="left", foreground=COULEUR["dim"],
                  text="Mask the scale bar, captions or "
                       "artefacts so that they are not counted.\n"
                       "Polygon: left click = vertex, right click = close."
                  ).pack(anchor="w")
        fr = ttk.Labelframe(self.droit, text=" Tool ", padding=6)
        fr.pack(fill="x", pady=10)
        self.var_outil = tk.StringVar(value="rectangle")
        ttk.Radiobutton(fr, text="Rectangle (click and drag)",
                        variable=self.var_outil,
                        value="rectangle").pack(anchor="w")
        ttk.Radiobutton(fr, text="Polygon",
                        variable=self.var_outil,
                        value="polygone").pack(anchor="w")
        self.lbl_nzones = ttk.Label(self.droit)
        self.lbl_nzones.pack(anchor="w", pady=(0, 8))
        bouton(self.droit, "↶  Undo the last one",
               self._zone_undo).pack(fill="x", pady=2)
        bouton(self.droit, "🗑  Clear all",
               self._zones_clear, kind="danger").pack(fill="x", pady=2)
        bouton(self.droit, "⬇  Importer ROIs ImageJ…",
               self.importer_rois_zones).pack(fill="x", pady=2)
        ttk.Label(self.droit, foreground=COULEUR["dim"],
                  font=(FONT_UI, 8),
                  wraplength=self.LARGEUR_DROIT - 24, justify="left",
                  text="\nOptional step: click « Next » to "
                       "go straight to the settings."
                  ).pack(anchor="w", side="bottom")

        img = self.session.image_bgr
        self._ax_z = self.dessiner_image(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            "Exclusion zones: the red zones will be ignored")
        self._mode_canvas = "zones"
        self._poly, self._poly_lines = [], []
        self._rect0, self._rect_patch = None, None
        self._zone_artists = []
        for z in self.session.zones:
            self._dessiner_zone(z)
        self._maj_nzones()

    def _maj_nzones(self):
        n = len(self.session.zones)
        self.lbl_nzones.config(
            text=f"{n} zone{'s' if n != 1 else ''} defined.",
            foreground=COULEUR["good"] if n else COULEUR["dim"])

    def _dessiner_zone(self, pts):
        xs = list(pts[:, 0, 0]) + [pts[0, 0, 0]]
        ys = list(pts[:, 0, 1]) + [pts[0, 0, 1]]
        ln, = self._ax_z.plot(xs, ys, color="#E24B4A", lw=2)
        fill = self._ax_z.fill(xs, ys, color="#E24B4A", alpha=0.28)[0]
        self._zone_artists.append([ln, fill])
        self.canvas.draw_idle()

    def _zones_press(self, e):
        if e.inaxes != self._ax_z or e.xdata is None or self.toolbar.mode:
            return
        if self.var_outil.get() == "polygone":
            if e.button == 1:
                self._poly.append((e.xdata, e.ydata))
                if len(self._poly) > 1:
                    ln, = self._ax_z.plot(
                        [self._poly[-2][0], e.xdata],
                        [self._poly[-2][1], e.ydata],
                        color="#E24B4A", lw=1.5, ls="--")
                    self._poly_lines.append(ln)
                    self.canvas.draw_idle()
            elif e.button == 3:
                if len(self._poly) >= 3:
                    self._zone_commit(self._poly)
                self._poly_clear()
        elif e.button == 1:
            self._rect0 = (e.xdata, e.ydata)

    def _zones_motion(self, e):
        if self.var_outil.get() != "rectangle" or self._rect0 is None \
           or e.inaxes != self._ax_z or e.xdata is None:
            return
        if self._rect_patch:
            self._rect_patch.remove()
        x0, y0 = self._rect0
        import matplotlib.patches as mpatches
        self._rect_patch = self._ax_z.add_patch(mpatches.Rectangle(
            (min(x0, e.xdata), min(y0, e.ydata)),
            abs(e.xdata - x0), abs(e.ydata - y0),
            lw=1.5, edgecolor="#E24B4A", facecolor="#E24B4A", alpha=0.25))
        self.canvas.draw_idle()

    def _zones_release(self, e):
        if self.var_outil.get() != "rectangle" or self._rect0 is None:
            return
        x0, y0 = self._rect0
        self._rect0 = None
        if self._rect_patch:
            self._rect_patch.remove()
            self._rect_patch = None
        if e.inaxes == self._ax_z and e.xdata is not None \
           and abs(e.xdata - x0) > 5 and abs(e.ydata - y0) > 5:
            self._zone_commit([(x0, y0), (e.xdata, y0),
                               (e.xdata, e.ydata), (x0, e.ydata)])
        self.canvas.draw_idle()

    def _zone_commit(self, points):
        h, w = self.session.image_bgr.shape[:2]
        pts = np.array(points, np.int32).reshape(-1, 1, 2)
        pts[:, :, 0] = np.clip(pts[:, :, 0], 0, w - 1)
        pts[:, :, 1] = np.clip(pts[:, :, 1], 0, h - 1)
        self.session.zones.append(pts)
        self.session.resultats = None
        self.faites.discard(4)
        self._dessiner_zone(pts)
        self.faites.add(2)
        self.stepper.maj(self.etape, self.faites)
        self._maj_nzones()

    def _poly_clear(self):
        for ln in self._poly_lines:
            try:
                ln.remove()
            except Exception:
                pass
        self._poly_lines.clear()
        self._poly.clear()
        self.canvas.draw_idle()

    def _zone_undo(self):
        if self._poly:
            self._poly_clear()
            return
        if self.session.zones:
            self.session.zones.pop()
            for a in self._zone_artists.pop():
                try:
                    a.remove()
                except Exception:
                    pass
            self.canvas.draw_idle()
            self._maj_nzones()

    def _zones_clear(self):
        self._poly_clear()
        self.session.zones.clear()
        for arts in self._zone_artists:
            for a in arts:
                try:
                    a.remove()
                except Exception:
                    pass
        self._zone_artists.clear()
        self.canvas.draw_idle()
        self._maj_nzones()

    # =========================================================================
    # ÉTAPE 3 — RÉGLAGES  (curseurs métier + aperçu superposé)
    # =========================================================================
    CONVENTIONS_GOUTTES = {"Outer edge (capsules, isolated)": "ext",
                           "Ring middle (packed emulsions)": "milieu"}

    def _construire_etape3_gouttes(self):
        """Settings of the Droplets & capsules mode (v1.7.5)."""
        s = self.session
        L = self.LARGEUR_DROIT - 24
        sens, sep, tmin = curseurs_depuis_params(s.params, s.mode)
        self._curseurs_init = (sens, sep, tmin)
        fd = ttk.Labelframe(self.droit, text=" Droplet detection ", padding=8)
        fd.pack(fill="x", pady=(0, 6))
        ttk.Label(fd, wraplength=L - 16, justify="left",
                  foreground=COULEUR["dim"], font=(FONT_UI, 8),
                  text="Transmitted light: droplets or capsules seen as a DARK "
                       "RING whose inside is as bright as the liquid. "
                       "Filled-looking objects: use a particle recipe."
                  ).pack(anchor="w")
        ttk.Label(fd, text="Diameter measured at").pack(anchor="w", pady=(6, 0))
        inv = {v: k for k, v in self.CONVENTIONS_GOUTTES.items()}
        self.var_g_conv = tk.StringVar(
            value=inv.get(s.params.get("convention", "ext")))
        cb = ttk.Combobox(fd, textvariable=self.var_g_conv, state="readonly",
                          values=list(self.CONVENTIONS_GOUTTES))
        cb.pack(fill="x", pady=(2, 4))
        cb.bind("<<ComboboxSelected>>", lambda _e: self._changer_gouttes())
        infobulle(cb, "The dark ring is an optical effect, the interface lies "
                      "inside it. Outer edge: capsules, isolated droplets with "
                      "a thick rim. Ring middle: packed emulsions (neighbours "
                      "share one ring). Both diameters are kept, their gap "
                      "(often 10-15 %) is reported: check once against "
                      "V = Q / f (flow rate / droplet frequency).")
        fh = ttk.Frame(fd)
        fh.pack(fill="x", pady=(2, 0))
        ttk.Label(fh, text="Channel height (µm)").pack(side="left")
        h = float(s.params.get("hauteur_canal_um") or 0)
        self.var_g_h = tk.StringVar(value=f"{h:g}" if h > 0 else "")
        e = ttk.Entry(fh, textvariable=self.var_g_h, width=8)
        e.pack(side="right")
        e.bind("<Return>", lambda _e: self._changer_gouttes())
        e.bind("<FocusOut>", lambda _e: self._changer_gouttes())
        infobulle(e, "Optional. Droplets larger than the channel are squeezed "
                     "into discs: volume of a disc of that height with a "
                     "rounded rim. Empty = spheres.")
        self.var_sens = self._slider(
            fd, "Ring sensitivity", 0, 100, sens, lambda v: f"{v:.0f}",
            "Higher = fainter rings accepted (out-of-focus droplets, low "
            "contrast). Too high: noise and channel walls become rings.")
        self.var_sep = tk.DoubleVar(value=sep)
        self.var_tmin = tk.DoubleVar(value=tmin)
        self.var_g_hough = tk.BooleanVar(value=bool(s.params.get("hough", True)))
        ck = ttk.Checkbutton(fd, text="Recover broken rings (Hough)",
                             variable=self.var_g_hough,
                             command=self._changer_gouttes)
        ck.pack(anchor="w", pady=(6, 0))
        infobulle(ck, "Searches circles of the typical size where the ring is "
                      "interrupted (touching droplets, bubbles, dirt); each "
                      "candidate must look like the detected population.")
        self._construire_etape3_fin()

    def _changer_gouttes(self):
        p = self.session.params
        p["convention"] = self.CONVENTIONS_GOUTTES.get(self.var_g_conv.get(), "ext")
        txt = self.var_g_h.get().strip().replace(",", ".")
        try:
            h = float(txt) if txt else 0.0
        except ValueError:
            self.statut("Channel height: a number in µm, or empty.", "bad")
            return
        p["hauteur_canal_um"] = max(0.0, h)
        p["hough"] = bool(self.var_g_hough.get())
        self._planifier_apercu(immediat=True)

    def construire_etape3(self):
        self.vider_droit()
        self.titre_droit("4 · Settings")
        if self.session.mode == "gouttes":
            return self._construire_etape3_gouttes()
        u = self.session.unite()
        sens, sep, tmin = curseurs_depuis_params(self.session.params,
                                                 self.session.mode)
        self._curseurs_init = (sens, sep, tmin)
        if self.session.mode in ("meb", "optique", "nano"):
            fm = ttk.Labelframe(self.droit, text=" Segmentation engine ",
                                padding=8)
            fm.pack(fill="x", pady=(0, 6))
            dispo = stardist_disponible()
            self.var_moteur = tk.StringVar(
                value=MOTEURS_INV.get(moteur_effectif(self.session.params),
                                      list(MOTEURS)[0]))
            cbm = ttk.Combobox(fm, textvariable=self.var_moteur,
                               values=list(MOTEURS) if dispo else
                               [MOTEURS_INV["watershed"]], state="readonly")
            cbm.pack(fill="x")
            cbm.bind("<<ComboboxSelected>>", lambda _e: self._changer_moteur())
            infobulle(cbm, "Classic: threshold + h-maxima watershed (fast, no "
                           "extra dependency). Deep learning: StarDist "
                           "predicts every particle as a star-convex polygon; "
                           "much better on touching, textured or shaded "
                           "particles (spheres, compact powders). Sliders: "
                           "Sensitivity = object probability threshold, "
                           "Separation = allowed overlap between neighbours.")
            if dispo and self.session.mode == "meb":
                # taille typique (v1.7.3) : vide = estimation automatique
                ft = ttk.Frame(fm)
                ft.pack(fill="x", pady=(6, 0))
                ttk.Label(ft, text="Typical size (µm)").pack(side="left")
                dnm = self.session.params.get("stardist_diametre_nm")
                self.var_dtyp = tk.StringVar(
                    value="" if not dnm else f"{float(dnm) / 1000:.3g}")
                e_t = ttk.Entry(ft, textvariable=self.var_dtyp, width=8)
                e_t.pack(side="right")
                e_t.bind("<Return>", lambda _e: self._changer_dtyp())
                e_t.bind("<FocusOut>", lambda _e: self._changer_dtyp())
                infobulle(e_t,
                          "Typical particle diameter used by the deep-learning "
                          "engine to choose its working scale. Empty = "
                          "automatic estimate (shown below). Enter a value "
                          "when the preview finds no or very few particles "
                          "although they are visible: the estimate can fail "
                          "on unusual images (e.g. one large aggregate made "
                          "of small grains).")
                self.lbl_dtyp = ttk.Label(fm, text="", font=(FONT_UI, 8),
                                          foreground=COULEUR["dim"])
                self.lbl_dtyp.pack(anchor="w")
            if not dispo:
                ttk.Label(fm, text="Deep learning: pip install onnxruntime",
                          foreground=COULEUR["dim"],
                          font=(FONT_UI, 8)).pack(anchor="w", pady=(4, 0))
        fr = ttk.Labelframe(self.droit, text=" Basic settings ", padding=8)
        fr.pack(fill="x")
        self.var_sens = self._slider(
            fr, "Detection sensitivity", 0, 100, sens,
            lambda v: f"{v:.0f}",
            "Higher = detects more particles (more permissive threshold). "
            "Too high: the background noise becomes particles.")
        self.var_sep = self._slider(
            fr, "Particle separation", 0, 100, sep,
            lambda v: f"{v:.0f}",
            "Higher = separates touching particles more. "
            "Too high: cuts particles into pieces.")
        tmax = {"nano": 100.0, "meb": 300.0}.get(self.session.mode, 200.0)
        taille_off = bool(self.session.params.get("taille_min_off", False))
        if taille_off:                 # position mémorisée du curseur
            tmin = float(self.session.params.get("taille_min_memo", tmin) or 1.0)
            self._curseurs_init = (sens, sep, tmin)
        self.var_tmin = self._slider(
            fr, f"Minimum size ({u})", 1, tmax, tmin,
            lambda v: f"{v:.0f}",
            "Diameter below which objects are ignored "
            "(dust, noise).")
        if cle_taille_min(self.session.mode):
            self.var_tmin_on = tk.BooleanVar(value=not taille_off)
            chk_t = ttk.Checkbutton(
                fr, text="Apply the minimum size", variable=self.var_tmin_on,
                command=self._basculer_taille_min)
            chk_t.pack(anchor="w", pady=(2, 0))
            infobulle(chk_t,
                      "Untick to keep every object, however small: useful "
                      "for fine powders or a fine tail of the distribution. "
                      "Objects only a few pixels across are still removed "
                      "(pixel floor of the mode): their size cannot be "
                      "measured. Check the preview: dust and noise may then "
                      "appear as particles.")
            if taille_off:
                self.var_tmin._scale.state(["disabled"])
        if self.session.mode in ("optique", "meb", "nano"):
            mode_t = self.session.mode
            self.var_tmax = self._slider(
                fr, f"Maximum size ({u})", 0, 100,
                taille_max_vers_pos(self.session.params.get("taille_max", 0.0),
                                    mode_t),
                lambda v, m=mode_t: fmt_taille_max(pos_vers_taille_max(float(v), m)),
                "Diameter above which objects are ignored (agglomerates, "
                "debris, contamination). Logarithmic scale; right end = no "
                "limit. In the preview, red contours = above the maximum.")
            self.var_tmax.trace_add(
                "write", lambda *_a, m=mode_t: self.session.params.__setitem__(
                    "taille_max", pos_vers_taille_max(self.var_tmax.get(), m)))
        if self.session.mode == "meb":
            self.var_combler = tk.BooleanVar(
                value=bool(self.session.params.get("combler_convexe", False)))
            chk_c = ttk.Checkbutton(
                fr, text="Complete spherical particles",
                variable=self.var_combler, command=self._basculer_combler)
            chk_c.pack(anchor="w", pady=(8, 0))
            infobulle(chk_c,
                      "SPHERICAL powders: in SEM, the shaded bottom of each "
                      "sphere merges with the background and is missing from the mask. "
                      "This option closes that crescent with the convex hull "
                      "and restores the true diameter and circularity. "
                      "Avoid it for angular or flaky particles.")
        if self.session.mode == "fibres":
            self.var_pores = tk.BooleanVar(
                value=bool(self.session.params.get("combler_pores", False)))
            chk_p = ttk.Checkbutton(
                fr, text="Porous fibres (fill + measure pores)",
                variable=self.var_pores, command=self._basculer_pores)
            chk_p.pack(anchor="w", pady=(8, 0))
            infobulle(chk_p,
                      "POROUS fibres: closes the ribbon edges and fills "
                      "the pores to measure the diameter of the full ribbon "
                      "(otherwise the skeleton fragments in the pores → Ø "
                      "underestimated). Also quantifies the porosity (fraction, "
                      "number and size of the pores, highlighted in red).")

        self._construire_etape3_fin()

    def _construire_etape3_fin(self):
        """Display, quality badge, advanced mode, archiving, preview: common
        to every mode."""
        fr2 = ttk.Labelframe(self.droit, text=" Display ", padding=8)
        fr2.pack(fill="x", pady=8)
        self.var_opac = self._slider(fr2, "Overlay opacity",
                                     0, 100, 65, lambda v: f"{v:.0f}", None)
        self.var_voir_masque = tk.BooleanVar(value=True)
        ttk.Checkbutton(fr2, text="Tint the detected areas",
                        variable=self.var_voir_masque,
                        command=self._redessiner_apercu).pack(anchor="w")

        self.lbl_badge = tk.Label(self.droit, text="Quality: –",
                                  font=(FONT_UI, 10, "bold"),
                                  pady=8, padx=6)
        self.lbl_badge.pack(fill="x", pady=(4, 2))
        self.lbl_napercu = ttk.Label(self.droit, foreground=COULEUR["dim"])
        self.lbl_napercu.pack(anchor="w")

        bouton(self.droit, "⚙  Advanced mode…",
               self.mode_avance).pack(fill="x", pady=(10, 2))
        self.var_autoarch = getattr(self, "var_autoarch",
                                    tk.BooleanVar(value=True))
        ttk.Checkbutton(self.droit,
                        text="Archive automatically\n(report + CSV + parameters)",
                        variable=self.var_autoarch).pack(anchor="w",
                                                         pady=(8, 0))
        ttk.Label(self.droit, foreground=COULEUR["dim"],
                  font=(FONT_UI, 8),
                  wraplength=self.LARGEUR_DROIT - 24, justify="left",
                  text="\nThe preview is computed on a reduced image; "
                       "the final analysis (Analyse button) uses the "
                       "full resolution.").pack(anchor="w", side="bottom")

        self._derniere_seg = None
        self._preparer_apercu()

    def _slider(self, parent, label, vmin, vmax, init, fmt, tip):
        ttk.Label(parent, text=label).pack(anchor="w", pady=(6, 0))
        fr = ttk.Frame(parent)
        fr.pack(fill="x")
        var = tk.DoubleVar(value=float(init))
        lbl = ttk.Label(fr, width=5, text=fmt(float(init)),
                        font=("Courier", 9))

        def on_move(v):
            lbl.config(text=fmt(float(v)))
            self._planifier_apercu()

        # the value label follows the variable, also when set by code
        var.trace_add("write", lambda *_a: lbl.config(text=fmt(float(var.get()))))
        sc = ttk.Scale(fr, from_=vmin, to=vmax, variable=var,
                       command=on_move)
        sc.pack(side="left", fill="x", expand=True)
        var._scale = sc              # pour pouvoir griser le curseur
        lbl.pack(side="left", padx=4)
        if tip:
            infobulle(sc, tip)
        return var

    def _preparer_apercu(self):
        """Prépare l'image d'aperçu réduite (et le rolling-ball plein format
        en optique) dans un thread, puis lance le premier aperçu."""
        s = self.session

        def travail(progress):
            progress(20, "Preparing the preview…")
            if s.mode == "optique":
                if self._img_pretraitee is None:
                    img = rolling_ball(s.image_bgr, s.params)
                    if s.params.get("inverser_couleurs"):
                        img = cv2.bitwise_not(img)
                    self._img_pretraitee = img
                base = self._img_pretraitee
            else:
                base = s.image_bgr
            h, w = base.shape[:2]
            sc = min(1.0, self.APERCU_MAX_MODE.get(s.mode, self.APERCU_MAX)
                     / max(h, w))
            prev = cv2.resize(base, (int(w * sc), int(h * sc)),
                              interpolation=cv2.INTER_AREA) \
                   if sc < 1.0 else base.copy()
            return prev, sc

        def fini(res):
            self._apercu = res
            self.statut("Adjust the sliders: the preview updates "
                        "automatically.")
            self._planifier_apercu(immediat=True)

        self.lancer_tache(travail, fini, "Preparing the preview…")

    def _changer_dtyp(self):
        """Taille typique saisie (µm) ; vide = automatique."""
        txt = self.var_dtyp.get().strip().replace(",", ".")
        p = self.session.params
        ancien = p.get("stardist_diametre_nm")
        try:
            v = float(txt) if txt else None
        except ValueError:
            self.statut("Typical size: enter a number in µm, or leave empty.",
                        "bad")
            return
        if v is None or v <= 0:
            p.pop("stardist_diametre_nm", None)
            self.var_dtyp.set("")
        else:
            p["stardist_diametre_nm"] = v * 1000.0
        if p.get("stardist_diametre_nm") != ancien:
            self._planifier_apercu(immediat=True)

    def _changer_moteur(self):
        """Switches the segmentation engine and rebuilds the settings (the
        sliders take the meaning of the chosen engine)."""
        moteur = MOTEURS.get(self.var_moteur.get(), "watershed")
        self.session.params["moteur"] = moteur
        if moteur == "stardist":
            self.statut("Deep learning engine: the first preview loads the "
                        "model (a few seconds).")
        self.construire_etape3()
        self._planifier_apercu(immediat=True)

    def _basculer_taille_min(self):
        """Active/désactive le filtre de taille minimale ; la position du
        curseur est mémorisée pour être restaurée."""
        s = self.session
        cle = cle_taille_min(s.mode)
        if cle is None:
            return
        if self.var_tmin_on.get():
            s.params.pop("taille_min_off", None)
            s.params[cle] = float(self.var_tmin.get())
            self.var_tmin._scale.state(["!disabled"])
        else:
            s.params["taille_min_off"] = True
            s.params["taille_min_memo"] = float(self.var_tmin.get())
            s.params[cle] = 0.0
            self.var_tmin._scale.state(["disabled"])
        self._planifier_apercu(immediat=True)

    def _basculer_combler(self):
        """Active/désactive la complétion par coque convexe (poudres
        sphériques) et rafraîchit l'aperçu."""
        self.session.params["combler_convexe"] = bool(self.var_combler.get())
        self._planifier_apercu(immediat=True)

    def _basculer_pores(self):
        """Active/désactive le comblage + mesure des pores (fibres poreuses)
        et rafraîchit l'aperçu."""
        self.session.params["combler_pores"] = bool(self.var_pores.get())
        self._planifier_apercu(immediat=True)

    def _planifier_apercu(self, immediat=False):
        if self.etape != 3 or self._apercu is None:
            return
        if self._after_id:
            self.root.after_cancel(self._after_id)
        self._after_id = self.root.after(
            0 if immediat else self.DEBOUNCE_MS, self._lancer_apercu)

    def _curseurs_actifs(self):
        """Quels curseurs ont bougé depuis l'entrée dans l'étape ?"""
        if not hasattr(self, "_curseurs_init"):
            return (False, False, False)
        i = self._curseurs_init
        v = (self.var_sens.get(), self.var_sep.get(), self.var_tmin.get())
        return tuple(abs(v[k] - i[k]) > 0.5 for k in range(3))

    def _lancer_apercu(self):
        self._after_id = None
        s = self.session
        s.params = appliquer_curseurs(
            s.params, s.mode, self.var_sens.get(),
            self.var_sep.get(), self.var_tmin.get(),
            actifs=self._curseurs_actifs())
        s.resultats = None
        self.faites.discard(4)
        prev, sc = self._apercu
        p = dict(s.params)

        def travail(_progress):
            return self._segmenter_apercu(prev, sc, p)

        def fini(seg):
            if self.etape != 3:            # on a quitté l'étape entre-temps
                return
            self._derniere_seg = seg
            self._redessiner_apercu()
            wc = {"good": COULEUR["good"], "warn": COULEUR["warn"],
                  "bad": COULEUR["bad"]}
            self.lbl_badge.config(
                text=f"Quality: {seg['score']:.0f}/100",
                bg=wc.get(seg["st"], COULEUR["warn"]), fg="white")
            obj = "droplet(s)" if self.session.mode == "gouttes" else "particle(s)"
            self.lbl_napercu.config(text=f"~{seg['n']} {obj}, {seg['msg']}")
            self.statut("Preview up to date.")
            e = seg.get("echelle_sd")
            if e and getattr(self, "lbl_dtyp", None) is not None:
                try:
                    self.lbl_dtyp.config(
                        text=("imposed" if e[2] else "automatic estimate") +
                        f": {e[0]:.3g} µm")
                except tk.TclError:
                    pass
                # échelle de travail très réduite et presque rien trouvé :
                # l'estimation de taille a probablement échoué
                if not e[2] and e[1] <= 0.25 and seg["n"] < 10:
                    self.statut("Very few particles: the automatic size "
                                f"estimate ({e[0]:.3g} µm) may be wrong; enter "
                                "a typical size under the engine.", "warn")

        self.lancer_tache(travail, fini, "Preview…", canal="apercu")

    def _segmenter_apercu(self, prev_bgr, sc, p):
        """Segmentation rapide sur image réduite → masque + frontières.
        L'aperçu se passe de l'augmentation par miroirs de StarDist (TTA,
        +50 % de temps environ) ; l'analyse finale la garde."""
        p = dict(p, stardist_tta=False)
        s = self.session
        echelle_sd = None
        zones_sc = [(z * sc).astype(np.int32) for z in s.zones] \
                   if (s.zones and sc < 1.0) else s.zones
        if s.mode == "gouttes":
            r = detecter_gouttes_mode(prev_bgr, s.ppm * sc if s.ppm > 0 else 0,
                                      p, zones_sc)
            g = r["gouttes"]
            H_, W_ = prev_bgr.shape[:2]
            conv = r["convention"]
            dpx = g["d_milieu"] if conv == "milieu" else np.where(
                np.isfinite(g["d_ext"]), g["d_ext"], g["d_milieu"])
            bords = np.zeros((H_, W_), np.uint8)
            coupes = np.zeros((H_, W_), np.uint8)
            for x, y, dd, b in zip(g["cx"], g["cy"], dpx, g["bord"]):
                cv2.circle(coupes if b else bords,
                           (int(round(x)), int(round(y))), int(round(dd / 2)),
                           1, max(1, int(round(max(H_, W_) / 700))))
            st_g = r["gouttes_stats"]
            msg = (f"CV {st_g['cv_pct']:.1f} %" if st_g.get("n", 0) > 1
                   else "no droplet")
            return {"base": cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2RGB),
                    "masque": r["masque"] > 0, "bords": bords > 0,
                    "bords_grands": coupes > 0, "score": r["score_ws"],
                    "msg": msg, "st": r["statut_ws"], "n": r["n_particules"],
                    "zones": zones_sc, "titre": "Preview: green = measured "
                    "droplets, red = cut by the edge"}
        if s.mode == "fibres":
            from skimage.morphology import skeletonize as _sk
            gris = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
            y = fibres._databar_y(gris, p.get("couper_databar", True))
            m = fibres.segmenter_fibres(gris[:y], p)
            masque = np.zeros(gris.shape, bool)
            masque[:y] = m
            skel = np.zeros(gris.shape, bool)
            skel[:y] = _sk(m)
            masque = appliquer_zones(masque.astype(np.uint8) * 255,
                                     zones_sc) > 0
            base = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2RGB)
            cov = 100.0 * m.mean()
            score = max(0.0, min(100.0, 100.0 - abs(cov - 45.0) * 2.0))
            st = "good" if 25 < cov < 65 else "warn"
            return {"base": base, "masque": masque, "bords": skel,
                    "score": score, "msg": f"coverage {cov:.0f}%",
                    "st": st, "n": int(skel.sum()), "zones": zones_sc}
        if s.mode == "nano" and moteur_effectif(p) == "stardist":
            # Two-step nanocarbon pipeline on the preview image, exactly the
            # method of the final analysis: aggregates (orange) and primary
            # particles (circles)
            r = analyser_nanocarbone(prev_bgr, s.ppm * sc, dict(p), zones_sc)
            nr = r["nano"]
            g_ = nr["gray_pre"]
            H_, W_ = g_.shape
            base = np.zeros((H_, W_, 3), np.uint8)
            base[:] = cv2.cvtColor(g_, cv2.COLOR_GRAY2RGB)
            masque = (r["masque"][:H_, :W_] > 0).astype(np.uint8) * 255
            bords = find_boundaries(r["labels_ws"][:H_, :W_] if r["labels_ws"].any()
                                    else (masque > 0).astype(int), mode="outer")
            cercles = np.zeros((H_, W_), np.uint8)
            for (x, y, rr) in nr["cercles_hough"]:
                cv2.circle(cercles, (int(x), int(y)), max(1, int(rr)), 1, 1)
            bords = bords | (cercles > 0)
            return {"base": base, "masque": masque > 0, "bords": bords,
                    "bords_grands": None, "score": r["score_ws"],
                    "msg": f"{nr['imagerie']}: {r['n_particules']} primaries, "
                           f"deep learning",
                    "st": r["statut_ws"], "n": r["n_particules"],
                    "zones": zones_sc}
        if s.mode in ("nano", "meb"):
            p_n = params_vers_nano(p) if s.mode == "meb" else p
            gray, _ = nano.pretraiter_meb(prev_bgr, p_n)
            ppnm_eff = (s.ppm / 1000.0) * sc
            if moteur_effectif(p_n) == "stardist" and s.mode == "meb":
                brut = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
                seg = segmenter_stardist(gray, max(ppnm_eff, 1e-6), p_n,
                                         gray_brut=brut)
                inf_sd = seg.get("stardist") or {}
                d_um = inf_sd.get("diametre_typique_px", 0) / max(
                    ppnm_eff, 1e-9) / 1000.0
                echelle_sd = (d_um, inf_sd.get("echelle", 1.0),
                              bool(p_n.get("stardist_diametre_nm")))
            else:
                seg = nano.segmenter_hmax(gray, max(ppnm_eff, 1e-6), p_n)
            masque = appliquer_zones(seg["masque"], zones_sc)
            lws = seg["labels"]
            base = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            n_det = len(seg["d_regs_nm"])
        else:
            img_am = ameliorer(prev_bgr, p)
            gris = cv2.cvtColor(img_am, cv2.COLOR_BGR2GRAY)
            k = int(p["blur_ksize"]); k = k if k % 2 == 1 else k + 1
            flou = cv2.GaussianBlur(gris, (k, k), sigmaX=1.5)
            bin_ = seuiller(flou, p)
            t = max(3, int(min(gris.shape) * 0.012))
            t = t if t % 2 == 1 else t + 1
            noy = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (t, t))
            masque = cv2.morphologyEx(bin_, cv2.MORPH_CLOSE, noy,
                                      iterations=2)
            masque = cv2.morphologyEx(masque, cv2.MORPH_OPEN, noy,
                                      iterations=1)
            masque = appliquer_zones(masque, zones_sc)
            dist = cv2.distanceTransform(masque, cv2.DIST_L2, 5)
            dist_n = dist / (dist.max() + 1e-9)
            rmin = max(3, int(p["diametre_min_um"] * s.ppm * sc / 2))
            coords = peak_local_max(dist_n, min_distance=rmin,
                                    threshold_abs=p["watershed_dist_pct"],
                                    labels=masque.astype(bool))
            mm = np.zeros(dist_n.shape, bool)
            if len(coords):
                mm[tuple(coords.T)] = True
            ml, _ = ndi.label(mm)
            lws = watershed(-dist, ml, mask=masque.astype(bool))
            if moteur_effectif(p) == "stardist":
                lws = labels_stardist_optique(gris, p)
            base = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2RGB)
            n_det = max(0, len(np.unique(lws)) - 1)
        # Maximum size: regions above it are drawn in red and not counted
        bords_grands = None
        dmax = float(p.get("taille_max", 0.0) or 0.0)
        if dmax > 0 and s.mode in ("optique", "meb") and lws.max() > 0:
            aires = np.bincount(lws.ravel())[1:].astype(float)
            d_unite = 2.0 * np.sqrt(aires / np.pi) / max(s.ppm * sc, 1e-9)
            grands = np.nonzero((d_unite > dmax) & (aires > 0))[0] + 1
            if grands.size:
                mg = np.isin(lws, grands)
                bords_grands = find_boundaries(np.where(mg, lws, 0),
                                               mode="outer")
                n_det = max(0, n_det - int(grands.size))
        dist = cv2.distanceTransform(masque, cv2.DIST_L2, 5)
        dist_n = dist / (dist.max() + 1e-9)
        score, msg, st = evaluer_ws(masque, dist_n, 0.3)
        bords = find_boundaries(lws, mode="outer")
        return {"base": base, "masque": masque > 0, "bords": bords,
                "bords_grands": bords_grands,
                "score": score, "msg": msg, "st": st, "n": n_det,
                "zones": zones_sc, "echelle_sd": echelle_sd}

    def _redessiner_apercu(self):
        seg = self._derniere_seg
        if seg is None:
            return
        out = seg["base"].astype(np.float32)
        a = self.var_opac.get() / 100.0
        if self.var_voir_masque.get() and a > 0:
            m = seg["masque"]
            teinte = np.array([30, 200, 120], np.float32)
            out[m] = out[m] * (1 - 0.45 * a) + teinte * (0.45 * a)
        out[seg["bords"]] = np.array([255, 120, 30], np.float32)
        if seg.get("bords_grands") is not None:
            out[seg["bords_grands"]] = np.array([230, 50, 50], np.float32)
        if seg["zones"]:
            ov = out.copy()
            for z in seg["zones"]:
                cv2.fillPoly(ov, [z], (226, 75, 74))
            out = out * 0.75 + ov * 0.25
        if seg.get("titre"):
            self.dessiner_image(np.clip(out, 0, 255).astype(np.uint8),
                                seg["titre"])
            return
        self.dessiner_image(
            np.clip(out, 0, 255).astype(np.uint8),
            ("Preview (deep learning): orange = aggregate contours and "
             "primary particles, green = carbon" if (self.session.mode == "nano"
             and moteur_effectif(self.session.params) == "stardist") else
             "Preview: orange contours = separated particles, "
             "green = detected area") + (", red = above the maximum size"
                                       if seg.get("bords_grands") is not None
                                       else ""))

    # ── Mode avancé ──────────────────────────────────────────────────────────
    LIBELLES_AVANCE = {
        "convention": "Diameter at (ext/milieu)",
        "hauteur_canal_um": "Channel height (µm)",
        "k_anneau": "Ring threshold k",
        "hough": "Hough recovery",
        "taille_max": "Maximum size (0 = no limit)",
        "diametre_min_um": "Minimum diameter (µm)", "circularite_min":
        "Minimum circularity", "clahe_clip": "CLAHE clip",
        "clahe_grid": "CLAHE grid", "gamma": "Gamma",
        "watershed_dist_pct": "Watershed threshold [0-1]",
        "blur_ksize": "Gaussian blur (px)",
        "rolling_ball_radius": "Rolling ball radius (px)",
        "rolling_ball_enabled": "Rolling ball enabled",
        "inverser_couleurs": "Invert the colours",
        "threshold_mode": "Threshold mode",
        "threshold_manual": "Manual threshold (0-255)",
        "otsu_offset": "Otsu offset [-0.25..0.25]",
        "particules_claires": "Bright particles (SE)",
        "couper_databar": "Crop the databar",
        "denoise_nlm_h": "NLM denoising (strength)",
        "flatten": "Flatten the background",
        "sauvola_window": "Sauvola window (px)",
        "sauvola_k": "Sauvola k", "seuil_plancher_rel":
        "Global floor (× Otsu)", "h_rel": "h-maxima (rel. P90)",
        "fermeture_iter": "Morphological closing (iter.)",
        "d_min_nm": "Minimum diameter (nm)",
        "d_min_um": "Minimum diameter (µm)",
        "polarite": "Polarity (auto/bright/dark)",
        "aire_min_px": "Minimum area (px²)",
        "circ_min": "Minimum circularity",
        "circ_aire_min_px": "Circularity reliability area (px²)",
        "hough_d_min_nm": "Hough Ø min (nm)",
        "hough_d_max_nm": "Hough Ø max (nm)",
        "hough_sensibilite": "Hough sensitivity [0-1]",
        "granulo_points": "Granulometry: number of scales",
        "granulo_d_max_nm": "Granulo Ø max (nm)",
        "granulo_dim_max": "Granulo image max (px)",
    }

    def mode_avance(self):
        top = tk.Toplevel(self.root)
        top.title("Advanced mode: expert parameters")
        top.transient(self.root)
        top.grab_set()
        pan = PanneauDefilant(top, padding=14,
                              hauteur_max=self.root.winfo_screenheight() - 170)
        pan.pack(fill="both", expand=True)
        fr = pan.interieur
        entries = {}
        cles = sorted(self.session.params)
        for i, k in enumerate(cles):
            v = self.session.params[k]
            ttk.Label(fr, text=self.LIBELLES_AVANCE.get(k, k)
                      ).grid(row=i % 12, column=(i // 12) * 2,
                             sticky="w", padx=(0, 6), pady=3)
            col = (i // 12) * 2 + 1
            if isinstance(v, bool):
                var = tk.BooleanVar(value=v)
                ttk.Checkbutton(fr, variable=var).grid(row=i % 12,
                                                       column=col,
                                                       sticky="w",
                                                       padx=(0, 18))
            elif k == "threshold_mode":
                var = tk.StringVar(value=str(v))
                ttk.Combobox(fr, textvariable=var, values=["otsu", "manual"],
                             state="readonly", width=8
                             ).grid(row=i % 12, column=col,
                                    sticky="w", padx=(0, 18))
            else:
                var = tk.StringVar(value="" if v is None else str(v))
                ttk.Entry(fr, textvariable=var, width=9
                          ).grid(row=i % 12, column=col,
                                 sticky="w", padx=(0, 18))
            entries[k] = (var, v)

        def ok():
            p = dict(self.session.params)
            for k, (var, anc) in entries.items():
                try:
                    if isinstance(anc, bool):
                        p[k] = bool(var.get())
                    elif k == "threshold_mode":
                        p[k] = var.get()
                    elif anc is None or isinstance(anc, float):
                        txt = str(var.get()).strip().replace(",", ".")
                        p[k] = None if txt == "" else float(txt)
                    elif isinstance(anc, int):
                        p[k] = int(float(str(var.get()).replace(",", ".")))
                    else:
                        p[k] = var.get()
                except ValueError:
                    messagebox.showerror(
                        "Error", f"Invalid value for « {k} ».",
                        parent=top)
                    return
            self.session.params = p
            self._img_pretraitee = None      # rolling-ball à recalculer
            top.destroy()
            self.construire_etape3()

        bf = ttk.Frame(top, padding=(0, 0, 0, 12))
        bf.pack()
        bouton(bf, "Apply", ok, kind="primary").pack(side="left", padx=5)
        bouton(bf, "Cancel", top.destroy).pack(side="left", padx=5)
        self.root.wait_window(top)
