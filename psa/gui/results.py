# -*- coding: utf-8 -*-
"""PSA GUI, Full analysis and step 5: results."""
from psa.widgets import *  # noqa: F401,F403


class MixinResults:
    """Methods of AppPSA: full analysis and step 5: results."""

    # =========================================================================
    # ANALYSE COMPLÈTE  (thread, pleine résolution)
    # =========================================================================
    def lancer_analyse(self):
        s = self.session
        self.annuler_canal("apercu")       # un aperçu tardif ne doit pas
        #                                    s'afficher sur les résultats
        s.operateur = self.var_oper.get()
        if hasattr(self, "var_sens"):
            s.params = appliquer_curseurs(
                s.params, s.mode, self.var_sens.get(),
                self.var_sep.get(), self.var_tmin.get(),
                actifs=self._curseurs_actifs())

        def _detection(progress):
            md = mode_de(s)
            if s.mode == "optique" and self._img_pretraitee is None:
                progress(2, "Background correction…")
                self._img_pretraitee = md.pretraiter(s.image_bgr, s.params)
            return md.detecter(s.image_bgr, s.ppm, dict(s.params), s.zones,
                               progress, img_pretraitee=self._img_pretraitee)

        def fini(res):
            s.resultats = res
            self.faites |= {0, 1, 3}
            if s.zones:
                self.faites.add(2)
            self.faites.add(4)
            if res["n_particules"] == 0:
                diag = diagnostiquer_echec(res, s)
                journal("warning", f"{s.nom_image} : n=0 — {diag}")
                self.aller_etape(4)
                self.statut("0 particle kept: " + diag, "bad")
                return
            journal("info", f"{s.nom_image} mode={s.mode} "
                            f"ppm={s.ppm:.4g} n={res['n_particules']} "
                            f"score={res['score_ws']:.0f}")
            objets = {"fibres": "fibre measurement(s)",
                      "gouttes": "droplet(s)"}.get(res["mode"], "particle(s)")
            msg = (f"{res['n_particules']} {objets} — score "
                   f"{res['score_ws']:.0f}/100")
            if self.var_autoarch.get():
                try:
                    dossier = archiver_analyse(s)
                    msg += f"  ·  archived: {dossier}"
                except Exception as e:
                    msg += f"  ·  archiving impossible ({e})"
            self.statut(msg, res["statut_ws"])
            self.aller_etape(4)

        def travail(progress):
            return filtrer_taille_max(_detection(progress), s.params)

        self.lancer_tache(travail, fini, "Full-resolution analysis…")

    # =========================================================================
    # ÉTAPE 4 — RÉSULTATS
    # =========================================================================
    def construire_etape4(self):
        self.vider_droit()
        res = self.session.resultats
        d = res["diametres"]
        u = res["unite"]
        st = stats_distribution(d)
        rep = evaluer_repr(d)
        wc = {"good": COULEUR["good"], "warn": COULEUR["warn"],
              "bad": COULEUR["bad"]}

        self.titre_droit("5 · Results")
        POP_NOM = {"primaires": "primaries", "agregats": "aggregates",
                   "agglomerats": "agglomerates"}
        nom_pop = ("fibre sites" if res["mode"] == "fibres" else
                   "droplets" if res["mode"] == "gouttes" else
                   POP_NOM.get(res.get("population"), "particles"))
        badge = badge_label(self.droit,
                            f"Score {res['score_ws']:.0f}/100 · "
                            f"{res['n_particules']} {nom_pop}",
                            res["statut_ws"])
        badge.pack(fill="x")
        infobulle(badge, res["msg_ws"])
        ttk.Label(self.droit, text=rep["message"],
                  foreground=wc.get(rep["verdict"]),
                  wraplength=self.LARGEUR_DROIT - 24,
                  font=(FONT_UI, 8, "italic")).pack(anchor="w",
                                                        pady=(3, 8))

        # Sélecteur de population (mode nano) : les statistiques portent sur
        # la population choisie. Pour un nanocarbone, les PRIMAIRES sont la
        # grandeur pertinente ; les régions ne donnent que les agrégats.
        if res.get("populations"):
            POP_LABEL = {"primaires": "Primary particles",
                         "agregats": "Aggregates (regions)",
                         "agglomerats": "Agglomerates"}
            fr_pop = ttk.Labelframe(self.droit,
                                    text=" Analysed population ", padding=6)
            fr_pop.pack(fill="x", pady=(0, 8))
            opts = [POP_LABEL[k] for k in ("primaires", "agregats",
                                           "agglomerats")
                    if len(res["populations"].get(k, []))]
            self.var_population = tk.StringVar(
                value=POP_LABEL.get(res.get("population")))
            cbp = ttk.Combobox(fr_pop, textvariable=self.var_population,
                               values=opts, state="readonly", width=20)
            cbp.pack(anchor="w")
            cbp.bind("<<ComboboxSelected>>",
                     lambda _e: self._changer_population())
            ttk.Label(fr_pop, wraplength=self.LARGEUR_DROIT - 44,
                      justify="left", foreground=COULEUR["dim"],
                      font=(FONT_UI, 8),
                      text="The statistics below refer to this "
                           "population. Primaries = elementary particles "
                           "(no segmentation, Hough transform); "
                           "Aggregates = watershed regions; Agglomerates = "
                           "connected masses.").pack(anchor="w", pady=(4, 0))
        if len(d):
            lignes = [
                ("Mean Ø", f"{st['norm'][0]:.2f} ± {st['norm'][1]:.2f} {u}"),
                ("Dn50 / Dv50", f"{st['dn50']:.2f} / {st['dv50']:.2f} {u}"),
                ("D[3,2] Sauter", f"{st['D32']:.2f} {u}"),
                ("D[4,3]", f"{st['D43']:.2f} {u}"),
                ("Span", f"{st['span']:.2f}"),
            ]
            if st["lognorm"][1] > 1.0:
                lignes.append(("σg (log-norm.)", f"{st['lognorm'][1]:.2f}"))
            if st.get("meilleur_fit"):
                lignes.append(("Best fit", NOMS_FIT.get(st["meilleur_fit"], st["meilleur_fit"])))
            uc = incertitude_calibration(self.session)
            lignes.append(("Calibration (k = 2)",
                           f"±{100 * uc['U_rel']:.2f} %" if uc["U_rel"] > 0
                           else "not evaluated"))
            grille = ttk.Frame(self.droit)
            grille.pack(fill="x")
            for i, (lb, vl) in enumerate(lignes):
                ttk.Label(grille, text=lb, foreground=COULEUR["dim"],
                          font=(FONT_UI, 8)).grid(row=i, column=0,
                                                      sticky="w")
                ttk.Label(grille, text=vl,
                          font=(FONT_UI, 9, "bold")
                          ).grid(row=i, column=1, sticky="e", padx=(8, 0))
            grille.columnconfigure(1, weight=1)

        # Spheroidization (powder modes)
        sph = sphericite(res, getattr(self.session, "sphero", None))
        if sph:
            frs = ttk.Labelframe(self.droit, text=" Spheroidization ", padding=6)
            frs.pack(fill="x", pady=(8, 0))
            gs_ = ttk.Frame(frs)
            gs_.pack(fill="x")
            for i, (lb, vl) in enumerate((
                    ("Rate, number", f"{sph['taux_nombre_hors_sat']:.1f} %"),
                    ("Rate, mass", f"{sph['taux_masse_hors_sat']:.1f} %"),
                    ("Satellites", f"{sph['satellites_nombre']:.1f} %"),
                    ("Bearing satellites", f"{sph['porteuses_pct']:.1f} %"))):
                ttk.Label(gs_, text=lb, foreground=COULEUR["dim"],
                          font=(FONT_UI, 8)).grid(row=i, column=0, sticky="w")
                ttk.Label(gs_, text=vl, font=(FONT_UI, 9, "bold")).grid(
                    row=i, column=1, sticky="e", padx=(8, 0))
            gs_.columnconfigure(1, weight=1)
            cr = sph["criteres"]
            fc = ttk.Frame(frs)
            fc.pack(fill="x", pady=(6, 0))
            fc.columnconfigure(1, weight=1)
            self._var_sph = {}
            for j, (cle, lab, tip) in enumerate((
                    ("rondeur_min", "Roundness 4A/(πF²) ≥", "Roundness 4A/(πF²max): 1 for "
                     "a disc, lower for elongated or angular particles."),
                    ("aspect_max", "Aspect ratio ≤", "Major / minor axis."),
                    ("satellite_ratio", "Satellite size ratio ≤", "Size ratio to the "
                     "touching particle below which a particle is a satellite."))):
                ttk.Label(fc, text=lab, font=(FONT_UI, 8)).grid(row=j, column=0,
                                                                sticky="w")
                v = tk.StringVar(value=f"{cr[cle]:g}")
                e = ttk.Entry(fc, textvariable=v, width=7)
                e.grid(row=j, column=1, sticky="e", pady=1)
                infobulle(e, tip)
                self._var_sph[cle] = v
            b = bouton(frs, "Apply criteria", self._appliquer_sphero)
            b.pack(fill="x", pady=(6, 0))
            if res.get("moteur") != "stardist":
                ttk.Label(frs, text="Classic engine: satellites are merged with "
                                    "their carrier, the rate is NOT reliable. "
                                    "Use the deep learning engine.",
                          foreground=PDF_C["red"], font=(FONT_UI, 8),
                          wraplength=self.LARGEUR_DROIT - 40,
                          justify="left").pack(anchor="w", pady=(4, 0))

        # Gouttes & capsules (microfluidique) : monodispersité, convention,
        # volume
        if res["mode"] == "gouttes":
            sg = res.get("gouttes_stats") or {}
            frg = ttk.Labelframe(self.droit, text=" Droplets & capsules ",
                                 padding=6)
            frg.pack(fill="x", pady=(8, 2))
            gg = ttk.Frame(frg)
            gg.pack(fill="x")
            if sg.get("n", 0) > 1:
                cv_ = sg["cv_pct"]
                classe = ("monodisperse" if cv_ < 3 else
                          "narrow" if cv_ < 5 else "polydisperse")
                h = res.get("hauteur_canal_um")
                vol = sg["v_moy"] * 1e-3
                lignes_g = [
                    ("CV (size)", f"{cv_:.2f} %  ·  {classe}"),
                    ("Measured at", "outer edge" if res["convention"] == "ext"
                     else "ring middle"),
                    ("Outer − middle", f"{abs(sg['ecart_convention']):.2f} {u}"
                     f" ({100 * abs(sg['ecart_convention']) / sg['d_moy']:.0f} %)"),
                    ("Mean volume", f"{vol:.4g} pL"),
                    ("Volume model", f"squeezed, h = {h:g} µm" if h else
                     "spheres"),
                    ("Cut by the edge", f"{res.get('n_bord', 0)} (excluded)")]
            else:
                lignes_g = [("Droplets", "fewer than 2 measured")]
            for i, (lb, vl) in enumerate(lignes_g):
                ttk.Label(gg, text=lb, foreground=COULEUR["dim"],
                          font=(FONT_UI, 8)).grid(row=i, column=0, sticky="w")
                ttk.Label(gg, text=vl, font=(FONT_UI, 9, "bold")).grid(
                    row=i, column=1, sticky="e", padx=(8, 0))
            gg.columnconfigure(1, weight=1)
            ttk.Label(frg, wraplength=self.LARGEUR_DROIT - 44, justify="left",
                      foreground=COULEUR["dim"], font=(FONT_UI, 8),
                      text="Monodisperse: CV < 3 %, narrow: < 5 %. The gap "
                           "between outer edge and ring middle is the "
                           "uncertainty of the diameter convention: check "
                           "it once against V = Q / f.").pack(anchor="w",
                                                              pady=(4, 0))

        # Métriques propres aux fibres électrospinnées
        if res["mode"] == "fibres" and res.get("fibres"):
            fr = res["fibres"]
            frf = ttk.Labelframe(self.droit,
                                 text=" Electrospun fibres ", padding=6)
            frf.pack(fill="x", pady=(4, 2))
            gf = ttk.Frame(frf)
            gf.pack(fill="x")
            lignes_f = [
                ("Median fibre Ø", fmt_diam_fibre(fr['diam_median_um'])),
                ("D10 – D90", (f"{fmt_diam_fibre(np.percentile(fr['diam_um'], 10))} – "
                               f"{fmt_diam_fibre(np.percentile(fr['diam_um'], 90))}")
                 if len(fr["diam_um"]) else "—"),
                ("Beads",
                 f"{fr['n_billes']} · {fr['densite_billes_mm2']:.0f}/mm²"),
                ("Alignment", f"{fr['alignement']:.2f}"),
                ("Orientation", f"{fr['orientation_deg']:.0f}°"),
                ("Mat porosity", f"{fr['porosite_pct']:.0f} %")]
            if fr.get("pores_mask") is not None and fr.get("n_pores", 0) > 0:
                lignes_f += [
                    ("Fibre porosity",
                     f"{fr.get('porosite_fibre_pct', 0):.0f} %"),
                    ("Pores",
                     f"{fr['n_pores']} · median Ø "
                     f"{fr.get('pore_d_median_um', 0)*1000:.0f} nm")]
            for i, (lb, vl) in enumerate(lignes_f):
                ttk.Label(gf, text=lb, foreground=COULEUR["dim"],
                          font=(FONT_UI, 8)).grid(row=i, column=0,
                                                      sticky="w")
                ttk.Label(gf, text=vl, font=(FONT_UI, 9, "bold")
                          ).grid(row=i, column=1, sticky="e", padx=(8, 0))
            gf.columnconfigure(1, weight=1)
            note = ("Diameter measured edge to edge, without threshold, by "
                    "pairing the two edges of each fibre (coloured "
                    "segments on the image); one site = one independent "
                    "measurement along the fibres. Alignment: 0 = random, "
                    "1 = aligned.")
            if fr.get("methode") == "masque":
                note = ("Diameter from the skeleton and distance map of the "
                        "mask (legacy method, threshold dependent).")
            if fr.get("pores_mask") is not None:
                note = ("Porous fibres: Ø measured on the filled ribbon; "
                        "pores highlighted in red. « Fibre porosity » = pore "
                        "area / ribbon area.")
            ttk.Label(frf, wraplength=self.LARGEUR_DROIT - 44,
                      justify="left", foreground=COULEUR["dim"],
                      font=(FONT_UI, 8), text=note).pack(
                anchor="w", pady=(4, 0))

        # Validation croisée BET — pertinente pour les matériaux carbonés
        # uniquement (le N₂ sonde les primaires) : masquée par défaut, et
        # sans objet pour les fibres.
        self.var_carbone = tk.BooleanVar(
            value=self.session.carbone or self.session.bet_m2g is not None)
        chk_c = ttk.Checkbutton(self.droit,
                                text="Carbon material: BET validation",
                                variable=self.var_carbone,
                                command=self._basculer_bet)
        if res["mode"] not in ("fibres", "gouttes"):
            chk_c.pack(anchor="w", pady=(8, 0))
        infobulle(chk_c, "The geometric SSA ↔ BET comparison only makes sense "
                         "when BET probes the primary particles "
                         "(nanocarbons). Leave unchecked for other "
                         "SEM uses (coatings, core-shell, oxides…).")
        fr_ssa = ttk.Labelframe(self.droit, text=" Validation BET ",
                                padding=6)
        self._fr_bet = fr_ssa
        if self.var_carbone.get():
            fr_ssa.pack(fill="x", pady=(2, 2))
        g = ttk.Frame(fr_ssa)
        g.pack(fill="x")
        ttk.Label(g, text="ρ (g/cm³) :",
                  font=(FONT_UI, 8)).grid(row=0, column=0, sticky="w")
        self.var_rho = tk.StringVar(value=f"{self.session.rho_gcm3:g}")
        en_rho = ttk.Entry(g, textvariable=self.var_rho, width=6)
        en_rho.grid(row=0, column=1, sticky="w", padx=4)
        self.lbl_ssa = ttk.Label(g, font=(FONT_UI, 9, "bold"))
        self.lbl_ssa.grid(row=1, column=0, columnspan=2, sticky="w",
                          pady=(0, 4))
        ttk.Label(g, text="BET (m²/g) :",
                  font=(FONT_UI, 8)).grid(row=2, column=0, sticky="w")
        self.var_bet = tk.StringVar(
            value="" if self.session.bet_m2g is None
            else f"{self.session.bet_m2g:g}")
        en_bet = ttk.Entry(g, textvariable=self.var_bet, width=6)
        en_bet.grid(row=2, column=1, sticky="w", padx=4, pady=2)
        self.lbl_dbet = ttk.Label(g, font=(FONT_UI, 9, "bold"))
        self.lbl_dbet.grid(row=3, column=0, columnspan=2, sticky="w")
        self.lbl_bet_msg = ttk.Label(fr_ssa,
                                     font=(FONT_UI, 8, "italic"),
                                     wraplength=self.LARGEUR_DROIT - 44,
                                     justify="left")
        self.lbl_bet_msg.pack(anchor="w", pady=(2, 0))
        infobulle(en_rho, "Material density (carbon black "
                          "≈ 1.8–2.1 g/cm³). Used for the geometric surface "
                          "SSA = 6/(ρ·D[3,2]).")
        infobulle(en_bet, "Measured BET specific surface area (N₂ adsorption). "
                          "N₂ reaches the surfaces between primaries: "
                          "BET probes the PRIMARY size, independently "
                          "of the image. The ratio diagnoses porosity, "
                          "sintering or segmentation errors.")
        for w_ in (en_rho, en_bet):
            w_.bind("<KeyRelease>", lambda _e: self._maj_ssa())
        self._maj_ssa()

        # Vues
        fr_v = ttk.Labelframe(self.droit, text=" View ", padding=6)
        fr_v.pack(fill="x", pady=8)
        self._fr_vue = fr_v
        for nom, cle in [("Annotated image", "image"),
                         ("Statistics", "stats"),
                         ("Distribution", "distribution"),
                         ("Pipeline", "pipeline")]:
            bouton(fr_v, nom,
                   lambda c=cle: self._changer_vue(c)).pack(fill="x", pady=2)

        # Options de distribution
        fr_d = ttk.Labelframe(self.droit, text=" Distribution options ",
                              padding=6)
        fr_d.pack(fill="x")
        ttk.Label(fr_d, text="Weighting:",
                  font=(FONT_UI, 8)).pack(anchor="w")
        self.var_pond = tk.StringVar(value="Number (q0)")
        cb = ttk.Combobox(fr_d, textvariable=self.var_pond,
                          values=["Number (q0)", "Volume (q3)"],
                          state="readonly", width=16)
        cb.pack(anchor="w", pady=(0, 4))
        cb.bind("<<ComboboxSelected>>", lambda _e: self._changer_vue(
            "distribution"))
        ttk.Label(fr_d, text="Size axis:", font=(FONT_UI, 8)).pack(anchor="w")
        g0 = reglages_graphe(self.session)
        self.var_xscale = tk.StringVar(
            value="Logarithmic" if g0["xscale"] == "log" else "Linear")
        cbx = ttk.Combobox(fr_d, textvariable=self.var_xscale,
                           values=["Logarithmic", "Linear"], state="readonly",
                           width=16)
        cbx.pack(anchor="w", pady=(0, 4))
        cbx.bind("<<ComboboxSelected>>", lambda _e: self._regler_graphe(
            xscale="log" if self.var_xscale.get() == "Logarithmic"
            else "linear"))
        infobulle(cbx, "Scale of the size axis, on screen and in the report.")
        self.var_fit_ln = tk.BooleanVar(value=bool(g0.get("lognormale", True)))
        self.var_fit_n = tk.BooleanVar(value=bool(g0.get("normale", False)))
        self.var_kde = tk.BooleanVar(value=bool(g0.get("courbe", True)))
        for txt, var, tip in [
                ("Log-normal fit", self.var_fit_ln,
                 "The physical default for powders (multiplicative growth / "
                 "fragmentation). σg in the legend."),
                ("Normal fit (Gaussian)", self.var_fit_n,
                 "To compare with the log-normal: if the Gaussian fits better "
                 "(AIC), the distribution is narrow and symmetric."),
                ("Smoothed curve (KDE, assumption-free)", self.var_kde,
                 "Kernel density estimate: reveals the bimodal "
                 "distributions that fits hide.")]:
            c = ttk.Checkbutton(fr_d, text=txt, variable=var,
                                command=lambda: self._regler_graphe(
                                    lognormale=self.var_fit_ln.get(),
                                    normale=self.var_fit_n.get(),
                                    courbe=self.var_kde.get()))
            c.pack(anchor="w")
            infobulle(c, tip)
        b_g = bouton(fr_d, "⚙  Graph settings…", self._dialogue_graphe)
        b_g.pack(fill="x", pady=(6, 0))
        infobulle(b_g, "Scales, axis limits, size classes, displayed "
                       "elements, titles. Also by double-clicking the graph.")

        # Exports
        bouton(self.droit, "📤  Report & export…", self.export_pdf,
               kind="primary").pack(fill="x", pady=(10, 2))
        bouton(self.droit, "📥  Export CSV…",
               self.export_csv).pack(fill="x", pady=2)
        bouton(self.droit, "🗂  Archive now",
               self.archiver_gui).pack(fill="x", pady=2)
        # L'édition n'est proposée que là où les entités cliquables
        # coïncident avec la population mesurée : optique (contours) et MEB
        # micronique (régions). En nano, les statistiques portent sur les
        # primaires (Hough) et non sur les régions cliquables — éditer
        # désalignerait les tableaux ; en fibres, il n'y a pas de particule
        # discrète à éditer.
        if res["mode"] in ("optique", "meb"):
            bouton(self.droit, "✏  Manual editing",
                   self.ouvrir_edition).pack(fill="x", pady=2)
        bouton(self.droit, "⚖  History / Compare…",
               self.ouvrir_comparaison).pack(fill="x", pady=2)
        bouton(self.droit, "🟡  ROIs ImageJ (.zip)…",
               self.export_rois).pack(fill="x", pady=2)
        bouton(self.droit, "📁  Analyse a batch of images…",
               self.ouvrir_lot).pack(fill="x", pady=2)
        bouton(self.droit, "↺  Adjust the settings",
               lambda: self.aller_etape(3)).pack(fill="x", pady=(8, 0))

        self._changer_vue(self._vue_resultats)

    def _maj_ssa(self):
        res = self.session.resultats
        if res is None or not len(res["diametres"]):
            for lbl in (self.lbl_ssa, self.lbl_dbet, self.lbl_bet_msg):
                lbl.config(text="")
            return
        st = stats_distribution(res["diametres"])
        u = res["unite"]
        try:
            rho = float(self.var_rho.get().replace(",", "."))
            if rho <= 0:
                raise ValueError
        except ValueError:
            self.lbl_ssa.config(text="ρ ?")
            self.lbl_dbet.config(text="")
            self.lbl_bet_msg.config(text="")
            return
        self.session.rho_gcm3 = rho
        d32_um = st["D32"] / 1000.0 if u == "nm" else st["D32"]
        self.lbl_ssa.config(text=f"Geometric SSA ≈ {6.0/(rho*d32_um):.1f} m²/g")
        txt = self.var_bet.get().strip().replace(",", ".")
        if not txt:
            self.session.bet_m2g = None
            self.lbl_dbet.config(text="")
            self.lbl_bet_msg.config(
                text="Enter your measured BET for the cross-validation "
                     "(N₂ probes the primaries).",
                foreground=COULEUR["dim"])
            return
        try:
            bet = float(txt)
            if bet <= 0:
                raise ValueError
        except ValueError:
            self.lbl_dbet.config(text="?")
            self.lbl_bet_msg.config(text="")
            return
        self.session.bet_m2g = bet
        cb = comparer_bet(st["D32"], u, rho, bet)
        wc = {"good": COULEUR["good"], "warn": COULEUR["warn"],
              "bad": COULEUR["bad"]}
        col = wc.get(cb["verdict"], COULEUR["dim"])
        self.lbl_dbet.config(
            text=f"d_BET = {cb['d_bet']:.1f} {u}   F = {cb['facteur']:.2f}",
            foreground=col)
        self.lbl_bet_msg.config(text=cb["message"], foreground=col)

    def _changer_population(self):
        """Bascule la population dont dérivent les statistiques (mode nano)."""
        res = self.session.resultats
        if not res or not res.get("populations"):
            return
        inv = {"Primary particles": "primaires",
               "Aggregates (regions)": "agregats",
               "Agglomerates": "agglomerats"}
        pop = inv.get(self.var_population.get())
        if pop is None or pop not in res["populations"]:
            return
        res["population"] = pop
        res["diametres"] = res["populations"][pop]
        res["n_particules"] = int(len(res["diametres"]))
        vue = getattr(self, "_vue_resultats", "stats")
        self.construire_etape4()      # recalcule tout le panneau statistique
        self._changer_vue(vue)        # rafraîchit la vue centrale

    # =========================================================================
    # SCIENTIFIC DISTRIBUTION VIEW + GRAPH SETTINGS DIALOG (double-click)
    # =========================================================================
    def _appliquer_sphero(self):
        try:
            c = {k: float(v.get().replace(",", ".")) for k, v in self._var_sph.items()}
        except ValueError:
            self.statut("Invalid spheroidization criterion.", "warn")
            return
        if not (0 < c["rondeur_min"] <= 1 and c["aspect_max"] >= 1 and
                0 < c["satellite_ratio"] < 1):
            self.statut("Criteria: 0 < roundness ≤ 1, aspect ≥ 1, "
                        "0 < satellite ratio < 1.", "warn")
            return
        self.session.sphero = c
        self.construire_etape4()
        self.statut("Spheroidization recomputed with the new criteria.", "good")

    def _dessiner_distribution_sci(self, res):
        """Same scientific plot as in the report, white 'paper' background,
        settings from the session (dialog / Distribution options)."""
        g = reglages_graphe(self.session)
        pond = "volume" if "olume" in self.var_pond.get() else "nombre"
        fib = res["mode"] == "fibres"
        with matplotlib.rc_context(STYLE_SCI):
            self.fig.set_facecolor("white")
            ax = self.fig.add_axes([0.10, 0.13, 0.78, 0.74])
            if len(res["diametres"]):
                tracer_psd(ax, res["diametres"], res["unite"], pond,
                           g["xscale"], "Fibre diameter" if fib
                           else "Equivalent diameter", g=g)
            self.fig.text(0.5, 0.015, "Double-click the graph to edit it "
                          "(scales, limits, classes, titles)", ha="center",
                          fontsize=7.5, color=GRIS, style="italic")

    def _regler_graphe(self, **kw):
        g = reglages_graphe(self.session)
        g.update(kw)
        self.session.graphe = g
        if self._outil == "export":
            self._apercu_rapport()
        else:
            self._changer_vue("distribution")

    def _dialogue_graphe(self):
        """Graph settings pop-up: weighting, scale, axis limits, classes,
        displayed elements, titles, font and line sizes."""
        if self.session.resultats is None:
            return
        g = reglages_graphe(self.session)
        top = tk.Toplevel(self.root)
        top.title("Graph settings")
        top.transient(self.root)
        top.resizable(False, False)
        # contenu défilant si l'écran est trop bas (portables 13 pouces)
        pan = PanneauDefilant(top, padding=12,
                              hauteur_max=self.root.winfo_screenheight() - 170)
        pan.pack(fill="both", expand=True)
        fr = pan.interieur
        V = {}

        def ligne(r, lab, widget):
            ttk.Label(fr, text=lab).grid(row=r, column=0, sticky="w", pady=3)
            widget.grid(row=r, column=1, sticky="ew", pady=3, padx=(10, 0))

        def champ(cle, val):
            V[cle] = tk.StringVar(value="" if val in (None, "") else
                                  (f"{val:g}" if isinstance(val, (int, float))
                                   and not isinstance(val, bool) else str(val)))
            return ttk.Entry(fr, textvariable=V[cle], width=22)
        r = 0
        V["pond"] = tk.StringVar(value=self.var_pond.get())
        ligne(r, "Weighting", ttk.Combobox(fr, textvariable=V["pond"], width=20,
                                           values=["Number (q0)", "Volume (q3)"],
                                           state="readonly")); r += 1
        V["xscale"] = tk.StringVar(value="Logarithmic" if g["xscale"] == "log"
                                   else "Linear")
        ligne(r, "Size axis", ttk.Combobox(fr, textvariable=V["xscale"], width=20,
                                           values=["Logarithmic", "Linear"],
                                           state="readonly")); r += 1
        ttk.Label(fr, text="Leave empty for automatic", foreground=COULEUR["dim"],
                  font=(FONT_UI, 8)).grid(row=r, column=1, sticky="w"); r += 1
        for cle, lab in (("xmin", "Size axis min"), ("xmax", "Size axis max"),
                         ("ymax", "Density axis max (%)"),
                         ("bins", "Number of size classes")):
            ligne(r, lab, champ(cle, g.get(cle))); r += 1
        ttk.Separator(fr).grid(row=r, column=0, columnspan=2, sticky="ew",
                               pady=6); r += 1
        for cle, lab in (("barres", "Bars (density per class)"),
                         ("courbe", "Smoothed curve"),
                         ("cumul", "Cumulative curve"),
                         ("reperes", "D10 / D50 / D90 markers"),
                         ("lognormale", "Log-normal fit"),
                         ("normale", "Normal fit")):
            V[cle] = tk.BooleanVar(value=bool(g.get(cle)))
            ttk.Checkbutton(fr, text=lab, variable=V[cle]).grid(
                row=r, column=0, columnspan=2, sticky="w"); r += 1
        ttk.Separator(fr).grid(row=r, column=0, columnspan=2, sticky="ew",
                               pady=6); r += 1
        for cle, lab in (("titre", "Title"), ("xlabel", "Size axis label"),
                         ("ylabel", "Density axis label")):
            ligne(r, lab, champ(cle, g.get(cle))); r += 1
        for cle, lab in (("police", "Font size (pt)"), ("trait", "Line width")):
            ligne(r, lab, champ(cle, g.get(cle))); r += 1
        V["rapport"] = tk.BooleanVar(value=bool(g.get("rapport", True)))
        ttk.Checkbutton(fr, text="Use these settings in the report",
                        variable=V["rapport"]).grid(row=r, column=0,
                                                    columnspan=2, sticky="w",
                                                    pady=(6, 0)); r += 1

        def lire():
            n = dict(g)
            n["xscale"] = "log" if V["xscale"].get() == "Logarithmic" else "linear"
            for cle in ("xmin", "xmax", "ymax", "bins", "police", "trait"):
                t = V[cle].get().strip().replace(",", ".")
                try:
                    n[cle] = (None if not t else
                              (int(float(t)) if cle == "bins" else float(t)))
                except ValueError:
                    messagebox.showerror("Graph settings",
                                         f"Invalid number: {t}", parent=top)
                    return None
            if n["xscale"] == "log" and n.get("xmin") is not None and n["xmin"] <= 0:
                messagebox.showerror("Graph settings", "A logarithmic axis "
                                     "needs a positive minimum.", parent=top)
                return None
            if n.get("bins") is not None and not 3 <= n["bins"] <= 200:
                messagebox.showerror("Graph settings", "Number of classes: "
                                     "between 3 and 200.", parent=top)
                return None
            n["police"] = n["police"] or GRAPHE_DEFAUT["police"]
            n["trait"] = n["trait"] or GRAPHE_DEFAUT["trait"]
            for cle in ("barres", "courbe", "cumul", "reperes", "lognormale",
                        "normale", "rapport"):
                n[cle] = bool(V[cle].get())
            for cle in ("titre", "xlabel", "ylabel"):
                n[cle] = V[cle].get().strip()
            return n

        def appliquer(fermer=False):
            n = lire()
            if n is None:
                return
            self.var_pond.set(V["pond"].get())
            self.var_xscale.set("Logarithmic" if n["xscale"] == "log" else "Linear")
            self.var_fit_ln.set(n["lognormale"])
            self.var_fit_n.set(n["normale"])
            self.var_kde.set(n["courbe"])
            self._regler_graphe(**n)
            if fermer:
                top.destroy()

        def defaut():
            self.session.graphe = dict(GRAPHE_DEFAUT)
            top.destroy()
            self._regler_graphe()
            self._dialogue_graphe()
        bas = ttk.Frame(fr)
        bas.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        bouton(bas, "Defaults", defaut).pack(side="left")
        bouton(bas, "Close", top.destroy).pack(side="right")
        bouton(bas, "OK", lambda: appliquer(True), kind="primary").pack(
            side="right", padx=4)
        bouton(bas, "Apply", appliquer).pack(side="right")
        top.bind("<Return>", lambda _e: appliquer(True))
        top.bind("<Escape>", lambda _e: top.destroy())
        self._dlg_graphe = top
        return top

    def _changer_vue(self, vue):
        self._vue_resultats = vue
        res = self.session.resultats
        if res is None:
            return
        self.fig.clf()
        self.fig.set_facecolor(T("BG"))
        if vue == "stats":
            self._dessiner_stats(res)
        elif vue == "distribution":
            self._dessiner_distribution_sci(res)
        elif vue == "pipeline":
            self._dessiner_pipeline(res)
        else:
            self._dessiner_image_annotee(res)
        self.canvas.draw_idle()

    def _dessiner_image_annotee(self, res):
        if res["mode"] == "gouttes":
            ax = self.fig.add_subplot(111)
            ax.imshow(cv2.cvtColor(res["image_base"], cv2.COLOR_BGR2RGB))
            sg = res.get("gouttes_stats") or {}
            ax.set_title(f"{res['n_particules']} droplets measured (green), "
                         f"{res.get('n_bord', 0)} cut by the edge (red)"
                         + (f" · CV {sg['cv_pct']:.1f} %" if sg.get("n", 0) > 1
                            else ""), fontsize=9)
            ax.axis("off")
            self.fig.tight_layout()
            return
        if res["mode"] == "fibres":
            ax = self.fig.add_subplot(111)
            ax.imshow(cv2.cvtColor(res["image_base"], cv2.COLOR_BGR2RGB))
            fr = res["fibres"]
            ax.set_title(
                f"Median fibre Ø {fmt_diam_fibre(fr['diam_median_um'])} · "
                f"edge-to-edge measurement segments (colour = Ø) · "
                f"{fr['n_billes']} beads (green) · "
                f"alignment {fr['alignement']:.2f}", fontsize=9)
            ax.axis("off")
            self.fig.tight_layout()
            return
        if res["mode"] == "meb":
            base = cv2.cvtColor(res["gray_pre"], cv2.COLOR_GRAY2RGB)
            base[find_boundaries(res["labels_ws"], mode="outer")] = \
                (255, 120, 30)
            ax = self.fig.add_subplot(111)
            ax.imshow(base)
            ax.set_title(f"{res['n_particules']} particles (orange "
                         "contours), " + ("StarDist (deep learning)"
                                          if res.get("moteur") == "stardist"
                                          else "Sauvola + h-maxima watershed"),
                         fontsize=9)
            ax.axis("off")
            self.fig.tight_layout()
            return
        if res["mode"] == "nano":
            nr = res["nano"]
            gray = nr["gray_pre"]
            base = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            base[find_boundaries(res["labels_ws"], mode="outer")] = \
                (255, 120, 30)
            deux = nr.get("methode") == "stardist_deux_etapes"
            for (x, y, r) in nr["cercles_hough"]:
                cv2.circle(base, (int(x), int(y)), max(1, int(r)),
                           (60, 200, 90) if deux else (226, 75, 74), 1)
            ax = self.fig.add_subplot(111)
            ax.imshow(base)
            if deux:
                titre = (f"{nr['imagerie']} image · {res['n_regions']} complete "
                         f"aggregates (orange) · {res['n_particules']} primary "
                         f"particles (green, StarDist nanocarbon model)")
                if nr["imagerie"] == "SE":
                    titre += " · SE: experimental"
            else:
                titre = (f"{res.get('n_regions', res['n_particules'])} regions "
                         f"(orange) · {len(nr['d_hough_nm'])} Hough primaries "
                         f"(red)")
            if nr["df"]:
                titre += f" · Df={nr['df']:.2f}"
            ax.set_title(titre, fontsize=9)
            ax.axis("off")
        else:
            ax = self.fig.add_subplot(111)
            ax.imshow(cv2.cvtColor(res["image_base"], cv2.COLOR_BGR2RGB))
            for i, (cx, cy) in enumerate(res["centres_px"][:250]):
                ax.text(cx, cy - 12, str(i + 1), fontsize=6.5,
                        color="#1A5AC7", ha="center", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="#1A5AC7", lw=0.4, alpha=0.85))
            ax.set_title(f"{res['n_particules']} particles: "
                         "blue contours · green crosses (centroid) · "
                         "orange chord (Feret)", fontsize=9)
            ax.axis("off")
        self.fig.tight_layout()

    def _dessiner_pipeline(self, res):
        if res["mode"] == "gouttes":
            g = res["gouttes"]
            axes = [self.fig.add_subplot(2, 2, i + 1) for i in range(4)]
            axes[0].imshow(res["gray_pre"], cmap="gray")
            axes[0].set_title("① Original", fontsize=8.5)
            axes[1].imshow(g["anneaux"], cmap="gray")
            axes[1].set_title("② Dark-ring map", fontsize=8.5)
            axes[2].imshow(cv2.cvtColor(res["image_base"], cv2.COLOR_BGR2RGB))
            axes[2].set_title("③ Fitted circles (sub-pixel)", fontsize=8.5)
            for a in axes[:3]:
                a.axis("off")
            ok = ~g["bord"]
            rel = 100 * g["rms"][ok] / np.maximum(g["d_milieu"][ok] / 2, 1e-6)
            if rel.size:
                axes[3].hist(rel, bins=25, color="#4C8FD6", edgecolor="white",
                             linewidth=0.3)
            axes[3].set_title("④ Ring-fit residual (% of radius)", fontsize=8.5)
            self.fig.tight_layout()
            return
        if res["mode"] == "fibres":
            fr = res["fibres"]
            etapes = [("Fibre mask", fr["masque"].astype(float)),
                      ("Skeleton (axes)", fr["squelette"].astype(float)),
                      ("Distance map (local radius)", fr["dist"])]
            for i, (titre, im) in enumerate(etapes, 1):
                ax = self.fig.add_subplot(2, 2, i)
                ax.imshow(im, cmap="magma" if i == 3 else "gray")
                ax.set_title(f"{i}. {titre}", fontsize=8)
                ax.axis("off")
            ax = self.fig.add_subplot(2, 2, 4)
            if len(res["diametres"]):
                ax.hist(res["diametres"] * 1000.0, bins=30,
                        color="#4C8FD6", edgecolor="white", linewidth=0.3)
            ax.set_title("4. Ø distribution (nm)", fontsize=8)
            ax.set_xlabel("Fibre Ø (nm)", fontsize=7)
            self.fig.tight_layout()
            return
        rng = np.random.default_rng(42)
        lws = res["labels_ws"]
        lrgb = np.zeros((*lws.shape, 3), np.uint8)
        for lbl in np.unique(lws):
            if lbl == 0:
                continue
            lrgb[lws == lbl] = rng.integers(60, 255, 3)
        if res["mode"] == "meb":
            data = [("① Preprocessed (databar, NLM, background)",
                     res["gray_pre"], "gray"),
                    ("② Sauvola mask + floor", res["masque"], "gray"),
                    ("③ Watershed h-maxima", lrgb, None),
                    ("④ Distance map", res["dist"], "plasma")]
            for i, (titre, img, cm) in enumerate(data):
                ax = self.fig.add_subplot(2, 2, i + 1)
                ax.imshow(img, cmap=cm)
                ax.set_title(titre, fontsize=8.5)
                ax.axis("off")
            self.fig.tight_layout()
            return
        if res["mode"] == "nano":
            nr = res["nano"]
            axes = [self.fig.add_subplot(2, 2, i + 1) for i in range(4)]
            axes[0].imshow(nr["gray_pre"], cmap="gray")
            axes[0].set_title("① Preprocessed (databar, NLM, background)",
                              fontsize=8.5)
            axes[1].imshow(res["masque"], cmap="gray")
            axes[1].set_title("② Sauvola mask + floor", fontsize=8.5)
            axes[2].imshow(lrgb)
            axes[2].set_title("③ Watershed h-maxima", fontsize=8.5)
            g = nr["granulo"]
            if g:
                axes[3].bar(g["d_nm"], g["ps"],
                            width=np.gradient(g["d_nm"]) * 0.85,
                            color="#1A5AC7", alpha=0.75)
                for q in ("dx10", "dx50", "dx90"):
                    axes[3].axvline(g[q], ls="--", lw=1.1, color="#14915B")
                axes[3].set_title(
                    f"④ Morphological granulometry "
                    f"Dx50={g['dx50']:.0f} nm (area weighted)", fontsize=8.5)
                axes[3].set_xlabel("Ø (nm)", fontsize=8)
            else:
                axes[3].set_title("④ Granulometry unavailable",
                                  fontsize=8.5)
            for ax in axes[:3]:
                ax.axis("off")
        else:
            panneaux = [
                ("① Original",
                 cv2.cvtColor(self.session.image_bgr, cv2.COLOR_BGR2RGB),
                 None),
                ("② CLAHE + γ",
                 cv2.cvtColor(res["image_amelioree"], cv2.COLOR_BGR2RGB),
                 None),
                ("③ Masque", res["masque"], "gray"),
                ("④ Watershed", lrgb, None),
            ]
            for i, (titre, img, cmap) in enumerate(panneaux):
                ax = self.fig.add_subplot(2, 2, i + 1)
                ax.imshow(img, cmap=cmap)
                ax.set_title(titre, fontsize=8.5)
                ax.axis("off")
        self.fig.tight_layout()

    # ── Exports GUI (avec garde-fous) ────────────────────────────────────────
    def _verifier_export(self):
        av = avertissements_qualite(self.session.resultats)
        if not av:
            return True
        return messagebox.askyesno(
            "Insufficient quality?",
            "\n\n".join("⚠ " + a for a in av) +
            "\n\nExport anyway? The warnings will be "
            "written in the report.",
            parent=self.root)

    def export_csv(self):
        self.ouvrir_export()

    def export_pdf(self):
        self.ouvrir_export()
