# -*- coding: utf-8 -*-
"""PSA GUI, Thickness measurements tool."""
from psa.widgets import *  # noqa: F401,F403


class MixinMeasurements:
    """Methods of AppPSA: thickness measurements tool."""

    # =========================================================================
    # OUTIL MESURES D'ÉPAISSEUR  (coatings PVD, core-shell, dimensions)
    # =========================================================================
    def ouvrir_mesures(self):
        self._outil = "mesure"
        self.stepper.maj(self.etape, self.faites, "mesure")
        if self.session.image_bgr is None or self.session.ppm <= 0:
            self.statut("Load an image and calibrate the scale before "
                        "measuring.", "warn")
            return
        self._mode_canvas = "mesure"
        self._mesure_p0 = None
        self._mesure_tmp = None
        self._gris_mesure = cv2.cvtColor(self.session.image_bgr,
                                         cv2.COLOR_BGR2GRAY)
        self.vider_droit()
        self.titre_droit("📏 · Thickness measurements")
        ttk.Label(self.droit, wraplength=self.LARGEUR_DROIT - 24,
                  justify="left", foreground=COULEUR["dim"],
                  text="Draw a line ACROSS the layer (coating, "
                       "shell…). Both interfaces are detected "
                       "on the grey-level profile; repeat along the deposit "
                       "to get the uniformity.").pack(anchor="w")
        self.var_auto_interf = tk.BooleanVar(value=True)
        chk = ttk.Checkbutton(self.droit,
                              text="Automatic interface detection",
                              variable=self.var_auto_interf)
        chk.pack(anchor="w", pady=(6, 4))
        infobulle(chk, "Checked: the thickness is measured between the two "
                       "strongest grey-level steps along the line "
                       "(independent of the click accuracy). Unchecked or "
                       "insufficient contrast: length of the drawn line.")
        sep = ttk.Separator(self.droit, orient="horizontal")
        sep.pack(fill="x", pady=(8, 6))
        ttk.Label(self.droit, wraplength=self.LARGEUR_DROIT - 24,
                  justify="left", font=(FONT_UI, 8, "bold"),
                  text="Automatic measurement ⚡").pack(anchor="w")
        ttk.Label(self.droit, wraplength=self.LARGEUR_DROIT - 24,
                  justify="left", foreground=COULEUR["dim"],
                  text="Draw ONE reference line across the "
                       "layer (with some margin on both sides), "
                       "then run the sweep: hundreds of parallel "
                       "profiles are measured along the layer.").pack(
            anchor="w", pady=(0, 4))
        bouton(self.droit, "⚡  Sweep along the layer",
               self._mesure_auto, kind="primary").pack(fill="x", pady=(0, 2))
        ttk.Label(self.droit, wraplength=self.LARGEUR_DROIT - 24,
                  justify="left", foreground=COULEUR["dim"], font=(FONT_UI, 8),
                  text="CURVED coating (bright ribbon following the grain): "
                       "isolates the band crossed by the line and follows it "
                       "with its skeleton.").pack(anchor="w", pady=(4, 0))
        frb = ttk.Frame(self.droit)
        frb.pack(fill="x", pady=(2, 0))
        ttk.Label(frb, text="Brightness", font=(FONT_UI, 8)).pack(side="left")
        self.var_ruban_pct = tk.IntVar(value=90)
        ttk.Scale(frb, from_=80, to=97, orient="horizontal",
                  variable=self.var_ruban_pct, length=120).pack(
            side="left", fill="x", expand=True, padx=4)
        self.lbl_ruban_pct = ttk.Label(frb, text="90", width=3,
                                       font=(FONT_UI, 8))
        self.lbl_ruban_pct.pack(side="left")
        self.var_ruban_pct.trace_add(
            "write", lambda *_: self.lbl_ruban_pct.config(
                text=str(self.var_ruban_pct.get())))
        bouton(self.droit, "⚡  Measure the bright ribbon",
               self._mesure_ruban, kind="primary").pack(fill="x", pady=(2, 2))
        self.lbl_auto_stats = ttk.Label(
            self.droit, wraplength=self.LARGEUR_DROIT - 24, justify="left",
            font=(FONT_UI, 9, "bold"), foreground=T("GREEN"))
        self.lbl_auto_stats.pack(anchor="w", pady=(2, 0))
        ttk.Separator(self.droit, orient="horizontal").pack(fill="x",
                                                            pady=(8, 6))
        self.lst_mesures = tk.Listbox(self.droit, height=9,
                                      bg=T("SURFACE"), fg=T("TEXT"),
                                      selectbackground=T("BLUE"),
                                      highlightthickness=0,
                                      font=("Courier", 9))
        self.lst_mesures.pack(fill="x", pady=6)
        self.lbl_mes_stats = ttk.Label(self.droit,
                                       wraplength=self.LARGEUR_DROIT - 24,
                                       justify="left",
                                       font=(FONT_UI, 9, "bold"))
        self.lbl_mes_stats.pack(anchor="w")
        bouton(self.droit, "↶  Undo the last one",
               self._mesure_undo).pack(fill="x", pady=(8, 2))
        bouton(self.droit, "🗑  Clear all",
               self._mesures_clear, kind="danger").pack(fill="x", pady=2)
        bouton(self.droit, "📥  Export CSV…",
               self._mesures_export, kind="primary").pack(fill="x", pady=2)
        bouton(self.droit, "✓  Finish the measurements",
               lambda: self.aller_etape(self.etape)).pack(fill="x",
                                                          pady=(10, 0))

        self.fig.clf()
        self.fig.set_facecolor(T("BG"))
        gs = self.fig.add_gridspec(5, 1, hspace=0.35)
        self._ax_mes = self.fig.add_subplot(gs[:4])
        self._ax_prof = self.fig.add_subplot(gs[4])
        self._ax_mes.imshow(cv2.cvtColor(self.session.image_bgr,
                                         cv2.COLOR_BGR2RGB))
        self._ax_mes.axis("off")
        self._ax_mes.set_title("Measurements: click and drag across "
                               "the layer", fontsize=9)
        self._ax_prof.set_title("Grey-level profile of the last line",
                                fontsize=8)
        self._ax_prof.tick_params(labelsize=7)
        self._mesure_artists = []
        self._auto_artists = []
        for i, m in enumerate(self.session.mesures):
            self._dessiner_mesure(m, i + 1)
        self._mesures_maj_liste()
        if self.session.mesures_auto:
            self._dessiner_auto()
            self._maj_auto_stats()
        else:
            self.lbl_auto_stats.config(text="")
        self.canvas.draw_idle()
        self.statut("Measuring tool active: lines accumulate, "
                    "CSV export on the right.")

    def _mesure_press(self, e):
        if e.inaxes != self._ax_mes or e.button != 1 or self.toolbar.mode:
            return
        self._mesure_p0 = (e.xdata, e.ydata)

    def _mesure_motion(self, e):
        if (self._mesure_p0 is None or e.inaxes != self._ax_mes
                or e.xdata is None):
            return
        if self._mesure_tmp:
            self._mesure_tmp.remove()
        self._mesure_tmp, = self._ax_mes.plot(
            [self._mesure_p0[0], e.xdata], [self._mesure_p0[1], e.ydata],
            color=T("YELLOW"), lw=1.8, ls="--")
        self.canvas.draw_idle()

    def _mesure_release(self, e):
        if self._mesure_p0 is None:
            return
        p0 = self._mesure_p0
        self._mesure_p0 = None
        if self._mesure_tmp:
            self._mesure_tmp.remove()
            self._mesure_tmp = None
        if (e.inaxes != self._ax_mes or e.xdata is None
                or math.hypot(e.xdata - p0[0], e.ydata - p0[1]) < 4):
            self.canvas.draw_idle()
            return
        self._mesure_commit(p0, (e.xdata, e.ydata))

    def _mesure_commit(self, p1, p2):
        s = self.session
        seg_px = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        prof, fa, fb, ep_px = profil_epaisseur(self._gris_mesure, p1, p2)
        auto = bool(self.var_auto_interf.get() and ep_px is not None)
        long_px = ep_px if auto else seg_px
        m = {"p1": (float(p1[0]), float(p1[1])),
             "p2": (float(p2[0]), float(p2[1])),
             "um": long_px / s.ppm, "auto": auto,
             "fa": fa if auto else None, "fb": fb if auto else None}
        s.mesures.append(m)
        self._dessiner_mesure(m, len(s.mesures))
        self._mesures_maj_liste()
        self._tracer_profil(prof, m, seg_px / s.ppm)
        self.canvas.draw_idle()

    def _dessiner_mesure(self, m, idx):
        p1, p2 = m["p1"], m["p2"]
        arts = []
        ln, = self._ax_mes.plot([p1[0], p2[0]], [p1[1], p2[1]],
                                color=T("YELLOW"), lw=1.6)
        arts.append(ln)
        if m["auto"] and m["fa"] is not None:
            for f in (m["fa"], m["fb"]):
                px = p1[0] + f * (p2[0] - p1[0])
                py = p1[1] + f * (p2[1] - p1[1])
                pt, = self._ax_mes.plot(px, py, "o", ms=4,
                                        color=T("GREEN"))
                arts.append(pt)
        tx = self._ax_mes.text(
            (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2 - 8,
            f"{idx}: {fmt_long(m['um'])}", fontsize=7.5,
            fontweight="bold", color=T("YELLOW"), ha="center",
            bbox=dict(boxstyle="round,pad=0.18", fc=T("BG"),
                      ec=T("YELLOW"), lw=0.5, alpha=0.85))
        arts.append(tx)
        self._mesure_artists.append(arts)

    def _tracer_profil(self, prof, m, seg_um):
        ax = self._ax_prof
        ax.clear()
        x = np.linspace(0, seg_um, len(prof))
        ax.plot(x, prof, color=T("BLUE"), lw=1.2)
        if m["auto"] and m["fa"] is not None:
            for f in (m["fa"], m["fb"]):
                ax.axvline(f * seg_um, color=T("GREEN"), ls="--", lw=1.2)
            ax.set_title(f"Grey-level profile, detected thickness: "
                         f"{fmt_long(m['um'])}", fontsize=8,
                         color=T("GREEN"))
        else:
            ax.set_title(f"Grey-level profile, drawn length: "
                         f"{fmt_long(m['um'])} (interfaces not detected)",
                         fontsize=8, color=T("YELLOW"))
        ax.set_xlabel("Position (µm)", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3, ls=":")

    def _mesures_maj_liste(self):
        self.lst_mesures.delete(0, "end")
        for i, m in enumerate(self.session.mesures):
            self.lst_mesures.insert(
                "end", f"{i+1:>2}  {fmt_long(m['um']):>11}  "
                       f"{'auto' if m['auto'] else 'line'}")
        arr = np.array([m["um"] for m in self.session.mesures])
        if len(arr) == 0:
            self.lbl_mes_stats.config(text="No measurement.")
        elif len(arr) == 1:
            self.lbl_mes_stats.config(text=f"1 measurement: {fmt_long(arr[0])}")
        else:
            mu, sd = float(np.mean(arr)), float(np.std(arr, ddof=1))
            cv = sd / mu * 100 if mu > 0 else 0
            self.lbl_mes_stats.config(
                text=(f"n={len(arr)}  ·  {fmt_long(mu)} ± {fmt_long(sd)}\n"
                      f"Uniformity: CV = {cv:.1f}%"))

    def _mesure_undo(self):
        if not self.session.mesures:
            return
        self.session.mesures.pop()
        for a in self._mesure_artists.pop():
            try:
                a.remove()
            except Exception:
                pass
        self._mesures_maj_liste()
        self.canvas.draw_idle()

    def _mesures_clear(self):
        self.session.mesures.clear()
        self.session.mesures_auto = {}
        for arts in self._mesure_artists:
            for a in arts:
                try:
                    a.remove()
                except Exception:
                    pass
        self._mesure_artists.clear()
        for art in getattr(self, "_auto_artists", []):
            try:
                art.remove()
            except Exception:
                pass
        self._auto_artists = []
        if hasattr(self, "lbl_auto_stats"):
            self.lbl_auto_stats.config(text="")
        self._ax_prof.clear()
        self._mesures_maj_liste()
        self.canvas.draw_idle()

    def _mesures_export(self, chemin=None):
        if not self.session.mesures and not self.session.mesures_auto:
            self.statut("No measurement to export.", "warn")
            return
        if chemin is None:
            chemin = filedialog.asksaveasfilename(
                parent=self.root, defaultextension=".csv",
                filetypes=[("CSV", "*.csv")],
                initialfile=f"mesures_{datetime.date.today()}.csv")
            if not chemin:
                return
        arr = np.array([m["um"] for m in self.session.mesures])
        with open(chemin, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow([APP_TITLE, "Thickness measurements",
                        f"Image : {self.session.nom_image}",
                        f"Operator: {self.var_oper.get() or '-'}",
                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M")])
            w.writerow(["#", "Thickness (µm)", "Thickness (nm)",
                        "Method", "x1", "y1", "x2", "y2"])
            for i, m in enumerate(self.session.mesures):
                w.writerow([i + 1, f"{m['um']:.4f}", f"{m['um']*1000:.1f}",
                            "auto interfaces" if m["auto"] else "line",
                            f"{m['p1'][0]:.0f}", f"{m['p1'][1]:.0f}",
                            f"{m['p2'][0]:.0f}", f"{m['p2'][1]:.0f}"])
            w.writerow([])
            w.writerow(["n", len(arr)])
            w.writerow(["Mean (µm)", f"{float(np.mean(arr)):.4f}"])
            if len(arr) > 1:
                sd = float(np.std(arr, ddof=1))
                w.writerow(["Standard deviation (µm)", f"{sd:.4f}"])
                w.writerow(["CV / uniformity (%)",
                            f"{sd/float(np.mean(arr))*100:.2f}"])
            w.writerow(["Calibration (px/µm)", f"{self.session.ppm:.4f}"])
            auto = self.session.mesures_auto
            if auto:
                st = auto["stats"]
                vals = np.asarray(auto["eps_um"], float)
                titre = ("Bright ribbon (curved coating), thickness "
                         "along the skeleton"
                         if auto.get("methode") == "ruban"
                         else "Automatic sweep along the layer")
                unite = ("points" if auto.get("methode") == "ruban"
                         else "profiles")
                w.writerow([])
                w.writerow([f"=== {titre} ==="])
                w.writerow([f"n {unite} retenus", st["n"],
                            f"n {unite} measured", st["n_total"]])
                w.writerow(["Mean (µm)", f"{st['moy']:.4f}",
                            "Standard deviation (µm)", f"{st['ec']:.4f}"])
                w.writerow(["CV / uniformity (%)", f"{st['cv']:.2f}"])
                w.writerow(["Median (µm)", f"{st['med']:.4f}",
                            "Min (µm)", f"{st['min']:.4f}",
                            "Max (µm)", f"{st['max']:.4f}"])
                w.writerow(["D10 (µm)", f"{st['p10']:.4f}",
                            "D90 (µm)", f"{st['p90']:.4f}"])
                w.writerow([])
                w.writerow(["# profile", "Thickness (µm)", "Thickness (nm)"])
                for i, v in enumerate(vals):
                    w.writerow([i + 1, f"{v:.4f}", f"{v*1000:.1f}"])
        self.statut(f"Measurements exported: {chemin}", "good")

    def _mesure_auto(self):
        """Balaye des centaines de profils parallèles le long de la couche,
        à partir du dernier trait de référence tracé en travers."""
        s = self.session
        if not s.mesures:
            self.statut("Draw ONE reference line across "
                        "the layer first, then run the sweep again.", "warn")
            return
        p1, p2 = s.mesures[-1]["p1"], s.mesures[-1]["p2"]
        self.statut("Sweeping profiles along the layer…")
        self.root.update_idletasks()
        eps_px, interfaces, taux = profils_epaisseur_auto(
            self._gris_mesure, p1, p2)
        st = stats_epaisseur(eps_px, s.ppm)
        if st is None or st["n"] < 5:
            self.statut("Sweep: too few usable profiles. "
                        "Check that the line crosses the layer, "
                        "with some margin on both sides.", "warn")
            return
        s.mesures_auto = {"stats": st, "interfaces": interfaces,
                          "ref": (p1, p2), "taux": float(taux),
                          "eps_um": st["valeurs"].tolist()}
        self._dessiner_auto()
        self._maj_auto_stats()
        rejet = st["n_total"] - st["n"]
        msg = f"Sweep finished: {st['n']} profiles measured"
        msg += f" ({rejet} outliers rejected)." if rejet else "."
        self.statut(msg, "good")

    def _mesure_ruban(self):
        """Mesure un coating COURBE : isole le ruban brillant que le dernier
        trait de référence croise, et l'épaisseur le long de son squelette."""
        s = self.session
        if not s.mesures:
            self.statut("Draw ONE line across the coating first "
                        "(the bright band), then run again.", "warn")
            return
        p1, p2 = s.mesures[-1]["p1"], s.mesures[-1]["p2"]
        self.statut("Extracting the ribbon + skeleton…")
        self.root.update_idletasks()
        pct = float(self.var_ruban_pct.get())
        eps_px, band, skel = epaisseur_ruban_guide(
            self._gris_mesure, p1, p2, s.ppm, pct=pct)
        st = stats_epaisseur(eps_px, s.ppm)
        if st is None or st["n"] < 10:
            self.statut("Ribbon: insufficient segmentation. Check that the "
                        "line crosses the bright band, or adjust the "
                        "brightness.", "warn")
            return
        ys, xs = np.where(skel)
        cnts, _ = cv2.findContours(band.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        s.mesures_auto = {
            "methode": "ruban", "stats": st,
            "eps_um": st["valeurs"].tolist(), "taux": 1.0, "ref": (p1, p2),
            "skel_xy": np.column_stack([xs, ys]).astype(int).tolist(),
            "band_cnt": [c.reshape(-1, 2).astype(int).tolist()
                         for c in cnts if len(c) > 2]}
        self._dessiner_auto()
        self._maj_auto_stats()
        self.statut(f"Ribbon measured: {st['n']} points along the coating.",
                    "good")

    def _maj_auto_stats(self):
        a = self.session.mesures_auto
        if not a:
            self.lbl_auto_stats.config(text="")
            return
        st = a["stats"]
        unite = "points" if a.get("methode") == "ruban" else "profiles"
        fiab = "" if a.get("taux", 1) >= 0.4 else f"  ⚠ few sharp {unite}"
        self.lbl_auto_stats.config(
            text=(f"n={st['n']} {unite}  ·  {fmt_long(st['moy'])} ± "
                  f"{fmt_long(st['ec'])}\n"
                  f"Uniformity: CV = {st['cv']:.1f}%{fiab}\n"
                  f"min {fmt_long(st['min'])} · max {fmt_long(st['max'])} · "
                  f"median {fmt_long(st['med'])}\n"
                  f"D10–D90: {fmt_long(st['p10'])} – {fmt_long(st['p90'])}"))

    def _dessiner_auto(self):
        """Trace l'enveloppe de la couche mesurée + l'histogramme."""
        a = self.session.mesures_auto
        if not a:
            return
        for art in getattr(self, "_auto_artists", []):
            try:
                art.remove()
            except Exception:
                pass
        self._auto_artists = []
        if a.get("methode") == "ruban":
            for cnt in a.get("band_cnt", []):
                arr = np.array(cnt, float)
                if len(arr) > 2:
                    ln, = self._ax_mes.plot(
                        np.append(arr[:, 0], arr[0, 0]),
                        np.append(arr[:, 1], arr[0, 1]),
                        color=T("GREEN"), lw=1.1, alpha=0.9)
                    self._auto_artists.append(ln)
            sk = np.array(a.get("skel_xy", []), float)
            if len(sk):
                pt, = self._ax_mes.plot(sk[:, 0], sk[:, 1], ".", ms=0.8,
                                        color=T("GREEN"), alpha=0.55)
                self._auto_artists.append(pt)
        else:
            inter = a.get("interfaces", [])
            if inter:
                hauts = np.array([p[0] for p in inter], float)
                bas = np.array([p[1] for p in inter], float)
                for pts in (hauts, bas):
                    ln, = self._ax_mes.plot(pts[:, 0], pts[:, 1],
                                            color=T("GREEN"), lw=1.0, alpha=0.9)
                    self._auto_artists.append(ln)
        ax = self._ax_prof
        ax.clear()
        vals = np.asarray(a["eps_um"], float)
        nb = max(10, int(1 + 3.322 * math.log10(len(vals)))) if len(vals) > 1 \
            else 10
        ax.hist(vals, bins=nb, color=T("BLUE"), edgecolor="white",
                linewidth=0.4, alpha=0.85)
        st = a["stats"]
        unite = "points" if a.get("methode") == "ruban" else "profiles"
        ax.axvline(st["moy"], color=T("GREEN"), lw=1.6)
        for off in (-st["ec"], st["ec"]):
            ax.axvline(st["moy"] + off, color=T("YELLOW"), ls="--", lw=1.0)
        ax.set_title(f"Thickness distribution, {st['n']} {unite} "
                     f"(CV {st['cv']:.1f}%)", fontsize=8, color=T("GREEN"))
        ax.set_xlabel("Thickness (µm)", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3, ls=":")
        self.canvas.draw_idle()

    def _basculer_bet(self):
        self.session.carbone = self.var_carbone.get()
        if self.session.carbone:
            self._fr_bet.pack(fill="x", pady=(2, 2), before=self._fr_vue)
            self._maj_ssa()
        else:
            self._fr_bet.pack_forget()
            self.session.bet_m2g = None
