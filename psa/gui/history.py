# -*- coding: utf-8 -*-
"""PSA GUI, History, comparison, ImageJ ROIs."""
from psa.widgets import *  # noqa: F401,F403


class MixinHistory:
    """Methods of AppPSA: history, comparison, ImageJ ROIs."""

    # =========================================================================
    # HISTORIQUE / COMPARAISON  (contrôle qualité)
    # =========================================================================
    def ouvrir_comparaison(self):
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        top = tk.Toplevel(self.root)
        top.title(f"History & comparison, {APP_TITLE}")
        ajuster_a_l_ecran(top, 900, 680)
        top.configure(bg=T("BG"))
        haut = ttk.Frame(top, padding=8)
        haut.pack(fill="x")
        ttk.Label(haut, text="Recipe:").pack(side="left")
        var_rec = tk.StringVar(value=self.session.recette or "(all)")
        cb = ttk.Combobox(haut, textvariable=var_rec, state="readonly",
                          width=30,
                          values=["(all)"] + db_recettes())
        cb.pack(side="left", padx=6)
        lignes_ids = []
        lst = tk.Listbox(top, height=8, bg=T("SURFACE"), fg=T("TEXT"),
                         selectbackground=T("BLUE"), font=("Courier", 9),
                         highlightthickness=0)
        lst.pack(fill="x", padx=8)

        def recharger(_e=None):
            lst.delete(0, "end")
            lignes_ids.clear()
            rec = None if var_rec.get() == "(all)" else var_rec.get()
            for (i, dte, img, rcet, u, n, dn50, sc) in db_lister(rec):
                lignes_ids.append(i)
                lst.insert("end",
                           f"{dte[:16]}  {img[:24]:<24} n={n:<4} "
                           f"Dn50={dn50:7.2f} {u}  [{rcet[:18]}]")
            if not lignes_ids:
                lst.insert("end", "  (no archived analysis for this "
                                  "recipe)")

        cb.bind("<<ComboboxSelected>>", recharger)
        recharger()

        fig = Figure(figsize=(8.6, 4.4), facecolor=T("BG"))
        cnv = FigureCanvasTkAgg(fig, master=top)
        cnv.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=4)
        lbl = ttk.Label(top, wraplength=860, justify="left",
                        font=(FONT_UI, 9, "bold"))
        lbl.pack(anchor="w", padx=10)

        def comparer():
            res = self.session.resultats
            if res is None or not len(res["diametres"]):
                lbl.config(text="No current analysis to compare.",
                           foreground=COULEUR["warn"])
                return
            sel = lst.curselection()
            if not sel or sel[0] >= len(lignes_ids):
                lbl.config(text="Select a reference analysis "
                                "in the list.",
                           foreground=COULEUR["warn"])
                return
            d_ref, u_ref, img_ref, dte = db_diametres(lignes_ids[sel[0]])
            cmpres = comparer_distributions(d_ref, res["diametres"])
            fig.clf()
            ax = fig.add_subplot(111)
            ax.set_facecolor(T("SURFACE"))
            ax.hist(d_ref, bins=20, density=True, alpha=0.55,
                    color="#888888", label=f"Reference {img_ref} ({dte[:10]},"
                                           f" n={len(d_ref)})")
            ax.hist(res["diametres"], bins=20, density=True, alpha=0.60,
                    color=T("BLUE"),
                    label=f"Current analysis (n={len(res['diametres'])})")
            note = "" if u_ref == res["unite"] else \
                   f"  ⚠ different units ({u_ref} vs {res['unite']})!"
            ax.set_title("Comparison with the reference" + note, fontsize=10)
            ax.set_xlabel(f"Diameter ({res['unite']})")
            ax.set_ylabel("Density")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3, ls=":")
            fig.tight_layout()
            cnv.draw_idle()
            lbl.config(text=cmpres["message"],
                       foreground=COULEUR.get(cmpres["verdict"]))

        def tendance():
            rec = None if var_rec.get() == "(all)" else var_rec.get()
            if rec is None:
                lbl.config(text="Choose a recipe for the trend.",
                           foreground=COULEUR["warn"])
                return
            rows = db_tendance(rec)
            if len(rows) < 3:
                lbl.config(text=f"{len(rows)} archived analysis(es): at least"
                                " 3 are needed for a trend.",
                           foreground=COULEUR["warn"])
                return
            y = np.array([r[1] for r in rows], float)
            mu, sd = float(np.mean(y)), float(np.std(y, ddof=1))
            fig.clf()
            ax = fig.add_subplot(111)
            ax.set_facecolor(T("SURFACE"))
            x = np.arange(1, len(y) + 1)
            ax.plot(x, y, "o-", color=T("BLUE"), lw=1.6, ms=5)
            ax.axhline(mu, color=T("GREEN"), lw=1.4,
                       label=f"moyenne {mu:.2f}")
            for s_, ls in [(3, "--")]:
                ax.axhline(mu + s_ * sd, color=T("RED"), lw=1.1, ls=ls,
                           label=f"± {s_}σ")
                ax.axhline(mu - s_ * sd, color=T("RED"), lw=1.1, ls=ls)
            hors = int(((y > mu + 3 * sd) | (y < mu - 3 * sd)).sum())
            ax.set_title(f"Dn50 control chart, {rec} "
                         f"({len(y)} analyses, {hors} outside ±3σ)",
                         fontsize=10)
            ax.set_xlabel("Analysis (chronological order)")
            ax.set_ylabel("Dn50")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3, ls=":")
            fig.tight_layout()
            cnv.draw_idle()
            lbl.config(text=f"CV inter-analyses : {sd/mu*100:.1f} %"
                            if mu > 0 else "",
                       foreground=COULEUR["good" if hors == 0 else "warn"])

        bas = ttk.Frame(top, padding=8)
        bas.pack(fill="x")
        bouton(bas, "⚖  Compare with the current analysis", comparer,
               kind="primary").pack(side="left", padx=4)
        bouton(bas, "📈  Dn50 trend (SPC)",
               tendance).pack(side="left", padx=4)

        def enregistrer():
            if self.session.resultats is None:
                return
            self.session.operateur = self.var_oper.get()
            i = db_enregistrer(self.session)
            lbl.config(text=f"Current analysis saved in "
                            f"the history (no. {i}).",
                       foreground=COULEUR["good"])
            recharger()

        bouton(bas, "💾  Save the current analysis",
               enregistrer).pack(side="left", padx=4)
        bouton(bas, "Close", top.destroy).pack(side="right", padx=4)
        top.bind("<Escape>", lambda _e: top.destroy())

    # =========================================================================
    # ROIs IMAGEJ
    # =========================================================================
    def export_rois(self):
        res = self.session.resultats
        if res is None:
            return
        if not ROI_OK:
            messagebox.showinfo(
                "ROIs ImageJ",
                "Install the « roifile » package:\n\n"
                "    pip install roifile\n\nthen restart the application.",
                parent=self.root)
            return
        p = filedialog.asksaveasfilename(
            parent=self.root, defaultextension=".zip",
            filetypes=[("ROIs ImageJ", "*.zip")],
            initialfile=f"rois_{Path(self.session.chemin).stem}.zip")
        if not p:
            return
        try:
            n = exporter_rois_imagej(res, p)
            self.statut(f"{n} ROI(s) exported, readable in the "
                        f"ImageJ ROI Manager: {p}", "good")
        except Exception as e:
            messagebox.showerror("ROIs ImageJ", str(e), parent=self.root)

    def importer_rois_zones(self):
        if not ROI_OK:
            messagebox.showinfo(
                "ROIs ImageJ",
                "Install the « roifile » package (pip install roifile).",
                parent=self.root)
            return
        p = filedialog.askopenfilename(
            parent=self.root,
            filetypes=[("ROIs ImageJ", "*.zip *.roi"), ("All", "*.*")])
        if not p:
            return
        try:
            zones = importer_rois_imagej(p)
        except Exception as e:
            messagebox.showerror("ROIs ImageJ", str(e), parent=self.root)
            return
        for z in zones:
            self._zone_commit([tuple(pt) for pt in z.reshape(-1, 2)])
        self.statut(f"{len(zones)} ImageJ ROI(s) imported as exclusion "
                    "zones.", "good")
