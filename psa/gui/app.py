# -*- coding: utf-8 -*-
"""PSA GUI, main window: assembles the screen mixins."""
from psa.widgets import *  # noqa: F401,F403
from psa.gui.image import MixinImage
from psa.gui.settings import MixinSettings
from psa.gui.results import MixinResults
from psa.gui.export import MixinExport
from psa.gui.measurements import MixinMeasurements
from psa.gui.editor import MixinHelpEditor
from psa.gui.history import MixinHistory


class AppPSA(MixinImage, MixinSettings, MixinResults, MixinExport, MixinMeasurements, MixinHelpEditor, MixinHistory):
    """Fenêtre unique : stepper | canvas matplotlib | panneau contextuel."""

    LARGEUR_DROIT = 290       # v1.7.2: 270 clipped "± … µm" in Results
    APERCU_MAX = 760          # max side of the preview image (Settings step)
    # SEM: at 760 px the h-maxima watershed merges particles of ~100 px
    # (66 found instead of 167 on the sphere test image); 1280 px gives a
    # faithful preview in ~1.5 s.
    APERCU_MAX_MODE = {"meb": 1280, "nano": 1280, "gouttes": 2000}
    DEBOUNCE_MS = 320

    def __init__(self, root):
        self.root = root
        root.title(APP_TITLE)
        icone = find_asset("psa.png")
        if icone:
            try:
                self._icone = tk.PhotoImage(master=root, file=icone)
                root.iconphoto(True, self._icone)
            except Exception as e:
                journal("warning", f"window icon not loaded: {e}")
        # Taille adaptée à l'écran (v1.7.2) : sur un portable 13 pouces
        # (1280 x 720 à 1366 x 768 utiles), 1500 x 940 débordait et le pied
        # de fenêtre (boutons Previous / Next) n'était plus visible.
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        self._compact = sw < 1450 or sh < 900
        w, h = min(1500, sw - 40), min(940, sh - 90)
        root.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 3)}")
        root.minsize(min(1000, sw - 40), min(560, sh - 90))
        if self._compact:
            try:
                root.state("zoomed")                    # Windows
            except tk.TclError:
                try:
                    root.attributes("-zoomed", True)    # Linux
                except tk.TclError:
                    pass

        self.session = Session()
        self.recettes, meta = charger_recettes()
        _THEME[0] = nom_theme_valide(meta.get("theme", THEME_DEFAUT))
        maj_couleurs()
        appliquer_theme_mpl()
        if TB:
            try:
                enregistrer_themes_mano(root.style)
                root.style.theme_use(_THEME[0])
            except Exception as e:
                journal("warning", f"Materia Nova themes not registered: {e}")
        else:
            appliquer_theme_tk(root)
        self._outil = None              # None | "mesure" | "export"
        self._logo_imgs = []
        self.etape = 0
        self.faites = set()
        self._mode_canvas = None        # "calib" | "zones" | None
        self._artists_tmp = []
        self._after_id = None
        self._apercu = None             # (img_bgr_reduit, échelle)
        self._img_pretraitee = None     # rolling-ball plein format (optique)
        self._vue_resultats = "image"

        # ── En-tête ─────────────────────────────────────────────────────────
        # ---- header (template: logos, title, actions, theme) ------------
        barre = ttk.Frame(root, padding=(10, 4) if self._compact else (12, 8))
        barre.pack(side="top", fill="x")
        boite = ttk.Frame(barre)
        boite.pack(side="left")
        self._placer_logos(boite)
        ttk.Label(barre, text=APP_NAME if self._compact else
                  f"{APP_NAME}  ·  SEM & optical granulometry",
                  font=(FONT_UI, 12 if self._compact else 15, "bold")
                  ).pack(side="left", padx=10 if self._compact else 18)
        droite = ttk.Frame(barre)
        droite.pack(side="right")
        bouton(droite, "📂  Open image", self.ouvrir_image,
               kind="primary").pack(side="left", padx=(14, 8))
        btn_aide = bouton(droite, "❓  Help", self.ouvrir_aide)
        btn_aide.pack(side="left")
        infobulle(btn_aide, "User guide and theoretical background (PDF), F1.")
        ttk.Label(droite, text="Theme").pack(side="left", padx=(14, 6))
        self.theme_cb = ttk.Combobox(droite, values=list(MANO_THEMES),
                                     width=11, state="readonly")
        self.theme_cb.set(_THEME[0])
        self.theme_cb.pack(side="left")
        self.theme_cb.bind("<<ComboboxSelected>>", self.changer_theme)
        ttk.Separator(root, orient="horizontal").pack(fill="x")

        # ── Pied : statut + progression + navigation ────────────────────────
        # empaqueté AVANT le corps : quand la fenêtre manque de hauteur, c'est
        # le canevas qui rétrécit, jamais les boutons de navigation
        bas = ttk.Frame(root, padding=(12, 7))
        bas.pack(side="bottom", fill="x")
        ttk.Separator(root, orient="horizontal").pack(side="bottom", fill="x")
        self.btn_prec = bouton(bas, "‹  Previous",
                               lambda: self.aller_etape(self.etape - 1))
        self.btn_prec.pack(side="left")
        self.btn_action = bouton(bas, "Next  ›", self.action_principale,
                                 kind="primary")
        self.btn_action.pack(side="right")
        self.progress = ttk.Progressbar(bas, length=180, mode="determinate")
        self.progress.pack(side="right", padx=12)
        self.lbl_statut = ttk.Label(bas, text="Choose a recipe, then "
                                              "load an image.",
                                    foreground=COULEUR["dim"])
        self.lbl_statut.pack(side="left", padx=14, fill="x", expand=True)

        # ---- body: grouped navigation | canvas | context panel ----------
        # les deux panneaux latéraux défilent verticalement (petits écrans)
        corps = ttk.Frame(root)
        corps.pack(fill="both", expand=True)
        self.panneau_gauche = PanneauDefilant(corps, padding=(10, 6))
        self.panneau_gauche.pack(side="left", fill="y")
        gauche = self.panneau_gauche.interieur
        ttk.Label(gauche, text="SESSION", font=(FONT_UI, 8, "bold"),
                  foreground=COULEUR["dim"]).pack(anchor="w", pady=(8, 4), padx=2)
        ttk.Label(gauche, text="Recipe", font=(FONT_UI, 8)).pack(anchor="w")
        self.var_recette = tk.StringVar()
        self.cb_recette = ttk.Combobox(gauche, textvariable=self.var_recette,
                                       values=list(self.recettes),
                                       state="readonly", width=26)
        self.cb_recette.pack(fill="x", pady=(0, 4))
        self.cb_recette.bind("<<ComboboxSelected>>",
                             lambda _e: self.appliquer_recette())
        if self.recettes:
            self.cb_recette.current(0)
        ttk.Label(gauche, text="Operator", font=(FONT_UI, 8)).pack(anchor="w")
        self.var_oper = tk.StringVar(value=meta.get("operateur", ""))
        ttk.Entry(gauche, textvariable=self.var_oper).pack(fill="x")
        ttk.Label(gauche, text="Sample", font=(FONT_UI, 8)).pack(anchor="w",
                                                                pady=(4, 0))
        self.var_echantillon = tk.StringVar()
        e_ech = ttk.Entry(gauche, textvariable=self.var_echantillon)
        e_ech.pack(fill="x")
        infobulle(e_ech, "Sample name: report title and file names "
                         "(<sample>_report.pdf …). Defaults to the image name.")
        self.var_echantillon.trace_add(
            "write", lambda *_a: setattr(self.session, "echantillon",
                                         self.var_echantillon.get()))
        self.stepper = NavBar(gauche, self)
        self.stepper.pack(fill="x")
        ttk.Label(gauche, text=f"{APP_VERSION}\nCtrl+1…5  steps\n"
                               "Ctrl+O  open image\nCtrl+E  export\nF1  help",
                  font=(FONT_UI, 8), foreground=COULEUR["dim"],
                  justify="left").pack(side="bottom", anchor="w", pady=6)
        ttk.Separator(corps, orient="vertical").pack(side="left", fill="y")

        self.panneau_droit = PanneauDefilant(corps, largeur=self.LARGEUR_DROIT,
                                             padding=12)
        self.panneau_droit.pack(side="right", fill="y")
        self.droit = self.panneau_droit.interieur
        ttk.Separator(corps, orient="vertical").pack(side="right", fill="y")

        centre = ttk.Frame(corps)
        centre.pack(side="left", fill="both", expand=True)
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg, NavigationToolbar2Tk)
        self.fig = Figure(figsize=(6, 4.5), facecolor=T("BG"))
        self.canvas = FigureCanvasTkAgg(self.fig, master=centre)
        # barre d'outils empaquetée avant le canevas : jamais rognée
        barre_outils = ttk.Frame(centre)
        barre_outils.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, barre_outils)
        self.toolbar.update()
        self.canvas.mpl_connect("button_press_event",   self._cv_press)
        self.canvas.mpl_connect("motion_notify_event",  self._cv_motion)
        self.canvas.mpl_connect("button_release_event", self._cv_release)


        root.bind("<Escape>", lambda _e: root.focus_set())
        root.bind("<Control-o>", lambda _e: self.ouvrir_image())
        root.bind("<F1>", lambda _e: self.ouvrir_aide())
        root.bind("<Control-e>", lambda _e: self.ouvrir_export())
        for _n in range(5):
            root.bind(f"<Control-Key-{_n + 1}>",
                      lambda _e, k=_n: self.aller_etape(k))
        root.protocol("WM_DELETE_WINDOW", self.fermer)

        self.appliquer_recette()
        self.aller_etape(0)

    # ── Utilitaires ──────────────────────────────────────────────────────────
    LOGOS = (("MANO_logo.png", "MANO", "materia nova"),
             ("UMONS_logo.png", "UMONS", "UMONS"),
             ("CHIPS_logo.png", "CHIPS", "CHIPS"))

    @staticmethod
    def _logo_plaque(src, height=40, pad=6, radius=7):
        """Trims the white margin of the artwork and lays it on a white
        rounded plate (template: clean on dark title bars)."""
        from PIL import Image, ImageChops, ImageDraw, ImageTk
        img = Image.open(src).convert("RGBA")
        rgb = img.convert("RGB")
        bbox = ImageChops.difference(
            rgb, Image.new("RGB", rgb.size, (255, 255, 255))).getbbox()
        if bbox:
            img = img.crop(bbox)
        w, h = img.size
        pw, ph = w + 2 * pad, h + 2 * pad
        plate = Image.new("RGBA", (pw, ph), (255, 255, 255, 255))
        plate.paste(img, (pad, pad), img)
        mask = Image.new("L", (pw, ph), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, pw - 1, ph - 1],
                                               radius=radius, fill=255)
        plate.putalpha(mask)
        plate = plate.resize((max(1, int(pw * height / ph)), height),
                             Image.LANCZOS)
        return ImageTk.PhotoImage(plate)

    def _placer_logos(self, parent):
        """Assets/*.png first, then the embedded copies, then a text label."""
        for fn, key, txt in self.LOGOS:
            path = find_asset(fn)
            photo = None
            if path or EMBEDDED_LOGOS.get(key):
                try:
                    src = path if path else _io.BytesIO(
                        base64.b64decode(EMBEDDED_LOGOS[key]))
                    photo = self._logo_plaque(
                        src, height=30 if self._compact else 40)
                except Exception:
                    try:
                        img = (tk.PhotoImage(master=self.root, file=path)
                               if path else tk.PhotoImage(
                                   master=self.root, data=EMBEDDED_LOGOS[key]))
                        f = max(1, round(img.height() / 40))
                        photo = img.subsample(f, f) if f > 1 else img
                    except Exception:
                        photo = None
            if photo is not None:
                self._logo_imgs.append(photo)
                ttk.Label(parent, image=photo).pack(side="left", padx=4)
            else:
                ttk.Label(parent, text=txt, font=(FONT_UI, 11, "bold"),
                          foreground=MN_LIGHT).pack(side="left", padx=6)

    def statut(self, msg, statut=None):
        self.lbl_statut.config(
            text=msg,
            foreground=COULEUR.get(statut, COULEUR["dim"]))

    def vider_droit(self):
        for w in self.droit.winfo_children():
            w.destroy()
        self.panneau_droit.haut()

    def titre_droit(self, texte):
        ttk.Label(self.droit, text=texte,
                  font=(FONT_UI, 10, "bold")).pack(anchor="w",
                                                       pady=(0, 8))

    def lancer_tache(self, fn, on_done, message="Calcul…", canal=None):
        """Exécute fn(progress) dans un thread ; on_done(résultat) au retour.
        progress(pct, msg) remonte vers la barre via une queue (thread-safe).

        v1.7.8 : canal (ex. "apercu") = tâches qui se remplacent. Une seule
        tourne à la fois par canal ; une demande arrivée pendant le calcul
        est mise en attente (seule la dernière est gardée) et le résultat
        devenu périmé n'est pas affiché. Sans cela, un aperçu ancien et lent
        pouvait écraser un aperçu plus récent, ou s'afficher sur les
        résultats. Les threads actifs sont suivis pour la fermeture."""
        if not hasattr(self, "_canaux"):
            self._canaux, self._threads = {}, set()
        if getattr(self, "_fermeture", False):
            return
        if canal is not None:
            c = self._canaux.setdefault(canal, {"gen": 0, "actif": False,
                                                "attente": None})
            c["gen"] += 1
            if c["actif"]:
                c["attente"] = (fn, on_done, message)
                return
            c["actif"] = True
            gen = c["gen"]
        q = queue.Queue()
        self.statut(message)
        self.progress["value"] = 0

        def progress(pct, msg=""):
            q.put(("prog", (pct, msg)))

        def run():
            try:
                q.put(("ok", fn(progress)))
            except Exception as e:          # remonté à l'UI, pas à la console
                import traceback
                journal("error", "task failed:\n"
                                 + traceback.format_exc())
                q.put(("err", e))

        th = threading.Thread(target=run, daemon=True)
        self._threads.add(th)
        th.start()

        def fin_canal():
            """Libère le canal ; True si le résultat est encore d'actualité."""
            if canal is None:
                return True
            c = self._canaux[canal]
            c["actif"] = False
            a_jour = (gen == c["gen"]) and c["attente"] is None
            if c["attente"] is not None and not self._fermeture_en_cours():
                fn2, done2, msg2 = c["attente"]
                c["attente"] = None
                c["gen"] -= 1                  # relancée avec le même numéro
                self.lancer_tache(fn2, done2, msg2, canal)
            return a_jour

        def poll():
            try:
                while True:
                    genre, val = q.get_nowait()
                    if genre == "prog":
                        pct, msg = val
                        self.progress["value"] = pct
                        if msg:
                            self.statut(msg)
                    elif genre == "ok":
                        self._threads.discard(th)
                        self.progress["value"] = 100
                        if fin_canal() and not self._fermeture_en_cours():
                            on_done(val)
                        return
                    else:
                        self._threads.discard(th)
                        fin_canal()
                        self.progress["value"] = 0
                        if self._fermeture_en_cours():
                            return
                        self.statut(f"Error: {val}", "bad")
                        messagebox.showerror("Error", str(val),
                                             parent=self.root)
                        return
            except queue.Empty:
                pass
            except tk.TclError:              # fenêtre détruite entre-temps
                return
            self.root.after(80, poll)

        poll()

    def annuler_canal(self, canal):
        """Rend périmée la tâche en cours sur ce canal (son résultat ne sera
        pas affiché) et oublie la demande en attente."""
        c = getattr(self, "_canaux", {}).get(canal)
        if c:
            c["gen"] += 1
            c["attente"] = None

    def _fermeture_en_cours(self):
        return getattr(self, "_fermeture", False)

    def taches_actives(self):
        return [t for t in getattr(self, "_threads", set()) if t.is_alive()]

    def dessiner_image(self, img_rgb, titre=""):
        self.fig.clf()
        self.fig.set_facecolor(T("BG"))
        ax = self.fig.add_subplot(111)
        ax.imshow(img_rgb)
        ax.set_title(titre, fontsize=10)
        ax.axis("off")
        self.fig.tight_layout()
        self.canvas.draw_idle()
        return ax

    # ── Recettes ─────────────────────────────────────────────────────────────
    def appliquer_recette(self):
        nom = self.var_recette.get()
        r = self.recettes.get(nom)
        if not r:
            return
        self.session.recette = nom
        self.session.mode = r.get("mode", "optique")
        self.session.params = mode_de(self.session.mode).params_defaut()
        self.session.params.update(r.get("params", {}))
        if r.get("ppm"):
            self.session.ppm = float(r["ppm"])
            self.session.calib_l_px = float(r.get("calib_l_px") or 0.0)
            self.session.calib_source = r.get("calib_source", "recipe")
            self.faites.add(1)
        self.session.u_instr_pct = float(r.get("u_instr_pct") or
                                         self.session.u_instr_pct or 0.0)
        if r.get("rho"):
            self.session.rho_gcm3 = float(r["rho"])
        self.session.carbone = bool(r.get("carbone", False))
        self.session.resultats = None
        self.faites.discard(4)
        if self.etape == 0:
            self.construire_etape0()
        self.statut(f"Recipe « {nom} » applied "
                    f"(mode {self.session.mode}).")

    def sauver_recette_courante(self):
        nom = tk.simpledialog.askstring(
            "Save the recipe",
            "Recipe name (e.g. « Nanocarbon X · SEM ×100k »):",
            parent=self.root)
        if not nom:
            return
        self.recettes[nom] = {
            "mode": self.session.mode,
            "ppm": self.session.ppm or None,
            "calib_l_px": self.session.calib_l_px or None,
            "calib_source": self.session.calib_source or None,
            "u_instr_pct": self.session.u_instr_pct or None,
            "rho": self.session.rho_gcm3,
            "carbone": self.session.carbone,
            "params": dict(self.session.params),
        }
        sauver_recettes(self.recettes, self.var_oper.get(), _THEME[0])
        self.cb_recette["values"] = list(self.recettes)
        self.var_recette.set(nom)
        self.session.recette = nom
        self.statut(f"Recipe « {nom} » saved "
                    f"(calibration and parameters included).", "good")

    # ── Navigation ──────────────────────────────────────────────────────────
    def aller_etape(self, i):
        i = max(0, min(4, i))
        if i >= 1 and self.session.image_bgr is None:
            self.statut("Load an image first.", "warn"); return
        if i >= 2 and self.session.ppm <= 0:
            self.statut("Calibrate the scale first (step 2).", "warn")
            i = 1
        if i == 4 and self.session.resultats is None:
            self.statut("Run the analysis first (Analyse button).", "warn")
            return
        if self._after_id:
            self.root.after_cancel(self._after_id)
            self._after_id = None
        self.etape = i
        self._mode_canvas = None
        self._outil = None
        self.stepper.maj(i, self.faites)
        self.btn_prec.config(state="normal" if i > 0 else "disabled")
        if TB:
            self.btn_action.config(text="▶  Analyse" if i == 3
                                   else "Next  ›")
        else:
            self.btn_action.config(text="Analyse" if i == 3 else "Next >")
        [self.construire_etape0, self.construire_etape1,
         self.construire_etape2, self.construire_etape3,
         self.construire_etape4][i]()

    def action_principale(self):
        if self.etape == 3:
            self.lancer_analyse()
        else:
            self.aller_etape(self.etape + 1)

    def fermer(self, _t0=None):
        """Fermeture : attend la fin des calculs en cours (au plus 60 s).
        Détruire Python pendant un calcul ONNX fait avorter le processus."""
        import time as _time
        if _t0 is None:
            self._fermeture = True
            for c in getattr(self, "_canaux", {}).values():
                c["attente"] = None
            sauver_recettes(self.recettes, self.var_oper.get(), _THEME[0])
            _t0 = _time.time()
        if self.taches_actives() and _time.time() - _t0 < 60:
            try:
                self.statut("Finishing the current computation before "
                            "closing…", "warn")
            except tk.TclError:
                pass
            self.root.after(150, lambda: self.fermer(_t0))
            return
        # annule les rappels encore programmés (sondages de tâches, aperçus)
        try:
            for a in self.root.tk.splitlist(self.root.tk.call("after", "info")):
                self.root.after_cancel(a)
        except tk.TclError:
            pass
        self.root.destroy()

    # ── Routeur d'événements canvas ──────────────────────────────────────────
    def _cv_press(self, e):
        # double-click on a distribution graph (results or report preview):
        # graph settings dialog
        if getattr(e, "dblclick", False) and self.session.resultats is not None \
                and (self._outil == "export" or (
                    self._outil is None and self.etape == 4 and
                    getattr(self, "_vue_resultats", None) == "distribution")):
            if not hasattr(self, "var_pond"):
                self.var_pond = tk.StringVar(value="Number (q0)")
                self.var_xscale = tk.StringVar(value="Logarithmic")
                self.var_fit_ln = tk.BooleanVar(value=False)
                self.var_fit_n = tk.BooleanVar(value=False)
                self.var_kde = tk.BooleanVar(value=True)
            self._dialogue_graphe()
            return
        if self._mode_canvas == "calib":
            self._calib_press(e)
        elif self._mode_canvas == "zones":
            self._zones_press(e)
        elif self._mode_canvas == "mesure":
            self._mesure_press(e)
        elif self._mode_canvas == "edition":
            self._ed_press(e)

    def _cv_motion(self, e):
        if self._mode_canvas == "calib":
            self._calib_motion(e)
        elif self._mode_canvas == "zones":
            self._zones_motion(e)
        elif self._mode_canvas == "mesure":
            self._mesure_motion(e)
        elif self._mode_canvas == "edition":
            self._ed_motion(e)

    def _cv_release(self, e):
        if self._mode_canvas == "calib":
            self._calib_release(e)
        elif self._mode_canvas == "zones":
            self._zones_release(e)
        elif self._mode_canvas == "mesure":
            self._mesure_release(e)
        elif self._mode_canvas == "edition":
            self._ed_release(e)


__all__ = [_n for _n in dir() if not _n.startswith("__")]
