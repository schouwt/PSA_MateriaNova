# -*- coding: utf-8 -*-
"""PSA, Shared GUI helpers and side navigation."""
from psa.report_latex import *  # noqa: F401,F403
from psa.batch import *  # noqa: F401,F403

# =============================================================================
# =============================================================================
#                                   GUI
# =============================================================================
# =============================================================================
COULEUR = {}


def maj_couleurs():
    COULEUR.update({"good": T("GREEN"), "warn": T("YELLOW"),
                    "bad": T("RED"), "accent": T("BLUE"),
                    "dim": T("TEXT_DIM")})


maj_couleurs()


def appliquer_theme_tk(root):
    """Thème sombre/clair pour ttk « clam » (utilisé sans ttkbootstrap)."""
    root.configure(bg=T("BG"))
    st = ttk.Style(root)
    try:
        st.theme_use("clam")
    except tk.TclError:
        pass
    b, s2 = T("BG"), T("SURFACE2")
    tx, bl = T("TEXT"), T("BLUE")
    st.configure(".", background=b, foreground=tx, fieldbackground=s2)
    st.configure("TFrame", background=b)
    st.configure("TLabel", background=b, foreground=tx)
    st.configure("TEntry", fieldbackground=s2, foreground=tx,
                 insertcolor=bl)
    st.configure("TButton", background=s2, foreground=tx, relief="flat",
                 padding=5)
    st.map("TButton", background=[("active", bl)],
           foreground=[("active", b if theme_sombre() else "#FFFFFF")])
    for w in ("TCheckbutton", "TRadiobutton"):
        st.configure(w, background=b, foreground=tx)
        st.map(w, background=[("active", b)])
    st.configure("TSeparator", background=s2)
    st.configure("TLabelframe", background=b, foreground=tx)
    st.configure("TLabelframe.Label", background=b, foreground=bl)
    st.configure("TCombobox", fieldbackground=s2, foreground=tx,
                 background=s2, arrowcolor=tx)
    st.map("TCombobox",
           fieldbackground=[("readonly", s2)],
           foreground=[("readonly", tx)],
           selectbackground=[("readonly", s2)],
           selectforeground=[("readonly", tx)])
    st.configure("Horizontal.TScale", background=b, troughcolor=s2)
    st.configure("Horizontal.TProgressbar", background=bl, troughcolor=s2)
    root.option_add("*TCombobox*Listbox.background", s2)
    root.option_add("*TCombobox*Listbox.foreground", tx)
    root.option_add("*TCombobox*Listbox.selectBackground", bl)


def bouton(parent, texte, cmd, kind="secondary", **kw):
    """Bouton ttkbootstrap si disponible, ttk sinon."""
    if TB:
        style = {"primary": "primary", "secondary": "info-outline",
                 "discret": "secondary-outline",
                 "danger": "danger-outline", "lien": "link"}.get(kind, kind)
        return tb.Button(parent, text=texte, command=cmd,
                         bootstyle=style, **kw)
    return ttk.Button(parent, text=texte, command=cmd, **kw)


def badge_label(parent, texte, statut):
    """Étiquette de statut pleine couleur (vert/jaune/rouge).

    Sous ttkbootstrap, on emploie les styles « inverse » du thème (couleur
    pleine + texte clair) : le rendu est cohérent et ne subit pas l'écrasement
    du fond des widgets tk classiques par le thème."""
    if TB:
        bs = {"good": "success", "warn": "warning",
              "bad": "danger"}.get(statut, "warning")
        return tb.Label(parent, text=texte, bootstyle=f"{bs}-inverse",
                        anchor="center", padding=6,
                        font=(FONT_UI, 10, "bold"))
    wc = {"good": COULEUR["good"], "warn": COULEUR["warn"],
          "bad": COULEUR["bad"]}
    return tk.Label(parent, text=texte, fg="white",
                    bg=wc.get(statut, COULEUR["warn"]),
                    font=(FONT_UI, 10, "bold"), pady=7)


def infobulle(widget, texte):
    """Tooltip minimal multiplateforme."""
    if TB:
        try:
            try:                        # ttkbootstrap récent
                from ttkbootstrap.widgets import ToolTip
            except ImportError:         # versions antérieures
                from ttkbootstrap.tooltip import ToolTip
            ToolTip(widget, text=texte, delay=400)
            return
        except Exception:
            pass
    tip = [None]

    def montrer(_e):
        if tip[0]:
            return
        t = tk.Toplevel(widget)
        t.wm_overrideredirect(True)
        t.wm_geometry(f"+{widget.winfo_rootx()+10}"
                      f"+{widget.winfo_rooty()+widget.winfo_height()+4}")
        tk.Label(t, text=texte, bg="#FFFBE6", fg="#333", relief="solid",
                 borderwidth=1, font=(FONT_UI, 8), justify="left",
                 wraplength=260, padx=6, pady=3).pack()
        tip[0] = t

    def cacher(_e):
        if tip[0]:
            tip[0].destroy()
            tip[0] = None

    widget.bind("<Enter>", montrer)
    widget.bind("<Leave>", cacher)


def fond_theme(widget):
    """Background colour of the current ttk theme (for tk.Canvas)."""
    try:
        return ttk.Style(widget).lookup("TFrame", "background") or T("BG")
    except Exception:
        return T("BG")


class PanneauDefilant(ttk.Frame):
    """Vertically scrollable panel (v1.7.2, small screens): the children go
    in `.interieur`. The scroll bar appears only when the content is taller
    than the panel, the mouse wheel scrolls while the pointer is over it.
    largeur: fixed width of the CONTENT (px); None = follows the content
    (dialogs). The bar has its own reserved strip, so the content width,
    and the wraplength of its labels, never change."""
    BARRE = 14

    def __init__(self, parent, largeur=None, padding=0, hauteur_max=None):
        super().__init__(parent)
        self._largeur = largeur
        self._hauteur_max = hauteur_max
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0,
                                background=fond_theme(self))
        self.barre = ttk.Scrollbar(self, orient="vertical",
                                   command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self._maj_barre)
        self.canvas.pack(side="left", fill="both", expand=True)
        self._bande = ttk.Frame(self, width=self.BARRE)   # place de la barre
        self._bande.pack(side="right", fill="y")
        self.interieur = ttk.Frame(self.canvas, padding=padding)
        self._fen = self.canvas.create_window(0, 0, window=self.interieur,
                                              anchor="nw")
        if largeur:
            self.canvas.configure(width=largeur)
        self.interieur.bind("<Configure>", self._sur_contenu)
        self.canvas.bind("<Configure>", self._sur_canvas)
        for w in (self.canvas, self.interieur):
            w.bind("<Enter>", self._activer_molette, add="+")
            w.bind("<Leave>", self._desactiver_molette, add="+")

    # -- géométrie ------------------------------------------------------------
    def _sur_contenu(self, _e=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        if not self._largeur:
            self.canvas.configure(width=self.interieur.winfo_reqwidth())
        if self._hauteur_max:
            self.canvas.configure(height=min(self.interieur.winfo_reqheight(),
                                             self._hauteur_max))

    def _sur_canvas(self, e):
        self.canvas.itemconfigure(self._fen, width=e.width)

    def _maj_barre(self, debut, fin):
        self.barre.set(debut, fin)
        tout = float(debut) <= 0.0 and float(fin) >= 1.0
        if tout and self.barre.winfo_ismapped():
            self.barre.pack_forget()
            self._bande.pack(side="right", fill="y")
        elif not tout and not self.barre.winfo_ismapped():
            self._bande.pack_forget()
            self.barre.pack(side="right", fill="y")

    def haut(self):
        """Back to the top (content rebuilt)."""
        self.canvas.yview_moveto(0)

    def maj_fond(self):
        self.canvas.configure(background=fond_theme(self))

    # -- molette ------------------------------------------------------------
    def _molette(self, e):
        if self.barre.winfo_ismapped():
            num = getattr(e, "num", 0)
            if num in (4, 5):                      # Linux
                n = -1 if num == 4 else 1
            else:                                  # Windows, macOS
                n = -1 if e.delta > 0 else 1
            self.canvas.yview_scroll(n * 3, "units")

    def _activer_molette(self, _e):
        self.canvas.bind_all("<MouseWheel>", self._molette)
        self.canvas.bind_all("<Button-4>", self._molette)
        self.canvas.bind_all("<Button-5>", self._molette)

    def _desactiver_molette(self, e):
        # le pointeur passe sur un enfant : on reste « dedans »
        x, y = self.winfo_pointerxy()
        w = self.winfo_containing(x, y)
        while w is not None:
            if w is self:
                return
            w = getattr(w, "master", None)
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")


def taille_ecran(root):
    """(width, height) of the screen in Tk units."""
    return root.winfo_screenwidth(), root.winfo_screenheight()


def ajuster_a_l_ecran(top, w=None, h=None, marge=(40, 90)):
    """Clamps a top-level window to the screen (13-inch laptops)."""
    top.update_idletasks()
    sw, sh = taille_ecran(top)
    w = min(w or top.winfo_reqwidth(), sw - marge[0])
    h = min(h or top.winfo_reqheight(), sh - marge[1])
    x = max(0, (sw - w) // 2)
    y = max(0, (sh - h) // 3)
    top.geometry(f"{int(w)}x{int(h)}+{x}+{y}")


class NavBar(ttk.Frame):
    """Grouped side navigation (template layout): workflow steps, tools and
    output. ✓ marks completed steps; the active entry is highlighted."""
    GROUPES = (("IMAGE", ((0, "Image"), (1, "Calibration"), (2, "Zones"))),
               ("ANALYSIS", ((3, "Settings"), (4, "Results"))),
               ("TOOLS", (("mesure", "Thickness measurements"),)),
               ("OUTPUT", (("export", "Report & export"),)))

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.boutons = {}
        kw = {"bootstyle": "secondary"} if TB else {}
        for groupe, items in self.GROUPES:
            ttk.Label(self, text=groupe, font=(FONT_UI, 8, "bold"),
                      foreground=COULEUR["dim"]).pack(anchor="w",
                                                      pady=(14, 4), padx=2)
            for cle, nom in items:
                if isinstance(cle, int):
                    cmd = (lambda k=cle: app.aller_etape(k))
                elif cle == "mesure":
                    cmd = app.ouvrir_mesures
                else:
                    cmd = app.ouvrir_export
                b = (tb if TB else ttk).Button(self, text=nom, width=24,
                                               command=cmd, **kw)
                b.pack(fill="x", pady=2)
                self.boutons[cle] = (b, nom)

    def maj(self, active, faites, outil=None):
        outil = outil if outil is not None else getattr(self.app, "_outil", None)
        for cle, (b, nom) in self.boutons.items():
            if isinstance(cle, int):
                actif = outil is None and cle == active
                fait = cle in faites and not actif
                txt = f"{'✓' if fait else '  '}  {cle + 1}.  {nom}"
            else:
                actif, txt = outil == cle, nom
            b.configure(text=txt)
            if TB:
                b.configure(bootstyle="primary" if actif else
                            ("success-outline" if (isinstance(cle, int) and
                                                   cle in faites) else
                             "secondary"))


__all__ = [_n for _n in dir() if not _n.startswith("__")]
