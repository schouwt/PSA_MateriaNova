# -*- coding: utf-8 -*-
"""PSA, Materia Nova branding: logos, Mano themes, palettes."""
from psa.base import *  # noqa: F401,F403

# =============================================================================
# THEMES AND BRANDING  (Materia Nova template, shared with Grating Analyzer
# and SMU 2400 Suite: ManoDark / ManoBlue / ManoWhite, partner logos)
# =============================================================================
import base64
import io as _io

MN_DARK, MN_LIGHT, MN_GRAY = "#102C54", "#3680B7", "#505050"
FONT_UI = "Segoe UI"
FONT_MONO = "Consolas"

# Partner logos embedded as base64 PNG: fallback when Assets/*.png are absent, so
# a single-file executable still shows them.
# Keys: MANO, UMONS, CHIPS. The block between the markers is rewritten by embed_logos.py.
# >>> EMBEDDED_LOGOS
EMBEDDED_LOGOS = {
    "MANO":
        "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"
        "uE60AAAAAElFTkSuQmCC"
    ,
    "UMONS":
        "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"
        "SUVORK5CYII="
    ,
    "CHIPS":
        "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"
    ,
}
# <<< EMBEDDED_LOGOS


MANO_THEMES = {
    "ManoDark": dict(mode="dark", colors={
        "primary": "#3680B7", "secondary": "#4a5a6e", "success": "#2fa572",
        "info": "#3680B7", "warning": "#e0a83e", "danger": "#d9534f",
        "light": "#2a3a4d", "dark": "#0d1826", "bg": "#1b2838", "fg": "#e6ecf3",
        "selectbg": "#3680B7", "selectfg": "#ffffff", "border": "#34455a",
        "inputfg": "#e6ecf3", "inputbg": "#26374a", "active": "#2a3a4d"}),
    "ManoBlue": dict(mode="dark", colors={
        "primary": "#1fb6c1", "secondary": "#2c6470", "success": "#3ac29b",
        "info": "#29c3d0", "warning": "#e6b450", "danger": "#e06c6c",
        "light": "#164048", "dark": "#08181d", "bg": "#0f2a33", "fg": "#e3f2f4",
        "selectbg": "#1fb6c1", "selectfg": "#06222a", "border": "#245c66",
        "inputfg": "#e3f2f4", "inputbg": "#12333c", "active": "#164048"}),
    "ManoWhite": dict(mode="light", colors={
        "primary": "#3680B7", "secondary": "#6c7a8a", "success": "#2fa572",
        "info": "#3680B7", "warning": "#d6912a", "danger": "#d9534f",
        "light": "#e9eef4", "dark": "#102C54", "bg": "#f4f6f9", "fg": "#102C54",
        "selectbg": "#3680B7", "selectfg": "#ffffff", "border": "#c7d2df",
        "inputfg": "#102C54", "inputbg": "#ffffff", "active": "#e9eef4"}),
}


def _palette_from_mano(spec):
    c = spec["colors"]
    sombre = spec["mode"] == "dark"
    return {"BG": c["bg"], "SURFACE": c["inputbg"], "SURFACE2": c["light"],
            "TEXT": c["fg"], "TEXT_DIM": "#8a97a8" if sombre else "#6c7a8a",
            "BLUE": c["primary"], "GREEN": c["success"],
            "YELLOW": c["warning"], "RED": c["danger"], "GRID": c["border"]}


THEMES = {name: _palette_from_mano(spec) for name, spec in MANO_THEMES.items()}
THEME_DEFAUT = "ManoDark"
_THEME = [THEME_DEFAUT]

# Publication palette for exported reports (white background, as in the
# Grating Analyzer reports)
LIGHT_PALETTE = dict(bg="#ffffff", fg="#1a1a1a", grid="#d5dbe3",
                     panel="#ffffff", primary=MN_LIGHT, muted="#8a8a8a")
PLOT_COLORS = ["#3680B7", "#e0a83e", "#2fa572", "#d9534f", "#9b7fd4"]
# Report colours on a white background (Materia Nova)
NOMS_FIT = {"log-normale": "log-normal", "normale": "normal"}
NOMS_OUTILS_ED = {"suppr": "delete", "ajout": "add", "scind": "split",
                  "fusion": "merge"}
PDF_C = {"blue": MN_LIGHT, "navy": MN_DARK, "text": "#1a1a1a",
         "dim": "#8a8a8a", "green": "#2fa572", "yellow": "#d6912a",
         "red": "#d9534f"}


def theme_sombre():
    return MANO_THEMES.get(_THEME[0], MANO_THEMES[THEME_DEFAUT])["mode"] == "dark"


def nom_theme_valide(nom):
    """Old configurations stored "dark" / "light"."""
    return {"dark": "ManoDark", "light": "ManoWhite"}.get(nom, nom) \
        if nom in MANO_THEMES or nom in ("dark", "light") else THEME_DEFAUT


def enregistrer_themes_mano(style):
    """Registers ManoDark / ManoBlue / ManoWhite in ttkbootstrap (1.x and 2.x)."""
    try:
        from ttkbootstrap.style import ThemeDefinition
    except Exception:
        return
    noms = set(style.theme_names())
    for name, spec in MANO_THEMES.items():
        if name in noms:
            continue
        for kw in ("mode", "themetype"):        # 2.x uses mode, 1.x themetype
            try:
                style.register_theme(ThemeDefinition(
                    name=name, colors=spec["colors"], **{kw: spec["mode"]}))
                break
            except TypeError:
                continue
            except Exception:
                break


def apply_palette(fig, axes, pal):
    """Colours of figure, axes, grid and legends (same rules as SMU 2400 Suite).
    Image axes are flagged with ax._nogrid = True."""
    fig.set_facecolor(pal["bg"])
    if getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.set_color(pal["fg"])
    for ax in axes:
        try:
            ax.set_facecolor(pal["panel"])
            ax.tick_params(colors=pal["fg"], which="both")
            for sp in ax.spines.values():
                sp.set_color(pal["grid"])
            ax.xaxis.label.set_color(pal["fg"])
            ax.yaxis.label.set_color(pal["fg"])
            ax.title.set_color(pal["fg"])
            if ax.axison and not getattr(ax, "_nogrid", False):
                ax.grid(True, which="major", color=pal["grid"], lw=0.6, alpha=0.9)
            leg = ax.get_legend()
            if leg is not None:
                leg.get_frame().set_facecolor(pal["bg"])
                leg.get_frame().set_edgecolor(pal["grid"])
                for t in leg.get_texts():
                    t.set_color(pal["fg"])
        except Exception:
            pass



def find_asset(fn):
    """Assets/<fn> next to the script / executable, or in the PyInstaller bundle."""
    bases = [_DOSSIER_APP]
    if getattr(sys, "frozen", False):
        bases.insert(0, os.path.dirname(sys.executable))
        bases.append(getattr(sys, "_MEIPASS", ""))
    for b in bases:
        for p in (os.path.join(b, "Assets", fn), os.path.join(b, fn)):
            if b and os.path.exists(p):
                return p
    return None


def T(k):
    return THEMES[_THEME[0]][k]


def appliquer_theme_mpl():
    matplotlib.rcParams.update({
        "figure.facecolor":  T("BG"),
        "axes.facecolor":    T("SURFACE"),
        "axes.edgecolor":    T("SURFACE2"),
        "axes.labelcolor":   T("TEXT"),
        "axes.titlecolor":   T("TEXT"),
        "xtick.color":       T("TEXT_DIM"),
        "ytick.color":       T("TEXT_DIM"),
        "text.color":        T("TEXT"),
        "grid.color":        T("GRID"),
        "legend.facecolor":  T("SURFACE2"),
        "legend.edgecolor":  T("SURFACE2"),
        "legend.labelcolor": T("TEXT"),
    })


def fmt_long(um):
    """Formate une longueur stockée en µm (bascule en nm sous le micron)."""
    return f"{um*1000:.0f} nm" if um < 1.0 else f"{um:.2f} µm"


__all__ = [_n for _n in dir() if not _n.startswith("__")]
