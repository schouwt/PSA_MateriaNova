# -*- coding: utf-8 -*-
"""PSA, Session state and recipes."""
from psa.sphero import *  # noqa: F401,F403

# =============================================================================
# SESSION
# =============================================================================
@dataclass
class Session:
    chemin:     str = ""
    image_bgr:  "np.ndarray | None" = None
    mode:       str = "optique"            # "optique" | "nano"
    ppm:        float = 0.0                # px / µm
    zones:      list = field(default_factory=list)
    params:     dict = field(default_factory=lambda: dict(PARAMS_OPTIQUE))
    resultats:  "dict | None" = None
    operateur:  str = ""
    recette:    str = ""
    echantillon: str = ""                  # sample name (report, file names)
    calib_l_px: float = 0.0                # scale bar length used (px)
    calib_source: str = ""                 # manual | auto | metadata | recipe
    u_instr_pct: float = 0.0               # instrument magnification std (%)
    sphero:     dict = field(default_factory=dict)   # spheroidization criteria
    graphe:     dict = field(default_factory=lambda: {"xscale": "log"})
    rho_gcm3:   float = 1.9            # masse volumique (noir de carbone)
    bet_m2g:    "float | None" = None  # SSA BET mesurée (validation croisée)
    carbone:    bool = False           # active la validation BET (carbone)
    mesures:    list = field(default_factory=list)  # épaisseurs (outil 📏)
    mesures_auto: dict = field(default_factory=dict)  # balayage auto épaisseur

    @property
    def nom_echantillon(self):
        """Sample name, by default the image file name without extension."""
        return (self.echantillon or "").strip() or \
            (Path(self.chemin).stem if self.chemin else "sample")

    @property
    def nom_image(self):
        return Path(self.chemin).name if self.chemin else "—"

    def unite(self):
        return "nm" if self.mode == "nano" else "µm"

# =============================================================================
# RECETTES
# =============================================================================
RECETTES_DEFAUT = {
    "Powder · optical microscopy": {
        "mode": "optique", "ppm": None,
        "params": dict(PARAMS_OPTIQUE),
    },
}
if NANO_OK:
    RECETTES_DEFAUT["Nanocarbon · SEM"] = {
        "mode": "nano", "ppm": None, "carbone": True, "rho": 1.9,
        "params": dict(PARAMS_NANO),
    }
    RECETTES_DEFAUT["Metal powder · SEM"] = {
        "mode": "meb", "ppm": None,
        "params": dict(PARAMS_MEB),
    }
    RECETTES_DEFAUT["Spherical powder · SEM"] = {
        "mode": "meb", "ppm": None,
        "params": {**PARAMS_MEB, "combler_convexe": True},
    }
RECETTES_DEFAUT["Droplets & capsules · microfluidics"] = {
    "mode": "gouttes", "ppm": None, "params": dict(PARAMS_GOUTTES),
}
if FIB_OK:
    RECETTES_DEFAUT["Electrospun fibres · SEM"] = {
        "mode": "fibres", "ppm": None,
        "params": dict(fibres.FIBRES_PARAMS),
    }
    RECETTES_DEFAUT["Porous fibres · SEM"] = {
        "mode": "fibres", "ppm": None,
        "params": {**fibres.FIBRES_PARAMS, "combler_pores": True},
    }


# Default recipe names before the English version (v1.0): saved files are
# migrated so that the user's own settings are kept, without duplicates.
ANCIENS_NOMS_RECETTES = {
    "Poudre · microscopie optique": "Powder · optical microscopy",
    "Nanocarbone · MEB": "Nanocarbon · SEM",
    "Poudre métallique · MEB": "Metal powder · SEM",
    "Poudre sphérique · MEB": "Spherical powder · SEM",
    "Fibres électrospinnées · MEB": "Electrospun fibres · SEM",
    "Fibres poreuses · MEB": "Porous fibres · SEM",
}


def charger_recettes():
    rec = {k: json.loads(json.dumps(v)) for k, v in RECETTES_DEFAUT.items()}
    meta = {"operateur": "", "theme": THEME_DEFAUT}
    if FICHIER_RECETTES.exists():
        try:
            data = json.loads(FICHIER_RECETTES.read_text(encoding="utf-8"))
            for nom, r in data.get("recettes", {}).items():
                rec[ANCIENS_NOMS_RECETTES.get(nom, nom)] = r
            meta["operateur"] = data.get("operateur", "")
            meta["theme"] = nom_theme_valide(data.get("theme", THEME_DEFAUT))
        except (json.JSONDecodeError, OSError):
            pass
    return rec, meta


def _ecrire_atomique(chemin, texte, encoding="utf-8"):
    """Writes a file through a temporary file and os.replace: a crash during
    the write cannot corrupt the recipes."""
    tmp = Path(str(chemin) + ".tmp")
    tmp.write_text(texte, encoding=encoding)
    os.replace(tmp, chemin)


def sauver_recettes(recettes, operateur="", theme=None):
    DOSSIER_CONF.mkdir(parents=True, exist_ok=True)
    perso = {k: v for k, v in recettes.items() if k not in RECETTES_DEFAUT}
    _ecrire_atomique(FICHIER_RECETTES, 
        json.dumps({"recettes": perso, "operateur": operateur,
                    "theme": theme or _THEME[0]},
                   indent=2, ensure_ascii=False),
        encoding="utf-8")


# -----------------------------------------------------------------------------
# Calibration uncertainty (GUM): every length is proportional to 1 / ppm, so
# the relative uncertainty of the calibration applies unchanged to every size
# (diameters, D10/D50/D90, D[3,2]...), twice to areas; counts and shapes are
# not affected. Two terms:
#   reading of the scale bar: u_L / L, with u_L = sqrt(2) x (standard
#     uncertainty of one bar end): automatic detection 0.6 px (integer edge
#     positions, ±1 px uniform), manual line 1.2 px (±2 px uniform);
#     metadata pixel size: no reading term;
#   instrument magnification accuracy (user input, %): 0 when unknown.
# -----------------------------------------------------------------------------
U_EXTREMITE_PX = {"auto": 0.6, "manual": 1.2}


def incertitude_calibration(session):
    """Returns a dict: u_lecture, u_instr, u_rel (standard, relative),
    U_rel (expanded, k = 2), and the inputs of the budget."""
    src = session.calib_source or ("manual" if session.calib_l_px else "")
    L = float(session.calib_l_px or 0.0)
    if src in U_EXTREMITE_PX and L > 0:
        u_l = np.sqrt(2.0) * U_EXTREMITE_PX[src] / L
    else:
        u_l = 0.0
    u_i = float(session.u_instr_pct or 0.0) / 100.0
    u = float(np.hypot(u_l, u_i))
    return {"source": src or "unknown", "L_px": L, "u_lecture": u_l,
            "u_instr": u_i, "u_rel": u, "U_rel": 2.0 * u, "k": 2,
            "instr_inclus": u_i > 0}


__all__ = [_n for _n in dir() if not _n.startswith("__")]
