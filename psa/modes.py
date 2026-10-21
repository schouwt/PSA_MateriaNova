# -*- coding: utf-8 -*-
"""PSA, registry of the analysis modes (v1.8.0, audit lot B).

What depends on the mode and used to be repeated in several files (the
interface, the batch, the PDF and LaTeX reports) is described ONCE here:
default parameters, detection call, names of the objects and of the
diameter, main weighting of the distribution, rows of the key-results
table, extra rows of the analysis table. Adding a mode means adding one
entry here (plus its own settings and result panels if it needs them).

Explicit imports only: this module can be checked by pyflakes.
"""
import cv2
import numpy as np

from psa.params import PARAMS_OPTIQUE, PARAMS_MEB, PARAMS_GOUTTES
from psa.base import FIB_OK, fibres
import psa_nano_meb as _nano


# Lignes du tableau « résultats clés » (clés de lignes_resume ; une clé
# absente des résultats est simplement sautée). Union de ce que montraient
# le rapport PDF et le rapport LaTeX avant la v1.8.0.
_CLES_POUDRE = ["n", "mean", "std", "dn10", "dn50", "dn90", "dv10", "dv50",
                "dv90", "d32", "d43", "span", "sigma_g", "circ_ms", "sph_n",
                "sph_m", "sat_n", "imaging", "dp_gm", "dp_sg", "n_agg", "df",
                "quality", "representativeness"]
_CLES_NANO2 = ["n", "imaging", "dp_gm", "dp_sg", "dn10", "dn50", "dn90", "d32",
               "n_agg", "np_moy", "rg_moy", "df", "kf", "quality",
               "representativeness"]
_CLES_FIBRES = ["n", "mean", "std", "dn10", "dn50", "dn90", "beads",
                "bead_density", "alignment", "orientation", "mat_porosity",
                "fibre_porosity", "pores", "pore_d50", "quality",
                "representativeness"]
_CLES_GOUTTES = ["n", "mean", "std", "dn10", "dn50", "dn90", "drop_cv",
                 "drop_class", "drop_conv", "drop_gap_pct", "drop_vol",
                 "drop_vmodel", "quality", "representativeness"]


class Mode:
    """One analysis mode. Subclasses override what differs."""
    cle = "optique"
    nom = "Optical microscopy"
    objets = "particles"               # pluriel, dans les phrases
    nom_diametre = "Equivalent diameter"
    ponderation = "volume"             # pondération du graphe principal
    unite = "µm"

    def params_defaut(self):
        return dict(PARAMS_OPTIQUE)

    def pretraiter(self, img, params):
        """Background correction (rolling ball) and optional inversion,
        cached by the interface between previews and analysis."""
        from psa.detection import rolling_ball
        out = rolling_ball(img, params)
        if params.get("inverser_couleurs"):
            out = cv2.bitwise_not(out)
        return out

    def detecter(self, img, ppm, params, zones=None, progress=None,
                 img_pretraitee=None):
        """Detection on the full image. img_pretraitee: background-corrected
        image already computed by the interface (optical mode only)."""
        from psa.detection import ameliorer, detecter_optique
        if progress:
            progress(2, "Background correction…")
        if img_pretraitee is None:
            img_pretraitee = self.pretraiter(img, params)
        return detecter_optique(img, ameliorer(img_pretraitee, params), ppm,
                                dict(params), zones, progress)

    def nom_diametre_de(self, res):
        return self.nom_diametre

    def cles_resultats(self, res):
        return list(_CLES_POUDRE)

    def lignes_info(self, res):
        """Extra rows of the « Sample and analysis » table."""
        sd = res.get("moteur") == "stardist"
        return [["Segmentation", "StarDist (deep learning) [1-3]" if sd
                 else "threshold + watershed"]]

    def ponderation_figure(self, res):
        return self.ponderation

    def agreger(self, agg, ok, sm):
        """Batch: mode-specific indicators of the images pooled (agg is the
        pooled result, ok the per-image results). Powders: spheroidization
        recomputed on the classes of all the particles."""
        sph = [r["_sph_image"] for r in ok if r.get("_sph_image")]
        if not sph:
            return
        from psa.sphero import taux_sphero
        c = taux_sphero(np.concatenate([x["classes"] for x in sph]),
                        np.concatenate([x["porteuses"] for x in sph]),
                        np.concatenate([x["d"] for x in sph]),
                        sph[0]["criteres"])
        # cache attendu par sphericite() pour ce résultat agrégé
        c["_cle"] = (int(agg["n_particules"]), str(agg.get("editions", {})),
                     str(sorted((getattr(sm, "sphero", None) or {}).items())))
        agg["_sphero"] = c if c["n"] == agg["n_particules"] else None


class ModeMeb(Mode):
    cle, nom = "meb", "SEM powders"

    def params_defaut(self):
        return dict(PARAMS_MEB)

    def detecter(self, img, ppm, params, zones=None, progress=None,
                 img_pretraitee=None):
        from psa.detection import detecter_meb
        return detecter_meb(img, ppm, dict(params), zones, progress)


class ModeNano(Mode):
    cle, nom, unite = "nano", "SEM nanoparticles", "nm"

    def params_defaut(self):
        return dict(_nano.NANO_PARAMS)

    def detecter(self, img, ppm, params, zones=None, progress=None,
                 img_pretraitee=None):
        from psa.detection import detecter_nano
        return detecter_nano(img, ppm, dict(params), zones, progress)

    @staticmethod
    def deux_etapes(res):
        return (res.get("nano") or {}).get("methode") == "stardist_deux_etapes"

    def cles_resultats(self, res):
        return list(_CLES_NANO2) if self.deux_etapes(res) else list(_CLES_POUDRE)

    def lignes_info(self, res):
        sd = res.get("moteur") == "stardist"
        return [["Segmentation", ("StarDist (deep learning) [1-3]" if sd else
                                  "threshold + watershed")
                 + (", two steps [4]" if self.deux_etapes(res) else "")]]

    def ponderation_figure(self, res):
        return "nombre" if self.deux_etapes(res) else "volume"

    def nom_diametre_de(self, res):
        return ("Primary particle diameter" if self.deux_etapes(res)
                else self.nom_diametre)

    def agreger(self, agg, ok, sm):
        agg["nano"] = {"methode": ok[0]["nano"].get("methode"),
                       "imagerie": ok[0]["nano"].get("imagerie"),
                       "agregats": [a for r in ok for a in r["nano"]["agregats"]],
                       "df": None}


class ModeFibres(Mode):
    cle, nom = "fibres", "Electrospun fibres"
    objets, nom_diametre, ponderation = ("measurement sites", "Fibre diameter",
                                         "nombre")

    def params_defaut(self):
        return dict(fibres.FIBRES_PARAMS) if FIB_OK else dict(PARAMS_OPTIQUE)

    def detecter(self, img, ppm, params, zones=None, progress=None,
                 img_pretraitee=None):
        from psa.detection import detecter_fibres
        return detecter_fibres(img, ppm, dict(params), zones, progress)

    def cles_resultats(self, res):
        return list(_CLES_FIBRES)

    def lignes_info(self, res):
        return []

    def agreger(self, agg, ok, sm):
        """Fibre indicators: counts summed, densities and fractions
        averaged over the images."""
        d = agg["diametres"]
        frs = [r["fibres"] for r in ok]

        def moy(k):
            v = [f[k] for f in frs if f.get(k) is not None]
            return float(np.mean(v)) if v else 0.0
        fr = {"n_billes": int(sum(f.get("n_billes") or 0 for f in frs)),
              "densite_billes_mm2": moy("densite_billes_mm2"),
              "alignement": moy("alignement"),
              "orientation_deg": moy("orientation_deg"),
              "porosite_pct": moy("porosite_pct"),
              "diam_um": d,
              "diam_median_um": float(np.median(d)) if d.size else 0}
        if any(f.get("pores") for f in frs):
            fr.update(pores_mask=True,
                      n_pores=int(sum(f.get("n_pores") or 0 for f in frs)),
                      porosite_fibre_pct=moy("porosite_fibre_pct"),
                      pore_d_median_um=moy("pore_d_median_um"))
        agg["fibres"] = fr


class ModeGouttes(Mode):
    cle, nom = "gouttes", "Microfluidic droplets & capsules"
    objets, nom_diametre, ponderation = "droplets", "Droplet diameter", "nombre"

    def params_defaut(self):
        return dict(PARAMS_GOUTTES)

    def detecter(self, img, ppm, params, zones=None, progress=None,
                 img_pretraitee=None):
        from psa.detection import detecter_gouttes_mode
        return detecter_gouttes_mode(img, ppm, dict(params), zones, progress)

    def cles_resultats(self, res):
        return list(_CLES_GOUTTES)

    def lignes_info(self, res):
        return [["Detection", "dark rings, sub-pixel circle fit"]]

    def agreger(self, agg, ok, sm):
        """Droplet statistics of the pooled diameters; the convention gap is
        the mean of the images, weighted by their number of droplets."""
        from psa.droplets import volume_goutte
        d = agg["diametres"]
        h = ok[0].get("hauteur_canal_um")
        st_i = [r.get("gouttes_stats", {}) for r in ok]
        ecarts = [g["ecart_convention"] for g in st_i if g.get("n", 0) > 1]
        poids = [g["n"] for g in st_i if g.get("n", 0) > 1]
        st = {"n": int(d.size), "n_bord": agg["n_bord"]}
        if d.size > 1:
            st.update(d_moy=float(d.mean()), d_std=float(d.std(ddof=1)),
                      cv_pct=float(100 * d.std(ddof=1) / d.mean()),
                      d10=float(np.percentile(d, 10)),
                      d50=float(np.median(d)), d90=float(np.percentile(d, 90)),
                      ecart_convention=float(np.average(ecarts, weights=poids))
                      if ecarts else 0.0,
                      v_moy=float(np.mean(volume_goutte(d, h))),
                      convention=ok[0].get("convention"), h_um=h,
                      unite=agg["unite"])
        agg["gouttes_stats"] = st
        agg["ressemble"] = all(r.get("ressemble", True) for r in ok)
        agg["um_par_px"] = ok[0].get("um_par_px", 1.0)


MODES = {m.cle: m for m in (Mode(), ModeMeb(), ModeNano(), ModeFibres(),
                             ModeGouttes())}


def mode_de(x):
    """Mode object from a key, a session or a result dict (optical by
    default, as before)."""
    if isinstance(x, str):
        cle = x
    elif isinstance(x, dict):
        cle = x.get("mode", "optique")
    else:
        cle = getattr(x, "mode", "optique")
    return MODES.get(cle, MODES["optique"])


MODE_NOMS = {k: m.nom for k, m in MODES.items()}
