# -*- coding: utf-8 -*-
"""
Test suite of the Particle Size Analyser (PSA), Materia Nova / CHIPS.

Run:   python -m pytest tests -q        (or)    python tests/test_psa.py
Real images: folder given by the PSA_TEST_IMAGES environment variable
(default: ./images_test). Tests whose image is missing are SKIPPED, not
failed. Synthetic tests always run. GUI tests need a display (on Linux
without screen: xvfb-run -a python -m pytest tests).
"""
import os
import sys
import time
import tempfile
import warnings
import json
from pathlib import Path

import numpy as np
import cv2

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ICI))
warnings.simplefilter("ignore")

import psa_fibres as F            # noqa: E402
import psa_nano_meb as nano       # noqa: E402
import psa_app as A               # noqa: E402

IMAGES = os.environ.get("PSA_TEST_IMAGES",
                        os.path.join(os.path.dirname(ICI), "images_test"))

try:
    import pytest
    _skip = pytest.skip
except ImportError:                                     # exécution directe
    pytest = None

    class _Saut(Exception):
        pass

    def _skip(msg):
        raise _Saut(msg)


def _image(nom):
    chemin = os.path.join(IMAGES, nom)
    if not os.path.exists(chemin):
        _skip(f"missing image: {chemin}")
    img = cv2.imread(chemin, cv2.IMREAD_UNCHANGED)
    if img is None:
        import tifffile
        img = tifffile.imread(chemin)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def _meb(nom, longueur_barre_um):
    """Image MEB + calibration lue sur la barre d'échelle."""
    a = _image(nom)
    g = a if a.ndim == 2 else cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    y = nano.detecter_databar(g)
    ppm = nano.detecter_barre_echelle(g, y) / longueur_barre_um
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), ppm


# ----------------------------------------------------------------------------
# Fibres synthétiques à diamètre connu (profil MEB à liseré brillant)
# ----------------------------------------------------------------------------
def fibres_synthetiques(d, H=700, W=900, n=14, seed=0, bruit=8.0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img = 30 + 12 * (xx / W) + 8 * np.sin(yy / 160)
    for _ in range(n):
        a = rng.uniform(0, np.pi)
        cx, cy = rng.uniform(0.15 * W, 0.85 * W), rng.uniform(0.15 * H, 0.85 * H)
        u = np.abs(((xx - cx) * -np.sin(a) + (yy - cy) * np.cos(a)) / (d / 2))
        prof = 120 + 85 * np.clip(u, 0, 1) ** 6 + 30 * (1 - np.clip(u, 0, 1) ** 2)
        img = np.where(u <= 1, prof, img)
    img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.2)
    img += rng.normal(0, bruit, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_fibres_diametre_synthetique():
    """Le diamètre médian retrouve le diamètre vrai (±5 %, ±15 % à 6 px)."""
    for d, tol in ((6, 0.15), (12, 0.05), (24, 0.05), (48, 0.05)):
        r = F.analyser_fibres(fibres_synthetiques(d, seed=d), 1.0,
                              {"couper_databar": False})
        med = float(np.median(r["diam_px"]))
        assert abs(med - d) <= tol * d, f"Ø {d} px measured as {med:.1f} px"


def test_fibres_independance_du_seuil():
    """Le diamètre ne dépend plus du seuil (défaut historique corrigé)."""
    img = fibres_synthetiques(24, seed=3)
    meds = [float(np.median(F.analyser_fibres(
        img, 1.0, {"couper_databar": False, "seuil_facteur": f})["diam_px"]))
        for f in (0.65, 0.8, 1.0)]
    assert max(meds) - min(meds) <= 0.06 * 24, meds


def test_fibres_mesures_operateur_A67():
    """Deux mesures posées au curseur MEB par l'opérateur sur A67."""
    img, ppm = _meb("A67_i005.tif", 10.0)
    r = F.analyser_fibres(img, ppm)
    R = r["rayons"]
    mid = (R["dep"] + R["arr"]) / 2
    for (y0, y1, x0, x1), ref in (((250, 340, 690, 755), 1.49),
                                  ((150, 330, 760, 860), 1.06)):
        z = ((mid[:, 0] > y0) & (mid[:, 0] < y1) & (mid[:, 1] > x0) &
             (mid[:, 1] < x1) & (R["larg"] / ppm > 0.6))
        assert z.sum() >= 5, "operator fibre not measured"
        med = float(np.median(R["larg"][z])) / ppm
        assert abs(med - ref) <= 0.08 * ref, f"{med:.2f} µm instead of {ref}"


def test_fibres_poreuses_A44b():
    img, ppm = _meb("A44b_i007.tif", 2.0)
    r = F.analyser_fibres(img, ppm, {"combler_pores": True})
    assert r["n_pores"] > 20 and r["porosite_fibre_pct"] > 5
    assert 0.5 < r["diam_median_um"] < 1.6


def test_fibres_annotation():
    img = fibres_synthetiques(12, seed=1)
    r = F.analyser_fibres(img, 1.0, {"couper_databar": False})
    vis = F.image_annotee_fibres(img, r)
    assert vis.shape[:2] == img.shape[:2]


# ----------------------------------------------------------------------------
# Épaisseurs de couche
# ----------------------------------------------------------------------------
def test_epaisseur_balayage_synthetique():
    H, W = 600, 1000
    img = np.full((H, W), 30, np.float32)
    img[330:, :] = 90
    for x in range(W):
        haut = 270 + int(8 * np.sin(x / 90))
        ep = 55 + int(15 * np.sin(x / 140) + 6 * np.cos(x / 40))
        img[haut:haut + ep, x] = 200
    img += np.random.default_rng(0).normal(0, 6, img.shape)
    g = np.clip(img, 0, 255).astype(np.uint8)
    eps, _, taux = A.profils_epaisseur_auto(g, (500, 250), (500, 360))
    st = A.stats_epaisseur(eps, 10.0)
    assert st["n"] > 100 and abs(st["moy"] - 5.55) < 0.4 and taux > 0.8


def test_epaisseur_ruban_coating_AB32():
    a = _image("AB32-251026_i007.tif")
    g = a if a.ndim == 2 else cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    y = nano.detecter_databar(g)
    eps, band, skel = A.epaisseur_ruban_guide(g[:y - 2], (350, 330),
                                              (470, 560), 507 / 5.0, pct=88)
    st = A.stats_epaisseur(eps, 507 / 5.0)
    assert st["n"] > 100 and 0.110 < st["moy"] < 0.165


# ----------------------------------------------------------------------------
# Particules (MEB, nano), distribution, édition
# ----------------------------------------------------------------------------
def test_meb_spheres_completion():
    img = _image("spheres_meb.png")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    p = dict(A.PARAMS_MEB, moteur="watershed")      # classic engine test
    p["combler_convexe"] = True
    r = A.detecter_meb(img, 1.0, p)
    assert r["n_particules"] > 120
    assert float(np.median(r["circularites"])) > 0.82


def test_nano_classique():
    """Classic nano engine kept (h-maxima watershed + Hough)."""
    img, ppm = _meb("86_sonic_i014.tif", 0.5)
    r = A.detecter_nano(img, ppm, dict(nano.NANO_PARAMS, moteur="watershed"))
    assert r["n_particules"] > 30
    assert 30 < float(np.median(r["diametres"])) < 150      # nm


def test_nano_deux_etapes_te():
    """Two-step nanocarbon pipeline on a real TE image: aggregates, then
    StarDist primaries; carbon-black primaries of ~25-45 nm."""
    _sans_stardist()
    img, ppm = _meb("86_sonic_i014.tif", 0.5)
    r = A.detecter_nano(img, ppm, dict(nano.NANO_PARAMS))
    nr = r["nano"]
    assert nr["methode"] == "stardist_deux_etapes" and nr["imagerie"] == "TE"
    assert r["n_particules"] > 100
    assert r["masque"].any()                      # aggregates segmented
    gm = float(np.exp(np.mean(np.log(r["diametres"]))))
    assert 25 < gm < 45, gm
    assert set(r["populations"]) == {"primaires", "agregats"}


def test_nano_modele_synthetique_verite_connue():
    """Validation on synthetic TE aggregates of KNOWN primaries (not seen in
    training): geometric mean primary diameter within 12 %."""
    _sans_stardist()
    from psa.synth_nano import image_te
    ecarts = []
    for graine in (7101, 7102, 7103, 7104):
        im, _, vrais = image_te(seed=graine)
        bgr = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        r = A.detecter_nano(bgr, 1000.0, dict(nano.NANO_PARAMS,
                                               couper_databar=False))
        d = np.asarray(r["diametres"], float)      # nm = px at 1 px/nm
        assert d.size >= 0.25 * len(vrais)
        ecarts.append(np.exp(np.mean(np.log(d))) /
                      np.exp(np.mean(np.log(vrais))) - 1)
    assert abs(float(np.median(ecarts))) < 0.12, ecarts


def test_distribution_volume_affiche_les_ajustements():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    d = np.random.default_rng(0).lognormal(np.log(60), 0.4, 400)
    for pond in ("nombre", "volume"):
        fig = Figure()
        axh, axc = fig.add_subplot(121), fig.add_subplot(122)
        A.dessiner_distribution(axh, axc, d, "nm", clair=True,
                                ponderation=pond,
                                fits=("lognormale", "normale"), kde=True)
        noms = [ln.get_label() for ln in axh.get_lines()]
        assert any("Log-normal" in s for s in noms), pond


def test_edition_fusion_sans_perte():
    for ecart in (80, 95, 200):
        lab = np.zeros((220, 460), np.int32)
        cv2.circle(lab, (110, 110), 45, 1, -1)
        cv2.circle(lab, (110 + ecart, 110), 45, 2, -1)
        res = {"mode": "meb", "unite": "µm", "labels_ws": lab, "ids": [1, 2],
               "masque": (lab > 0).astype(np.uint8) * 255,
               "diametres": np.array([90., 90.]),
               "circularites": np.array([1., 1.]),
               "aspects": np.array([1., 1.]), "solidites": np.array([1., 1.]),
               "centres_px": [(110, 110), (110 + ecart, 110)],
               "n_particules": 2,
               "editions": {"suppr": 0, "ajout": 0, "scind": 0, "fusion": 0}}
        assert A.edition_fusionner(res, 0, 1, 1.0)
        xs = np.nonzero(res["labels_ws"] > 0)[1]
        assert res["n_particules"] == 1
        assert xs.min() < 110 and xs.max() > 110 + ecart - 10, ecart


def test_curseur_sensibilite_fibres_aller_retour():
    base = dict(F.FIBRES_PARAMS)
    s, _, _ = A.curseurs_depuis_params(base, "fibres")
    p = A.appliquer_curseurs(base, "fibres", s, 50, 0,
                             actifs=(True, False, False))
    assert abs(p["seuil_facteur"] - base["seuil_facteur"]) < 0.03


def _billes_synthetiques():
    img = np.full((500, 700), 25, np.uint8)
    for (x, y, r) in [(80, 90, 12), (200, 90, 16), (330, 100, 20), (460, 90, 14),
                      (100, 300, 18), (250, 320, 13), (520, 330, 70)]:
        cv2.circle(img, (x, y), r, 210, -1)
    img = cv2.GaussianBlur(img, (0, 0), 1.2)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def test_taille_max_meb_et_optique():
    """The maximum size removes the large object everywhere, consistently."""
    img = _billes_synthetiques()
    p = dict(A.PARAMS_MEB, couper_databar=False, moteur="watershed")
    r0 = A.detecter_meb(img, 1.0, p)
    r1 = A.filtrer_taille_max(A.detecter_meb(img, 1.0, p),
                              dict(p, taille_max=80.0))
    assert r0["n_particules"] - r1["n_particules"] == 1
    assert r1["n_trop_grands"] == 1 and max(r1["diametres"]) < 80
    assert len(r1["ids"]) == len(r1["diametres"]) == len(r1["circularites"])
    assert len(A.contours_du_resultat(r1)) == r1["n_particules"]
    po = dict(A.PARAMS_OPTIQUE, moteur="watershed")
    img_am = A.ameliorer(A.rolling_ball(img, po), po)
    o0 = A.detecter_optique(img, img_am, 1.0, po)
    o1 = A.filtrer_taille_max(A.detecter_optique(img, img_am, 1.0, po),
                              dict(po, taille_max=80.0))
    assert o0["n_particules"] - o1["n_particules"] >= 1
    assert len(o1["contours"]) == len(o1["diametres"]) == o1["n_particules"]
    assert all(d <= 80 for d in o1["diametres"])
    # 0 = no limit
    assert A.filtrer_taille_max(A.detecter_meb(img, 1.0, p),
                                p)["n_particules"] == r0["n_particules"]


def test_curseur_taille_max_log():
    for mode in ("optique", "meb", "nano"):
        assert A.pos_vers_taille_max(100, mode) == 0.0          # no limit
        for v in (5.0, 80.0, 1200.0):
            pos = A.taille_max_vers_pos(v, mode)
            assert abs(A.pos_vers_taille_max(pos, mode) - v) < 0.01 * v


def _sans_stardist():
    if not A.stardist_disponible():
        _skip("StarDist engine unavailable (pip install onnxruntime)")


def test_stardist_spheres_lisses():
    """Learned segmentation: whole spheres, shaded bottom included."""
    _sans_stardist()
    img = _image("spheres_meb.png")
    r = A.detecter_meb(img, 1.0, dict(A.PARAMS_MEB, couper_databar=False,
                                      moteur="stardist"))
    assert 140 <= r["n_particules"] <= 185, r["n_particules"]
    assert float(np.median(r["circularites"])) > 0.85
    assert abs(float(np.median(r["diametres"])) - 103) < 0.08 * 103
    assert len(r["ids"]) == r["n_particules"]


def test_stardist_spheres_texturees():
    """The case the watershed fails: touching, textured spheres."""
    _sans_stardist()
    img = _image("spheres_texturees.png")
    r = A.detecter_meb(img, 1.0, dict(A.PARAMS_MEB, couper_databar=False,
                                      moteur="stardist"))
    assert 24 <= r["n_particules"] <= 34, r["n_particules"]
    assert float(np.median(r["circularites"])) > 0.85


def test_stardist_optique():
    _sans_stardist()
    img = _image("spheres_meb.png")
    po = dict(A.PARAMS_OPTIQUE, moteur="stardist")
    img_am = A.ameliorer(A.rolling_ball(img, po), po)
    r = A.detecter_optique(img, img_am, 1.0, po)
    assert r["n_particules"] > 120 and r["moteur"] == "stardist"
    assert len(r["contours"]) == r["n_particules"]


def test_stardist_onnx_sans_tensorflow():
    """Default backend: ONNX network + our post-processing, no TensorFlow."""
    _sans_stardist()
    if A.backend_stardist() != "onnx":
        _skip("ONNX backend not available")
    img = _image("spheres_texturees.png")
    r = A.detecter_meb(img, 1.0, dict(A.PARAMS_MEB, couper_databar=False))
    assert r["moteur"] == "stardist" and r["stardist"]["backend"] == "onnx"
    assert r["stardist"]["tta"] is True
    assert 24 <= r["n_particules"] <= 32


def test_curseurs_stardist_aller_retour():
    p = dict(A.PARAMS_MEB, moteur="stardist")
    q = A.appliquer_curseurs(p, "meb", 70, 30, 2, actifs=(True, True, True))
    assert q["sauvola_k"] == p["sauvola_k"]          # watershed untouched
    s, e, _ = A.curseurs_depuis_params(q, "meb")
    assert abs(s - 70) < 0.5 and abs(e - 30) < 0.5


def test_rapport_multipage_echantillon():
    """Multi-page report named after the sample, with the StarDist
    references when that engine is used."""
    img = _image("spheres_meb.png")
    s = A.Session()
    s.chemin, s.image_bgr, s.mode, s.ppm = "/tmp/x.png", img, "meb", 1.0
    s.params = dict(A.PARAMS_MEB, couper_databar=False)
    s.echantillon = "Lot 42 / test"
    s.rapport_latex = False            # this test checks the built-in report
    s.resultats = A.detecter_meb(img, 1.0, s.params)
    with tempfile.TemporaryDirectory() as dossier:
        stem = A.safe_name(s.nom_echantillon)
        f = A.exporter_resultats(dossier, stem, s, dpi=60)
        noms = [os.path.basename(x) for x in f]
        assert f"{stem}_report.pdf" in noms
        assert all(n.startswith(stem) for n in noms)
        pdf = open(os.path.join(dossier, f"{stem}_report.pdf"), "rb").read()
        n_pages = pdf.count(b"/Type /Page") - pdf.count(b"/Type /Pages")
        attendu = 5 if A.sphericite(s.resultats) else 4   # + spheroidization page
        assert n_pages == attendu, n_pages
        assert sum(n.startswith(f"{stem}_report_p") for n in noms) == attendu
    figs = A.pages_rapport(s)
    textes = " ".join(t.get_text() for fg in figs for t in fg.texts)
    if s.resultats.get("moteur") == "stardist":
        assert "10.1007/978-3-030-00934-2_30" in textes
        assert "10.1109/WACV45572.2020.9093435" in textes
        assert "10.1109/ISBIC56247.2022.9854534" in textes


def test_sphero_poudre_synthetique():
    """Spheroidization rate on synthetic powders of KNOWN composition
    (deep learning engine): within 7 points by number, 8 by mass."""
    _sans_stardist()
    from psa.synth_poudre import image_poudre
    for graine, frac in ((11, 0.9), (13, 0.5)):
        img, v = image_poudre(seed=graine, frac_sph=frac)
        r = A.detecter_meb(img, 1.0, dict(A.PARAMS_MEB, couper_databar=False))
        c = A.classer_sphericite(r)
        assert abs(c["taux_nombre_hors_sat"] - v["taux_nombre_hors_sat"]) <= 7
        assert abs(c["taux_masse_hors_sat"] - v["taux_masse_hors_sat"]) <= 8
        assert (c["classes"] == "satellite").sum() >= 0.6 * v["n_satellites"]


def test_incertitude_calibration():
    """GUM budget: reading sqrt(2) x 1.2 px / L (manual), instrument term,
    k = 2; applied to every size in the report."""
    s = A.Session()
    s.ppm, s.calib_l_px, s.calib_source = 5.0, 500.0, "manual"
    u = A.incertitude_calibration(s)
    assert abs(u["u_lecture"] - np.sqrt(2) * 1.2 / 500) < 1e-12
    assert abs(u["U_rel"] - 2 * np.sqrt(2) * 1.2 / 500) < 1e-12
    s.u_instr_pct = 2.0
    u = A.incertitude_calibration(s)
    assert abs(u["U_rel"] - 2 * np.hypot(np.sqrt(2) * 1.2 / 500, 0.02)) < 1e-12
    s.calib_source, s.calib_l_px = "metadata", 0.0
    assert abs(A.incertitude_calibration(s)["U_rel"] - 0.04) < 1e-12
    s.u_instr_pct = 0.0
    assert A.incertitude_calibration(s)["U_rel"] == 0.0
    # in the report: sizes given with their expanded uncertainty
    img = _image("spheres_meb.png")
    s2 = A.Session()
    s2.chemin, s2.image_bgr, s2.mode, s2.ppm = "/tmp/x.png", img, "meb", 1.0
    s2.calib_l_px, s2.calib_source, s2.u_instr_pct = 400.0, "auto", 1.5
    s2.params = dict(A.PARAMS_MEB, couper_databar=False, moteur="watershed")
    s2.resultats = A.detecter_meb(img, 1.0, s2.params)
    textes = " ".join(t.get_text() for fg in A.pages_rapport(s2)
                      for ax in fg.axes for tb in ax.tables
                      for t in [c.get_text() for c in tb.get_celld().values()])
    assert " ± " in textes and "3.0" in textes          # U = 2 x 1.5 % ≈ 3.0 %


def test_rapport_latex_template():
    """LaTeX report with the Materia Nova template: sources always exported
    (Overleaf-ready zip with the template logos), PDF compiled when
    pdflatex is available."""
    import zipfile
    assert A.tex("a_b & 5 % ≥ 2 µm") == r"a\_b \& 5 \% $\geq$ 2 \textmu{}m"
    img = _image("spheres_meb.png")
    s = A.Session()
    s.chemin, s.image_bgr, s.mode, s.ppm = "/tmp/x.png", img, "meb", 1.0
    s.calib_l_px, s.calib_source, s.u_instr_pct = 400.0, "auto", 1.5
    s.params = dict(A.PARAMS_MEB, couper_databar=False, moteur="watershed")
    s.echantillon = "Lot #3 & test_A"
    s.resultats = A.detecter_meb(img, 1.0, s.params)
    with tempfile.TemporaryDirectory() as d:
        pdf, z, msg = A.rapport_latex(s, d, "lot3")
        noms = zipfile.ZipFile(z).namelist()
        assert "report.tex" in noms and "Logo_MateriaNova.png" in noms
        assert any(n.startswith("figures/") and n.endswith(".pdf") for n in noms)
        src = zipfile.ZipFile(z).read("report.tex").decode("utf-8")
        assert r"Lot \#3 \& test\_A" in src and "MNbluedark" in src
        if A.trouver_pdflatex():
            assert pdf and os.path.getsize(pdf) > 20000, msg
            assert b"LaTeX" in open(pdf, "rb").read()


def test_tuiles_equivalentes():
    """Tiled network prediction (memory-safe on large images) gives the same
    objects as a single block."""
    _sans_stardist()
    import psa.learned as L
    img = _image("spheres_meb.png")
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x = L._normaliser(cv2.resize(g, None, fx=0.41, fy=0.41,
                                 interpolation=cv2.INTER_AREA))
    a = L.nms_candidats(*L.candidats_stardist(x, A.STARDIST_MODELE, 0.479, False,
                                              tuile=4096), x.shape, 0.3)
    b = L.nms_candidats(*L.candidats_stardist(x, A.STARDIST_MODELE, 0.479, False,
                                              tuile=256, marge=128), x.shape, 0.3)
    assert abs(len(a[0]) - len(b[0])) <= 1
    da = np.linalg.norm(a[1] - a[2][:, :, None], axis=1).mean(1)
    db = np.linalg.norm(b[1] - b[2][:, :, None], axis=1).mean(1)
    assert abs(np.median(da) - np.median(db)) < 0.02 * np.median(da)


def test_sauvegarde_recettes_atomique(tmp_path=None):
    import psa.session as S
    d = tempfile.mkdtemp()
    ancien = S.FICHIER_RECETTES
    try:
        S.FICHIER_RECETTES = Path(d) / "recettes.json"
        S.sauver_recettes({"Test": {"mode": "meb", "ppm": None, "params": {}}},
                          "op", "ManoDark")
        assert S.FICHIER_RECETTES.exists()
        assert not Path(str(S.FICHIER_RECETTES) + ".tmp").exists()
        assert "Test" in json.loads(S.FICHIER_RECETTES.read_text("utf-8"))["recettes"]
    finally:
        S.FICHIER_RECETTES = ancien


# ----------------------------------------------------------------------------
# Interface (bout en bout)
# ----------------------------------------------------------------------------
def _app():
    if not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
        _skip("no display (run under xvfb-run)")
    import ttkbootstrap as tb
    root = tb.Window(themename="darkly")
    return root, A.AppPSA(root)


def _attendre(root, cond, delai=180):
    t0 = time.time()
    while not cond():
        root.update()
        time.sleep(0.02)
        if time.time() - t0 > delai:
            raise TimeoutError("timeout")


def test_gui_fibres_bout_en_bout():
    img, ppm = _meb("A67_i005.tif", 10.0)
    root, app = _app()
    try:
        app.session.chemin = "A67"
        app.session.image_bgr = img
        app.var_recette.set("Electrospun fibres · SEM")
        app.appliquer_recette()
        app.session.image_bgr, app.session.ppm = img, ppm
        app.faites = {0, 1}
        app.aller_etape(3)
        _attendre(root, lambda: app._apercu is not None)
        app.var_autoarch.set(False)
        app.lancer_analyse()
        _attendre(root, lambda: app.session.resultats is not None
                  and app.etape == 4)
        fr = app.session.resultats["fibres"]
        assert 0.6 < fr["diam_median_um"] < 1.6
        assert app.session.resultats["n_particules"] >= 30
    finally:
        root.destroy()


def test_gui_mesures_epaisseur():
    a = _image("AB32-251026_i007.tif")
    g = a if a.ndim == 2 else cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    y = nano.detecter_databar(g)
    root, app = _app()
    try:
        app.session.chemin = "AB32"
        app.session.image_bgr = cv2.cvtColor(g[:y - 2], cv2.COLOR_GRAY2BGR)
        app.session.ppm = 507 / 5.0
        app.ouvrir_mesures()
        root.update()
        app._mesure_commit((350, 330), (470, 560))
        app.var_ruban_pct.set(88)
        app._mesure_ruban()
        root.update()
        st = app.session.mesures_auto["stats"]
        assert st["n"] > 100
        fd, chemin = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        app._mesures_export(chemin)
        with open(chemin, encoding="utf-8") as f:
            assert "Bright ribbon" in f.read()
        os.remove(chemin)
    finally:
        root.destroy()



def test_gui_taille_max():
    img = _billes_synthetiques()
    root, app = _app()
    try:
        app.session.chemin = "billes"
        app.session.image_bgr = img
        app.var_recette.set("Metal powder · SEM")
        app.appliquer_recette()
        app.session.image_bgr, app.session.ppm = img, 1.0
        app.session.params["couper_databar"] = False
        app.session.params["moteur"] = "watershed"
        app.faites = {0, 1}
        app.aller_etape(3)
        _attendre(root, lambda: app._apercu is not None
                  and app._derniere_seg is not None)
        app.var_tmax.set(A.taille_max_vers_pos(80.0, "meb"))
        app._planifier_apercu(immediat=True)
        _attendre(root, lambda: app._derniere_seg is not None
                  and app._derniere_seg.get("bords_grands") is not None)
        app.var_autoarch.set(False)
        app.lancer_analyse()
        _attendre(root, lambda: app.session.resultats is not None
                  and app.etape == 4)
        res = app.session.resultats
        assert res["n_trop_grands"] == 1 and max(res["diametres"]) < 80
    finally:
        root.destroy()


def test_gui_stardist():
    _sans_stardist()
    img = _image("spheres_texturees.png")
    root, app = _app()
    try:
        app.session.chemin = "texturees"
        app.session.image_bgr = img
        app.var_recette.set("Metal powder · SEM")
        app.appliquer_recette()
        app.session.image_bgr, app.session.ppm = img, 1.0
        app.session.params["couper_databar"] = False
        app.faites = {0, 1}
        app.aller_etape(3)
        assert app.var_moteur.get() == "Deep learning (StarDist)"
        _attendre(root, lambda: app._apercu is not None
                  and app._derniere_seg is not None)
        app.var_autoarch.set(False)
        app.lancer_analyse()
        _attendre(root, lambda: app.session.resultats is not None
                  and app.etape == 4)
        res = app.session.resultats
        assert res["moteur"] == "stardist" and 24 <= res["n_particules"] <= 34
    finally:
        root.destroy()


def test_gui_reglages_graphe():
    """Log / linear scale, graph settings dialog opened by a double-click,
    settings applied on screen and in the report."""
    img = _image("spheres_meb.png")
    root, app = _app()
    try:
        app.session.chemin = "spheres"
        app.session.image_bgr = img
        app.var_recette.set("Metal powder · SEM")
        app.appliquer_recette()
        app.session.image_bgr, app.session.ppm = img, 1.0
        app.session.params["couper_databar"] = False
        app.faites = {0, 1}
        app.aller_etape(3)
        _attendre(root, lambda: app._apercu is not None)
        app.var_autoarch.set(False)
        app.lancer_analyse()
        _attendre(root, lambda: app.session.resultats is not None
                  and app.etape == 4)
        app._changer_vue("distribution")
        app.var_xscale.set("Linear")
        app._regler_graphe(xscale="linear")
        assert app.session.graphe["xscale"] == "linear"
        assert app.fig.axes[0].get_xscale() == "linear"

        class Ev:                                  # simulated double-click
            dblclick, button, inaxes = True, 1, None
            xdata = ydata = None
        app._cv_press(Ev())
        root.update()
        dlg = app._dlg_graphe
        assert dlg.winfo_exists()
        champs = {}
        def descendants(w):
            for c in w.winfo_children():
                yield c
                yield from descendants(c)
        # le formulaire peut être dans un panneau défilant (v1.7.2)
        form = next(f for f in descendants(dlg) if any(
            isinstance(c, A.ttk.Entry) for c in f.winfo_children()))
        for w in form.winfo_children():
            if isinstance(w, A.ttk.Entry) and not isinstance(w, A.ttk.Combobox):
                champs[len(champs)] = w
        # Entries order: xmin, xmax, ymax, bins, title, xlabel, ylabel, font, line
        champs[3].delete(0, "end"); champs[3].insert(0, "18")
        champs[1].delete(0, "end"); champs[1].insert(0, "200")
        champs[4].delete(0, "end"); champs[4].insert(0, "Batch 3")
        def boutons(w):
            for c in w.winfo_children():
                yield c
                yield from boutons(c)
        ok = [b for b in boutons(dlg) if b.winfo_class().endswith("Button")
              and str(b.cget("text")) == "OK"]
        ok[0].invoke()
        root.update()
        g = app.session.graphe
        assert g["bins"] == 18 and g["xmax"] == 200 and g["titre"] == "Batch 3"
        ax = app.fig.axes[0]
        assert ax.get_xscale() == "linear" and abs(ax.get_xlim()[1] - 200) < 1e-6
        assert len(ax.patches) == 18                  # 18 size classes
        # report: linear axis, same classes
        figs = A.pages_rapport(app.session)
        ax_r = [a for a in figs[1].axes if a.get_xlabel().startswith("Equivalent")]
        assert ax_r and all(a.get_xscale() == "linear" for a in ax_r)
        assert len(ax_r[0].patches) == 18
    finally:
        root.destroy()


def test_gui_sphero_criteres():
    """Spheroidization frame in the results; stricter criteria lower the
    rate."""
    _sans_stardist()
    from psa.synth_poudre import image_poudre
    img, v = image_poudre(seed=12, frac_sph=0.7)
    root, app = _app()
    try:
        app.session.chemin = "poudre"
        app.session.image_bgr = img
        app.var_recette.set("Metal powder · SEM")
        app.appliquer_recette()
        app.session.image_bgr, app.session.ppm = img, 1.0
        app.session.params["couper_databar"] = False
        app.faites = {0, 1}
        app.aller_etape(3)
        _attendre(root, lambda: app._apercu is not None)
        app.var_autoarch.set(False)
        app.lancer_analyse()
        _attendre(root, lambda: app.session.resultats is not None
                  and app.etape == 4)
        t0 = A.sphericite(app.session.resultats, app.session.sphero)["taux_nombre_hors_sat"]
        app._var_sph["rondeur_min"].set("0.95")
        app._appliquer_sphero()
        root.update()
        t1 = A.sphericite(app.session.resultats, app.session.sphero)["taux_nombre_hors_sat"]
        assert t1 < t0
    finally:
        root.destroy()


def test_gui_nano_moteur_apprentissage():
    """Nanocarbon mode: the Settings screen shows the deep learning engine
    and its preview uses the two-step pipeline of the final analysis."""
    _sans_stardist()
    img, ppm = _meb("86_sonic_i014.tif", 0.5)
    root, app = _app()
    try:
        app.session.chemin = "nano"
        app.session.image_bgr = img
        app.var_recette.set("Nanocarbon · SEM")
        app.appliquer_recette()
        app.session.image_bgr, app.session.ppm = img, ppm
        app.faites = {0, 1}
        app.aller_etape(3)
        assert app.var_moteur.get() == "Deep learning (StarDist)"
        _attendre(root, lambda: app._apercu is not None
                  and app._derniere_seg is not None)
        seg = app._derniere_seg
        assert "deep learning" in seg["msg"] and seg["msg"].startswith("TE")
        app.var_autoarch.set(False)
        app.lancer_analyse()
        _attendre(root, lambda: app.session.resultats is not None
                  and app.etape == 4)
        res = app.session.resultats
        assert res["nano"]["methode"] == "stardist_deux_etapes"

        assert abs(res["n_particules"] - seg["n"]) <= 0.3 * res["n_particules"]
        # the results screen itself must build (population selector)
        for _ in range(10):
            root.update()
        assert app.var_population.get() == "Primary particles"
        if len(res["populations"].get("agregats", [])):
            app.var_population.set("Aggregates (regions)")
            app._changer_population()
            root.update()
            assert app.session.resultats["population"] == "agregats"
    finally:
        root.destroy()

# ----------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    ok = ko = saut = 0
    for nom, f in tests:
        t0 = time.time()
        try:
            f()
            ok += 1
            print(f"  OK       {nom}  ({time.time() - t0:.1f} s)")
        except (Exception, BaseException) as e:      # pytest.skip = BaseException
            if isinstance(e, KeyboardInterrupt):
                raise
            if type(e).__name__ in ("_Saut", "Skipped"):
                saut += 1
                print(f"  SKIPPED  {nom}: {e}")
            else:
                ko += 1
                print(f"  FAILED   {nom}: {type(e).__name__}: {e}")
    print(f"\n{ok} passed, {ko} failed, {saut} skipped")
    sys.exit(1 if ko else 0)
