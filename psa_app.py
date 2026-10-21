# -*- coding: utf-8 -*-
"""
=============================================================================
PARTICLE SIZE ANALYSER  –  Materia Nova 2026  –  v0.6  « Workbench »
Centre de Recherche Materia Nova – Mons, Belgique
=============================================================================
Refonte ergonomique pour déploiement technicien :

  • FENÊTRE UNIQUE persistante — navigation par étapes (stepper cliquable),
    un seul canvas matplotlib central qui change de rôle selon l'étape.
  • RECETTES : presets matériau/microscope en JSON (~/.psa_materia/),
    le technicien choisit une recette, charge l'image, clique Analyser.
  • CURSEURS MÉTIER : Sensibilité / Séparation / Taille min — les paramètres
    experts (CLAHE, Sauvola, h_rel…) restent derrière « Mode avancé… ».
  • APERÇU SUPERPOSÉ : détections dessinées sur l'image avec opacité réglable,
    pastille qualité permanente (vert/orange/rouge).
  • ANALYSE EN THREAD : l'interface ne gèle jamais ; barre de progression
    dans la fenêtre, plus aucune dépendance à la console.
  • MODE LOT : un dossier entier avec la même recette → distribution agrégée
    (atteint n ≥ 30 exigé par la représentativité).
  • TRAÇABILITÉ : archivage auto (CSV + PDF + paramètres JSON + opérateur
    + horodatage) à côté de l'image. Garde-fous à l'export (n<30, score bas).
  • MODE NANO/MEB : via le module psa_nano_meb (Sauvola + watershed h-maxima,
    granulométrie morphologique, Hough, dimension fractale).

Dépendances : opencv-python, numpy, matplotlib, scipy, scikit-image
Optionnel   : ttkbootstrap (thème moderne), psa_nano_meb.py (mode nano)
=============================================================================
"""
# Entry point. The code lives in the psa/ package; everything is re-exported
# here so that "import psa_app" keeps working (tests, scripts, self-test).
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from psa.gui.app import *  # noqa: F401,F403,E402


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================
def autotest(chemin_rapport=None):
    """Vérification rapide sans interface : modules présents et analyses
    fonctionnelles sur des images synthétiques. Utilisé après construction de
    l'exécutable (psa.exe --autotest) ; écrit un rapport et renvoie 0 si tout
    est bon. Un exécutable fenêtré n'a pas de console : d'où le fichier."""
    import tempfile
    import traceback
    lignes, ok = [], True

    def etape(nom, f):
        nonlocal ok
        try:
            lignes.append(f"OK     {nom} : {f()}")
        except Exception:
            ok = False
            lignes.append(f"FAILED {nom}\n" + traceback.format_exc())

    etape("versions", lambda: (f"Python {sys.version.split()[0]}, "
                               f"numpy {np.__version__}, cv2 {cv2.__version__}"))
    etape("tkinter", lambda: f"Tk {tk.TkVersion}, ttkbootstrap {TB}")
    etape("modules", lambda: f"nano={NANO_OK}, fibres={FIB_OK}")

    def _meb():
        img = np.full((400, 500), 25, np.uint8)
        for i, (x, y) in enumerate([(100, 100), (250, 120), (400, 110),
                                    (120, 280), (280, 290), (420, 300)]):
            cv2.circle(img, (x, y), 40 + 3 * i, 200, -1)
        img = cv2.GaussianBlur(img, (0, 0), 1.5)
        r = detecter_meb(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 1.0,
                         dict(PARAMS_MEB, couper_databar=False))
        return f"{r['n_particules']} particles"
    if NANO_OK:
        etape("SEM analysis", _meb)

    def _fib():
        img = np.full((400, 500), 30, np.uint8)
        for k in range(6):
            cv2.line(img, (0, 40 + 60 * k), (499, 80 + 55 * k), 190, 14)
        img = cv2.GaussianBlur(img, (0, 0), 1.2)
        r = fibres.analyser_fibres(img, 1.0, {"couper_databar": False})
        return f"median Ø {np.median(r['diam_px']):.1f} px (expected ~14)"
    if FIB_OK:
        etape("fibre analysis", _fib)

    def _gui():
        # construit réellement la fenêtre (détecte les modules chargés à la
        # volée par ttkbootstrap / PIL, absents d'un exécutable mal construit)
        root = tb.Window(themename="darkly") if TB else tk.Tk()
        root.withdraw()
        try:
            AppPSA(root)
            root.update()
        finally:
            root.destroy()
        return "main window built"
    etape("interface", _gui)

    def _sd():
        if not stardist_disponible():
            return "not available (pip install onnxruntime): classic engine used"
        img = np.full((300, 400), 20, np.uint8)
        for x, y in ((80, 80), (200, 90), (320, 80), (90, 210), (220, 220)):
            cv2.circle(img, (x, y), 34, 200, -1)
        img = cv2.GaussianBlur(img, (0, 0), 1.2)
        r = detecter_meb(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 1.0,
                         dict(PARAMS_MEB, couper_databar=False,
                              moteur="stardist"))
        return (f"StarDist ({backend_stardist()}, {os.cpu_count()} threads) "
                f"{r['n_particules']} particles (expected 5)")
    etape("deep learning", _sd)

    def _ep():
        g = np.full((300, 300), 30, np.uint8)
        g[120:170, :] = 200
        eps, _, _ = profils_epaisseur_auto(g, (150, 100), (150, 200))
        return f"thickness {np.median(eps):.1f} px (expected 50)"
    etape("thickness", _ep)

    texte = ("PSA SELF-TEST " + ("PASSED" if ok else "FAILED") + "\n"
             + "\n".join(lignes) + "\n")
    chemin = chemin_rapport or os.path.join(tempfile.gettempdir(),
                                            "psa_autotest.txt")
    with open(chemin, "w", encoding="utf-8") as f:
        f.write(texte)
    print(texte)
    return 0 if ok else 1


def main():
    if "--autotest" in sys.argv:
        i = sys.argv.index("--autotest")
        rapport = sys.argv[i + 1] if len(sys.argv) > i + 1 else None
        sys.exit(autotest(rapport))
    init_journal()

    def _crochet(exctype, val, tb_):
        import traceback
        journal("error", "unhandled exception:\n"
                + "".join(traceback.format_exception(exctype, val, tb_)))
        sys.__excepthook__(exctype, val, tb_)

    sys.excepthook = _crochet
    import tkinter.simpledialog  # noqa: F401 — requis : rend tk.simpledialog accessible (askstring, nom de recette)
    if TB:
        root = tb.Window(themename="darkly")
    else:
        root = tk.Tk()
        try:
            ttk.Style(root).theme_use("clam")
        except tk.TclError:
            pass
    AppPSA(root)
    root.mainloop()


if __name__ == "__main__":
    main()
