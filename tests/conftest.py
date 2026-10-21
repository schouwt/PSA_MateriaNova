# -*- coding: utf-8 -*-
"""Test markers (v1.8.1).

    pytest -m "not lent and not gui"     quick check while working (< 1 min)
    pytest                               full suite, REQUIRED before a release

gui  : builds the main window (Linux: run under xvfb-run);
lent : longer than about 2.5 s on a single core (measured, v1.8.1).
The markers are set here, from the test names, so that the test files stay
unchanged; a new slow test is simply added to LENTS.
"""
import pytest

LENTS = {
    "test_detection_et_diametre",
    "test_detection_format_commun",
    "test_detection_reseau_dense",
    "test_fibres_annotation",
    "test_fibres_diametre_synthetique",
    "test_fibres_independance_du_seuil",
    "test_fibres_mesures_operateur_A67",
    "test_fibres_poreuses_A44b",
    "test_fraction_surfacique_exacte",
    "test_gui_fibres_bout_en_bout",
    "test_gui_nano_moteur_apprentissage",
    "test_gui_recette_gouttes_bout_en_bout",
    "test_gui_reglages_graphe",
    "test_gui_sphero_criteres",
    "test_gui_stardist",
    "test_gui_taille_typique",
    "test_homogeneite_sans_seuil",
    "test_incertitude_calibration",
    "test_lot_fibres_identique_a_l_analyse_seule",
    "test_lot_gouttes_rapport_standard",
    "test_lot_mode_optique",
    "test_lot_poudre_sphero_agregee",
    "test_meb_spheres_completion",
    "test_nano_classique",
    "test_nano_deux_etapes_te",
    "test_nano_modele_synthetique_verite_connue",
    "test_nms_rapide_contre_stardist_officiel",
    "test_nms_rapide_identique_a_l_exhaustive",
    "test_outil_bout_en_bout",
    "test_rapport_chaque_mode",
    "test_rapport_latex_template",
    "test_rapport_multipage_echantillon",
    "test_reconnaissance_type_image",
    "test_sphero_poudre_synthetique",
    "test_stardist_onnx_sans_tensorflow",
    "test_stardist_spheres_lisses",
    "test_stardist_spheres_texturees",
    "test_taches_d_un_canal_serialisees",
    "test_taille_typique_imposee",
    "test_verdict_structure_connue",
}
GUI_EXTRA = {"test_taches_d_un_canal_serialisees", "test_fermeture_attend_le_calcul",
             "test_fenetre_tient_dans_l_ecran", "test_taille_typique_imposee"}


def pytest_configure(config):
    config.addinivalue_line("markers", "gui: builds the main window")
    config.addinivalue_line("markers", "lent: slow test (> 2.5 s on one core)")


def pytest_collection_modifyitems(config, items):
    for it in items:
        nom = it.originalname if hasattr(it, "originalname") else it.name
        if "gui" in nom or nom in GUI_EXTRA:
            it.add_marker(pytest.mark.gui)
        if nom in LENTS:
            it.add_marker(pytest.mark.lent)
