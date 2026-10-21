PARTICLE SIZE ANALYSER  ·  Materia Nova, CHIPS department (UMONS)  ·  v1.8.1
SEM & optical granulometry
=============================================================================

CONTENTS (keep this folder structure)
  psa_app.py           entry point (run this file)
  psa/                 application package (see CODE STRUCTURE below)
  psa_nano_meb.py      SEM module (micron + nano), required for the SEM modes
  psa_fibres.py        electrospun fibre module, required for the fibre mode
  Models/stardist/     pretrained StarDist model (deep learning engine):
                       model.onnx (inference, no TensorFlow needed) and the
                       original Keras weights (future training)
  guide_psa.pdf        full guide (opened by the Help button / F1)
  build.bat            builds the Windows executable (see below)
  requirements.txt     dependencies (onnxruntime included: deep learning)
  requirements-dl.txt  only for TRAINING models (tools/train_nanocarbon.py)
  tools/               model training (nanocarbon, our SEM images) and
                       command-line tools: evaluation against an annotation,
                       filler dispersion, code checks (see COMMAND-LINE
                       TOOLS below)
  CHANGELOG.md         changes per version
  tests/               automatic test suite;  images_test/: test images
  psa.ico, psa.png     application icon
  Assets/              MANO_logo.png and CHIPS_logo.png: high-resolution logos
                       of the Materia Nova LaTeX template (report, header);
                       add UMONS_logo.png (high resolution) to show it too

FOR A COLLEAGUE: NO INSTALLATION NEEDED
  Copy the whole dist\PSA_MateriaNova folder built with build.bat (see
  WINDOWS EXECUTABLE) and run PSA_MateriaNova.exe: it needs no Python.

INSTALLATION (once per environment, to run from the sources or build)
  Use Python 3.12 or 3.13 (python.org, keep the "py launcher" option).
  A too recent Python (3.15) has no precompiled scikit-image package yet:
  pip would then need a C compiler. build.bat picks a suitable installed
  version automatically (3.13, 3.12, 3.14, 3.11) and never compiles.
  In the terminal of your environment (PyCharm: Terminal tab, prompt
  starting with (.venv)):
     pip install -r requirements.txt
  The deep learning engine (StarDist) runs with onnxruntime, installed by
  requirements.txt: no TensorFlow, every CPU core is used.

RUN
  python psa_app.py
  Self-test without GUI window:   python psa_app.py --autotest

INTERFACE (Materia Nova template, as Grating Analyzer and SMU 2400 Suite)
  Header: partner logos, Open image, Help, Theme (ManoDark, ManoBlue,
  ManoWhite). Left panel: SESSION (recipe, operator), IMAGE (1 Image,
  2 Calibration, 3 Zones), ANALYSIS (4 Settings, 5 Results), TOOLS
  (thickness measurements), OUTPUT (Report & export).
  Shortcuts: Ctrl+1…5 steps, Ctrl+O open image, Ctrl+E export, Ctrl+S
  export now (on the export screen), F1 help.
  BATCH ANALYSIS: the images of a batch are usually COMPARED
  (same magnification, different process conditions). Output folder
  PSA_batch_<date>/ in the image folder:
  - images/<image>/: the FULL report of every image (PDF, CSV, summary,
    parameters), exactly as when exported from the interface;
  - batch_comparison_report.pdf: key results per image side by side
    (n, mean, D10/D50/D90, CV, score and the indicators of the recipe:
    spheroidization rate, droplet volume, beads and alignment), cumulative
    distributions overlaid, box plots with the 95 % confidence interval of
    each median, Kruskal-Wallis test and matrix of the pairwise median
    differences (Mann-Whitney, Holm correction), annotated thumbnails,
    method and warnings per image; annex: images pooled, meaningful only
    for replicates of the same sample;
  - batch_images.csv, batch_objects.csv, batch_pairwise.csv,
    batch_pooled_summary.csv, batch_parameters.json.
  One calibration for the whole batch: same magnification required (an
  image whose metadata disagree by more than 2 % is flagged). Batch results
  are identical to single analyses, in every mode.
  DROPLETS & CAPSULES recipe ("Droplets & capsules · microfluidics"): microfluidic droplets or capsules seen in transmitted
  light as a dark ring whose inside is as bright as the liquid. Same
  workflow as the particles: image, calibration, zones, Settings (diameter
  at the outer edge, default for capsules, or at the ring middle for packed
  emulsions; channel height for squeezed droplets; ring sensitivity; Hough
  recovery of broken rings; live preview), Results with the usual graphs
  plus a Droplets & capsules block (size CV and class: monodisperse < 3 %,
  narrow < 5 %; gap between the two conventions; mean volume), specialised
  PDF report (volume distribution, deformation, diameter across the field,
  convention comparison, fit quality) and per-droplet CSV. Batch mode
  supported. Filled-looking objects: use the particle recipes.
  SMALL SCREENS (13-inch laptops): the window adapts to the screen
  (maximised and compact header below about 1450 x 900), the Previous /
  Next bar always stays visible, and the left and right panels SCROLL
  (mouse wheel over the panel, or the scroll bar shown when needed).
  Dialogs (graph settings, advanced mode, help, history) fit the screen.

THE ANALYSIS MODES
  - Optical microscopy (µm): well-contrasted, separable powders.
  - SEM, micron powders (µm): metal powders 5-200 µm ("Spherical powder"
    recipe: completes the shaded crescent of spheres).
  - SEM, nano / agglomerated (nm): NANOCARBON (carbon black, soot), new
    two-step pipeline following the soot TEM literature (Sipkens et al.
    2021): (1) aggregates from the transmission t = I / I_background
    (Beer-Lambert), with projected area, equivalent diameter, radius of
    gyration, number of primaries (Köylü et al. 1995) and the fractal law
    Np = kf (2Rg/dp)^Df (ensemble fit, at least 5 complete aggregates);
    (2) primary particles by a StarDist model trained for TE images of
    carbon aggregates (Models/stardist/nanocarbon_te). Primary diameter =
    2 x 90th percentile of the polygon rays (the free part of the contour;
    the visible area underestimates overlapped primaries by ~19 %).
    Validation on synthetic aggregates of KNOWN primaries, unseen in
    training: F1 0.66 (published SAGE method: 0.68), geometric mean
    diameter median error -1.3 %, mean absolute error 4.9 %.
    TE images (dispersed, dark on bright) are the reference case; SE images
    of bulk powder (no background) are EXPERIMENTAL (primaries only).
    The "Analysed population" selector switches to aggregates.
    Classic engine (watershed + Hough) still available (engine setting).
    Retraining:  python tools/train_nanocarbon.py  (needs requirements-dl).
  - SEM, electrospun fibres (µm): FIBRE DIAMETER measured edge to edge,
    without threshold (pairing of the two edges of each fibre), beads,
    alignment, mat porosity; "Porous fibres" recipe for porous ribbons.
  Particle modes: Minimum AND Maximum size sliders (log scale for the
  maximum, right end = no limit); objects above the maximum are drawn in
  red in the preview and counted in the report as excluded.
  The minimum size can be SWITCHED OFF ("Apply the minimum size"
  box under the slider, optical, SEM micro and nano modes): every object
  is kept whatever its physical size; only objects a few pixels across
  are still removed (pixel floor, their size cannot be measured). The
  slider position is remembered, the setting is stored with the recipe,
  honoured in batch use, and the report states "no minimum".

SEGMENTATION ENGINE (optical and SEM powder modes, Settings step)
  - Deep learning (StarDist), the DEFAULT: predicts every particle as a
    star-convex polygon (Schmidt et al. 2018, Weigert et al. 2020, 2022;
    cited in the report). Much better on touching, textured or shaded
    particles: whole particles, shaded bottom included, no fragments in
    the texture. The network runs as ONNX on every CPU core, with test-time
    augmentation (the image and its 3 mirror images are predicted and
    averaged): about 4x the computing time, fewer false detections.
    Sliders: Sensitivity = object probability threshold, Separation =
    allowed overlap between neighbours. The working scale is chosen from
    the typical particle size; regions are measured by the same code as
    the watershed. Best for compact, roughly convex particles.
  - Classic (threshold + watershed): for very elongated or branched shapes
    (not star-convex), or when onnxruntime is missing (automatic fallback).

SPHEROIDIZATION RATE (optical and SEM powder modes)
  Every particle is classified SPHERICAL, IRREGULAR or SATELLITE:
    roundness R = 4A / (pi Fmax^2) >= 0.80 and aspect ratio <= 1.20 ->
    spherical (R is 1 for a disc and drops for elongated AND angular
    shapes; the pixel circularity and, with StarDist, the solidity are not
    reliable for this); a particle touching one at least ~3x larger (size
    ratio <= 0.35) -> satellite.
  Rate by NUMBER and by MASS (d^3, same density), satellites excluded from
  the rate (main particles), satellites and particles bearing them given
  apart. Criteria adjustable in the Results panel ("Apply criteria").
  Validation on synthetic powders of known composition: within 5 points by
  number and 6 by mass with the deep learning engine; the classic engine
  merges satellites with their carrier and is NOT reliable for this rate.
  The report has a dedicated page (classes coloured on the image).

COMMAND-LINE TOOLS (run from the sources: python tools/<tool>.py -h)
  Not yet in the graphical interface nor in the executable.
  - eval_annotations.py  compares the PSA with a hand annotation (napari
    label image, ImageJ ROI set, Cellpose _seg.npy): F1, count error and
    size bias with the protocol of the meb_grains comparison report
    (matching by centre, border objects excluded), plus IoU >= 0.5
    matching, error types (missed, merged, split / satellite, background)
    and recall per size class. Variants: automatic scale, StarDist diameter
    forced to the true median (scale diagnosis), watershed.
  - analyse_dispersion.py  dispersion of a filler in a polymer or a resin
    (SEM image of a section): area fraction, local area fraction versus
    window size, distance from the matrix to the filler, agglomerates,
    Clark-Evans ratio, pair correlation g(r) and L(r), Voronoi cells,
    Saltykov unfolding; every indicator compared with the SAME particles
    placed at random (Monte Carlo, global test on L(r)), and a
    threshold-free homogeneity measure for composition-contrast images.
    Engines: stardist, threshold, hybrid (StarDist instances clipped to the
    threshold mask). Calibration read from the Hitachi .txt file.
    CAUTION: on a CRYOFRACTURE the surface is not a random section (the
    crack follows weak interfaces): the area fraction is not the volume
    fraction, Saltykov does not apply, the spatial statistics are for
    COMPARING samples prepared the same way. In BSE, the beam sees about
    1 um under the surface of a polymer at 15 kV: lower the voltage
    (about 5 kV) when the particles are of that size.
  - verifier_noms.py  finds the names that would raise a NameError but
    are hidden by the star imports (run by the tests).
  The droplet command-line tool was replaced by the Droplets & capsules
  recipe (batch mode included).

TRAINING STARDIST ON OUR OWN IMAGES (training PC only)
  1. Annotation kit:  python tools/kit_annotation.py RAW_FOLDER --out KIT
     Hitachi TIFF + .txt: the image family is deduced from the acquisition
     (lowmag_powder, highmag_powder, se_nano_crystals, te_section), tiles
     are cut (data bar excluded) with a PROPOSAL of the current model to
     correct; one image per family is kept for VALIDATION (annotated from
     scratch). Rules: KIT/ANNOTATION_GUIDE.md.
  2. Annotation in napari:  python KIT/annotate_napari.py [family]
     (pip install "napari[all]"); S = save <tile>_label.tif and next.
  3. Training:  pip install -r requirements-dl.txt, then
     python tools/train_meb.py KIT --name meb_mano [--families a,b]
     Tiles prepared exactly as the PSA sees them (polarity, working scale,
     normalisation); thresholds optimised; ONNX export to
     Models/stardist/meb_mano/; the new model is compared with the current
     one on the validation tiles (F1, count error, size bias per family,
     Models/stardist/meb_mano/validation.txt). A GPU is strongly advised
     (minutes instead of hours); the model then runs on any CPU.
  A trained model is used by the PSA through the recipe parameter
  "stardist_modele"; a model selector in the Settings step will come with
  the first validated model.

STARDIST WORKING SCALE AND TYPICAL SIZE
  The deep-learning engine resizes the image so that the particles are
  about 45 px, from a quick estimate of their typical size (dense fields,
  where the particles touch, are handled). Field "Typical size (µm)" under
  the engine (SEM micro mode): empty = automatic (the estimate is shown
  below the field); enter the size by hand when the preview finds no or
  very few visible particles. A warning is shown when the working scale is
  very small and almost nothing is found. Limit: on ONE large aggregate of
  small grains (e.g. TCP at x10k), the estimate measures the aggregate:
  enter the grain size.

KNOWN ISSUE
  On synthetic images, StarDist enlarges small objects by about 1.5 px on
  the diameter (+13 % at 12 px, +6 % at 24 px, so about 3 % at 50 px).
  Under investigation; for small particles in pixels, prefer a
  higher magnification.

CALIBRATION UNCERTAINTY
  Every size is proportional to 1 / calibration: its relative uncertainty
  applies to every diameter and D value (counts and shapes unaffected).
  Budget (GUM, expanded k = 2): reading of the scale bar, sqrt(2) x
  uncertainty of one bar end / bar length (automatic detection 0.6 px,
  manual line 1.2 px; none for metadata), and the MAGNIFICATION ACCURACY
  of the microscope, entered in the Calibration step (typically 1-3 % for
  an SEM; left empty = not included, stated in the report).
  Report: sizes given as value ± U, budget detailed on the Method page.
  Stored with the recipes.

GRAPHS (Results step, Distribution view)
  Same scientific plot as in the report: density per size class (bars),
  smoothed curve, cumulative (dashed), D10 / D50 / D90 markers.
  Distribution options: weighting (number / volume), SIZE AXIS
  (logarithmic / linear), log-normal and normal fits, smoothed curve.
  DOUBLE-CLICK the graph (or "Graph settings…") to edit it: weighting,
  scale, axis limits, number of size classes, displayed elements, title and
  axis labels, font size, line width, and "Use these settings in the
  report". Also available by double-clicking the report preview.

OPTIONAL LaTeX VERSION OF THE REPORT
  The default report is the built-in multi-page scientific report. On the
  Report & export screen, "Also a LaTeX version (Materia Nova template)"
  additionally writes the report in Template_Materia_Nova (compiled with
  pdflatex when MiKTeX or TeX Live is installed, replacing the built-in
  PDF) and its sources, <sample>_report_latex.zip (Overleaf-ready).
  The user guide (Help, guide_psa.pdf) is written in this template.

REPORT & EXPORT
  Enter the SAMPLE name in the left panel (default: image name): it is the
  report title and the prefix of every file, in a <sample>_analysis folder
  next to the image (or a chosen folder):
    _report.pdf               multi-page A4 scientific report with the
                              Materia Nova, UMONS and CHIPS logos:
                              1 summary (sample, key results, images,
                              summary distribution), 2 size distributions
                              (number, volume), 3 shape descriptors (or
                              fibre network), 4 method, quality, references
    _report_p1..p4.png        the same pages as images
    _particles.csv            one row per particle
                              (fibres: _fibre_sites.csv, _fibre_segments.csv)
    _summary.csv / .txt       all the statistics
    _parameters.json          mode, recipe, calibration, parameters
  CSV format: International ( , separator, . decimal) or Excel FR/BE
  ( ; separator, , decimal). The export screen previews every page.

FIRST USE, POINTS OF ATTENTION
  - Calibration: draw along the scale bar or click "Detect the scale bar"
    (SEM databar). This is the most critical step.
  - 16-bit TIFF and accented paths are supported.
  - Polarity (bright / dark particles) is detected automatically.
  - 0 particle kept: the status bar gives the probable cause.
  - Recipes saved by earlier (French) versions are renamed automatically,
    your settings are kept.

FILES CREATED BY THE APPLICATION (folder ~/.psa_materia)
  recettes.json        custom recipes, operator, theme
  historique.sqlite    database of archived analyses (comparison / SPC)
  journal.log          versions, analyses, detailed errors
                       -> ATTACH IT TO ANY PROBLEM REPORT.

CODE STRUCTURE (package psa/, each layer imports only the previous ones)
  base        imports, optional dependencies, constants, log
  modes       REGISTRY of the analysis modes (defaults, detection, names,
              key results, batch aggregation): explicit imports
  theme       Materia Nova branding: logos, Mano themes, palettes
  params      default parameters, sliders <-> expert parameters
  shapes      shape descriptors, annotation
  learned     deep learning segmentation (StarDist: ONNX inference, own
              post-processing, test-time augmentation; TensorFlow fallback)
  nanocarbon  two-step nanocarbon pipeline (aggregates, then primaries)
  synth_nano  synthetic TE aggregates of known primaries (training, tests)
  detection   image processing, particle and fibre detectors, size filter
  editing     manual editing of the detections
  stats       distribution statistics and plots
  database    SQLite history, statistical comparison, ImageJ ROIs
  session     session state, recipes
  report      multi-page scientific report, exports, archiving
  batch       batch analysis of a folder (individual reports + comparison
              report): explicit imports
  widgets     shared GUI helpers, side navigation
  gui/        main window (app.py) assembled from one class per screen:
              image, settings, results, export, measurements, editor,
              history
  psa_app.py re-exports everything: "import psa_app" keeps working.

AUTOMATIC TESTS
  Quick check while working (about 30 s, no window):
     python -m pytest tests -q -m "not lent and not gui"
  Full suite, REQUIRED before distributing a version (about 6 min):
     python -m pytest tests -q
  (Linux: xvfb-run python -m pytest tests -q for the GUI tests.)
  PyCharm: right click on tests > Run 'pytest in tests'. The markers gui
  (builds the window) and lent (slow) are set in tests/conftest.py.
  The suite also checks that no undefined name hides behind the star
  imports (tools/verifier_noms.py) and that the version number is the same
  in psa/base.py, README.txt, CHANGELOG.md and the guide.
  Real-image tests use images_test/ (or the folder given by the
  PSA_TEST_IMAGES environment variable); without the images they are
  SKIPPED, the synthetic tests always run. StarDist tests are skipped when
  the deep learning engine is not installed.

WINDOWS EXECUTABLE (.exe)
  Double click build.bat. It uses the project .venv when present, ignores a
  foreign PYTHONPATH (e.g. ChemOffice), installs PyInstaller, builds the
  executable, then runs an AUTOTEST of the built exe (modules, SEM, fibre,
  deep learning and thickness analyses, GUI construction) and shows the
  report.
     build.bat            dist\PSA_MateriaNova\ (recommended: fast start;
                          distribute the WHOLE folder), about 490 MB,
                          deep learning engine included (ONNX)
     build.bat onefile    single dist\PSA_MateriaNova.exe (slower start,
                          more often flagged by antivirus software)
     build.bat console    keeps a console window (diagnostics)
  Manual check at any time: PSA_MateriaNova.exe --autotest
  (report in %TEMP%\psa_autotest.txt).
