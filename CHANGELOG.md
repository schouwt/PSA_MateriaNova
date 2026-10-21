# Particle Size Analyser (PSA), changes per version

Materia Nova, CHIPS department (UMONS). The README describes the current
version; this file keeps the history. Newest first.

## v1.8.1 (code audit, lot C: process)
- Test markers `gui` and `lent` (tests/conftest.py): quick check in about
  30 s with `python -m pytest tests -q -m "not lent and not gui"`; the full
  suite stays required before distributing a version.
- This changelog, separated from the README (which now only describes the
  current version).
- User guide updated to v1.8.1: droplets & capsules recipe, batch analysis
  and comparison report, Hitachi calibration, minimum size switch, typical
  size, small screens, training on our own images, command-line tools.
- A test checks that the version number is the same in psa/base.py,
  README.txt, CHANGELOG.md and the guide.

## v1.8.0 (code audit, lot B: restructuring)
- `psa/modes.py`: registry of the analysis modes (default parameters,
  detection, names, main weighting, key results, batch aggregation), used
  by the interface, the batch and both reports.
- PDF report and batch comparison report built page by page.
- LaTeX report on par with the PDF for the droplets recipe.
- FIXED: the PDF report of the SEM nano mode (two steps) stopped on page 3
  (undefined name hidden by the star imports, present since v1.6.4). New
  check `tools/verifier_noms.py`, run by the tests.
- `psa/batch.py`, `psa/modes.py`: explicit imports (pyflakes-clean).
- Recipes start from the defaults of their own mode (fibre recipes started
  from the optical ones).

## v1.7.8 (code audit, lot A: corrections)
- StarDist candidate suppression skips candidates already covered:
  x100 powder analysis 45 s -> 26 s, same particles (validated against the
  official StarDist on the same network outputs). Preview without the
  mirror augmentation (30 s -> 10 s); the final analysis keeps it.
- Calibration read from the Hitachi .txt file (PixelSize). Batch: images
  whose metadata disagree with the batch calibration by more than 2 % are
  flagged.
- Previews serialized (only the latest is shown, never over the results);
  closing during a computation waits for its end (it could abort).
- Errors formerly ignored are logged; dead code removed (old landscape
  report, droplet command-line tool); dependency versions capped.

## v1.7.7
- Batch: full report of every image (images/<name>/) plus a comparison
  report: key results side by side, overlaid distributions, box plots with
  the 95 % CI of the median, Kruskal-Wallis and pairwise Mann-Whitney
  (Holm), thumbnails, pooled annex for replicates only.

## v1.7.6
- Batch report at the standard of the single-image report.
- FIXED: fibre recipes were analysed with the optical pipeline in batch.

## v1.7.5
- Droplets & capsules RECIPE (microfluidics): settings with live preview,
  results, specialised report, per-droplet CSV, batch. Replaces the v1.7.4
  tool.

## v1.7.4
- Droplets & capsules tool in the interface (replaced by the recipe).

## v1.7.3
- FIXED: StarDist working scale on dense fields (estimate 7 to 9 times too
  large, no particle found: resin x20k, powders x100). New "Typical size"
  field and warning.

## v1.7.2
- Small screens: window adapted to the screen, scrolling side panels,
  navigation bar always visible, dialogs fitted to the screen.

## v1.7.1
- Annotation kit (`tools/kit_annotation.py`, napari) and training on our
  SEM images (`tools/train_meb.py`, ONNX export, validation per family).

## v1.7.0
- Minimum size can be switched off (optical, SEM micro, nano).
- Command-line tools: evaluation against a hand annotation, filler
  dispersion analysis, microfluidic droplets.
- Known issue documented: StarDist enlarges small objects (about 1.5 px).

## v1.6.4
- Version received for the audit (34 tests).
