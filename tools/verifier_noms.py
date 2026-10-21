# -*- coding: utf-8 -*-
"""PSA, detection of undefined names hidden by the star imports (v1.8.0).

The package is a chain of `from psa.x import *`: pyflakes cannot tell a real
undefined name (a NameError waiting to happen) from a name inherited from the
chain. This tool replaces, in a temporary copy of each module, every star
import by the explicit list of the names it actually provides, then runs
pyflakes and reports only "undefined name" errors.

    python tools/verifier_noms.py        (exit code 1 if a name is undefined)
"""
import importlib
import os
import re
import subprocess
import sys
import tempfile

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)
ETOILE = re.compile(r"^from ([\w.]+) import \*.*$", re.M)


def noms_exportes(module):
    m = importlib.import_module(module)
    noms = getattr(m, "__all__", None) or [n for n in dir(m) if not n.startswith("_")]
    return sorted(set(noms))


def verifier(fichiers):
    erreurs = []
    with tempfile.TemporaryDirectory() as d:
        for f in fichiers:
            src = open(f, encoding="utf-8").read()

            def remplace(mt):
                noms = noms_exportes(mt.group(1))
                return "from {} import ({})  # noqa".format(
                    mt.group(1), ", ".join(noms))
            tmp = os.path.join(d, os.path.basename(f))
            open(tmp, "w", encoding="utf-8").write(ETOILE.sub(remplace, src))
            out = subprocess.run([sys.executable, "-m", "pyflakes", tmp],
                                 capture_output=True, text=True).stdout
            for ligne in out.splitlines():
                if "undefined name" in ligne:
                    erreurs.append(ligne.replace(tmp, os.path.relpath(f, RACINE)))
    return erreurs


def fichiers_du_paquet():
    out = []
    for base, _, noms in os.walk(os.path.join(RACINE, "psa")):
        out += [os.path.join(base, n) for n in noms if n.endswith(".py")]
    out += [os.path.join(RACINE, n) for n in ("psa_app.py", "psa_nano_meb.py",
                                               "psa_fibres.py")]
    return sorted(out)


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore")
    err = verifier(fichiers_du_paquet())
    print("\n".join(err) if err else "No undefined name.")
    sys.exit(1 if err else 0)
