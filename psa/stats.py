# -*- coding: utf-8 -*-
"""PSA, Distribution statistics and distribution plots."""
from psa.editing import *  # noqa: F401,F403

def stats_distribution(d):
    """
    Statistiques granulométriques complètes (ISO 9276) :
      • percentiles en NOMBRE (Dn10/50/90) et en VOLUME (Dv10/50/90),
      • diamètres moyens pondérés : D[1,0] arithmétique,
        D[3,2] Sauter (→ surface spécifique, comparable BET),
        D[4,3] De Brouckere (comparable diffraction laser),
      • span de polydispersité (Dv90−Dv10)/Dv50,
      • ajustements normal et LOG-NORMAL (défaut physique des poudres :
        croissance/fragmentation multiplicatives) avec choix par AIC.
    """
    sps = _sps
    d = np.asarray(d, float)
    d = d[d > 0]
    n = len(d)
    out = {"n": n}
    if n == 0:
        return out
    out["D10"] = float(np.mean(d))
    out["D32"] = float((d ** 3).sum() / (d ** 2).sum())
    out["D43"] = float((d ** 4).sum() / (d ** 3).sum())
    out["dn10"], out["dn50"], out["dn90"] = \
        (float(np.percentile(d, p)) for p in (10, 50, 90))
    out["dv10"], out["dv50"], out["dv90"] = dx_volume(d)
    out["span"] = ((out["dv90"] - out["dv10"]) / out["dv50"]
                   if out["dv50"] > 0 else 0.0)
    mu = float(np.mean(d))
    sig = float(np.std(d, ddof=1)) if n > 1 else 0.0
    out["norm"] = (mu, sig)
    ln = np.log(d)
    mug = float(np.mean(ln))
    sigg = float(np.std(ln, ddof=1)) if n > 1 else 0.0
    out["lognorm"] = (math.exp(mug), math.exp(sigg))   # médiane géo., σg
    if n > 3 and sig > 0 and sigg > 0:
        ll_n = float(np.sum(sps.norm.logpdf(d, mu, sig)))
        ll_l = float(np.sum(sps.lognorm.logpdf(d, sigg,
                                               scale=math.exp(mug))))
        out["aic"] = {"normale": 4 - 2 * ll_n, "log-normale": 4 - 2 * ll_l}
        out["meilleur_fit"] = ("log-normale" if ll_l > ll_n else "normale")
    return out


def dessiner_distribution(ax_h, ax_c, d, unite, clair=False,
                          ponderation="nombre",
                          fits=("lognormale",), kde=False):
    """
    Histogramme + cumulative, pondération « nombre » (q0) ou « volume »
    (q3, poids d³). Superpositions optionnelles (dans les deux pondérations) :
    ajustement log-normal, ajustement normal, estimation par noyaux (KDE) ;
    en volume, les courbes sont pondérées d³ et renormalisées pour coïncider
    avec l'histogramme de densité volumique.
    Les lignes de percentiles affichées sont COHÉRENTES avec la pondération
    choisie (Dn en nombre, Dv en volume).
    """
    sps = _sps
    d = np.asarray(d, float)
    d = d[d > 0]
    n = len(d)
    if n == 0:
        ax_h.text(0.5, 0.5, "No particle", ha="center", va="center")
        return
    cb = PDF_C["blue"] if clair else T("BLUE")
    COUL = {"10": "#E53935", "50": "#B07D00", "90": "#14915B"}
    vol = str(ponderation).lower().startswith("vol")
    st = stats_distribution(d)
    if vol:
        p10, p50, p90 = st["dv10"], st["dv50"], st["dv90"]
        pref = "Dv"
    else:
        p10, p50, p90 = st["dn10"], st["dn50"], st["dn90"]
        pref = "Dn"

    n_bins = max(10, int(1 + 3.322 * math.log10(n))) if n > 1 else 10
    poids = d ** 3 if vol else None
    cnt, bords, _ = ax_h.hist(d, bins=n_bins, weights=poids, color=cb,
                              edgecolor="white", linewidth=0.5,
                              alpha=0.85, density=vol)
    largeur = float(bords[1] - bords[0]) if len(bords) > 1 else 1.0

    xs = np.linspace(max(1e-9, d.min() * 0.7), d.max() * 1.15, 300)
    pas = float(xs[1] - xs[0])

    def _echelle(pdf_nombre):
        """Met une densité-nombre à l'échelle de l'histogramme affiché :
        en volume, l'histogramme est une densité pondérée d³ (density=True),
        donc on pondère la PDF par x³ et on la renormalise ; en nombre,
        l'histogramme est en effectifs, donc on multiplie par n·largeur."""
        if vol:
            fv = xs ** 3 * pdf_nombre
            aire = float(np.sum(fv)) * pas
            return fv / aire if aire > 0 else fv
        return pdf_nombre * n * largeur

    if "lognormale" in fits and st["lognorm"][1] > 1.0:
        med_g, sig_g = st["lognorm"]
        pdf = sps.lognorm.pdf(xs, math.log(sig_g), scale=med_g)
        ax_h.plot(xs, _echelle(pdf), color="#7B2FBE", lw=2.0,
                  label=f"Log-normal (σg={sig_g:.2f})")
    if "normale" in fits and st["norm"][1] > 0:
        mu, sg = st["norm"]
        ax_h.plot(xs, _echelle(sps.norm.pdf(xs, mu, sg)),
                  color="#D85A30", lw=1.6, ls="--",
                  label=f"Normal ({mu:.1f}±{sg:.1f})")
    if kde and n > 4:
        d_kde = (np.random.default_rng(0).choice(d, 4000, replace=False)
                 if n > 4000 else d)
        k = sps.gaussian_kde(d_kde)
        ax_h.plot(xs, _echelle(k(xs)), color="#0F6E56",
                  lw=1.4, ls=":", label="KDE")
    if fits or kde:
        ax_h.legend(fontsize=7.5)

    ym = ax_h.get_ylim()[1] or 1
    rng = float(d.max() - d.min()) or 1.0
    for suf, vl in [("10", p10), ("50", p50), ("90", p90)]:
        ax_h.axvline(vl, color=COUL[suf], lw=1.5, ls="--")
        ax_h.text(vl + rng * 0.01, ym * 0.92, f"{pref}{suf}\n{vl:.1f}",
                  color=COUL[suf], fontsize=7.5, fontweight="bold", va="top")
    ax_h.set_xlabel(f"Equivalent diameter ({unite})")
    if vol:
        ax_h.set_ylabel("Volume density")
        ax_h.set_title(f"Volume distribution q3 (d³ weighting, n={n}, "
                       f"span={st['span']:.2f})", fontsize=9)
    else:
        ax_h.set_ylabel("Count")
        ax_h.set_title(
            f"Number distribution q0 (n={n}, "
            f"mean={st['norm'][0]:.1f}±{st['norm'][1]:.1f} {unite}, "
            f"D[3,2]={st['D32']:.1f})", fontsize=9)
    ax_h.grid(axis="y", alpha=0.3, ls=":")

    d_s = np.sort(d)
    if vol:
        cum = np.cumsum(d_s ** 3); cum = cum / cum[-1] * 100
        ax_c.set_ylabel("Cumulative volume (%)")
        ax_c.set_title("Cumulative volume (spheres assumed, ISO 9276-2)",
                       fontsize=9)
    else:
        cum = np.arange(1, n + 1) / n * 100
        ax_c.set_ylabel("Cumulative number (%)")
        ax_c.set_title("Cumulative number", fontsize=9)
    ax_c.plot(d_s, cum, color=cb, lw=2.2)
    ax_c.fill_between(d_s, cum, alpha=0.12, color=cb)
    for suf, vl, pct in [("10", p10, 10), ("50", p50, 50), ("90", p90, 90)]:
        ax_c.axvline(vl, color=COUL[suf], lw=1.3, ls="--", alpha=0.85)
        ax_c.plot(vl, pct, "o", color=COUL[suf], ms=6)
    ax_c.set_xlabel(f"Equivalent diameter ({unite})")
    ax_c.set_ylim(0, 105)
    ax_c.grid(alpha=0.3, ls=":")


__all__ = [_n for _n in dir() if not _n.startswith("__")]
