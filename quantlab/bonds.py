"""Module 16 — Obligations, forwards & futures (cours EFREI, chapitre 1.2).

Le cours décrit les obligations (coupons, prix inverse aux taux, risque de crédit)
et les dérivés à terme ; ce module en donne les formules de marché standard :

  • Prix d'une obligation à coupons : P = Σ c·N/f / (1+y/f)^k + N/(1+y/f)^n
  • Zéro-coupon, taux actuariel (rendement à l'échéance, YTM) par inversion
  • Duration de Macaulay, duration modifiée, sensibilité (DV01), convexité
  • Variation de prix pour un choc de taux : ΔP/P ≈ -D_mod·Δy + ½·C·Δy²
  • Spread de crédit (rendement obligation - rendement souverain)
  • Prix forward / futures : F = S·e^{(r - q)T} (q = dividende ou coût de portage)
    et valeur d'un forward déjà conclu
  • Lecture de marché via ETF : la duration d'un ETF obligataire se déduit de sa
    sensibilité empirique aux variations du taux 10 ans (^TNX)

Conventions : taux annuels en décimal (0.04 = 4 %), f = coupons par an.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def bond_cashflows(face: float, coupon_rate: float, years: float, freq: int = 1):
    n = max(int(round(years * freq)), 1)
    c = face * coupon_rate / freq
    times = np.arange(1, n + 1) / freq
    flows = np.full(n, c)
    flows[-1] += face
    return times, flows


def bond_price(face: float, coupon_rate: float, years: float, ytm: float, freq: int = 1) -> float:
    """Prix = valeur actuelle des coupons + du nominal au taux actuariel."""
    times, flows = bond_cashflows(face, coupon_rate, years, freq)
    return float(np.sum(flows / (1 + ytm / freq) ** (times * freq)))


def zero_coupon_price(face: float, years: float, rate: float, continuous: bool = False) -> float:
    return face * (math.exp(-rate * years) if continuous else (1 + rate) ** -years)


def ytm(price: float, face: float, coupon_rate: float, years: float, freq: int = 1,
        lo: float = -0.99, hi: float = 1.0) -> float:
    """Taux actuariel tel que bond_price(y) = prix (dichotomie : le prix baisse quand y monte)."""
    f = lambda y: bond_price(face, coupon_rate, years, y, freq) - price
    if f(lo) < 0 or f(hi) > 0:
        raise ValueError("prix hors de l'intervalle de taux [-99 %, 100 %]")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def durations(face: float, coupon_rate: float, years: float, y: float, freq: int = 1) -> dict:
    """Macaulay, modifiée, DV01 (variation de prix pour +1 pb) et convexité."""
    times, flows = bond_cashflows(face, coupon_rate, years, freq)
    disc = (1 + y / freq) ** (times * freq)
    pv = flows / disc
    price = float(pv.sum())
    macaulay = float((times * pv).sum() / price)
    modified = macaulay / (1 + y / freq)
    k = times * freq
    convexity = float((pv * k * (k + 1)).sum() / (price * (1 + y / freq) ** 2 * freq ** 2))
    return {"price": price, "macaulay": macaulay, "modified": modified,
            "dv01": modified * price * 1e-4, "convexity": convexity}


def price_change(modified: float, convexity: float, dy: float) -> float:
    """ΔP/P ≈ -D_mod·Δy + ½·C·Δy² (approximation de Taylor du cours de taux)."""
    return -modified * dy + 0.5 * convexity * dy ** 2


def credit_spread(bond_ytm: float, govt_ytm: float) -> float:
    return bond_ytm - govt_ytm


def forward_price(spot: float, r: float, T: float, q: float = 0.0) -> float:
    """Prix à terme sans arbitrage : F = S·e^{(r-q)T} (q : dividende/rendement de l'actif)."""
    return spot * math.exp((r - q) * T)


def forward_value(spot: float, K: float, r: float, T: float, q: float = 0.0, long: bool = True) -> float:
    """Valeur d'un forward existant de prix d'exercice K : S·e^{-qT} - K·e^{-rT}."""
    v = spot * math.exp(-q * T) - K * math.exp(-r * T)
    return v if long else -v


def empirical_duration(etf_close: pd.Series, yield_pct: pd.Series, window: int = 252) -> float:
    """Duration effective d'un ETF obligataire : -pente de ses rendements sur Δ(taux).

    yield_pct : taux en % (ex. ^TNX de Yahoo = 4.25 pour 4,25 %).
    """
    df = pd.DataFrame({"r": etf_close.pct_change(), "dy": yield_pct.diff() / 100}).dropna()
    df = df.tail(window)
    if len(df) < 30 or df["dy"].var() == 0:
        return float("nan")
    beta = np.cov(df["r"], df["dy"])[0, 1] / df["dy"].var()
    return float(-beta)


BOND_ETFS = {"SHY": "Trésor US 1-3 ans", "IEF": "Trésor US 7-10 ans",
             "TLT": "Trésor US 20+ ans", "TIP": "Trésor indexé inflation",
             "LQD": "Entreprises investment grade", "HYG": "Entreprises haut rendement"}


def report(loader, years: int = 2) -> str:
    lines = ["=" * 78, "MODULE 16 — OBLIGATIONS, FORWARDS & FUTURES (cours EFREI)", "=" * 78,
             "", "— Exemple : obligation 10 ans, coupon 4 %, nominal 1 000, taux 4,5 % —"]
    d = durations(1000, 0.04, 10, 0.045)
    lines.append(f"  prix {d['price']:,.2f} | duration Macaulay {d['macaulay']:.2f} ans | "
                 f"modifiée {d['modified']:.2f} | DV01 {d['dv01']:.3f} | convexité {d['convexity']:.1f}")
    for dy in (-0.01, 0.01, 0.02):
        lines.append(f"  choc de taux {dy * 1e4:+.0f} pb → prix {price_change(d['modified'], d['convexity'], dy) * 100:+.2f} %"
                     f" (exact {(bond_price(1000, 0.04, 10, 0.045 + dy) / d['price'] - 1) * 100:+.2f} %)")
    lines += ["", "— ETF obligataires : duration effective mesurée sur le marché (vs taux 10 ans) —"]
    try:
        tnx, src = loader("^TNX", years)
        if src == "synthetic":
            raise ValueError("pas de données réelles pour ^TNX")
        lines.append(f"  taux 10 ans US actuel : {float(tnx['Close'].iloc[-1]):.2f} %")
        for sym, label in BOND_ETFS.items():
            df, s = loader(sym, years)
            if s == "synthetic":
                continue
            dur = empirical_duration(df["Close"], tnx["Close"])
            shock = -dur * 0.01 * 100 if math.isfinite(dur) else float("nan")
            lines.append(f"  {sym:<4} {label:<30} duration ≈ {dur:5.1f} ans → +1 % de taux ≈ {shock:+.1f} %")
    except Exception as e:
        lines.append(f"  données de marché indisponibles : {str(e)[:80]}")
    S, r = 100.0, 0.04
    lines += ["", "— Forwards / futures —",
              f"  actif à 100, taux 4 %, 1 an : forward {forward_price(S, r, 1):.2f} "
              f"(avec 1,5 % de dividende : {forward_price(S, r, 1, 0.015):.2f})",
              "  Rappel : prix et taux varient en sens inverse ; plus la duration est longue,",
              "  plus l'obligation (ou l'ETF) est sensible aux mouvements de taux."]
    return "\n".join(lines)
