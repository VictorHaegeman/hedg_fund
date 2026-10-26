"""Module 15 — Pricing d'options & stratégies (cours EFREI « Introduction to Financial Markets »).

Formules du cours, implémentées et testées :
  • Payoffs, P&L acheteur/vendeur, points morts, gains/pertes max
  • Black-Scholes : d1, d2, call, put, parité call-put, bornes d'arbitrage,
    probabilité d'exercice Q(S_T > K) = N(d2), options digitales
  • Modèle de Bachelier (dynamique arithmétique S_t = S_0 + σW_t)
  • Grecques : Δ, Γ, ν (vega), Θ, ρ — et la relation Θ = -½σ²S²Γ - rSΔ + rV
  • Volatilité réalisée annualisée (log-rendements, facteur 252) et implicite
    (inversion de Black-Scholes)
  • P&L d'une option delta-hedgée : dΠ = ½ΓS²(σ_réal² - σ_impl²)dt
  • Monte-Carlo : estimateur, erreur standard, variables antithétiques,
    variable de contrôle ; options exotiques (asiatique, barrière) ; Heston
  • Stratégies : bull spread, straddle, strangle, butterfly, risk reversal
    (payoff, prime nette, points morts, gain/perte max, grecques agrégées)
  • Analyse statistique des rendements (projet Bloom) : moments, asymétrie,
    kurtosis, test de normalité de Shapiro-Wilk, corrélations

Conventions : T en années, r et σ annualisés, prix par unité de sous-jacent
(un contrat d'option US = 100 unités).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sps

N = sps.norm.cdf
n_pdf = sps.norm.pdf
TRADING_DAYS = 252


# ── 1. Payoffs & P&L ────────────────────────────────────────────────────────
def payoff(kind: str, S_T, K: float):
    """Payoff à l'échéance : call (S_T-K)+, put (K-S_T)+."""
    S_T = np.asarray(S_T, dtype=float)
    return np.maximum(S_T - K, 0.0) if kind == "call" else np.maximum(K - S_T, 0.0)


def breakeven(kind: str, K: float, premium: float) -> float:
    """Point mort : call K + p, put K - p."""
    return K + premium if kind == "call" else K - premium


# ── 2. Black-Scholes ────────────────────────────────────────────────────────
def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return d1(S, K, T, r, sigma) - sigma * math.sqrt(T)


def bs_price(S: float, K: float, T: float, r: float, sigma: float, kind: str = "call") -> float:
    """C0 = S N(d1) - K e^{-rT} N(d2) ; P0 = K e^{-rT} N(-d2) - S N(-d1)."""
    if T <= 0 or sigma <= 0:   # à l'échéance (ou sans vol) : valeur intrinsèque actualisée
        fwd = S - K * math.exp(-r * max(T, 0.0))
        return max(fwd, 0.0) if kind == "call" else max(-fwd, 0.0)
    a, b = d1(S, K, T, r, sigma), d2(S, K, T, r, sigma)
    if kind == "call":
        return S * N(a) - K * math.exp(-r * T) * N(b)
    return K * math.exp(-r * T) * N(-b) - S * N(-a)


def prob_itm(S: float, K: float, T: float, r: float, sigma: float, kind: str = "call") -> float:
    """Probabilité risque-neutre d'exercice : Q(S_T > K) = N(d2) (put : N(-d2))."""
    b = d2(S, K, T, r, sigma)
    return float(N(b) if kind == "call" else N(-b))


def digital_price(S: float, K: float, T: float, r: float, sigma: float, kind: str = "call") -> float:
    """Digitale 1{S_T>K} : e^{-rT} N(d2) ; put : e^{-rT} N(-d2)."""
    return math.exp(-r * T) * prob_itm(S, K, T, r, sigma, kind)


def parity_gap(C: float, P: float, S: float, K: float, T: float, r: float) -> float:
    """Écart à la parité call-put C - P - (S - K e^{-rT}) ; ≠ 0 → arbitrage théorique."""
    return C - P - (S - K * math.exp(-r * T))


def call_bounds(S: float, K: float, T: float, r: float) -> tuple[float, float]:
    """Bornes d'arbitrage : max(S - K e^{-rT}, 0) ≤ C ≤ S."""
    return max(S - K * math.exp(-r * T), 0.0), S


def bachelier_price(S: float, K: float, T: float, sigma_abs: float, kind: str = "call",
                    r: float = 0.0) -> float:
    """Modèle de Bachelier (S_T = S_0 + σ W_T, σ en unités de prix) : formule normale."""
    sd = sigma_abs * math.sqrt(T)
    if sd <= 0:
        return max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)
    d = (S - K) / sd
    call = (S - K) * N(d) + sd * n_pdf(d)
    price = call if kind == "call" else call - (S - K)
    return math.exp(-r * T) * price


# ── 3. Grecques ─────────────────────────────────────────────────────────────
@dataclass
class Greeks:
    delta: float
    gamma: float
    vega: float      # par point de volatilité absolu (σ += 1.00) ; /100 → par point de %
    theta: float     # par an ; /365 → par jour
    rho: float

    def __add__(self, o: "Greeks") -> "Greeks":
        return Greeks(*(a + b for a, b in zip(self.as_tuple(), o.as_tuple())))

    def scale(self, k: float) -> "Greeks":
        return Greeks(*(k * a for a in self.as_tuple()))

    def as_tuple(self):
        return self.delta, self.gamma, self.vega, self.theta, self.rho


def greeks(S: float, K: float, T: float, r: float, sigma: float, kind: str = "call") -> Greeks:
    a, b = d1(S, K, T, r, sigma), d2(S, K, T, r, sigma)
    disc = math.exp(-r * T)
    gamma = n_pdf(a) / (S * sigma * math.sqrt(T))
    vega = S * n_pdf(a) * math.sqrt(T)
    common = -S * sigma * n_pdf(a) / (2 * math.sqrt(T))
    if kind == "call":
        return Greeks(N(a), gamma, vega, common - r * K * disc * N(b), K * T * disc * N(b))
    return Greeks(N(a) - 1, gamma, vega, common + r * K * disc * N(-b), -K * T * disc * N(-b))


def theta_from_pde(S: float, r: float, sigma: float, V: float, g: Greeks) -> float:
    """Relation thêta-gamma-delta issue de l'EDP : Θ = -½σ²S²Γ - rSΔ + rV."""
    return -0.5 * sigma ** 2 * S ** 2 * g.gamma - r * S * g.delta + r * V


def gamma_pnl(gamma: float, S: float, sigma_real: float, sigma_impl: float, dt: float) -> float:
    """P&L d'une option delta-hedgée : ½ΓS²(σ_réal² - σ_impl²)dt.

    Long gamma gagne si la vol réalisée dépasse la vol implicite payée.
    """
    return 0.5 * gamma * S ** 2 * (sigma_real ** 2 - sigma_impl ** 2) * dt


# ── 4. Volatilité ───────────────────────────────────────────────────────────
def realized_vol(close: pd.Series, n: int | None = 20, periods: int = TRADING_DAYS) -> float:
    """σ_réal = sqrt(252/(N-1) Σ (r_i - r̄)²), r_i = ln(S_i/S_{i-1})."""
    r = np.log(close / close.shift(1)).dropna()
    if n:
        r = r.tail(n)
    if len(r) < 2:
        return float("nan")
    return float(math.sqrt(periods / (len(r) - 1) * ((r - r.mean()) ** 2).sum()))


def implied_vol(price: float, S: float, K: float, T: float, r: float, kind: str = "call",
                lo: float = 1e-4, hi: float = 5.0, tol: float = 1e-8) -> float | None:
    """σ_impl tel que C_BS(σ) = prix de marché (dichotomie : C_BS croît avec σ).

    None si le prix viole les bornes d'arbitrage (aucune vol ne le reproduit).
    """
    intrinsic = bs_price(S, K, T, r, 1e-12, kind)
    upper = S if kind == "call" else K * math.exp(-r * T)
    if not (intrinsic - 1e-9 <= price < upper):
        return None
    f = lambda s: bs_price(S, K, T, r, s, kind) - price
    if f(lo) > 0 or f(hi) < 0:
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


# ── 5. Monte-Carlo ──────────────────────────────────────────────────────────
@dataclass
class MCResult:
    price: float
    std_error: float
    n: int

    @property
    def ci95(self) -> tuple[float, float]:
        return self.price - 1.96 * self.std_error, self.price + 1.96 * self.std_error


def simulate_terminal(S: float, T: float, r: float, sigma: float, z: np.ndarray) -> np.ndarray:
    """S_T = S_0 exp[(r - σ²/2)T + σ√T Z] (log-normalité sous Q)."""
    return S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * math.sqrt(T) * z)


def mc_european(S: float, K: float, T: float, r: float, sigma: float, kind: str = "call",
                n: int = 100_000, seed: int = 0, antithetic: bool = False,
                control_variate: bool = False) -> MCResult:
    """Prix MC d'une européenne ; réductions de variance du cours en option.

    • antithétique : (f(Z) + f(-Z))/2
    • variable de contrôle : Y = e^{-rT} S_T, E[Y] = S_0, c* = Cov(X,Y)/Var(Y)
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    disc = math.exp(-r * T)
    x = disc * payoff(kind, simulate_terminal(S, T, r, sigma, z), K)
    if antithetic:
        x = 0.5 * (x + disc * payoff(kind, simulate_terminal(S, T, r, sigma, -z), K))
    if control_variate:
        y = disc * simulate_terminal(S, T, r, sigma, z)
        c = np.cov(x, y)[0, 1] / np.var(y, ddof=1)
        x = x - c * (y - S)
    return MCResult(float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x))), len(x))


def simulate_paths(S: float, T: float, r: float, sigma: float, n_steps: int, n_paths: int,
                   seed: int = 0) -> np.ndarray:
    """Trajectoires GBM (schéma exact par pas) : tableau (n_paths, n_steps+1)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    inc = (r - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * rng.standard_normal((n_paths, n_steps))
    return S * np.exp(np.hstack([np.zeros((n_paths, 1)), np.cumsum(inc, axis=1)]))


def mc_path_option(payoff_fn, S: float, T: float, r: float, sigma: float, n_steps: int = 100,
                   n_paths: int = 20_000, seed: int = 0) -> MCResult:
    """Options exotiques dépendant du chemin (asiatique, barrière…)."""
    paths = simulate_paths(S, T, r, sigma, n_steps, n_paths, seed)
    x = math.exp(-r * T) * np.apply_along_axis(payoff_fn, 1, paths)
    return MCResult(float(x.mean()), float(x.std(ddof=1) / math.sqrt(n_paths)), n_paths)


def asian_call_payoff(K: float):
    return lambda path: max(path[1:].mean() - K, 0.0)


def up_and_out_call_payoff(K: float, barrier: float):
    return lambda path: 0.0 if path.max() >= barrier else max(path[-1] - K, 0.0)


def heston_mc(S: float, K: float, T: float, r: float, v0: float, kappa: float, theta: float,
              xi: float, rho: float, kind: str = "call", n_steps: int = 200,
              n_paths: int = 20_000, seed: int = 0) -> MCResult:
    """Modèle de Heston (vol stochastique), Euler avec troncature de la variance :
    dS = rS dt + √v S dW^S ; dv = κ(θ - v)dt + ξ√v dW^v ; Corr = ρ."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    s = np.full(n_paths, math.log(S))
    v = np.full(n_paths, v0)
    for _ in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rho * z1 + math.sqrt(1 - rho ** 2) * rng.standard_normal(n_paths)
        vp = np.maximum(v, 0.0)
        s += (r - 0.5 * vp) * dt + np.sqrt(vp * dt) * z1
        v += kappa * (theta - vp) * dt + xi * np.sqrt(vp * dt) * z2
    x = math.exp(-r * T) * payoff(kind, np.exp(s), K)
    return MCResult(float(x.mean()), float(x.std(ddof=1) / math.sqrt(n_paths)), n_paths)


# ── 6. Stratégies ───────────────────────────────────────────────────────────
@dataclass
class Leg:
    kind: str            # "call" | "put"
    K: float
    qty: float           # +1 achat, -1 vente (ex. butterfly : +1, -2, +1)
    premium: float | None = None   # prix payé/reçu ; None → prix BS


@dataclass
class StrategyReport:
    name: str
    legs: list[Leg]
    net_premium: float            # Σ primes achetées - Σ primes vendues (>0 : débit)
    breakevens: list[float]
    max_gain: float               # inf si illimité
    max_loss: float               # inf si illimitée
    greeks: Greeks
    prob_profit: float            # probabilité (sous la loi utilisée) de finir en gain
    pnl_grid: pd.DataFrame = field(repr=False, default=None)


def strategy_payoff(legs: list[Leg], S_T):
    S_T = np.asarray(S_T, dtype=float)
    return sum(l.qty * payoff(l.kind, S_T, l.K) for l in legs)


def make_strategy(name: str, S: float, K: float | None = None, K1: float | None = None,
                  K2: float | None = None, K3: float | None = None) -> list[Leg]:
    """Les 5 stratégies du cours (chapitre 5.5)."""
    K = K or S
    if name == "bull_spread":       # (S-K1)+ - (S-K2)+
        return [Leg("call", K1, 1), Leg("call", K2, -1)]
    if name == "straddle":          # |S - K|
        return [Leg("call", K, 1), Leg("put", K, 1)]
    if name == "strangle":          # (K1-S)+ + (S-K2)+
        return [Leg("put", K1, 1), Leg("call", K2, 1)]
    if name == "butterfly":         # (S-K1)+ - 2(S-K2)+ + (S-K3)+
        return [Leg("call", K1, 1), Leg("call", K2, -2), Leg("call", K3, 1)]
    if name == "risk_reversal":     # (S-K2)+ - (K1-S)+
        return [Leg("call", K2, 1), Leg("put", K1, -1)]
    raise ValueError(f"stratégie inconnue : {name}")


def analyze_strategy(name: str, legs: list[Leg], S: float, T: float, r: float, sigma: float,
                     sigma_real: float | None = None) -> StrategyReport:
    """Prime nette, points morts, extrêmes, grecques, probabilité de gain.

    La probabilité de gain est évaluée sous une loi log-normale de vol
    `sigma_real` (vol réalisée si fournie, sinon σ) : c'est la vol que l'on
    ANTICIPE, à comparer à la vol implicite payée (cf. P&L de gamma).
    """
    prem = [l.premium if l.premium is not None else bs_price(S, l.K, T, r, sigma, l.kind)
            for l in legs]
    net = float(sum(l.qty * p for l, p in zip(legs, prem)))
    g = Greeks(0, 0, 0, 0, 0)
    for l in legs:
        g = g + greeks(S, l.K, T, r, sigma, l.kind).scale(l.qty)

    strikes = sorted({l.K for l in legs})
    grid = np.unique(np.concatenate([np.linspace(0, strikes[-1] * 3, 3001), strikes]))
    pnl = strategy_payoff(legs, grid) - net
    # pente au-delà du plus grand strike → gain/perte illimité(e)
    slope_up = sum(l.qty for l in legs if l.kind == "call")
    max_gain = math.inf if slope_up > 0 else float(pnl.max())
    max_loss = math.inf if slope_up < 0 else float(-min(pnl.min(), 0.0))
    sign = np.sign(pnl)
    bes = []
    for i in range(len(grid) - 1):
        if sign[i] != sign[i + 1] and sign[i] != 0:
            x0, x1, y0, y1 = grid[i], grid[i + 1], pnl[i], pnl[i + 1]
            bes.append(round(float(x0 - y0 * (x1 - x0) / (y1 - y0)), 4))

    vol = sigma_real if sigma_real else sigma
    z = np.random.default_rng(1).standard_normal(200_000)
    st = simulate_terminal(S, T, r, vol, z)
    p_win = float((strategy_payoff(legs, st) - net * math.exp(r * T) > 0).mean())

    scen = np.array([0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]) * S
    pnl_grid = pd.DataFrame({"S_T": scen.round(2),
                             "P&L": (strategy_payoff(legs, scen) - net).round(2)})
    return StrategyReport(name, legs, net, bes, max_gain, max_loss, g, p_win, pnl_grid)


# ── 7. Statistiques des rendements (projet Bloom) ───────────────────────────
def return_stats(close: pd.Series) -> dict:
    """Moments des rendements journaliers + test de normalité de Shapiro-Wilk."""
    r = close.pct_change().dropna()
    sample = r.tail(5000)  # Shapiro est conçu pour n ≤ 5000
    p = float(sps.shapiro(sample).pvalue) if len(sample) >= 3 else float("nan")
    return {"mean_daily": float(r.mean()), "std_daily": float(r.std()),
            "ann_return": float((1 + r.mean()) ** TRADING_DAYS - 1),
            "realized_vol": realized_vol(close, n=None),
            "skew": float(r.skew()), "excess_kurtosis": float(r.kurt()),
            "shapiro_p": p, "normal": bool(p > 0.05)}


def correlation_matrix(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Corrélations de Pearson des rendements journaliers (dates communes)."""
    closes = pd.DataFrame({s: df["Close"] for s, df in prices.items()}).dropna()
    return closes.pct_change().dropna().corr()


# ── Rapport CLI ─────────────────────────────────────────────────────────────
STRATEGY_STRIKES = {   # strikes par défaut, en fraction du spot
    "bull_spread": dict(K1=1.00, K2=1.05),
    "straddle": dict(K=1.00),
    "strangle": dict(K1=0.95, K2=1.05),
    "butterfly": dict(K1=0.95, K2=1.00, K3=1.05),
    "risk_reversal": dict(K1=0.95, K2=1.05),
}


def _fmt(x: float) -> str:
    return "illimité(e)" if math.isinf(x) else f"{x:,.2f}"


def report(df: pd.DataFrame, symbol: str, source: str, days: int = 30,
           r: float = 0.04, sigma_impl: float | None = None) -> str:
    S = float(df["Close"].iloc[-1])
    T = days / 365
    sig_r = realized_vol(df["Close"], 20)
    sig = sigma_impl or sig_r
    st = return_stats(df["Close"])
    lines = [
        "=" * 78,
        f"MODULE 15 — PRICING D'OPTIONS & STRATÉGIES (cours EFREI) — {symbol} [{source}]",
        "=" * 78,
        f"Spot {S:,.2f} | échéance {days} j | r = {r:.2%} | vol réalisée 20 j "
        f"{sig_r:.1%} | vol de pricing {sig:.1%}"
        + ("" if sigma_impl else " (= réalisée, faute de vol implicite)"),
        "",
        "— Rendements (projet Bloom) —",
        f"  rendement annualisé {st['ann_return']:+.1%} | vol réalisée {st['realized_vol']:.1%} | "
        f"asymétrie {st['skew']:+.2f} | kurtosis excédentaire {st['excess_kurtosis']:.2f}",
        f"  Shapiro-Wilk p = {st['shapiro_p']:.3g} → "
        + ("rendements compatibles avec la loi normale" if st["normal"] else
           "rendements NON normaux (queues épaisses) : Black-Scholes sous-estime les extrêmes"),
        "",
        "— Option à la monnaie (Black-Scholes) —",
    ]
    for kind in ("call", "put"):
        p = bs_price(S, S, T, r, sig, kind)
        g = greeks(S, S, T, r, sig, kind)
        lines.append(
            f"  {kind:<4} K={S:,.2f} prix {p:,.2f} | point mort {breakeven(kind, S, p):,.2f} | "
            f"P(exercice) {prob_itm(S, S, T, r, sig, kind):.0%} | Δ {g.delta:+.2f} "
            f"Γ {g.gamma:.4f} ν {g.vega / 100:.2f}/pt Θ {g.theta / 365:+.3f}/j ρ {g.rho / 100:+.3f}")
    c, pt = bs_price(S, S, T, r, sig), bs_price(S, S, T, r, sig, "put")
    mc = mc_european(S, S, T, r, sig, n=50_000, antithetic=True, control_variate=True)
    lines += [f"  Parité call-put : écart {parity_gap(c, pt, S, S, T, r):+.2e} (≈ 0 attendu)",
              f"  Monte-Carlo (antithétique + contrôle) : {mc.price:,.3f} ± {1.96 * mc.std_error:.3f}"
              f" vs Black-Scholes {c:,.3f}", "", "— Stratégies du cours —"]
    for name, ks in STRATEGY_STRIKES.items():
        legs = make_strategy(name, S, **{k: v * S for k, v in ks.items()})
        rep = analyze_strategy(name, legs, S, T, r, sig, sig_r)
        lines.append(
            f"  {name:<14} prime nette {rep.net_premium:+,.2f} | points morts "
            f"{', '.join(f'{b:,.2f}' for b in rep.breakevens) or '—'} | gain max "
            f"{_fmt(rep.max_gain)} | perte max {_fmt(rep.max_loss)} | Δ {rep.greeks.delta:+.2f} "
            f"ν {rep.greeks.vega / 100:+.2f} | P(gain) {rep.prob_profit:.0%}")
    lines += ["", "Rappel du cours : une position long gamma delta-hedgée gagne "
              "½ΓS²(σ_réal² − σ_impl²) par unité de temps → acheter de la vol n'a de "
              "sens que si l'on anticipe σ_réal > σ_impl."]
    return "\n".join(lines)
