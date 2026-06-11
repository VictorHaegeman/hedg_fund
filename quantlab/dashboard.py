"""Tableau de bord HTML — état de « mon argent » en un coup d'œil.

Génère dashboard.html (autonome, aucune dépendance JS/CSS externe) :
  • compte papier live : équité, positions, derniers trades
  • simulation d'entraînement du fonds : courbe d'équité, drawdown,
    métriques, rendements annuels, contributions
"""

from __future__ import annotations

import html
import os
from datetime import datetime

import numpy as np
import pandas as pd

from .backtest import BacktestResult, yearly_returns
from .broker import Account
from .drawdown import drawdown_series

# Publié via GitHub Pages (dossier docs/) : le dashboard est servi en ligne
OUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "index.html")


def _downsample(s: pd.Series, n: int = 700) -> pd.Series:
    if len(s) <= n:
        return s
    step = len(s) / n
    idx = (np.arange(n) * step).astype(int)
    return s.iloc[idx]


def _svg_area(s: pd.Series, w: int = 920, h: int = 260, color: str = "#4ade80",
              fmt: str = "{:,.0f} $", baseline: float | None = None) -> str:
    s = _downsample(s.dropna())
    lo, hi = float(s.min()), float(s.max())
    span = (hi - lo) or 1.0
    pad = span * 0.06
    lo, hi = lo - pad, hi + pad
    xs = np.linspace(45, w - 10, len(s))
    ys = (h - 22) - (s.to_numpy() - lo) / (hi - lo) * (h - 40)
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    area = f"{pts} {w - 10},{h - 22} 45,{h - 22}"
    # repères horizontaux
    grid = []
    for frac in (0.0, 0.5, 1.0):
        val = lo + (hi - lo) * frac
        y = (h - 22) - frac * (h - 40)
        grid.append(f'<line x1="45" y1="{y:.0f}" x2="{w - 10}" y2="{y:.0f}" '
                    f'stroke="#2a3447" stroke-width="1"/>'
                    f'<text x="40" y="{y + 4:.0f}" text-anchor="end" fill="#8b98ad" '
                    f'font-size="10">{fmt.format(val)}</text>')
    base = ""
    if baseline is not None and lo < baseline < hi:
        yb = (h - 22) - (baseline - lo) / (hi - lo) * (h - 40)
        base = (f'<line x1="45" y1="{yb:.0f}" x2="{w - 10}" y2="{yb:.0f}" '
                f'stroke="#facc15" stroke-dasharray="5,4" stroke-width="1.2"/>')
    d0, d1 = s.index[0], s.index[-1]
    labels = (f'<text x="45" y="{h - 6}" fill="#8b98ad" font-size="10">{d0:%Y-%m-%d}</text>'
              f'<text x="{w - 10}" y="{h - 6}" text-anchor="end" fill="#8b98ad" '
              f'font-size="10">{d1:%Y-%m-%d}</text>')
    return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">'
            f'{"".join(grid)}{base}'
            f'<polygon points="{area}" fill="{color}" opacity="0.12"/>'
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.8"/>'
            f'{labels}</svg>')


def _card(label: str, value: str, sub: str = "", color: str = "#e7ecf3") -> str:
    return (f'<div class="card"><div class="lbl">{html.escape(label)}</div>'
            f'<div class="val" style="color:{color}">{html.escape(value)}</div>'
            f'<div class="sub">{html.escape(sub)}</div></div>')


def _table(df: pd.DataFrame) -> str:
    return df.to_html(border=0, classes="tbl", justify="left", escape=True)


def _pos_color(x: float) -> str:
    return "#4ade80" if x >= 0 else "#f87171"


def render_html(fund_res: BacktestResult, account: Account,
                prices_now: dict[str, float], capital_sim: float) -> str:
    m = fund_res.metrics
    eq_live = account.equity(prices_now)
    pnl_live = eq_live - account.initial_capital
    pnl_live_pct = (eq_live / account.initial_capital - 1) * 100

    dd = drawdown_series(fund_res.equity)
    yr = yearly_returns(fund_res.equity)
    yr_bars = "".join(
        f'<div class="yr"><div class="yrv" style="color:{_pos_color(v)}">{v:+.1f}%</div>'
        f'<div class="yrl">{y}</div></div>' for y, v in yr.items())

    # Tables compte papier
    if account.positions:
        pos_rows = []
        for sym, p in account.positions.items():
            cur = prices_now.get(sym, p["entry"])
            pos_rows.append({
                "Actif": sym, "Stratégie": p["strategy"],
                "Unités": round(p["shares"], 4), "Entrée": round(p["entry"], 2),
                "Cours": round(cur, 2), "Stop": round(p["stop"], 2),
                "PnL latent $": round((cur - p["entry"]) * p["shares"], 2),
            })
        positions_html = _table(pd.DataFrame(pos_rows))
    else:
        positions_html = "<p class='muted'>Aucune position ouverte — le système attend un signal.</p>"

    if account.history:
        ht = pd.DataFrame(account.history[-10:])[
            ["exit_date", "symbol", "strategy", "entry", "exit", "reason", "pnl"]]
        ht.columns = ["Date", "Actif", "Stratégie", "Entrée", "Sortie", "Motif", "PnL $"]
        history_html = _table(ht)
    else:
        history_html = "<p class='muted'>Aucun trade clôturé pour l'instant.</p>"

    contrib_html = ""
    if len(fund_res.trades):
        by_sym = fund_res.trades.groupby("symbol")["pnl"].agg(["count", "sum"]).round(0)
        by_sym.columns = ["Trades", "PnL $"]
        by_strat = fund_res.trades.groupby("strategy")["pnl"].agg(["count", "sum"]).round(0)
        by_strat.columns = ["Trades", "PnL $"]
        contrib_html = (f'<div class="cols"><div><h3>Par actif</h3>{_table(by_sym)}</div>'
                        f'<div><h3>Par stratégie</h3>{_table(by_strat)}</div></div>')

    gen = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hedge Fund — Tableau de bord</title>
<style>
  body {{ background:#0d1320; color:#e7ecf3; font-family:'Segoe UI',system-ui,sans-serif;
         margin:0; padding:24px; }}
  h1 {{ font-size:22px; margin:0 0 4px; }} h2 {{ font-size:16px; margin:28px 0 10px;
       color:#aebacd; text-transform:uppercase; letter-spacing:.08em; }}
  h3 {{ font-size:13px; color:#8b98ad; margin:8px 0; }}
  .muted {{ color:#8b98ad; }} .gen {{ color:#8b98ad; font-size:12px; }}
  .cards {{ display:flex; flex-wrap:wrap; gap:12px; margin-top:14px; }}
  .card {{ background:#161e2e; border:1px solid #2a3447; border-radius:10px;
           padding:12px 18px; min-width:140px; }}
  .lbl {{ font-size:11px; color:#8b98ad; text-transform:uppercase; letter-spacing:.06em; }}
  .val {{ font-size:22px; font-weight:600; margin-top:2px; }}
  .sub {{ font-size:11px; color:#8b98ad; margin-top:2px; }}
  .panel {{ background:#161e2e; border:1px solid #2a3447; border-radius:10px;
            padding:14px; margin-top:10px; }}
  svg {{ width:100%; height:auto; display:block; }}
  .yrwrap {{ display:flex; gap:8px; flex-wrap:wrap; margin-top:8px; }}
  .yr {{ background:#161e2e; border:1px solid #2a3447; border-radius:8px;
         padding:8px 12px; text-align:center; }}
  .yrv {{ font-weight:600; font-size:14px; }} .yrl {{ font-size:11px; color:#8b98ad; }}
  .tbl {{ border-collapse:collapse; width:100%; font-size:13px; }}
  .tbl th, .tbl td {{ padding:6px 10px; text-align:left; border-bottom:1px solid #2a3447; }}
  .tbl th {{ color:#8b98ad; font-weight:600; }}
  .cols {{ display:flex; gap:24px; flex-wrap:wrap; }} .cols > div {{ flex:1; min-width:260px; }}
  .warn {{ color:#facc15; font-size:12px; margin-top:18px; }}
</style></head><body>
<h1>🏦 Hedge Fund — Tableau de bord</h1>
<div class="gen">Généré le {gen} · données réelles Yahoo Finance · paper trading</div>

<h2>Mon argent — compte papier (marché réel)</h2>
<div class="cards">
  {_card("Équité actuelle", f"{eq_live:,.2f} $", "valorisée aux derniers cours",
         _pos_color(pnl_live))}
  {_card("PnL depuis le départ", f"{pnl_live:+,.2f} $", f"{pnl_live_pct:+.2f} %",
         _pos_color(pnl_live))}
  {_card("Cash disponible", f"{account.cash:,.2f} $")}
  {_card("Positions ouvertes", f"{len(account.positions)}")}
  {_card("Trades clôturés", f"{len(account.history)}")}
</div>
<div class="panel">{positions_html}</div>
<h3>Derniers trades clôturés</h3>
<div class="panel">{history_html}</div>

<h2>Entraînement — simulation du fonds sur {m['years']:.0f} ans ({capital_sim:,.0f} $ de départ)</h2>
<div class="cards">
  {_card("Équité finale simulée", f"{m['final_equity']:,.0f} $",
         f"{m['total_return'] * 100:+.1f} % au total", _pos_color(m['total_return']))}
  {_card("CAGR", f"{m['cagr'] * 100:+.2f} %/an", "", _pos_color(m['cagr']))}
  {_card("Sharpe", f"{m['sharpe']:.2f}")}
  {_card("Max drawdown", f"{m['max_drawdown'] * 100:.1f} %", "", "#f87171")}
  {_card("Win rate", f"{m['win_rate'] * 100:.1f} %", f"{m['n_trades']} trades")}
  {_card("Profit factor", f"{m['profit_factor']:.2f}")}
</div>
<h3>Courbe d'équité du fonds (simulation)</h3>
<div class="panel">{_svg_area(fund_res.equity, baseline=capital_sim)}</div>
<h3>Drawdown (%)</h3>
<div class="panel">{_svg_area(dd * 100, h=150, color="#f87171", fmt="{:.1f} %")}</div>
<h3>Rendements annuels</h3>
<div class="yrwrap">{yr_bars}</div>
{contrib_html}

<div class="warn">⚠ Capital fictif — outil d'entraînement. Performance simulée/passée ≠ performance future.
Aucun ordre réel n'est passé.</div>
</body></html>"""


def save(fund_res: BacktestResult, account: Account, prices_now: dict[str, float],
         capital_sim: float, path: str = OUT_PATH) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_html(fund_res, account, prices_now, capital_sim))
    return path
