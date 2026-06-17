"""Mode LIVE — exécute un cycle de trading papier sur les vraies données de marché.

À lancer une fois par jour après la clôture US (~22h30 heure de Paris) :
    python -m quantlab live

Cycle :
  1. Télécharge les dernières données réelles (cache ignoré).
  2. Positions ouvertes : vérifie stop / target / signal de sortie → vente.
  3. Symboles plats : si une stratégie déclenche sur la dernière bougie ET que
     le régime de marché lui est favorable → achat (sizing en % de risque).
  4. Sauvegarde le compte (paper_account.json) et affiche le rapport.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from .broker import DEFAULT_STATE, Account, buy, load_account, save_account, sell
from .data import load
from .regime import detect
from .setups import _fits
from .strategies import ALL_STRATEGIES, get_strategy

DEFAULT_UNIVERSE = ["SPY", "QQQ", "GLD", "TLT",
                    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
                    "JPM", "V", "JNJ", "UNH", "XOM", "WMT", "HD", "BTC-USD"]


def run_cycle(symbols: list[str] = DEFAULT_UNIVERSE, capital: float = 10_000.0,
              risk_pct: float = 1.0, state_path: str = DEFAULT_STATE,
              loader=load, max_positions: int = 3) -> str:
    account = load_account(state_path, capital)
    lines = [
        "=" * 78,
        "MODE LIVE — CYCLE DE TRADING PAPIER (données de marché réelles)",
        f"Exécuté : {datetime.now().isoformat(timespec='seconds')} | "
        f"Univers : {', '.join(symbols)}",
        "=" * 78,
    ]

    data: dict[str, pd.DataFrame] = {}
    last_prices: dict[str, float] = {}
    for sym in symbols:
        df, src = loader(sym, years=2, fresh=True)
        if src == "synthetic":
            # Sécurité : jamais d'ordres sur des prix fictifs en mode live
            lines.append(f"  [!] {sym} : données réelles indisponibles → symbole "
                         "IGNORÉ ce cycle (aucun ordre sur prix synthétiques)")
            continue
        data[sym] = df
        last_prices[sym] = float(df["Close"].iloc[-1])

    equity_before = account.equity(last_prices)

    # ---- 1) Gestion des positions ouvertes ----
    for sym in list(account.positions):
        if sym not in data:
            continue
        df = data[sym]
        bar = df.iloc[-1]
        date = str(df.index[-1].date())
        pos = account.positions[sym]
        strat = get_strategy(pos["strategy"])
        exit_sig = bool(strat.generate_signals(df)["exit"].iloc[-1])

        if float(bar["Low"]) <= pos["stop"]:
            px = min(float(bar["Open"]), pos["stop"])
            t = sell(account, sym, px, date, "stop")
            lines.append(f"  VENTE {sym} @ {px:,.2f} (STOP) — PnL {t['pnl']:+,.2f} $ "
                         f"({t['pnl_pct']:+.1f}%)")
        elif pos["target"] and float(bar["High"]) >= pos["target"]:
            px = max(float(bar["Open"]), pos["target"])
            t = sell(account, sym, px, date, "target")
            lines.append(f"  VENTE {sym} @ {px:,.2f} (TARGET) — PnL {t['pnl']:+,.2f} $ "
                         f"({t['pnl_pct']:+.1f}%)")
        elif exit_sig:
            px = float(bar["Close"])
            t = sell(account, sym, px, date, "signal")
            lines.append(f"  VENTE {sym} @ {px:,.2f} (signal de sortie) — "
                         f"PnL {t['pnl']:+,.2f} $ ({t['pnl_pct']:+.1f}%)")

    # ---- 2) Nouvelles entrées (sauf filtre de risque géopolitique) ----
    from . import news as nw
    gauge = nw.risk_gauge(loader)
    risk_block = nw.blocks_new_entries(gauge)
    if gauge["vix"] is not None:
        lines.append(f"  Contexte risque : VIX {gauge['vix']} "
                     f"({int(gauge['vix_pct'] * 100)}e pct) → {gauge['level'].upper()}"
                     + ("  ⛔ nouvelles entrées bloquées (peur extrême)"
                        if risk_block else ""))
    for sym, df in data.items():
        if risk_block:
            break
        if sym in account.positions or len(account.positions) >= max_positions:
            continue
        regime = detect(df)
        date = str(df.index[-1].date())
        for key in ALL_STRATEGIES:
            strat = get_strategy(key)
            sig = strat.generate_signals(df).iloc[-1]
            if not bool(sig["entry"]) or not _fits(key, regime):
                continue
            price = float(df["Close"].iloc[-1])
            stop = float(sig["stop"])
            target = float(sig["target"]) if np.isfinite(sig["target"]) else None
            pos = buy(account, sym, price, stop, target, key, date,
                      risk_pct, equity=account.equity(last_prices))
            if pos:
                tgt = f"{target:,.2f}" if target else "suiveur"
                lines.append(
                    f"  ACHAT {sym} @ {price:,.2f} [{key}] — "
                    f"{pos['shares']:.4f} u | stop {stop:,.2f} | target {tgt} | "
                    f"risque {pos['risk_amount']:,.2f} $")
                break

    if not any(l.lstrip().startswith(("ACHAT", "VENTE")) for l in lines):
        lines.append("  Aucun ordre ce cycle (positions inchangées, pas de signal).")

    # ---- 3) État du compte ----
    account.last_cycle = datetime.now().isoformat(timespec="seconds")
    eq = account.equity(last_prices)
    # date de marché du cycle (dernière barre dispo), sinon date du jour
    mkt_date = (max(str(d.index[-1].date()) for d in data.values())
                if data else datetime.now().strftime("%Y-%m-%d"))
    account.record_equity(mkt_date, eq)
    save_account(account, state_path)
    lines += ["", account_summary(account, last_prices)]
    if abs(eq - equity_before) > 0.005:
        lines.append(f"Variation du cycle : {eq - equity_before:+,.2f} $")
    return "\n".join(lines)


def account_summary(account: Account, prices: dict[str, float] | None = None) -> str:
    prices = prices or {}
    eq = account.equity(prices)
    lines = [
        "— COMPTE PAPIER —",
        f"  Créé : {account.created} | Dernier cycle : {account.last_cycle or 'jamais'}",
        f"  Équité : {eq:,.2f} $ ({(eq / account.initial_capital - 1) * 100:+.2f}% "
        f"depuis le départ) | Cash : {account.cash:,.2f} $",
    ]
    if account.positions:
        lines.append("  Positions ouvertes :")
        for sym, p in account.positions.items():
            cur = prices.get(sym, p["entry"])
            upnl = (cur - p["entry"]) * p["shares"]
            lines.append(
                f"    {sym:<8} {p['shares']:.4f} u @ {p['entry']:,.2f} "
                f"[{p['strategy']}] | stop {p['stop']:,.2f} | "
                f"latent {upnl:+,.2f} $")
    else:
        lines.append("  Aucune position ouverte.")
    if account.history:
        pnls = [t["pnl"] for t in account.history]
        wins = sum(1 for x in pnls if x > 0)
        lines += [
            f"  Trades clôturés : {len(pnls)} | win rate {wins / len(pnls) * 100:.0f}% | "
            f"PnL réalisé total : {sum(pnls):+,.2f} $",
            "  5 derniers trades :",
        ]
        for t in account.history[-5:]:
            lines.append(
                f"    {t['exit_date']} {t['symbol']:<8} [{t['strategy']}] "
                f"{t['entry']:,.2f} → {t['exit']:,.2f} ({t['reason']}) "
                f"PnL {t['pnl']:+,.2f} $")
    return "\n".join(lines)


def account_report(state_path: str = DEFAULT_STATE,
                   symbols: list[str] = DEFAULT_UNIVERSE, loader=load) -> str:
    account = load_account(state_path)
    prices = {}
    for sym in set(list(account.positions) + symbols):
        try:
            df, _ = loader(sym, years=1, fresh=False)
            prices[sym] = float(df["Close"].iloc[-1])
        except Exception:
            pass
    return account_summary(account, prices)
