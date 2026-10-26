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
from .fund import TRAILING_STRATEGIES, vol_scale
from .regime import detect
from .setups import _fits
from .strategies import ALL_STRATEGIES, get_strategy

DEFAULT_UNIVERSE = ["SPY", "QQQ", "GLD", "TLT",
                    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB",
                    "BTC-USD"]


def dedupe_same_issuer(data: dict, candidates: list[str], keep: set[str] | None = None,
                       min_corr: float = 0.97, window: int = 120) -> list[str]:
    """Une seule ligne par émetteur : GOOG / GOOGL, BRK-A / BRK-B…

    Deux actions dont les rendements journaliers sont corrélés à plus de 0,97 sur
    ~6 mois sont la même société (classes d'actions) : on garde la première,
    pour ne pas doubler l'exposition. Basé sur les données, pas sur le nom."""
    kept = [k for k in (keep or set()) if k in data]
    out = []
    for s in candidates:
        if s not in data:
            continue
        r = data[s]["Close"].pct_change().tail(window)
        dup = False
        for k in kept + out:
            rk = data[k]["Close"].pct_change().tail(window)
            both = pd.concat([r, rk], axis=1).dropna()
            if len(both) >= 30 and both.corr().iloc[0, 1] >= min_corr:
                dup = True
                break
        if not dup:
            out.append(s)
    return out


def run_cycle(symbols: list[str] = DEFAULT_UNIVERSE, capital: float = 10_000.0,
              risk_pct: float = 1.0, state_path: str = DEFAULT_STATE,
              loader=load, max_positions: int = 3, screener=None,
              vol_target: float | None = None, vol_window: int = 20,
              trail: bool = False, after=None, screener_mode: str = "replace",
              max_screened_positions: int | None = None, earnings_guard=None,
              earnings_days: int = 5, core_pct: float = 0.0,
              sweep: str | None = None) -> str:
    """`after(account, last_prices) -> str` : action optionnelle en fin de cycle
    (ex. miroir des positions sur Alpaca). N'influence jamais les décisions."""
    account = load_account(state_path, capital)
    # Univers : si un screener FORWARD est fourni (ex. screener.candidates), on
    # trade ses candidats ∪ les positions ouvertes. Fail-safe : screener vide →
    # on garde l'univers fixe passé en argument. Réservé au live, jamais aux
    # backtests (cf. garde-fou d'intégrité dans screener.py).
    #   screener_mode="add" : univers fixe (ETF + BTC) ∪ actions screenées ∪ positions
    #   (le fonds trade alors ETF ET actions ; les ETF restent prioritaires).
    screened_set: set[str] = set()
    if screener is not None:
        screened = list(screener() or [])
        if screened:
            base = list(symbols) if screener_mode == "add" else []
            screened_set = set(screened) - set(base)
            symbols = list(dict.fromkeys(base + list(screened) + list(account.positions)))
    # positions ouvertes TOUJOURS suivies (stops), même si le screener ne répond pas
    symbols = list(dict.fromkeys(list(symbols) + list(account.positions)))
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

    if screened_set:   # classes d'actions d'un même émetteur : on n'en garde qu'une
        fixed = set(data) - screened_set
        uniq = set(dedupe_same_issuer(data, [s for s in data if s in screened_set],
                                      keep=fixed | set(account.positions)))
        dropped = [s for s in data if s in screened_set and s not in uniq
                   and s not in account.positions]
        for s in dropped:
            data.pop(s, None)
        if dropped:
            lines.append(f"  Doublons d'émetteur écartés : {', '.join(dropped)}")

    # Trésorerie (pas de cash qui dort) : prix du socle SPY et des bons du Trésor (BIL).
    # Ces actifs ne sont PAS ajoutés à `data` : aucune stratégie ne les trade.
    from . import treasury as tr
    use_treasury = core_pct > 0 or bool(sweep) or bool(account.treasury)
    if use_treasury:
        for sym in {tr.CORE_SYMBOL, sweep or tr.SWEEP_SYMBOL} | set(account.treasury):
            if sym not in last_prices:
                try:
                    df_t, src_t = loader(sym, years=2, fresh=True)
                    if src_t != "synthetic":
                        last_prices[sym] = float(df_t["Close"].iloc[-1])
                except Exception:
                    pass

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
        sigs = strat.generate_signals(df)
        exit_sig = bool(sigs["exit"].iloc[-1])

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
        elif trail and pos["strategy"] in TRAILING_STRATEGIES:
            # Stop suiveur en cliquet : remonte le stop sur la bande ATR close,
            # persisté pour le prochain cycle (ne descend jamais).
            new_stop = float(sigs["stop"].iloc[-1])
            if np.isfinite(new_stop) and new_stop > pos["stop"]:
                lines.append(f"  Stop suiveur {sym} : {pos['stop']:,.2f} → {new_stop:,.2f}")
                pos["stop"] = new_stop

    # ---- 2) Nouvelles entrées (sauf filtre de risque géopolitique) ----
    from . import news as nw
    gauge = nw.risk_gauge(loader)
    risk_block = nw.blocks_new_entries(gauge)
    if gauge["vix"] is not None:
        lines.append(f"  Contexte risque : VIX {gauge['vix']} "
                     f"({int(gauge['vix_pct'] * 100)}e pct) → {gauge['level'].upper()}"
                     + ("  ⛔ nouvelles entrées bloquées (peur extrême)"
                        if risk_block else ""))
    # Vol-targeting : module le risque par trade selon la vol réalisée du compte.
    scale = vol_scale([v for _, v in account.equity_curve], vol_target, vol_window)
    eff_risk = risk_pct * scale
    if vol_target is not None and abs(scale - 1.0) > 0.01:
        lines.append(f"  Vol-targeting : risque {risk_pct:.2f}% × {scale:.2f} "
                     f"= {eff_risk:.2f}% (cible {vol_target * 100:.0f}% de vol)")
    for sym, df in data.items():
        if risk_block:
            break
        if sym in account.positions or len(account.positions) >= max_positions:
            continue
        if sym in screened_set:
            n_stock = sum(1 for p in account.positions if p in screened_set)
            if max_screened_positions is not None and n_stock >= max_screened_positions:
                continue
            # Action individuelle : pas d'entrée juste avant ses résultats (risque de
            # gap qui saute le stop — le danger propre aux actions vs les ETF).
            if earnings_guard is not None:
                days = earnings_guard(sym)
                if days is not None and days <= earnings_days:
                    lines.append(f"  {sym} : résultats dans {days} j → pas d'entrée")
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
            eq_now = account.equity(last_prices)
            if account.treasury and price > stop:   # les stratégies passent avant la trésorerie
                need = eq_now * eff_risk / 100.0 / (price - stop) * price * 1.001
                lines += tr.raise_cash(account, need, last_prices)
            pos = buy(account, sym, price, stop, target, key, date,
                      eff_risk, equity=eq_now)
            if pos:
                tgt = f"{target:,.2f}" if target else "suiveur"
                lines.append(
                    f"  ACHAT {sym} @ {price:,.2f} [{key}] — "
                    f"{pos['shares']:.4f} u | stop {stop:,.2f} | target {tgt} | "
                    f"risque {pos['risk_amount']:,.2f} $")
                break

    if not any(l.lstrip().startswith(("ACHAT", "VENTE")) for l in lines):
        lines.append("  Aucun ordre ce cycle (positions inchangées, pas de signal).")

    if use_treasury:
        lines += tr.rebalance(account, last_prices, core_pct, sweep)

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
    if after is not None:
        lines.append(after(account, last_prices))
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
    if account.treasury:
        from . import treasury as tr
        lines.append(tr.summary_line(account, prices))
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
