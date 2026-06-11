"""CLI quantlab — point d'entrée des 10 modules.

Exemples :
  python -m quantlab strategies
  python -m quantlab backtest   --strategy trend --symbol SPY --years 10
  python -m quantlab riskreward --strategy meanrev --symbol QQQ
  python -m quantlab regime     --symbol BTC-USD
  python -m quantlab multifactor --symbols SPY QQQ GLD TLT BTC-USD
  python -m quantlab optimize   --strategy breakout --symbol SPY
  python -m quantlab portfolio  --symbols SPY QQQ GLD TLT --risk-tolerance medium
  python -m quantlab setups     --symbols SPY QQQ BTC-USD
  python -m quantlab montecarlo --strategy trend --symbol SPY
  python -m quantlab drawdown   --strategy trend --symbol SPY
  python -m quantlab macro
  python -m quantlab alpha      --symbol SPY
  python -m quantlab all        --symbol SPY
"""

from __future__ import annotations

import argparse
import sys

from . import alpha as al
from . import backtest as bt
from . import drawdown as dd
from . import macro as mcr
from . import montecarlo as mc
from . import multifactor as mf
from . import optimize as opt
from . import portfolio as pf
from . import regime as rg
from . import risk as rk
from . import setups as st
from .data import load
from .strategies import describe_all, get_strategy

DEFAULT_UNIVERSE = ["SPY", "QQQ", "GLD", "TLT", "BTC-USD"]


def _load_many(symbols: list[str], years: int):
    prices, sources = {}, {}
    for sym in symbols:
        df, src = load(sym, years)
        prices[sym], sources[sym] = df, src
        if src == "synthetic":
            print(f"  [!] {sym} : données Yahoo indisponibles → données SYNTHÉTIQUES "
                  "(démo hors-ligne)", file=sys.stderr)
    return prices, sources


def _run_bt(args):
    df, src = load(args.symbol, args.years)
    strat = get_strategy(args.strategy)
    return bt.run_backtest(df, strat, args.capital, args.risk, symbol=args.symbol,
                           source=src), df, src


def main(argv: list[str] | None = None) -> int:
    # Console Windows souvent en cp1252 : forcer UTF-8 pour les caractères → █ é
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
    p = argparse.ArgumentParser(prog="quantlab",
                                description="Boîte à outils quant — 10 modules de trading")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, strategy=False, symbol=True, symbols=False):
        if strategy:
            sp.add_argument("--strategy", default="trend",
                            choices=["trend", "meanrev", "breakout"])
        if symbol:
            sp.add_argument("--symbol", default="SPY")
        if symbols:
            sp.add_argument("--symbols", nargs="+", default=DEFAULT_UNIVERSE)
        sp.add_argument("--years", type=int, default=10)
        sp.add_argument("--capital", type=float, default=10_000.0)
        sp.add_argument("--risk", type=float, default=1.0,
                        help="risque par trade en %% de l'équité")

    common(sub.add_parser("strategies", help="Module 1 — génération de stratégies"),
           symbol=False)
    common(sub.add_parser("backtest", help="Module 2 — backtesting"), strategy=True)
    common(sub.add_parser("riskreward", help="Module 3 — analyse risque/rendement"),
           strategy=True)
    common(sub.add_parser("regime", help="Module 4 — régime de marché"))
    common(sub.add_parser("multifactor", help="Module 5 — multi-facteurs"),
           symbol=False, symbols=True)
    common(sub.add_parser("optimize", help="Module 6 — optimisation"), strategy=True)
    sp = sub.add_parser("portfolio", help="Module 7 — portefeuille")
    common(sp, symbol=False, symbols=True)
    sp.add_argument("--risk-tolerance", default="medium", choices=["low", "medium", "high"])
    sp.add_argument("--horizon", type=int, default=3, help="horizon en années")
    common(sub.add_parser("setups", help="Module 8 — trade setups"),
           symbol=False, symbols=True)
    sp = sub.add_parser("montecarlo", help="Module 9 — Monte Carlo")
    common(sp, strategy=True)
    sp.add_argument("--paths", type=int, default=5_000)
    common(sub.add_parser("drawdown", help="Module 10 — drawdowns"), strategy=True)
    common(sub.add_parser("macro", help="Module 11 — stratégie macro"), symbol=False)
    common(sub.add_parser("alpha", help="Module 12 — détection d'alpha/edge"))
    common(sub.add_parser("all", help="Exécute les 12 modules"), strategy=True)

    args = p.parse_args(argv)

    if args.cmd == "strategies":
        print(describe_all(args.capital, args.risk))

    elif args.cmd == "backtest":
        res, _, _ = _run_bt(args)
        print(bt.report(res))

    elif args.cmd == "riskreward":
        res, _, _ = _run_bt(args)
        print(rk.report(res, args.risk))

    elif args.cmd == "regime":
        df, src = load(args.symbol, args.years)
        print(rg.report(df, args.symbol, src))

    elif args.cmd == "multifactor":
        prices, sources = _load_many(args.symbols, args.years)
        print(mf.report(prices, sources))

    elif args.cmd == "optimize":
        df, src = load(args.symbol, args.years)
        print(opt.report(df, args.strategy, args.symbol, args.capital, args.risk))

    elif args.cmd == "portfolio":
        prices, _ = _load_many(args.symbols, args.years)
        print(pf.report(prices, args.risk_tolerance, args.horizon, args.capital))

    elif args.cmd == "setups":
        prices, _ = _load_many(args.symbols, args.years)
        print(st.report(prices, args.capital, args.risk))

    elif args.cmd == "montecarlo":
        res, _, _ = _run_bt(args)
        print(mc.report(res, args.paths))

    elif args.cmd == "drawdown":
        res, _, _ = _run_bt(args)
        print(dd.report(res))

    elif args.cmd == "macro":
        prices, sources = _load_many(mcr.PROXIES + ["SPY", "QQQ", "GLD", "TLT"],
                                     args.years)
        print(mcr.report(prices, sources, args.capital, args.risk))

    elif args.cmd == "alpha":
        df, src = load(args.symbol, args.years)
        print(al.report(df, args.symbol, src))

    elif args.cmd == "all":
        df, src = load(args.symbol, args.years)
        res = bt.run_backtest(df, get_strategy(args.strategy), args.capital,
                              args.risk, symbol=args.symbol, source=src)
        prices, sources = _load_many(DEFAULT_UNIVERSE, args.years)
        macro_prices, macro_sources = _load_many(
            mcr.PROXIES + ["SPY", "QQQ", "GLD", "TLT"], args.years)
        blocks = [
            describe_all(args.capital, args.risk),
            bt.report(res),
            rk.report(res, args.risk),
            rg.report(df, args.symbol, src),
            mf.report(prices, sources),
            opt.report(df, args.strategy, args.symbol, args.capital, args.risk),
            pf.report(prices, "medium", 3, args.capital),
            st.report(prices, args.capital, args.risk),
            mc.report(res),
            dd.report(res),
            mcr.report(macro_prices, macro_sources, args.capital, args.risk),
            al.report(df, args.symbol, src),
        ]
        print(("\n\n" + "#" * 78 + "\n\n").join(blocks))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
