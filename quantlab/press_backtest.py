"""Test historique des idées presse (module 17), dans les deux sens.

Rejoue chaque jour de bourse, avant l'ouverture (13h UTC), exactement le classement des idées
presse du live (articles Benzinga des 48 h via Alpaca, mentions, ton des titres, tendance,
élan 5 j, filtres prix ≥ 5 $ et volume ≥ 10 M$/jour), puis mesure ce qu'aurait rapporté :
  - LONG  : acheter les 8 meilleures idées au ton positif (bonnes nouvelles) à l'ouverture ;
  - SHORT : vendre à découvert les 8 pires (mauvaises nouvelles, ton négatif) ;
gardées 1, 3, 5 ou 10 jours, comparées au S&P 500 (SPY) sur la même période, après frais
(0,10 % aller-retour, 0,30 % sur les valeurs « niche »).

Pas de regard vers le futur : les indicateurs utilisent la clôture de la veille, l'entrée se
fait à l'ouverture du jour. Le t-stat est calculé sur des dates de départ qui ne se chevauchent
pas (une tous les H jours), car les idées d'un même jour sont liées.
"""

from __future__ import annotations

import bisect
import json
import math
import os
import re
import time
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import indicators as ind
from .alpaca import ROOT, AlpacaError
from .ideas import MIN_ADV_USD, MIN_PRICE, NICHE_ADV_USD, _TICKER
from .news import score_text

OUT_DIR = os.path.join(ROOT, "research")
HOLDS = (1, 3, 5, 10)
TOP_N = 8
TOP_MENTIONS = 80
COST_LIQUID, COST_NICHE = 0.0010, 0.0030
EARN = re.compile(r"\b(earnings|eps|results|quarter|q[1-4]|guidance|revenue|sales|beats?|miss(es)?)\b", re.I)


# ── données ─────────────────────────────────────────────────────────────────
def _call(client, path, params, tries=6, sleep=time.sleep):
    for k in range(tries):
        try:
            return client.data(path, dict(params))
        except AlpacaError as e:
            if re.search(r"→ (429|5\d\d)", str(e)) and k < tries - 1:
                sleep(15 * (k + 1))
                continue
            raise
    return {}


def fetch_news_day(client, day: date, max_pages: int = 200, pause: float = 0.3,
                   sleep=time.sleep) -> list[dict]:
    """Tous les articles d'un jour (UTC), réduits à : heure, tickers, ton, « résultats ? »."""
    params = {"start": f"{day}T00:00:00Z", "end": f"{day + timedelta(days=1)}T00:00:00Z",
              "limit": 50, "sort": "asc"}
    out = []
    for _ in range(max_pages):
        d = _call(client, "/v1beta1/news", params, sleep=sleep)
        for n in d.get("news", []):
            head = (n.get("headline") or "").strip()
            syms = [s for s in n.get("symbols", []) if _TICKER.match(s)]
            if not head or not syms or len(syms) > 6:
                continue
            out.append({"t": n.get("created_at", ""), "s": syms,
                        "tone": score_text(head + " " + (n.get("summary") or "")),
                        "earn": bool(EARN.search(head))})
        token = d.get("next_page_token")
        if not token:
            break
        params["page_token"] = token
        if pause:
            sleep(pause)
    return out


def fetch_bars(client, symbols: list[str], start: str, end: str, feed: str = "sip",
               sleep=time.sleep) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    syms = sorted(set(symbols))
    for i in range(0, len(syms), 100):
        params = {"symbols": ",".join(syms[i:i + 100]), "timeframe": "1Day", "start": start,
                  "end": end, "limit": 10000, "adjustment": "all", "feed": feed}
        raw: dict[str, list] = {}
        for _ in range(500):
            d = _call(client, "/v2/stocks/bars", params, sleep=sleep)
            for s, bl in (d.get("bars") or {}).items():
                raw.setdefault(s, []).extend(bl)
            tok = d.get("next_page_token")
            if not tok:
                break
            params["page_token"] = tok
        for s, bl in raw.items():
            if len(bl) < 30:
                continue
            df = pd.DataFrame(bl)
            df.index = pd.to_datetime(df["t"]).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
            out[s] = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close",
                                        "v": "Volume"})[["Open", "High", "Low", "Close", "Volume"]]
    return out


# ── signaux ─────────────────────────────────────────────────────────────────
def features(df: pd.DataFrame) -> pd.DataFrame:
    """Indicateurs du live (ideas.metrics), connus à la CLÔTURE de chaque jour."""
    c = df["Close"]
    e20, e50 = ind.ema(c, 20), ind.ema(c, 50)
    trend = np.where((c > e20) & (e20 > e50), 1, np.where((c < e20) & (e20 < e50), -1, 0))
    return pd.DataFrame({"price": c, "adv": (c * df["Volume"]).rolling(20).mean(),
                         "chg5": (c / c.shift(5) - 1) * 100, "trend": trend,
                         "n": np.arange(1, len(c) + 1)}, index=df.index)


def long_score(m: int, sent: float, trend: int, chg5: float) -> float:   # = ideas.score
    return 2 * math.log1p(m) + 3 * sent + trend + float(np.clip(chg5 / 5, -1, 1))


def short_score(m: int, sent: float, trend: int, chg5: float) -> float:  # miroir
    return 2 * math.log1p(m) - 3 * sent - trend + float(np.clip(-chg5 / 5, -1, 1))


def aggregate(arts: list[dict]) -> dict[str, dict]:
    agg: dict[str, dict] = {}
    for a in arts:
        for s in a["s"]:
            g = agg.setdefault(s, {"m": 0, "tones": [], "earn": False})
            g["m"] += 1
            g["tones"].append(a["tone"])
            g["earn"] |= a["earn"]
    for g in agg.values():
        g["sent"] = float(np.mean(g.pop("tones")))
    return agg


def picks(agg: dict, feats: dict, day: pd.Timestamp, top: int = TOP_N) -> dict[str, list]:
    cands = sorted(agg, key=lambda s: (-agg[s]["m"], s))[:TOP_MENTIONS]
    rows = []
    for s in cands:
        f = feats.get(s)
        if f is None:
            continue
        i = f.index.searchsorted(day) - 1            # clôture de la veille
        if i < 50:
            continue
        r = f.iloc[i]
        if not (r.price >= MIN_PRICE and r.adv >= MIN_ADV_USD) or not math.isfinite(r.chg5):
            continue
        g = agg[s]
        rows.append({"sym": s, "m": g["m"], "sent": g["sent"], "earn": g["earn"],
                     "chg5": float(r.chg5), "niche": bool(r.adv < NICHE_ADV_USD),
                     "long": long_score(g["m"], g["sent"], int(r.trend), float(r.chg5)),
                     "short": short_score(g["m"], g["sent"], int(r.trend), float(r.chg5))})
    longs = sorted([r for r in rows if r["sent"] > 0], key=lambda r: -r["long"])[:top]
    shorts = sorted([r for r in rows if r["sent"] < 0], key=lambda r: -r["short"])[:top]
    return {"long": longs, "short": shorts}


def outcome(df: pd.DataFrame, spy: pd.DataFrame, day: pd.Timestamp, hold: int):
    """(rendement brut, rendement SPY) entrée à l'ouverture du jour, sortie à la clôture J+hold-1."""
    if day not in df.index or day not in spy.index:
        return None
    i, j = df.index.get_loc(day), spy.index.get_loc(day)
    if i + hold - 1 >= len(df) or j + hold - 1 >= len(spy):
        return None
    r = df["Close"].iloc[i + hold - 1] / df["Open"].iloc[i] - 1
    b = spy["Close"].iloc[j + hold - 1] / spy["Open"].iloc[j] - 1
    return float(r), float(b)


def trades_table(days: list[pd.Timestamp], books: dict, bars: dict, spy: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d in days:
        for side in ("long", "short"):
            for p in books[d][side]:
                df = bars.get(p["sym"])
                if df is None:
                    continue
                for h in HOLDS:
                    o = outcome(df, spy, d, h)
                    if o is None:
                        continue
                    r, b = o
                    sign = 1 if side == "long" else -1
                    cost = COST_NICHE if p["niche"] else COST_LIQUID
                    rows.append({"day": d, "side": side, "hold": h, "sym": p["sym"],
                                 "niche": p["niche"], "earn": p["earn"], "chg5": p["chg5"],
                                 "ret": sign * r - cost, "excess": sign * (r - b) - cost})
    return pd.DataFrame(rows)


def summarize(t: pd.DataFrame, side: str, hold: int, mask=None) -> dict:
    s = t[(t.side == side) & (t.hold == hold)]
    if mask is not None:
        s = s[mask(s)]
    if s.empty:
        return {"n": 0}
    daily = s.groupby("day")["excess"].mean().sort_index()
    nonover = daily.iloc[::hold]
    tstat = (nonover.mean() / nonover.std() * math.sqrt(len(nonover))
             if len(nonover) > 2 and nonover.std() > 1e-9 else float("nan"))
    return {"n": int(len(s)), "jours": int(len(daily)),
            "brut_moy_pct": round(s["ret"].mean() * 100, 2),
            "vs_spy_moy_pct": round(s["excess"].mean() * 100, 2),
            "battent_spy_pct": round((s["excess"] > 0).mean() * 100, 1),
            "t_stat": round(float(tstat), 2)}


def report(t: pd.DataFrame, meta: dict) -> tuple[str, dict]:
    res: dict = {"meta": meta, "global": {}, "variantes": {}, "par_annee": {}}
    L = ["# Test historique des idées presse (achat ET vente à découvert)", "",
         f"Période : {meta['start']} → {meta['end']} · {meta['days']} jours de bourse · "
         f"{meta['articles']:,} articles lus".replace(",", " ") + " · frais déduits (0,10 % / 0,30 % niche)",
         "", "Écart vs S&P 500 = gain du trade moins celui du SPY sur la même période "
         "(pour un short : ce qu'on gagne quand l'action fait pire que le SPY). "
         "t-stat ≥ 2 : résultat probablement pas dû au hasard.", ""]
    for side, title in (("long", "ACHAT des bonnes nouvelles"), ("short", "VENTE À DÉCOUVERT des mauvaises nouvelles")):
        L += [f"## {title}", "", "| Garde | Trades | Gain moyen | Écart vs S&P | Battent le S&P | t-stat |",
              "|---|---|---|---|---|---|"]
        for h in HOLDS:
            r = summarize(t, side, h)
            res["global"][f"{side}_{h}j"] = r
            if r["n"]:
                L.append(f"| {h} j | {r['n']} | {r['brut_moy_pct']:+.2f} % | {r['vs_spy_moy_pct']:+.2f} pt | "
                         f"{r['battent_spy_pct']:.0f} % | {r['t_stat']:.1f} |")
        L.append("")
    variants = {
        "liquides": lambda s: ~s.niche, "niche": lambda s: s.niche,
        "news de résultats": lambda s: s.earn, "autres news": lambda s: ~s.earn,
        "sans grosse hausse/baisse avant (|5 j| ≤ 8 %)": lambda s: s.chg5.abs() <= 8,
        "après grosse hausse/baisse (|5 j| > 8 %)": lambda s: s.chg5.abs() > 8,
    }
    L += ["## Variantes (garde 5 jours)", "", "| Variante | Côté | Trades | Écart vs S&P | Battent | t-stat |",
          "|---|---|---|---|---|---|"]
    for name, m in variants.items():
        for side in ("long", "short"):
            r = summarize(t, side, 5, m)
            res["variantes"][f"{name}|{side}"] = r
            if r["n"]:
                L.append(f"| {name} | {side} | {r['n']} | {r['vs_spy_moy_pct']:+.2f} pt | "
                         f"{r['battent_spy_pct']:.0f} % | {r['t_stat']:.1f} |")
    L += ["", "## Stabilité par année (garde 5 jours)", "", "| Année | Côté | Trades | Écart vs S&P | t-stat |",
          "|---|---|---|---|---|"]
    for y in sorted({d.year for d in t.day}):
        for side in ("long", "short"):
            r = summarize(t, side, 5, lambda s, y=y: s.day.dt.year == y)
            res["par_annee"][f"{y}|{side}"] = r
            if r["n"]:
                L.append(f"| {y} | {side} | {r['n']} | {r['vs_spy_moy_pct']:+.2f} pt | {r['t_stat']:.1f} |")
    return "\n".join(L) + "\n", res


def run(client, start: date, end: date, budget_s: float = 4 * 3600, sleep=time.sleep,
        pause: float = 0.3, log=print) -> tuple[str, dict]:
    t0 = time.time()
    arts: list[dict] = []
    day, fetched = end, []
    while day >= start and time.time() - t0 < budget_s:      # du plus récent au plus ancien
        arts += fetch_news_day(client, day, pause=pause, sleep=sleep)   # week-ends compris
        fetched.append(day)
        if len(fetched) % 20 == 0:
            log(f"  news : {day} · {len(arts):,} articles · {time.time() - t0:,.0f} s")
        day -= timedelta(days=1)
    first = min(fetched) if fetched else end
    arts.sort(key=lambda a: a["t"])
    times = [a["t"] for a in arts]
    # jours de décision : jours de bourse avec 48 h de news complètes derrière
    bdays = [d for d in pd.bdate_range(first + timedelta(days=2), end)]
    windows = {}
    for d in bdays:
        hi = f"{d.date()}T13:00:00Z"
        lo = f"{(d - timedelta(hours=48)).date()}T13:00:00Z"
        windows[d] = aggregate(arts[bisect.bisect_left(times, lo):bisect.bisect_left(times, hi)])
    cands = set()
    for agg in windows.values():
        cands |= set(sorted(agg, key=lambda s: (-agg[s]["m"], s))[:TOP_MENTIONS])
    log(f"  barres : {len(cands)} actions")
    bstart = str(first - timedelta(days=120))
    bend = str(min(end, date.today() - timedelta(days=1)))
    try:
        bars = fetch_bars(client, list(cands) + ["SPY"], bstart, bend, "sip", sleep=sleep)
        feed = "sip"
    except AlpacaError:
        bars = fetch_bars(client, list(cands) + ["SPY"], bstart, bend, "iex", sleep=sleep)
        feed = "iex"
    spy = bars.get("SPY")
    if spy is None:
        return "SPY indisponible : test impossible\n", {}
    feats = {s: features(df) for s, df in bars.items()}
    days = [d for d in bdays if d in spy.index]
    books = {d: picks(windows[d], feats, d) for d in days}
    t = trades_table(days, books, bars, spy)
    meta = {"start": str(days[0].date()) if days else "", "end": str(days[-1].date()) if days else "",
            "days": len(days), "articles": len(arts), "feed": feed,
            "duree_s": round(time.time() - t0)}
    md, res = report(t, meta) if not t.empty else ("Aucun trade\n", {"meta": meta})
    return md, res


def main(years: float = 2.0) -> str:
    from .alpaca import Alpaca
    client = Alpaca.from_env("ALPACA", label="fonds") or Alpaca.from_env("ALPACA_AI", label="IA")
    if client is None:
        return "Clés Alpaca manquantes"
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=int(365 * years))
    md, res = run(client, start, end)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "press_backtest.md"), "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(OUT_DIR, "press_backtest.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    return md
