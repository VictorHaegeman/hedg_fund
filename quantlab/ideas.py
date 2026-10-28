"""Module 17 — Idées d'investissement tirées de la presse financière (actions « niche »).

Source : flux d'actualité Benzinga diffusé gratuitement par l'API Alpaca (~150
articles/heure, des centaines de sociétés, beaucoup de petites capitalisations).
Les sources payantes (Bloomberg, WSJ, X/Grok) ne sont pas lues : pas d'accès
gratuit ni légal.

Pipeline (transparent, chaque étape est un filtre lisible) :
  1. Articles des dernières heures → tickers cités (fournis par l'éditeur)
  2. Agrégation par société : nombre de mentions, nombre d'articles distincts,
     ton moyen des titres (lexique financier de news.py), titres récents
  3. Filtres de sécurité : action cotée (pas de crypto), prix ≥ 5 $, volume
     échangé moyen ≥ 10 M$/jour (sinon impossible de sortir proprement)
  4. Score = attention (log des mentions) + ton + élan du cours (tendance, 5 j)
  5. (optionnel) un LLM résume en une phrase pourquoi c'est intéressant et le risque

⚠ Ce sont des IDÉES, pas des signaux backtestés (l'actualité passée n'est pas
rejouable sans biais). Elles alimentent la poche IA (module 14) avec une taille de
position réduite pour les valeurs peu liquides (« niche »), jamais le fonds principal.
"""

from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import indicators as ind
from .news import score_text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDEAS_PATH = os.path.join(ROOT, "ideas.json")

MIN_PRICE = 5.0
MIN_ADV_USD = 10e6          # volume moyen 20 j en $ : en dessous, trop illiquide
NICHE_ADV_USD = 50e6        # en dessous : « niche » → taille de position réduite
_TICKER = re.compile(r"^[A-Z]{1,5}$")


def fetch_news(client, hours: int = 48, max_pages: int = 20) -> list[dict]:
    """Articles Benzinga des `hours` dernières heures via Alpaca (fail-safe)."""
    start = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {"start": start, "limit": 50, "sort": "desc"}
    out: list[dict] = []
    try:
        for _ in range(max_pages):
            d = client.data("/v1beta1/news", dict(params))
            out += d.get("news", [])
            token = d.get("next_page_token")
            if not token:
                break
            params["page_token"] = token
    except Exception:
        pass
    return out


def aggregate(news: list[dict], exclude: set[str] | None = None) -> dict[str, dict]:
    """Par ticker : mentions, articles distincts, ton moyen, titres récents."""
    exclude = exclude or set()
    agg: dict[str, dict] = {}
    for n in news:
        head = (n.get("headline") or "").strip()
        syms = [s for s in n.get("symbols", []) if _TICKER.match(s) and s not in exclude]
        if not head or not syms or len(syms) > 6:   # articles « fourre-tout » ignorés
            continue
        tone = score_text(head + " " + (n.get("summary") or ""))
        for s in syms:
            a = agg.setdefault(s, {"mentions": 0, "tones": [], "headlines": [], "last_at": ""})
            a["mentions"] += 1
            a["tones"].append(tone)
            if len(a["headlines"]) < 4 and head not in a["headlines"]:
                a["headlines"].append(head[:140])
            a["last_at"] = max(a["last_at"], n.get("created_at", ""))
    for a in agg.values():
        a["sentiment"] = float(np.mean(a.pop("tones"))) if a.get("tones") else 0.0
    return agg


def yf_bars(tickers: list[str], period: str = "3mo") -> dict[str, pd.DataFrame]:
    """Historique journalier (volume consolidé) via yfinance, en un seul appel."""
    if not tickers:
        return {}
    try:
        import yfinance as yf
        raw = yf.download(tickers, period=period, interval="1d", group_by="ticker",
                          auto_adjust=True, progress=False, threads=True)
    except Exception:
        return {}
    out = {}
    for t in tickers:
        try:
            df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            df = df.dropna(subset=["Close"])
            if len(df) >= 25:
                out[t] = df
        except Exception:
            continue
    return out


def metrics(df: pd.DataFrame) -> dict:
    close = df["Close"]
    price = float(close.iloc[-1])
    adv = float((close * df["Volume"]).tail(20).mean())
    e20, e50 = float(ind.ema(close, 20).iloc[-1]), float(ind.ema(close, 50).iloc[-1])
    trend = ("haussiere" if price > e20 > e50 else "baissiere" if price < e20 < e50 else "neutre")
    hv = float(np.log(close / close.shift(1)).tail(20).std() * math.sqrt(252))
    return {"price": round(price, 2), "adv_usd": adv,
            "chg_5d": round((price / float(close.iloc[-6]) - 1) * 100, 1),
            "chg_20d": round((price / float(close.iloc[-21]) - 1) * 100, 1),
            "rsi": round(float(ind.rsi(close, 14).iloc[-1]), 1), "trend": trend,
            "hv": hv if math.isfinite(hv) else 0.0}


def score(row: dict) -> float:
    s = 2.0 * math.log1p(row["mentions"]) + 3.0 * row["sentiment"]
    s += {"haussiere": 1.0, "baissiere": -1.0}.get(row["trend"], 0.0)
    return round(s + float(np.clip(row["chg_5d"] / 5, -1, 1)), 3)


def build_ideas(client, exclude: set[str] | None = None, top: int = 15,
                bars_fn=yf_bars, summarize=None, hours: int = 48) -> list[dict]:
    """Idées classées (fail-safe : liste vide si la presse ou les prix manquent)."""
    agg = aggregate(fetch_news(client, hours=hours), exclude)
    # on n'enrichit que les plus citées (coût réseau borné)
    cands = sorted(agg, key=lambda s: (-agg[s]["mentions"], s))[:80]
    bars = bars_fn(cands)
    rows = []
    for s in cands:
        df = bars.get(s)
        if df is None:
            continue
        try:
            m = metrics(df)
        except Exception:
            continue
        if m["price"] < MIN_PRICE or m["adv_usd"] < MIN_ADV_USD:
            continue
        row = {"symbol": s, **agg[s], **m, "niche": m["adv_usd"] < NICHE_ADV_USD}
        row["score"] = score(row)
        rows.append(row)
    rows.sort(key=lambda r: -r["score"])
    rows = rows[:top]
    if summarize and rows:
        try:
            notes = summarize(rows)
            for r in rows:
                n = notes.get(r["symbol"]) or {}
                for k in ("why", "risk", "why_en", "risk_en"):
                    r[k] = n.get(k, "")
        except Exception:
            pass
    return rows


def groq_summarizer(api_key: str):
    """Résumé IA (Groq) : pour chaque idée, une phrase « pourquoi » et une « risque »."""
    from .ai_trader import groq_decide

    def _summarize(rows: list[dict]) -> dict:
        lines = [f"{r['symbol']} | {r['mentions']} articles | ton {r['sentiment']:+.2f} | "
                 f"5j {r['chg_5d']:+}% | tendance {r['trend']} | titres: " + " / ".join(r["headlines"][:3])
                 for r in rows]
        prompt = ("Pour chaque action ci-dessous, à partir UNIQUEMENT des titres de presse "
                  "fournis, écris une phrase courte expliquant pourquoi elle attire l'attention "
                  "et une phrase sur le principal risque, en français ET en anglais. "
                  "N'invente aucun chiffre.\n\n"
                  + "\n".join(lines)
                  + '\n\nRéponds en JSON : {"TICKER": {"why": "...", "risk": "...", '
                    '"why_en": "...", "risk_en": "..."}, ...}')
        d = groq_decide(prompt, api_key, max_tokens=6000,
                        system="Tu es analyste actions. Factuel, prudent, en francais. "
                               "Tu reponds uniquement en JSON valide.")
        return {k: v for k, v in d.items() if isinstance(v, dict)}
    return _summarize


def save(rows: list[dict], path: str = IDEAS_PATH) -> None:
    payload = {"updated_at": datetime.now().isoformat(timespec="seconds"),
               "source": "Benzinga via Alpaca (gratuit)", "ideas": rows}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=float)


def load(path: str = IDEAS_PATH) -> dict:
    if not os.path.exists(path):
        return {"updated_at": None, "ideas": []}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"updated_at": None, "ideas": []}


def report(rows: list[dict]) -> str:
    lines = ["=" * 78, "MODULE 17 — IDÉES TIRÉES DE LA PRESSE (Benzinga via Alpaca)", "=" * 78]
    if not rows:
        return "\n".join(lines + ["  Aucune idée ne passe les filtres (ou presse indisponible)."])
    lines.append(f"  {'Action':<7}{'Score':>6}{'Art.':>5}{'Ton':>7}{'Prix':>9}{'5 j':>8}"
                 f"{'Vol. $/j':>11}  Tendance   Niche")
    for r in rows:
        lines.append(f"  {r['symbol']:<7}{r['score']:>6.2f}{r['mentions']:>5}{r['sentiment']:>+7.2f}"
                     f"{r['price']:>9.2f}{r['chg_5d']:>+7.1f}%{r['adv_usd'] / 1e6:>9.0f} M  "
                     f"{r['trend']:<10} {'oui' if r['niche'] else ''}")
        if r.get("why"):
            lines.append(f"         ↳ {r['why']}  ⚠ {r.get('risk', '')}")
        else:
            lines.append(f"         ↳ {r['headlines'][0] if r['headlines'] else ''}")
    lines.append("\n⚠ Idées issues de l'actualité : non backtestables. Utilisées par la poche IA "
                 "avec une taille réduite pour les valeurs « niche ».")
    return "\n".join(lines)
