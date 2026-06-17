"""Module 13 — Pouls de marché : actualité (scraping RSS) + contexte géopolitique.

Deux briques, volontairement séparées car de fiabilité très différente :

1. SENTIMENT D'ACTUALITÉ (scraping) — flux RSS Yahoo Finance par ticker, sans
   clé API. Score lexical (mots positifs/négatifs). C'est BRUITÉ : utilisé comme
   *contexte* affiché, jamais comme signal d'achat automatique.

2. JAUGE DE RISQUE GÉOPOLITIQUE (basée marché) — le marché price la tension
   géopolitique avant qu'on puisse la lire dans les titres : on la mesure via le
   VIX (indice de la peur), l'or et le pétrole. C'est ainsi que procèdent les
   desks. Utilisée comme *filtre défensif* : ne pas ouvrir de nouvelles positions
   quand la peur est extrême.

Tout est fail-safe : si le réseau échoue, on renvoie un état neutre — jamais
d'exception qui casserait le cycle de trading.
"""

from __future__ import annotations

import re
import urllib.request
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from . import indicators as ind

# --- Lexique financier compact ------------------------------------------------
POSITIVE = {
    "beat", "beats", "surge", "surges", "rally", "rallies", "gain", "gains",
    "jump", "jumps", "soar", "soars", "record", "profit", "profits", "growth",
    "upgrade", "upgraded", "outperform", "strong", "strength", "boost", "boosted",
    "win", "wins", "breakthrough", "optimism", "bullish", "rebound", "recovery",
    "expand", "expansion", "dividend", "buyback", "raises", "raised", "tops",
    "exceeds", "momentum", "approval", "approved", "partnership", "wins",
}
NEGATIVE = {
    "miss", "misses", "fall", "falls", "drop", "drops", "plunge", "plunges",
    "slump", "sink", "sinks", "loss", "losses", "decline", "declines", "downgrade",
    "downgraded", "underperform", "weak", "weakness", "cut", "cuts", "layoff",
    "layoffs", "lawsuit", "probe", "investigation", "fraud", "bearish", "recession",
    "crisis", "warning", "warns", "fears", "selloff", "tumble", "slashed",
    "default", "bankruptcy", "halt", "halts", "recall", "delay", "delayed",
}
# Mots à charge géopolitique (comptés dans les titres généraux du marché)
GEO_RISK = {
    "war", "conflict", "sanctions", "sanction", "tariff", "tariffs", "invasion",
    "attack", "missile", "strike", "strikes", "escalation", "tension", "tensions",
    "embargo", "coup", "unrest", "ceasefire", "nuclear", "geopolitical", "military",
    "troops", "border", "retaliation", "blockade",
}

_TOKEN = re.compile(r"[a-z']+")


def score_text(text: str) -> float:
    """Score de sentiment lexical d'un texte, dans [-1, 1]."""
    words = _TOKEN.findall(text.lower())
    pos = sum(w in POSITIVE for w in words)
    neg = sum(w in NEGATIVE for w in words)
    total = pos + neg
    return (pos - neg) / total if total else 0.0


def count_geo(text: str) -> int:
    return sum(w in GEO_RISK for w in _TOKEN.findall(text.lower()))


# --- Scraping RSS (fail-safe) -------------------------------------------------
def fetch_headlines(symbol: str, limit: int = 15, timeout: float = 8.0) -> list[str]:
    """Titres d'actualité récents pour un ticker via le flux RSS Yahoo Finance.

    Renvoie [] en cas d'échec réseau/parse — ne lève jamais."""
    url = (f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}"
           "&region=US&lang=en-US")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        root = ElementTree.fromstring(raw)
        titles = [t.text for t in root.iter("title") if t.text]
        return titles[1 : limit + 1]  # le 1er title est le nom du flux
    except Exception:
        return []


def headline_sentiment(symbol: str, fetch=fetch_headlines) -> dict:
    titles = fetch(symbol)
    if not titles:
        return {"symbol": symbol, "n": 0, "score": 0.0, "label": "n/a",
                "geo": 0, "titles": []}
    scores = [score_text(t) for t in titles]
    geo = sum(count_geo(t) for t in titles)
    avg = float(np.mean(scores))
    label = "positif" if avg > 0.15 else "négatif" if avg < -0.15 else "neutre"
    return {"symbol": symbol, "n": len(titles), "score": round(avg, 2),
            "label": label, "geo": geo, "titles": titles[:5]}


# --- Jauge de risque géopolitique basée marché --------------------------------
def risk_gauge(loader, lookback_pct: int = 504) -> dict:
    """Niveau de risque global via VIX (peur) + or + pétrole. Fail-safe → neutre."""
    out = {"level": "neutre", "vix": None, "vix_pct": None,
           "gold_1m": None, "oil_1m": None, "note": ""}
    try:
        vix, _ = loader("^VIX", years=3, fresh=False)
        v = vix["Close"].dropna()
        cur = float(v.iloc[-1])
        pct = float((v.tail(lookback_pct) < cur).mean())
        out["vix"], out["vix_pct"] = round(cur, 1), round(pct, 2)
    except Exception:
        return out  # sans VIX, on reste neutre (pas de filtre)

    for key, sym in (("gold_1m", "GLD"), ("oil_1m", "USO")):
        try:
            df, _ = loader(sym, years=1, fresh=False)
            c = df["Close"].dropna()
            out[key] = round(float(c.iloc[-1] / c.iloc[-21] - 1), 3)
        except Exception:
            pass

    # Classification : la peur extrême (VIX très haut) = risk-off
    if pct > 0.85:
        out["level"] = "risk-off"
        out["note"] = "peur extrême : prudence, ne pas initier de nouveaux longs"
    elif pct > 0.65:
        out["level"] = "prudence"
        out["note"] = "tension supérieure à la normale : tailles réduites conseillées"
    elif pct < 0.30:
        out["level"] = "risk-on"
        out["note"] = "marché calme : environnement favorable aux prises de position"
    else:
        out["level"] = "neutre"
        out["note"] = "tension dans la normale"
    # Renfort géopolitique : or qui s'envole pendant que le VIX monte
    if out["gold_1m"] and out["gold_1m"] > 0.05 and pct > 0.6:
        out["note"] += " · or en forte hausse (signe de stress géopolitique)"
    return out


def blocks_new_entries(gauge: dict) -> bool:
    """Filtre défensif : bloque les NOUVELLES entrées en peur extrême."""
    return gauge.get("level") == "risk-off"


def report(symbols: list[str], loader, fetch=fetch_headlines) -> str:
    gauge = risk_gauge(loader)
    lines = [
        "=" * 78,
        "MODULE 13 — POULS DE MARCHÉ : ACTUALITÉ + CONTEXTE GÉOPOLITIQUE",
        "=" * 78,
        "",
        "— Jauge de risque géopolitique (basée marché) —",
    ]
    if gauge["vix"] is not None:
        lines += [
            f"  VIX (indice de la peur) : {gauge['vix']} "
            f"({int(gauge['vix_pct'] * 100)}e percentile sur 2 ans)",
            f"  Or sur 1 mois : {gauge['gold_1m'] if gauge['gold_1m'] is not None else 'n/a'} | "
            f"Pétrole sur 1 mois : {gauge['oil_1m'] if gauge['oil_1m'] is not None else 'n/a'}",
            f"  NIVEAU : {gauge['level'].upper()} — {gauge['note']}",
            f"  → Nouvelles entrées : "
            f"{'BLOQUÉES (défensif)' if blocks_new_entries(gauge) else 'autorisées'}",
        ]
    else:
        lines.append("  VIX indisponible → jauge neutre, aucun filtre appliqué.")

    lines += ["", "— Sentiment d'actualité par actif (scraping RSS, contexte seulement) —"]
    geo_total = 0
    for sym in symbols:
        s = headline_sentiment(sym, fetch)
        geo_total += s["geo"]
        if s["n"] == 0:
            lines.append(f"  {sym:<7}: pas d'actualité récupérée")
            continue
        flag = f" | {s['geo']} mention(s) géopolitique(s)" if s["geo"] else ""
        lines.append(f"  {sym:<7}: {s['label']:<8} (score {s['score']:+.2f}, "
                     f"{s['n']} titres){flag}")
        if s["titles"]:
            lines.append(f"           ex: « {s['titles'][0][:70]} »")
    lines += [
        "",
        f"Total mentions géopolitiques dans les titres : {geo_total}",
        "",
        "Lecture : le sentiment d'actualité est un CONTEXTE bruité (à ne pas suivre",
        "aveuglément). Le filtre qui agit réellement sur le trading est la jauge de",
        "risque basée marché ci-dessus, pas l'interprétation des titres.",
    ]
    return "\n".join(lines)
