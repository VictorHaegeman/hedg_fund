"""Module 14 — Poche IA : options, vente à découvert, actions et crypto pilotées par un LLM.

⚠ Poche EXPÉRIMENTALE, volontairement séparée du fonds :
  • le fonds (live.py) applique des règles backtestées ; ici un LLM (Groq) décide
    à partir d'indicateurs + actualité. Aucun backtest possible de ses décisions :
    on mesure sa performance en conditions réelles (compte Alpaca paper) et on la
    compare au fonds sur le tableau de bord.
  • le code garde la main sur le risque : pré-tri des candidats, choix des strikes
    dans la vraie chaîne d'options, filtre de risque, sorties automatiques.

Un appel = un cycle (adapté à GitHub Actions) :
  1. gère les positions ouvertes (take-profit, stop-loss, échéance proche)
  2. photo du marché : ~50 actions/ETF pré-triées → 12 candidats + crypto
     (RSI, tendance, vol. historique/implicite, résultats, news Alpaca + RSS)
  3. le LLM choisit AU PLUS un trade (ou HOLD)
  4. le filtre de risque valide, puis l'ordre part sur Alpaca

Stratégies : BULL_CALL_SPREAD, BEAR_PUT_SPREAD, LONG_STRADDLE, LONG_STRANGLE
(perte max = coût payé), BUY_STOCK, SHORT_STOCK, BUY_CRYPTO (stop 2–5 %).
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from datetime import date, datetime, timedelta, timezone
from xml.etree import ElementTree

import pandas as pd

from . import indicators as ind
from . import pricing as pr
from .alpaca import ROOT, Alpaca, AlpacaError, load_env, position_symbol, to_alpaca_symbol

STATE_PATH = os.path.join(ROOT, "ai_trades.json")
GREEKS_PATH = os.path.join(ROOT, "ai_greeks.json")

STOCKS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD", "AVGO", "NFLX",
    "CRM", "ORCL", "ADBE", "INTC", "QCOM", "MU", "PLTR", "UBER", "SHOP", "COIN",
    "JPM", "BAC", "GS", "V", "MA", "PYPL",
    "UNH", "LLY", "JNJ", "PFE", "MRK",
    "WMT", "COST", "KO", "PEP", "MCD", "NKE", "DIS", "SBUX",
    "XOM", "CVX", "BA", "CAT",
    "SPY", "QQQ", "IWM", "XLE", "XLF", "GLD", "TLT",
]
ETFS = {"SPY", "QQQ", "IWM", "XLE", "XLF", "GLD", "TLT"}
# Cryptos liquides tradables sur Alpaca (stablecoins, or tokenisé et memecoins exclus)
CRYPTOS = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "DOGE/USD", "AVAX/USD", "LINK/USD",
           "LTC/USD", "BCH/USD", "DOT/USD", "ADA/USD", "UNI/USD", "AAVE/USD"]
TOP_CRYPTOS = 5                # cryptos envoyées au LLM après pré-tri
MAX_IDEAS = 8                  # idées « presse » ajoutées aux candidats (module 17)
IDEAS_MAX_AGE_H = 72           # idées plus vieilles : ignorées
IDEAS_REFRESH_H = 3            # le cycle IA reconstruit ideas.json au-delà de cet âge
NICHE_SIZE = 0.5               # valeur « niche » (peu liquide) → taille de position × 0,5

TOP_CANDIDATES = 12            # actions envoyées au LLM après pré-tri (quota Groq)
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# Modèles essayés dans l'ordre (le suivant sert si l'un est retiré ou hors quota).
# Surchargeable sans toucher au code : variable GROQ_MODEL (Secret ou .env).
GROQ_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]

MIN_CONFIDENCE = 0.65
MAX_OPEN_TRADES = 5
DAILY_LOSS_CAP = 0.03          # -3 % sur la journée → pas de nouvelle entrée
OPTION_MAX_RISK_PCT = 0.05     # perte max d'un trade options = 5 % de l'équité
SPOT_MAX_PCT = 0.10            # achat / vente à découvert = 10 % de l'équité max
MAX_CONTRACTS = 10

DTE_MIN, DTE_MAX, DTE_TARGET = 14, 45, 30
CLOSE_BEFORE_EXPIRY_DAYS = 5
OTM_PCT = 0.05
MAX_LEG_SPREAD_PCT = 0.15

SPREAD_TP_OF_MAX_GAIN = 0.50
SPREAD_SL_OF_COST = 0.50
VOL_TP_OF_COST = 0.50
VOL_SL_OF_COST = 0.40

OPTION_STRATEGIES = {"BULL_CALL_SPREAD", "BEAR_PUT_SPREAD", "LONG_STRADDLE", "LONG_STRANGLE",
                     "LONG_CALL_BUTTERFLY"}
LONG_VOL = {"LONG_STRADDLE", "LONG_STRANGLE"}
RISK_FREE = 0.04               # taux sans risque annuel pour Black-Scholes
VOL_EDGE_MAX = 1.15            # long vol refusé si prix payé > 1,15 × valeur BS à la vol réalisée
MAX_DELTA_PCT = 0.50           # |delta $ du portefeuille| ≤ 50 % de l'équité, sinon pas de nouveau trade
MAX_VEGA_PCT = 0.01            # |vega pour +1 pt de vol| ≤ 1 % de l'équité
MAX_THETA_PCT = 0.002          # érosion temporelle ≤ 0,2 % de l'équité par jour
SPOT_STRATEGIES = {"BUY_STOCK", "SHORT_STOCK", "BUY_CRYPTO"}

CRYPTO_RSS = ["https://www.coindesk.com/arc/outboundfeeds/rss/",
              "https://cointelegraph.com/rss"]
CRYPTO_KEYWORDS = {"BTC/USD": ["bitcoin", "btc"],
                   "ETH/USD": ["ethereum", "ether", "eth"],
                   "SOL/USD": ["solana", "sol"],
                   "XRP/USD": ["xrp", "ripple"],
                   "DOGE/USD": ["dogecoin", "doge"],
                   "AVAX/USD": ["avalanche", "avax"],
                   "LINK/USD": ["chainlink", "link"],
                   "LTC/USD": ["litecoin", "ltc"],
                   "BCH/USD": ["bitcoin cash", "bch"],
                   "DOT/USD": ["polkadot", "dot"],
                   "ADA/USD": ["cardano", "ada"],
                   "UNI/USD": ["uniswap", "uni"],
                   "AAVE/USD": ["aave"]}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _num(v, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def universe(fund_universe: list[str] | None = None) -> tuple[list[str], list[str]]:
    """Univers de la poche IA ; disjoint de celui du fonds si le compte est partagé."""
    if not fund_universe:
        return list(STOCKS), list(CRYPTOS)
    excl = {to_alpaca_symbol(s) for s in fund_universe}
    return [s for s in STOCKS if s not in excl], [c for c in CRYPTOS if c not in excl]


# ── indicateurs ─────────────────────────────────────────────────────────────
def metrics(closes: list[float], price: float, periods_per_year: int = 252) -> dict:
    s = pd.Series(closes, dtype=float)
    e20, e50 = float(ind.ema(s, 20).iloc[-1]), float(ind.ema(s, 50).iloc[-1])
    trend = ("haussiere" if price > e20 > e50 else
             "baissiere" if price < e20 < e50 else "neutre")
    hv = float(s.pct_change().tail(20).std() * math.sqrt(periods_per_year))
    return {"rsi": round(float(ind.rsi(s, 14).iloc[-1]), 1), "trend": trend,
            "hv": hv if math.isfinite(hv) else 0.0,
            "chg_5d": round((price / float(s.iloc[-6]) - 1) * 100, 1)}


def interest_score(r: dict) -> float:
    """Pré-tri fait par le code : met en avant ce qui bouge ou va bouger."""
    score = 0.0
    if r.get("earnings") is not None and r["earnings"] <= 10:
        score += 3
    score += 2 if abs(r["chg_5d"]) >= 5 else 1 if abs(r["chg_5d"]) >= 3 else 0
    if r["rsi"] <= 30 or r["rsi"] >= 70:
        score += 2
    if r["trend"] != "neutre":
        score += 1
    if r.get("idea"):
        score += 1.5
    return score + min(len(r.get("news", [])), 2) * 0.5


# ── données Alpaca ──────────────────────────────────────────────────────────
def _paged(client: Alpaca, path: str, params: dict, key: str, max_pages: int = 10) -> dict:
    out, params = {}, dict(params)
    for _ in range(max_pages):
        d = client.data(path, params)
        for s, items in (d.get(key) or {}).items():
            if isinstance(items, list):
                out.setdefault(s, []).extend(items)
            else:
                out[s] = items
        token = d.get("next_page_token")
        if not token:
            break
        params["page_token"] = token
    return out


def daily_closes(client: Alpaca, symbols: list[str], crypto: bool = False) -> dict:
    if not symbols:
        return {}
    start = (datetime.now(timezone.utc) - timedelta(days=120)).strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {"symbols": ",".join(symbols), "timeframe": "1Day", "start": start, "limit": 10000}
    if crypto:
        bars = _paged(client, "/v1beta3/crypto/us/bars", params, "bars")
    else:
        params.update(feed="iex", adjustment="split")
        bars = _paged(client, "/v2/stocks/bars", params, "bars")
    return {s: [float(b["c"]) for b in bl] for s, bl in bars.items()}


def last_prices(client: Alpaca, symbols: list[str], crypto: bool = False) -> dict:
    if not symbols:
        return {}
    try:
        if crypto:
            d = client.data("/v1beta3/crypto/us/latest/trades", {"symbols": ",".join(symbols)})
            return {s: float(t["p"]) for s, t in (d.get("trades") or {}).items()}
        d = client.data("/v2/stocks/snapshots", {"symbols": ",".join(symbols), "feed": "iex"})
        return {s: float(v["latestTrade"]["p"]) for s, v in d.items()
                if v and v.get("latestTrade")}
    except Exception:
        return {}


OCC_RE = re.compile(r"^([A-Z.]+?)(\d{6})([CP])(\d{8})$")


def parse_occ(symbol: str) -> dict | None:
    """AAPL261120C00250000 → sous-jacent, échéance, call/put, strike."""
    m = OCC_RE.match(symbol)
    if not m:
        return None
    root, ymd, cp, strike = m.groups()
    return {"underlying": root, "expiry": datetime.strptime(ymd, "%y%m%d").date(),
            "type": "call" if cp == "C" else "put", "strike": int(strike) / 1000}


def option_chain(client: Alpaca, underlying: str, spot: float) -> list[dict]:
    today = date.today()
    params = {"feed": "indicative", "limit": 1000,
              "expiration_date_gte": (today + timedelta(days=DTE_MIN)).isoformat(),
              "expiration_date_lte": (today + timedelta(days=DTE_MAX)).isoformat(),
              "strike_price_gte": round(spot * 0.8, 2),
              "strike_price_lte": round(spot * 1.2, 2)}
    snaps = _paged(client, f"/v1beta1/options/snapshots/{underlying}", params,
                   "snapshots", max_pages=5)
    chain = []
    for sym, snap in snaps.items():
        c = parse_occ(sym)
        if not c:
            continue
        q = (snap or {}).get("latestQuote") or {}
        c.update(symbol=sym, bid=_num(q.get("bp"), 0.0), ask=_num(q.get("ap"), 0.0),
                 iv=(snap or {}).get("impliedVolatility"))
        chain.append(c)
    return chain


def pick_expiry(chain: list[dict]):
    exps = sorted({c["expiry"] for c in chain})
    if not exps:
        return None
    target = date.today() + timedelta(days=DTE_TARGET)
    return min(exps, key=lambda e: abs((e - target).days))


def nearest(contracts: list[dict], target: float):
    return min(contracts, key=lambda c: abs(c["strike"] - target)) if contracts else None


def atm_iv(chain: list[dict], spot: float) -> float | None:
    exp = pick_expiry(chain)
    if not exp:
        return None
    ivs = []
    for kind in ("call", "put"):
        c = nearest([x for x in chain if x["expiry"] == exp and x["type"] == kind], spot)
        if c and c.get("iv"):
            ivs.append(float(c["iv"]))
    return sum(ivs) / len(ivs) if ivs else None


# ── actualité (fail-safe) ───────────────────────────────────────────────────
def stock_news(client: Alpaca, symbols: list[str]) -> dict:
    by = {s: [] for s in symbols}
    try:
        d = client.data("/v1beta1/news", {"symbols": ",".join(symbols), "limit": 50,
                                          "sort": "desc"})
        for n in d.get("news", []):
            for s in n.get("symbols", []):
                if s in by and len(by[s]) < 3:
                    by[s].append(n.get("headline", "")[:120])
    except Exception:
        pass
    return by


def market_headlines(client: Alpaca) -> list[str]:
    try:
        d = client.data("/v1beta1/news", {"limit": 8, "sort": "desc"})
        return [n.get("headline", "")[:120] for n in d.get("news", [])]
    except Exception:
        return []


def crypto_headlines(timeout: float = 8.0) -> list[str]:
    titles = []
    for url in CRYPTO_RSS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                root = ElementTree.fromstring(r.read())
            titles += [i.findtext("title", "").strip() for i in root.iter("item")][:10]
        except Exception:
            pass
    return [t for t in titles if t]


def crypto_news_for(symbol: str, headlines: list[str]) -> list[str]:
    kws = CRYPTO_KEYWORDS.get(symbol, [])
    return [h[:120] for h in headlines
            if any(re.search(rf"\b{k}\b", h.lower()) for k in kws)][:3]


def fear_greed() -> str:
    try:
        req = urllib.request.Request("https://api.alternative.me/fng/",
                                     headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as r:
            d = json.loads(r.read())["data"][0]
        return f"{d['value']}/100 ({d['value_classification']})"
    except Exception:
        return "indisponible"


_earnings_cache: dict = {}


def days_to_earnings(symbol: str) -> int | None:
    if symbol in ETFS:
        return None
    today = date.today()
    cached = _earnings_cache.get(symbol)
    if cached and cached[0] == today:
        return cached[1]
    val = None
    try:
        import yfinance as yf
        cal = yf.Ticker(symbol).calendar
        dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if dates:
            d0 = dates[0]
            d0 = d0.date() if isinstance(d0, datetime) else d0
            days = (d0 - today).days
            val = days if 0 <= days <= 60 else None
    except Exception:
        pass
    _earnings_cache[symbol] = (today, val)
    return val


# ── photo du marché ─────────────────────────────────────────────────────────
def build_snapshot(client: Alpaca, market_open: bool, stocks: list[str],
                   cryptos: list[str], open_underlyings: list[str],
                   earnings=days_to_earnings, crypto_news=crypto_headlines,
                   fng=fear_greed, ideas: list[dict] | None = None) -> dict:
    rows, prices, chains = [], {}, {}
    ideas = {i["symbol"]: i for i in (ideas or [])}
    if market_open and stocks:
        closes = daily_closes(client, stocks)
        prices.update(last_prices(client, stocks))
        news = stock_news(client, stocks)
        cands = []
        for s in stocks:
            c = closes.get(s) or []
            if len(c) < 55 or s in open_underlyings:
                continue
            price = prices.get(s) or c[-1]
            prices[s] = price
            idea = ideas.get(s)
            kind = ("ETF" if s in ETFS else
                    ("idee presse, niche" if idea.get("niche") else "idee presse") if idea
                    else "action")
            cands.append({"symbol": s, "kind": kind,
                          "price": round(price, 2), **metrics(c, price),
                          "earnings": earnings(s),
                          "news": news.get(s) or (idea or {}).get("headlines", [])[:3],
                          "idea": bool(idea), "niche": bool(idea and idea.get("niche")),
                          "iv": None, "iv_hv": None})
        cands.sort(key=interest_score, reverse=True)
        for r in cands[:TOP_CANDIDATES]:
            try:
                chain = option_chain(client, r["symbol"], r["price"])
                chains[r["symbol"]] = chain
                iv = atm_iv(chain, r["price"])
                if iv:
                    r["iv"] = round(iv * 100, 1)
                    r["iv_hv"] = round(iv / r["hv"], 2) if r["hv"] else None
            except Exception:
                pass
            rows.append(r)

    if cryptos:
        cc = daily_closes(client, cryptos, crypto=True)
        prices.update(last_prices(client, cryptos, crypto=True))
        heads = crypto_news()
        crows = []
        for s in cryptos:
            c = cc.get(s) or []
            if len(c) < 55:
                continue
            price = prices.get(s) or c[-1]
            prices[s] = price
            crows.append({"symbol": s, "kind": "crypto", "price": round(price, 2),
                          **metrics(c, price, 365), "earnings": None,
                          "news": crypto_news_for(s, heads), "iv": None, "iv_hv": None})
        crows = [r for r in crows if r["symbol"] not in open_underlyings]
        crows.sort(key=interest_score, reverse=True)
        rows += crows[:TOP_CRYPTOS]

    macro = (f"Fear & Greed crypto: {fng()}\nTitres du marche: "
             + " ; ".join(market_headlines(client)[:6]))
    return {"rows": rows, "prices": prices, "chains": chains, "macro": macro,
            "hv": {r["symbol"]: r["hv"] for r in rows}}


# ── LLM ─────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = ("Tu es un gerant de portefeuille prudent et discipline. Tu choisis AU PLUS "
                 "un trade par cycle, et HOLD est souvent la meilleure reponse. "
                 "Tu reponds uniquement en JSON valide.")


def build_prompt(snap: dict, ctx: dict) -> str:
    lines = []
    for r in snap["rows"]:
        parts = [f"{r['symbol']} ({r['kind']})", f"prix {r['price']}", f"5j {r['chg_5d']:+}%",
                 f"RSI {r['rsi']}", f"tendance {r['trend']}",
                 f"vol. historique {r['hv'] * 100:.1f}%"]
        if r.get("iv"):
            parts.append(f"vol. implicite {r['iv']}% (IV/HV {r['iv_hv']})")
        if r.get("earnings") is not None:
            parts.append(f"RESULTATS dans {r['earnings']} j")
        if r.get("news"):
            parts.append("news: " + " / ".join(r["news"]))
        lines.append(" | ".join(parts))
    return f"""DATE: {datetime.now().strftime('%Y-%m-%d %H:%M')}
MARCHE ACTIONS US: {"OUVERT" if ctx['market_open'] else "FERME (seulement BUY_CRYPTO ou HOLD)"}
CAPITAL: ${ctx['equity']:.0f} | P&L du jour: {ctx['daily_pnl_pct']:+.2f}%
DEJA EN POSITION SUR: {", ".join(ctx['open_underlyings']) or "rien"}

CONTEXTE:
{snap['macro']}

MARCHES SURVEILLES (pre-tries par le code):
{chr(10).join(lines) or "aucun"}

STRATEGIES POSSIBLES:
- BULL_CALL_SPREAD : action, vue haussiere moderee (perte max = cout)
- BEAR_PUT_SPREAD : action, vue baissiere moderee (perte max = cout)
- LONG_STRADDLE : gros mouvement attendu dans un sens inconnu (resultats proches), seulement si IV/HV < 1.3
- LONG_STRANGLE : comme le straddle, moins cher mais demande un mouvement plus fort
- LONG_CALL_BUTTERFLY : action, le prix devrait rester proche du cours actuel (perte max = cout)
- BUY_STOCK : achat d'une action en tendance haussiere claire
- SHORT_STOCK : vente a decouvert d'une action en tendance baissiere claire
- BUY_CRYPTO : achat d'une crypto en tendance haussiere claire
- HOLD : rien de convaincant

REGLES:
- Jamais de straddle/strangle si IV/HV > 1.5 : options deja cheres. Principe : une position
  long gamma gagne 1/2*Gamma*S^2*(vol_realisee^2 - vol_implicite^2) : acheter de la vol
  n'a de sens que si la vol realisee attendue depasse la vol implicite payee.
- Pas de trade sur un sous-jacent deja en position.
- Options, BUY_STOCK et SHORT_STOCK uniquement si le marche actions est OUVERT.
- SHORT_STOCK jamais juste avant des resultats : prefere BEAR_PUT_SPREAD.
- "idee presse" = societe tres citee dans la presse financiere ces derniers jours (souvent
  une petite valeur). Exige une tendance qui confirme les news ; "niche" = peu liquide,
  la position sera automatiquement deux fois plus petite.
- confidence entre 0 et 1, sois honnete. Sous {MIN_CONFIDENCE}, choisis HOLD.

Reponds avec ce JSON:
{{"action": "OPEN" ou "HOLD", "strategy": "...", "underlying": "NVDA", "confidence": 0.7,
  "stop_loss_pct": 0.03, "take_profit_pct": 0.06, "reasoning": "une phrase en francais"}}
stop_loss_pct / take_profit_pct : seulement pour BUY_STOCK, SHORT_STOCK, BUY_CRYPTO."""


def groq_decide(prompt: str, api_key: str, timeout: int = 60, system: str | None = None,
                max_tokens: int = 1500) -> dict:
    """Appelle Groq (API compatible OpenAI) ; bascule sur le petit modèle si quota."""
    last = ""
    custom = os.environ.get("GROQ_MODEL", "").strip()
    for model in ([custom] if custom else []) + GROQ_MODELS:
        payload = {
            "model": model, "temperature": 0.2, "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system or SYSTEM_PROMPT},
                         {"role": "user", "content": prompt}]}
        if "gpt-oss" in model:          # modèle à raisonnement : on le garde court
            payload["reasoning_effort"] = "low"
        body = json.dumps(payload).encode()
        req = urllib.request.Request(GROQ_URL, data=body, method="POST", headers={
            "Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
            "User-Agent": "quantlab"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.loads(r.read())
            d = json.loads(out["choices"][0]["message"]["content"])
            d["model"] = model
            return d
        except urllib.error.HTTPError as e:
            last = f"{model}: HTTP {e.code}"
            if e.code in (400, 404, 429, 503):   # retiré / non supporté / quota → suivant
                continue
            break
        except Exception as e:
            last = str(e)[:100]
            break
    return {"action": "HOLD", "reasoning": f"IA indisponible ({last})"}


# ── journal des trades ──────────────────────────────────────────────────────
def load_trades(path: str = STATE_PATH) -> list[dict]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_trades(trades: list[dict], path: str = STATE_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trades, f, indent=2, ensure_ascii=False)


def _close_record(trades: list[dict], trade_id: str, pnl, reason: str) -> dict | None:
    for t in trades:
        if t["id"] == trade_id:
            t.update(status="CLOSED", pnl=None if pnl is None else round(pnl, 2),
                     closed_at=now_iso(), close_reason=reason)
            return t
    return None


# ── stratégies options ──────────────────────────────────────────────────────
def leg_ok(c) -> bool:
    if not c or c["bid"] <= 0 or c["ask"] <= 0:
        return False
    mid = (c["bid"] + c["ask"]) / 2
    return (c["ask"] - c["bid"]) / mid <= MAX_LEG_SPREAD_PCT


def build_option_trade(strategy: str, chain: list[dict], spot: float):
    """Choisit les strikes dans la vraie chaîne d'options. Renvoie (plan, raison)."""
    exp = pick_expiry(chain)
    if not exp:
        return None, "Aucune echeance disponible"
    calls = [c for c in chain if c["expiry"] == exp and c["type"] == "call"]
    puts = [c for c in chain if c["expiry"] == exp and c["type"] == "put"]
    if strategy == "BULL_CALL_SPREAD":
        buy = nearest(calls, spot)
        sell = nearest([c for c in calls if buy and c["strike"] > buy["strike"]], spot * (1 + OTM_PCT))
        legs = [(buy, "buy"), (sell, "sell")]
    elif strategy == "BEAR_PUT_SPREAD":
        buy = nearest(puts, spot)
        sell = nearest([p for p in puts if buy and p["strike"] < buy["strike"]], spot * (1 - OTM_PCT))
        legs = [(buy, "buy"), (sell, "sell")]
    elif strategy == "LONG_STRADDLE":
        call = nearest(calls, spot)
        put = next((p for p in puts if call and p["strike"] == call["strike"]), None)
        legs = [(call, "buy"), (put, "buy")]
    elif strategy == "LONG_STRANGLE":
        call = nearest([c for c in calls if c["strike"] > spot], spot * (1 + OTM_PCT))
        put = nearest([p for p in puts if p["strike"] < spot], spot * (1 - OTM_PCT))
        legs = [(call, "buy"), (put, "buy")]
    elif strategy == "LONG_CALL_BUTTERFLY":    # (S-K1)+ - 2(S-K2)+ + (S-K3)+
        mid = nearest(calls, spot)
        low = nearest([c for c in calls if mid and c["strike"] < mid["strike"]], spot * (1 - OTM_PCT))
        high = nearest([c for c in calls if mid and low and
                        abs(c["strike"] - mid["strike"] - (mid["strike"] - low["strike"])) < 1e-9], 0) \
            if low else None
        legs = [(low, "buy"), (mid, "sell"), (high, "buy")]
    else:
        return None, f"Strategie inconnue: {strategy}"
    if any(c is None for c, _ in legs):
        return None, "Strikes introuvables dans la chaine"
    if not all(leg_ok(c) for c, _ in legs):
        return None, "Options pas assez liquides (ecart achat/vente trop grand)"
    ratio = [2 if (strategy == "LONG_CALL_BUTTERFLY" and s == "sell") else 1 for _, s in legs]
    debit = (sum(q * c["ask"] for (c, s), q in zip(legs, ratio) if s == "buy")
             - sum(q * c["bid"] for (c, s), q in zip(legs, ratio) if s == "sell"))
    if debit <= 0:
        return None, "Prix incoherent"
    if "SPREAD" in strategy:
        width = abs(legs[0][0]["strike"] - legs[1][0]["strike"])
    elif strategy == "LONG_CALL_BUTTERFLY":
        width = legs[1][0]["strike"] - legs[0][0]["strike"]   # gain max = K2 - K1 - débit
    else:
        width = None
    return {"strategy": strategy, "expiry": exp.isoformat(),
            "legs": [{"symbol": c["symbol"], "side": s, "strike": c["strike"],
                      "type": c["type"], "ratio": q, "iv": c.get("iv")}
                     for (c, s), q in zip(legs, ratio)],
            "debit": round(debit, 2), "width": width,
            "max_loss": round(debit * 100, 2),
            "max_gain": None if width is None else round((width - debit) * 100, 2)}, "OK"


def course_analysis(plan: dict, spot: float, hv: float | None) -> dict:
    """Analyse « cours EFREI » d'une structure : grecques, points morts, P(gain),
    et rapport prix payé / valeur Black-Scholes à la vol réalisée (avantage de vol)."""
    T = max((date.fromisoformat(plan["expiry"]) - date.today()).days, 1) / 365
    legs = []
    for l in plan["legs"]:
        sign = 1 if l["side"] == "buy" else -1
        legs.append(pr.Leg(l["type"], l["strike"], sign * l.get("ratio", 1)))
    ivs = [float(l["iv"]) for l in plan["legs"] if l.get("iv")]
    sig_i = sum(ivs) / len(ivs) if ivs else (hv or 0.3)
    try:
        rep = pr.analyze_strategy(plan["strategy"], legs, spot, T, RISK_FREE, sig_i, hv or None)
        theo_real = (sum(l.qty * pr.bs_price(spot, l.K, T, RISK_FREE, hv, l.kind) for l in legs)
                     if hv else None)
        g = rep.greeks
        return {"iv": round(sig_i, 4), "hv": None if hv is None else round(hv, 4),
                "delta": round(g.delta * 100, 2), "gamma": round(g.gamma * 100, 4),
                "vega_per_pt": round(g.vega, 2), "theta_per_day": round(g.theta / 365 * 100, 2),
                "breakevens": rep.breakevens, "prob_profit": round(rep.prob_profit, 3),
                "vol_edge": (round(plan["debit"] / theo_real, 3)
                             if theo_real and theo_real > 0 else None)}
    except Exception:
        return {}


def mleg_order(legs: list[dict], qty: int, opening: bool) -> dict:
    body = {"order_class": "mleg", "qty": str(qty), "type": "market",
            "time_in_force": "day", "legs": []}
    for leg in legs:
        if opening:
            side = leg["side"]
            intent = "buy_to_open" if side == "buy" else "sell_to_open"
        else:
            side = "sell" if leg["side"] == "buy" else "buy"
            intent = "sell_to_close" if leg["side"] == "buy" else "buy_to_close"
        body["legs"].append({"symbol": leg["symbol"], "side": side,
                             "ratio_qty": str(leg.get("ratio", 1)),
                             "position_intent": intent})
    return body


# ── grecques en continu (recalculées à chaque cycle) ───────────────────────
def live_greeks(client: Alpaca, trades: list[dict]) -> dict:
    """Grecques du portefeuille de la poche IA, recalculées avec le spot et la vol
    implicite du moment (Black-Scholes, formules du cours).

    Options : Δ, Γ, ν, Θ, ρ par position (× 100 unités par contrat, × ratio, signe).
    Actions/crypto : delta = quantité détenue (Γ = ν = Θ = 0).
    Unités : delta en équivalent-actions et en $, gamma par $ de spot, vega en $ pour
    +1 point de vol, theta en $ par jour, rho en $ pour +1 point de taux.
    """
    open_t = [t for t in trades if t["status"] == "OPEN"]
    out = {"updated_at": now_iso(), "positions": [],
           "total": {"delta_usd": 0.0, "gamma_usd": 0.0, "vega": 0.0, "theta_day": 0.0, "rho": 0.0}}
    if not open_t:
        return out
    opt_syms = [l["symbol"] for t in open_t if t["kind"] == "options" for l in t["legs"]]
    snaps = {}
    if opt_syms:
        try:
            snaps = (client.data("/v1beta1/options/snapshots",
                                 {"symbols": ",".join(opt_syms), "feed": "indicative"})
                     .get("snapshots") or {})
        except Exception:
            snaps = {}
    stock_syms = sorted({t["underlying"] for t in open_t if "/" not in t["underlying"]})
    crypto_syms = sorted({t["underlying"] for t in open_t if "/" in t["underlying"]})
    spots = {**last_prices(client, stock_syms), **last_prices(client, crypto_syms, crypto=True)}
    try:
        held = {p["symbol"]: _num(p.get("qty"), 0.0) for p in client.positions()}
    except Exception:
        held = {}
    tot = out["total"]
    for t in open_t:
        S = spots.get(t["underlying"]) or t.get("spot_at_entry") or t.get("entry_price")
        if not S:
            continue
        g = pr.Greeks(0, 0, 0, 0, 0)
        if t["kind"] == "options":
            T = max((date.fromisoformat(t["expiry"]) - date.today()).days, 1) / 365
            for l in t["legs"]:
                snap = snaps.get(l["symbol"]) or {}
                iv = _num(snap.get("impliedVolatility"), 0.0) or _num(l.get("iv"), 0.0) \
                    or _num((t.get("analysis") or {}).get("iv"), 0.3)
                sign = 1 if l["side"] == "buy" else -1
                n_units = sign * l.get("ratio", 1) * t["qty"] * 100
                g = g + pr.greeks(S, l["strike"], T, RISK_FREE, iv, l["type"]).scale(n_units)
        else:
            qty = held.get(t["position_symbol"], 0.0)
            g = pr.Greeks(qty, 0, 0, 0, 0)
        row = {"id": t["id"], "strategy": t["strategy"], "underlying": t["underlying"],
               "spot": round(S, 2), "delta": round(g.delta, 2),
               "delta_usd": round(g.delta * S, 2), "gamma_usd": round(g.gamma * S, 4),
               "vega": round(g.vega / 100, 2), "theta_day": round(g.theta / 365, 2),
               "rho": round(g.rho / 100, 2)}
        out["positions"].append(row)
        for k in tot:
            tot[k] = round(tot[k] + row[k], 2)
    return out


def save_greeks(greeks: dict, path: str = GREEKS_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(greeks, f, indent=2, ensure_ascii=False)


def greek_limits_ok(greeks: dict, equity: float) -> tuple[bool, str]:
    t = greeks.get("total") or {}
    if abs(t.get("delta_usd", 0)) > MAX_DELTA_PCT * equity:
        return False, f"Exposition delta trop forte ({t['delta_usd']:,.0f} $)"
    if abs(t.get("vega", 0)) > MAX_VEGA_PCT * equity:
        return False, f"Exposition vega trop forte ({t['vega']:,.0f} $/pt de vol)"
    if -t.get("theta_day", 0) > MAX_THETA_PCT * equity:
        return False, f"Érosion theta trop forte ({t['theta_day']:,.0f} $/jour)"
    return True, "OK"


# ── filtre de risque ────────────────────────────────────────────────────────
def validate(d: dict, ctx: dict) -> tuple[bool, str]:
    strat, und = d.get("strategy"), d.get("underlying")
    if strat not in OPTION_STRATEGIES | SPOT_STRATEGIES:
        return False, f"Strategie inconnue: {strat}"
    if strat == "BUY_CRYPTO" and und not in ctx["cryptos"]:
        return False, f"{und} n'est pas une crypto de la poche"
    if strat != "BUY_CRYPTO" and und not in ctx["stocks"]:
        return False, f"{und} n'est pas une action de la poche"
    if _num(d.get("confidence"), 0.0) < MIN_CONFIDENCE:
        return False, f"Confiance trop faible ({d.get('confidence')})"
    if strat != "BUY_CRYPTO" and not ctx["market_open"]:
        return False, "Marche actions ferme"
    if und in ctx["open_underlyings"]:
        return False, f"Deja en position sur {und}"
    if len(ctx["open_underlyings"]) >= MAX_OPEN_TRADES:
        return False, "Nombre max de positions atteint"
    if ctx.get("greeks"):
        ok, why = greek_limits_ok(ctx["greeks"], ctx["equity"])
        if not ok:
            return False, why
    return True, "OK"


# ── ouverture ───────────────────────────────────────────────────────────────
def open_option_trade(client: Alpaca, d: dict, ctx: dict, snap: dict, sleep=time.sleep):
    und, strat = d["underlying"], d["strategy"]
    spot = snap["prices"].get(und)
    if not spot:
        return None, f"Prix de {und} indisponible"
    chain = snap["chains"].get(und) or option_chain(client, und, spot)
    plan, why = build_option_trade(strat, chain, spot)
    if not plan:
        return None, why
    if plan["max_gain"] is not None and plan["max_gain"] < plan["max_loss"]:
        return None, f"Gain max ({plan['max_gain']}$) < perte max ({plan['max_loss']}$)"
    analysis = course_analysis(plan, spot, snap.get("hv", {}).get(und))
    if strat in LONG_VOL and analysis.get("vol_edge") and analysis["vol_edge"] > VOL_EDGE_MAX:
        return None, (f"Vol trop chère : débit {plan['debit']:.2f} = {analysis['vol_edge']:.2f}× "
                      f"la valeur Black-Scholes à la vol réalisée (max {VOL_EDGE_MAX})")
    budget = ctx["equity"] * OPTION_MAX_RISK_PCT * (NICHE_SIZE if und in ctx.get("niche", ()) else 1)
    qty = min(MAX_CONTRACTS, int(budget // plan["max_loss"]))
    if qty < 1:
        return None, f"1 contrat risque {plan['max_loss']:.0f}$, max autorise {budget:.0f}$"
    filled = client.wait_fill(client.submit_order(mleg_order(plan["legs"], qty, True))["id"],
                              sleep=sleep)
    if not filled:
        return None, "Ordre options non rempli, annule"
    debit = plan["debit"]
    fp = abs(_num(filled.get("filled_avg_price"), 0.0))
    if 0 < fp < plan["debit"] * 3:
        debit = fp
    max_loss = debit * 100 * qty
    if plan["width"]:
        max_gain = (plan["width"] - debit) * 100 * qty
        tp, sl = SPREAD_TP_OF_MAX_GAIN * max_gain, SPREAD_SL_OF_COST * max_loss
    else:
        max_gain, tp, sl = None, VOL_TP_OF_COST * max_loss, VOL_SL_OF_COST * max_loss
    return {"id": uuid.uuid4().hex[:10], "opened_at": now_iso(), "status": "OPEN",
            "kind": "options", "strategy": strat, "underlying": und, "qty": qty,
            "expiry": plan["expiry"], "legs": plan["legs"], "cost": round(max_loss, 2),
            "max_loss": round(max_loss, 2),
            "max_gain": None if max_gain is None else round(max_gain, 2),
            "tp_amount": round(tp, 2), "sl_amount": round(sl, 2), "spot_at_entry": spot,
            "analysis": analysis,
            "source": "presse" if und in ctx.get("ideas", ()) else None,
            "niche": und in ctx.get("niche", ()),
            "confidence": d.get("confidence"), "reasoning": d.get("reasoning"),
            "pnl": None}, "OK"


def open_spot_trade(client: Alpaca, d: dict, ctx: dict, snap: dict, sleep=time.sleep):
    und, strat = d["underlying"], d["strategy"]
    is_crypto, is_short = strat == "BUY_CRYPTO", strat == "SHORT_STOCK"
    sl = min(max(_num(d.get("stop_loss_pct"), 0.03), 0.02), 0.05)
    tp = max(_num(d.get("take_profit_pct"), 0.0), sl * 1.5)
    size = SPOT_MAX_PCT * (NICHE_SIZE if und in ctx.get("niche", ()) else 1)
    if is_short:
        asset = client.asset(und)
        if not (asset.get("shortable") and asset.get("easy_to_borrow")):
            return None, f"{und} ne peut pas etre vendue a decouvert"
        price = snap["prices"].get(und)
        if not price:
            return None, f"Prix de {und} indisponible"
        qty = int(ctx["equity"] * size // price)
        if qty < 1:
            return None, f"Capital trop petit pour vendre 1 {und}"
        notional = qty * price
        body = {"symbol": und, "qty": str(qty), "side": "sell", "type": "market",
                "time_in_force": "day"}
    else:
        notional = min(ctx["equity"] * size, ctx["cash"] * 0.95)
        if notional < 10:
            return None, "Pas assez de cash disponible"
        body = {"symbol": und, "notional": f"{notional:.2f}", "side": "buy",
                "type": "market", "time_in_force": "gtc" if is_crypto else "day"}
    if not client.wait_fill(client.submit_order(body)["id"], sleep=sleep):
        return None, "Ordre non rempli, annule"
    return {"id": uuid.uuid4().hex[:10], "opened_at": now_iso(), "status": "OPEN",
            "kind": "crypto" if is_crypto else "short" if is_short else "stock",
            "strategy": strat, "underlying": und, "position_symbol": position_symbol(und),
            "cost": round(notional, 2), "entry_price": snap["prices"].get(und),
            "sl_pct": round(sl, 4), "tp_pct": round(tp, 4),
            "source": "presse" if und in ctx.get("ideas", ()) else None,
            "niche": und in ctx.get("niche", ()),
            "confidence": d.get("confidence"), "reasoning": d.get("reasoning"),
            "pnl": None}, "OK"


# ── sorties automatiques ────────────────────────────────────────────────────
def manage_positions(client: Alpaca, trades: list[dict], market_open: bool,
                     sleep=time.sleep) -> list[str]:
    lines = []
    open_trades = [t for t in trades if t["status"] == "OPEN"]
    if not open_trades:
        return lines
    positions = {p["symbol"]: p for p in client.positions()}
    for t in open_trades:
        try:
            if t["kind"] == "options":
                legs = [positions.get(l["symbol"]) for l in t["legs"]]
                if not any(legs):
                    _close_record(trades, t["id"], None, "Position introuvable chez Alpaca")
                    lines.append(f"  {t['underlying']} : position introuvable, marquee fermee")
                    continue
                pnl = sum(_num(p["unrealized_pl"], 0.0) for p in legs if p)
                days_left = (date.fromisoformat(t["expiry"]) - date.today()).days
                reason = ("Take-profit" if pnl >= t["tp_amount"] else
                          "Stop-loss" if pnl <= -t["sl_amount"] else
                          "Echeance proche" if days_left <= CLOSE_BEFORE_EXPIRY_DAYS else None)
                if reason and market_open:
                    o = client.submit_order(mleg_order(t["legs"], t["qty"], False))
                    if client.wait_fill(o["id"], sleep=sleep):
                        _close_record(trades, t["id"], pnl, reason)
                        lines.append(f"  FERME {t['strategy']} {t['underlying']} "
                                     f"({reason}) PnL {pnl:+,.2f} $")
            else:
                p = positions.get(t["position_symbol"])
                if not p:
                    _close_record(trades, t["id"], None, "Position introuvable chez Alpaca")
                    lines.append(f"  {t['underlying']} : position introuvable, marquee fermee")
                    continue
                if t["kind"] in ("stock", "short") and not market_open:
                    continue
                pnl = _num(p.get("unrealized_pl"), 0.0)
                cost = abs(_num(p.get("cost_basis"), 0.0))
                plpc = pnl / cost if cost else _num(p.get("unrealized_plpc"), 0.0)
                reason = ("Take-profit" if plpc >= t["tp_pct"] else
                          "Stop-loss" if plpc <= -t["sl_pct"] else None)
                if reason:
                    client.close_position(t["underlying"])
                    _close_record(trades, t["id"], pnl, reason)
                    lines.append(f"  FERME {t['strategy']} {t['underlying']} "
                                 f"({reason}) PnL {pnl:+,.2f} $")
        except Exception as e:
            lines.append(f"  [!] gestion {t['underlying']} : {str(e)[:120]}")
    return lines


# ── idées presse (module 17) ────────────────────────────────────────────────
def fresh_ideas(path: str | None, exclude: set[str] = frozenset()) -> list[dict]:
    """Idées de ideas.json récentes (< IDEAS_MAX_AGE_H), hors univers déjà suivi."""
    if not path:
        return []
    from . import ideas as ideas_mod
    d = ideas_mod.load(path)
    try:
        age = (datetime.now() - datetime.fromisoformat(d["updated_at"])).total_seconds() / 3600
    except Exception:
        return []
    if age > IDEAS_MAX_AGE_H:
        return []
    out = [i for i in d.get("ideas", []) if i.get("symbol") and i["symbol"] not in exclude
           and i.get("score", 0) > 0]
    return out[:MAX_IDEAS]


def refresh_ideas(client: Alpaca, path: str, summarize=None, exclude: set[str] = frozenset(),
                  max_age_h: float = IDEAS_REFRESH_H) -> str:
    """Reconstruit ideas.json si trop ancien (fail-safe). Renvoie une ligne de rapport."""
    from . import ideas as ideas_mod
    d = ideas_mod.load(path)
    try:
        age = (datetime.now() - datetime.fromisoformat(d["updated_at"])).total_seconds() / 3600
    except Exception:
        age = float("inf")
    if age < max_age_h:
        return f"  Idées presse : à jour (il y a {age:.1f} h, {len(d.get('ideas', []))} idées)"
    try:
        rows = ideas_mod.build_ideas(client, exclude=exclude, summarize=summarize)
    except Exception as e:
        return f"  Idées presse : échec ({str(e)[:80]})"
    if rows or not d.get("ideas"):
        ideas_mod.save(rows, path)
    return f"  Idées presse : {len(rows)} idées rafraîchies"


# ── cycle ───────────────────────────────────────────────────────────────────
def run_cycle(client: Alpaca, decide, state_path: str = STATE_PATH,
              fund_universe: list[str] | None = None, sleep=time.sleep,
              greeks_path: str | None = None, ideas_path: str | None = None,
              **snapshot_kw) -> tuple[str, dict]:
    """Un cycle complet. `decide(prompt) -> dict` (Groq en prod, factice en test).

    Renvoie (rapport texte, dernière décision).
    """
    stocks, cryptos = universe(fund_universe)
    acct, clock = client.account(), client.clock()
    equity = float(acct["equity"])
    last_eq = _num(acct.get("last_equity"), equity) or equity
    daily = (equity / last_eq - 1) * 100
    market_open = bool(clock.get("is_open"))
    lines = ["=" * 78,
             "MODULE 14 — POCHE IA (Alpaca paper · LLM Groq) — EXPÉRIMENTAL",
             f"Exécuté : {now_iso()} | équité {equity:,.2f} $ ({daily:+.2f} % jour) | "
             f"bourse US {'OUVERTE' if market_open else 'FERMÉE'}"
             + (" | compte partagé avec le fonds (univers disjoints)" if fund_universe else ""),
             "=" * 78]

    trades = load_trades(state_path)
    lines += manage_positions(client, trades, market_open, sleep)
    save_trades(trades, state_path)

    greeks = live_greeks(client, trades)
    if greeks["positions"]:
        g = greeks["total"]
        lines.append(f"  Grecques du portefeuille : Δ {g['delta_usd']:+,.0f} $ | Γ {g['gamma_usd']:+,.2f} | "
                     f"ν {g['vega']:+,.0f} $/pt | Θ {g['theta_day']:+,.0f} $/j | ρ {g['rho']:+,.0f} $/pt")
    if greeks_path:
        save_greeks(greeks, greeks_path)

    decision: dict = {"action": "HOLD", "reasoning": ""}
    if daily <= -DAILY_LOSS_CAP * 100:
        decision["reasoning"] = f"Pause : perte du jour {daily:.2f} % ≤ -{DAILY_LOSS_CAP:.0%}"
        lines.append(f"  ⛔ {decision['reasoning']}")
        return "\n".join(lines), decision

    open_und = [t["underlying"] for t in trades if t["status"] == "OPEN"]
    ideas = fresh_ideas(ideas_path, exclude=set(stocks) | set(fund_universe or []))
    stocks = stocks + [i["symbol"] for i in ideas]
    if ideas:
        lines.append(f"  Idées presse ajoutées : {', '.join(i['symbol'] + ('*' if i.get('niche') else '') for i in ideas)}"
                     "  (* = niche, taille ÷ 2)")
    snap = build_snapshot(client, market_open, stocks, cryptos, open_und, ideas=ideas,
                          **snapshot_kw)
    ctx = {"equity": equity, "cash": _num(acct.get("non_marginable_buying_power"),
                                          _num(acct.get("cash"), 0.0)),
           "daily_pnl_pct": daily, "market_open": market_open,
           "open_underlyings": open_und, "stocks": stocks, "cryptos": cryptos,
           "greeks": greeks, "ideas": {i["symbol"] for i in ideas},
           "niche": {i["symbol"] for i in ideas if i.get("niche")}}
    lines.append(f"  Candidats envoyés à l'IA : {', '.join(r['symbol'] for r in snap['rows']) or 'aucun'}")

    decision = decide(build_prompt(snap, ctx))
    what = (f"{decision.get('strategy')} {decision.get('underlying')}"
            if decision.get("action") == "OPEN" else "HOLD")
    lines.append(f"  IA : {what} | confiance {decision.get('confidence')} | "
                 f"{decision.get('reasoning', '')}")
    if decision.get("action") != "OPEN":
        return "\n".join(lines), decision

    ok, why = validate(decision, ctx)
    if ok:
        try:
            opener = (open_option_trade if decision["strategy"] in OPTION_STRATEGIES
                      else open_spot_trade)
            rec, why = opener(client, decision, ctx, snap, sleep=sleep)
        except AlpacaError as e:
            rec, why = None, f"Alpaca : {str(e)[:140]}"
        if rec:
            trades.append(rec)
            save_trades(trades, state_path)
            a = rec.get("analysis") or {}
            extra = (f" | points morts {a.get('breakevens')} | P(gain) "
                     f"{a.get('prob_profit', 0):.0%} | Θ {a.get('theta_per_day')} $/j"
                     if a else "")
            lines.append(f"  OUVERT {rec['strategy']} {rec['underlying']} | coût "
                         f"{rec['cost']:,.0f} ${extra}")
            return "\n".join(lines), decision
    lines.append(f"  Trade refusé : {why}")
    decision["refused"] = why
    return "\n".join(lines), decision


def summary(state_path: str = STATE_PATH) -> dict:
    """Statistiques de la poche IA pour le tableau de bord."""
    trades = load_trades(state_path)
    closed = [t for t in trades if t["status"] == "CLOSED" and t.get("pnl") is not None]
    wins = sum(1 for t in closed if t["pnl"] > 0)
    greeks = None
    gp = os.path.join(os.path.dirname(state_path), "ai_greeks.json")
    if os.path.exists(gp):
        try:
            with open(gp, encoding="utf-8") as f:
                greeks = json.load(f)
        except Exception:
            pass
    return {"trades": trades, "open": [t for t in trades if t["status"] == "OPEN"],
            "greeks": greeks,
            "closed": closed, "realized": round(sum(t["pnl"] for t in closed), 2),
            "win_rate": wins / len(closed) if closed else None}


def main_cycle(fund_universe_if_shared: list[str]) -> tuple[str, dict]:
    """Point d'entrée CLI : choisit le compte et les clés depuis l'environnement."""
    load_env()
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    client = Alpaca.from_env("ALPACA_AI", label="poche IA")
    shared = None
    if client is None:
        client = Alpaca.from_env("ALPACA", label="poche IA (compte du fonds)")
        shared = fund_universe_if_shared
    if client is None or not groq_key:
        missing = [n for n, ok in (("ALPACA_API_KEY/ALPACA_SECRET_KEY", client),
                                   ("GROQ_API_KEY", groq_key)) if not ok]
        return f"Poche IA non lancée : clé(s) manquante(s) : {', '.join(missing)}", {}
    from .ideas import IDEAS_PATH, groq_summarizer
    note = refresh_ideas(client, IDEAS_PATH, summarize=groq_summarizer(groq_key),
                         exclude=set(STOCKS) | set(shared or []))
    rep, dec = run_cycle(client, lambda p: groq_decide(p, groq_key), fund_universe=shared,
                         greeks_path=GREEKS_PATH, ideas_path=IDEAS_PATH)
    return rep + "\n" + note, dec
