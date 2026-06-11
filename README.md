# quantlab — Boîte à outils quantitative de trading

Implémentation fonctionnelle des 10 modules d'analyse de trading (génération de
stratégies → analyse des drawdowns), en Python, avec données réelles Yahoo Finance
(cache local) et fallback synthétique hors-ligne.

> ⚠️ **Avertissement** : outil d'étude et de recherche. Rien ici n'est un conseil
> en investissement. Les performances passées ou simulées ne préjugent pas des
> performances futures.

## Installation

```powershell
pip install -r requirements.txt
python -m pytest tests/ -q        # vérification : 14 tests
```

## Les 10 modules

| # | Module | Commande |
|---|--------|----------|
| 1 | Génération de stratégies (règles exactes, SL/TP, edge) | `python -m quantlab strategies` |
| 2 | Backtesting (CAGR, Sharpe, max DD, win rate, table annuelle) | `python -m quantlab backtest --strategy trend --symbol SPY --years 10` |
| 3 | Analyse risque/rendement (+3 réductions de risque, +2 boosts) | `python -m quantlab riskreward --strategy trend --symbol SPY` |
| 4 | Détection du régime de marché (tendance/vol/volume + reco) | `python -m quantlab regime --symbol BTC-USD` |
| 5 | Stratégie multi-facteurs (momentum/value/vol/trend, poids, rebal) | `python -m quantlab multifactor --symbols SPY QQQ GLD TLT BTC-USD` |
| 6 | Optimisation (grid search, comparaison avant/après) | `python -m quantlab optimize --strategy meanrev --symbol SPY` |
| 7 | Construction de portefeuille (allocation %, rendement, risque) | `python -m quantlab portfolio --risk-tolerance medium` |
| 8 | Trade setups (entrée/stop/TP/R:R + raisonnement) | `python -m quantlab setups` |
| 9 | Monte Carlo (P(perte), distribution, pires scénarios, verdict) | `python -m quantlab montecarlo --strategy trend --symbol SPY` |
| 10 | Analyse des drawdowns (profondeur, récupération, sizing) | `python -m quantlab drawdown --strategy trend --symbol SPY` |

Tout enchaîner sur un actif :

```powershell
python -m quantlab all --strategy trend --symbol SPY
```

## Options communes

- `--symbol SPY` / `--symbols SPY QQQ ...` — tickers Yahoo Finance (`BTC-USD`, `EURUSD=X`, …)
- `--years 10` — profondeur d'historique
- `--capital 10000` — capital de départ
- `--risk 1.0` — risque par trade en % de l'équité (1–2 % recommandé)

## Les 3 stratégies intégrées

| Clé | Stratégie | Logique |
|-----|-----------|---------|
| `trend` | Trend Following | Croisement EMA 20/50 au-dessus de la SMA 200, filtre ADX > 20 ; stop 2×ATR, target 4×ATR |
| `meanrev` | Mean Reversion | RSI(2) < 10 dans une tendance haussière (SMA 200) ; sortie RSI > 70, stop 3×ATR |
| `breakout` | Breakout Donchian | Cassure du plus haut 55 j confirmée volume ; sortie plus bas 20 j, stop 2×ATR |

## Architecture

```
quantlab/
  data.py         # yfinance + cache CSV (data_cache/) + données synthétiques hors-ligne
  indicators.py   # SMA, EMA, RSI (Wilder), ATR, MACD, Bollinger, ADX, Donchian
  strategies.py   # Module 1 — les 3 stratégies + specs complètes
  backtest.py     # Module 2 — moteur bar-par-bar (exécution à l'open J+1, stops intraday,
                  #            sizing en % de risque, commissions) + métriques
  risk.py         # Module 3    regime.py      # Module 4    multifactor.py # Module 5
  optimize.py     # Module 6    portfolio.py   # Module 7    setups.py      # Module 8
  montecarlo.py   # Module 9    drawdown.py    # Module 10
  cli.py          # point d'entrée argparse
tests/test_all.py # 14 tests (déterministes, hors-ligne) dont absence de look-ahead
```

Points de rigueur du backtester : signal à la clôture J → exécution à l'open J+1
(pas de look-ahead, testé), stops/targets vérifiés en intraday avec gestion des
gaps, position sizing sur l'équité courante, commissions 5 bps par côté.
