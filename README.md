# quantlab — Boîte à outils quantitative de trading

Implémentation fonctionnelle des 12 modules d'analyse de trading (génération de
stratégies → détection d'alpha), en Python, avec données réelles Yahoo Finance
(cache local) et fallback synthétique hors-ligne.

> ⚠️ **Avertissement** : outil d'étude et de recherche. Rien ici n'est un conseil
> en investissement. Les performances passées ou simulées ne préjugent pas des
> performances futures.

## Installation

```powershell
pip install -r requirements.txt
python -m pytest tests/ -q        # vérification : 14 tests
```

## Les 12 modules

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
| 11 | Stratégie macro (taux ^TNX, inflation TIP/IEF, croissance XLY/XLP, quadrants) | `python -m quantlab macro` |
| 12 | Détection d'alpha/edge (4 anomalies mesurées avec t-stats, 2 stratégies) | `python -m quantlab alpha --symbol SPY` |

Tout enchaîner sur un actif :

```powershell
python -m quantlab all --strategy trend --symbol SPY
```

## Mode marché réel (paper trading)

Le système trade le **vrai marché** en compte papier : données réelles
rafraîchies à chaque exécution, ordres simulés, stops/targets suivis, état
persistant dans `paper_account.json`.

```powershell
python -m quantlab live              # un cycle : gère les positions + nouvelles entrées
python -m quantlab account           # état du compte (équité, positions, historique)
python -m quantlab live --reset      # remet le compte à $10 000
```

Logique d'un cycle : télécharge les dernières données (cache ignoré) → vérifie
stop/target/signaux de sortie des positions ouvertes → ouvre les nouvelles
positions dont le signal vient de se déclencher **et** dont le régime de marché
(module 4) est favorable, max 3 positions, sizing 1 % de risque par trade.

**Sécurité** : si les données réelles d'un symbole sont indisponibles, le
symbole est ignoré pour le cycle — aucun ordre n'est jamais passé sur des prix
synthétiques (testé).

**Automatisation** : le workflow GitHub Actions
[live-cycle.yml](.github/workflows/live-cycle.yml) exécute le cycle chaque jour
de semaine après la clôture US (22:30 UTC) et committe `paper_account.json` +
le journal dans `live_logs/`. Lançable aussi manuellement (onglet Actions →
Run workflow). En local, planifiable via le Planificateur de tâches Windows :
`python -m quantlab live` une fois par jour.

**Passage aux ordres réels** : il faudrait un compte de courtage avec API
(ex. Alpaca paper/live, Binance pour la crypto) et des clés API personnelles.
La couche `broker.py` est conçue pour être remplaçable par un connecteur réel —
mais aucun capital ne devrait y passer avant plusieurs mois de paper trading
satisfaisants.

## Univers dynamique — screener TradingView (source FORWARD)

Par défaut le cycle live trade un univers figé (4 ETF socle + 9 ETF sectoriels +
BTC). Le module `screener.py` permet de générer un **univers de candidats liquides
en direct** via le screener TradingView (lib `tradingview-screener`, sans clé API).

```powershell
python -m quantlab screen                       # liste les candidats actuels (inspection)
python -m quantlab screen --crypto              # cryptos les plus liquides
python -m quantlab live --screener --screen-limit 15   # cycle live sur l'univers screené
```

Les symboles TradingView (`NASDAQ:AAPL`, `BINANCE:BTCUSDT`) sont remappés au format
yfinance (`AAPL`, `BTC-USD`) avant chargement. Tout est **fail-safe** : si le screener
est indisponible (réseau/lib), le cycle retombe sur l'univers fixe.

> ⚠️ **Garde-fou d'intégrité** : le screener est une source *forward*, réservée au
> cycle **live**. Il n'alimente jamais `backtest` / `fund` / `validate` — sélectionner
> les actifs qui sortent du screener aujourd'hui puis les backtester dans le passé
> introduirait un biais de sélection / look-ahead. Voir l'en-tête de `screener.py`.

## Recherche assistée par IA — MCP TradingView (optionnel, local)

Pour **prototyper** des idées (screening, analyse technique, sentiment) directement
dans une session Claude Code locale, on peut brancher un serveur MCP TradingView. Ce
n'est **pas** utilisé par le pipeline automatisé (GitHub Actions reste headless) ; c'est
un outil de recherche pour l'humain. Workflow recommandé : *idée explorée via le MCP →
règles formalisées dans `strategies.py` → validée par le backtester sans look-ahead →
seulement ensuite éligible au live*.

Ajout opt-in dans un `.mcp.json` à la racine (lance un paquet externe via `uvx` ; à
ajouter délibérément) :

```json
{
  "mcpServers": {
    "tradingview": { "command": "uvx", "args": ["tradingview-mcp-server"] }
  }
}
```

> Le serveur de *pilotage de charts* TradingView Desktop (Chrome DevTools Protocol)
> a été volontairement écarté : il exige un desktop GUI + abonnement payant et n'a
> pas de backtest — incompatible avec un fonds headless. Rien ici n'est un conseil
> en investissement.

## Simulation du fonds & tableau de bord

```powershell
python -m quantlab fund --capital 100000 --years 10        # entraînement : le fonds complet sur l'historique
python -m quantlab dashboard --capital 100000 --open       # tableau de bord HTML (état de l'argent)
```

`fund` rejoue la logique exacte du mode live sur l'historique : 5 actifs,
3 stratégies en parallèle, capital partagé, max 5 positions, risque 1 %/trade.
`dashboard` génère `docs/index.html` (courbe d'équité, drawdown, rendements
annuels, positions du compte papier, derniers trades) — régénéré chaque jour
par le workflow live et **publié en ligne via GitHub Pages** :
<https://victorhaegeman.github.io/hedg_fund/>

Résultat de l'entraînement 2016→2026 avec 100 000 $ : équité finale ~474 000 $
(CAGR +16,8 %/an, Sharpe 0,94, max drawdown −20,3 %, 333 trades, win rate 60 %).
Performance simulée — aucune garantie future.

## Améliorations du modèle (overlays optionnels)

Deux surcouches robustes, **désactivées par défaut** (opt-in), pour améliorer le
risque ajusté sans toucher aux règles des stratégies :

- **Vol-targeting** (`--vol-target 0.12`) : module le risque par trade selon la
  volatilité réalisée récente du compte/fonds — on réduit l'exposition quand le
  marché s'agite, on la remonte quand il se calme (borné ×0,5–×2). Calculé sur le
  passé uniquement (pas de look-ahead).
- **Stop suiveur** (`--trail`) : stop en cliquet (ne descend jamais) qui suit la
  bande ATR de la stratégie, sur trend/breakout seulement (la mean reversion garde
  sa sortie RSI). Le risque initial reste figé pour le calcul du R-multiple.

Comparer avant/après sur le **même univers** (le vrai juge : Sharpe, Calmar, max DD) :

```powershell
python -m quantlab fund --compare --years 10                 # base vs amélioré
python -m quantlab fund --vol-target 0.12 --trail            # fonds avec overlays
python -m quantlab live --vol-target 0.12 --trail            # live avec overlays
```

> ⚠️ **À valider sur données réelles avant activation en production.** Sur les
> données synthétiques hors-ligne (sans persistance de tendance réaliste), ces
> overlays n'améliorent pas le risque ajusté ; et la cible de vol doit être réglée
> par rapport à la volatilité effective du fonds. C'est pourquoi le pipeline
> automatisé les laisse désactivés tant que `fund --compare` sur données réelles
> ne montre pas un gain net. Ne pas sur-ajuster les paramètres (piège classique).

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
  macro.py        # Module 11   alpha.py       # Module 12
  broker.py       # courtier papier persistant (paper_account.json)
  live.py         # cycle de trading sur le marché réel
  screener.py     # screener TradingView — univers FORWARD pour le live (fail-safe)
  cli.py          # point d'entrée argparse
tests/test_all.py # 19 tests (déterministes, hors-ligne) dont absence de look-ahead
.github/workflows # CI pytest + cycle live quotidien automatisé
```

Points de rigueur du backtester : signal à la clôture J → exécution à l'open J+1
(pas de look-ahead, testé), stops/targets vérifiés en intraday avec gestion des
gaps, position sizing sur l'équité courante, commissions 5 bps par côté.
