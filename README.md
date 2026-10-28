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

## Courtier réel Alpaca & poche IA (module 14)

### Miroir du fonds sur Alpaca

```powershell
python -m quantlab live --broker alpaca   # cycle normal + recopie des positions sur Alpaca
python -m quantlab alpaca                 # état des comptes Alpaca
```

Le compte papier (`paper_account.json`) reste la **source de vérité** : la logique
backtestée décide, puis `mirror.py` **réconcilie** le compte Alpaca sur ce
portefeuille (mis à l'échelle de l'équité Alpaca). Idempotent : un ordre raté est
rattrapé au cycle suivant, un ordre en attente n'est jamais doublé, seuls les
actifs de l'univers du fonds sont touchés. Sans clés, le miroir est ignoré.

### Poche IA (expérimentale, séparée du fonds)

```powershell
python -m quantlab ai                     # un cycle : gestion des positions + décision IA
```

Un LLM (Groq) choisit au plus un trade par cycle parmi : bull call spread, bear put
spread, straddle, strangle (perte max = coût payé), achat d'action, vente à découvert,
achat crypto. Le code garde la main sur le risque : pré-tri de ~50 actions/ETF vers
12 candidats, strikes choisis dans la vraie chaîne d'options, filtre de risque
(confiance ≥ 0,65, max 5 positions, 5 % de risque max par trade options, stop 2–5 %),
sorties automatiques (take-profit, stop-loss, fermeture 5 jours avant l'échéance).
Ses décisions ne sont **pas backtestables** : elles sont mesurées en réel et
comparées au fonds sur le tableau de bord, jamais mélangées avec lui.

Workflow : [ai-cycle.yml](.github/workflows/ai-cycle.yml) — toutes les 30 min pendant la
séance US, toutes les 3 h sinon (crypto). Journal : `ai_trades.json`, `ai_logs/`.

### Vente d'options et delta hedging (poche IA)

Comme les gérants : on **achète** des options quand elles sont bon marché (IV/HV ≤ 1 → bull call
spread / bear put spread de préférence à l'action), on les **vend** quand elles sont chères :
`BULL_PUT_SPREAD`, `BEAR_CALL_SPREAD`, `IRON_CONDOR` (perte max connue) et `COVERED_CALL` (call
vendu sur ≥ 100 actions détenues par la poche). Garde-fous : prime ≥ 1,05 × sa valeur
Black-Scholes à la vol réalisée, prime ≥ 20 % de la largeur du spread, jamais de vente si des
résultats tombent dans les 45 jours, rachat à 50 % de la prime gagnée, stop à 1 × la prime.
Un covered call est toujours racheté avant la vente des actions qu'il couvre.

**Delta hedging** : les stratégies neutres (straddle, strangle, iron condor, butterfly) sont
rendues delta-neutres à chaque cycle en achetant / vendant l'action (dès que |Δ $| dépasse
0,5 % de l'équité) ; la couverture est soldée à la fermeture et comptée dans le P&L. Sur un
straddle, c'est du « gamma scalping » : gain ≈ ½·Γ·S²·(σ_réalisée² − σ_implicite²).

### Clés (jamais dans le code)

| Où | Comment |
|----|---------|
| GitHub Actions | Settings → Secrets and variables → Actions : `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `GROQ_API_KEY` (+ optionnel `ALPACA_AI_API_KEY`, `ALPACA_AI_SECRET_KEY` pour un 2ᵉ compte paper dédié à la poche IA) |
| En local | fichier `.env` à la racine (ignoré par git), mêmes noms |

Sans compte dédié, la poche IA partage le compte du fonds avec un univers disjoint
(elle ne touche jamais SPY, QQQ, GLD, TLT, les ETF sectoriels ni BTC).

## Simulation du fonds & tableau de bord

```powershell
python -m quantlab fund --capital 100000 --years 10        # entraînement : le fonds complet sur l'historique
python -m quantlab dashboard --capital 100000 --open       # tableau de bord HTML (état de l'argent)
```

`fund` rejoue la logique exacte du mode live sur l'historique : 5 actifs,
3 stratégies en parallèle, capital partagé, max 5 positions, risque 1 %/trade.
`dashboard` génère le **terminal du gérant** `docs/index.html` (et l'ancienne page
`docs/classic.html`) — régénéré à chaque cycle IA (toutes les 30 min en séance) et
chaque soir par le workflow live, **publié en ligne via GitHub Pages** :
<https://victorhaegeman.github.io/hedg_fund/>

Onglets du terminal (page autonome, sans dépendance externe) :

| Onglet | Contenu |
|---|---|
| Vue d'ensemble | VL, P&L jour / mois / année vs S&P 500, cash, exposition, drawdown, répartition classe / secteur, alertes (stop proche, résultats, limites) |
| Positions | fonds + poche IA, poids, P&L, distance au stop, risque engagé, grecques des options |
| Risque | VaR / CVaR 95 % du portefeuille actuel, vol, bêta, stress (actions, taux, bitcoin, or, vol), limites, corrélations |
| Grecques | Δ Γ ν Θ ρ du portefeuille + options ATM 30 j de chaque actif (IV Alpaca sinon vol réalisée) |
| Marchés | ~100 actifs (univers du fonds, actions screenées, idées presse, 13 cryptos, univers IA). **Un clic** → graphique 6 mois, indicateurs, signaux des 3 stratégies, résultats, news, calculateur Black-Scholes (strike / jours / vol réglables) |
| Idées presse | module 17 (ci-dessous) |
| Activité, Fiche, Contexte | historique des trades, factsheet brute / nette, VIX, taux, dollar, pétrole, or, titres de presse |

### Idées « niche » tirées de la presse (module 17)

```powershell
python -m quantlab ideas            # idées des dernières 48 h (+ résumé IA si GROQ_API_KEY)
```

Source gratuite : le fil Benzinga d'Alpaca (centaines de sociétés par jour, beaucoup de
petites valeurs). Bloomberg, WSJ et X/Grok sont payants : non utilisés. Filtres : prix ≥ 5 $,
≥ 10 M$ échangés par jour ; score = attention (nombre d'articles) + ton des titres + tendance.
Les idées (≤ 8, rafraîchies toutes les 3 h) entrent dans la **poche IA**, jamais dans le
fonds principal ; une valeur « niche » (< 50 M$/jour) y est prise avec une **taille ÷ 2**.
Elles ne sont pas backtestables (l'actualité passée ne se rejoue pas sans biais).

Résultat de l'entraînement 2016→2026 avec 100 000 $ : équité finale ~474 000 $
(CAGR +16,8 %/an, Sharpe 0,94, max drawdown −20,3 %, 333 trades, win rate 60 %).
Performance simulée — aucune garantie future.

## Trésorerie : pas de cash qui dort (comme un vrai fonds)

Le cash que les stratégies n'utilisent pas est placé automatiquement à chaque cycle :
jusqu'à **30 % dans le S&P 500 (SPY, « socle »)**, le reste en **bons du Trésor US
très courts (BIL)**, 1 % gardé en cash. Les stratégies restent prioritaires : dès qu'une
entrée a besoin d'argent, on revend BIL puis SPY (`quantlab/treasury.py`). Ces lignes
ne comptent pas dans le nombre max de positions ni dans les statistiques de trades, et
sont recopiées sur Alpaca par le miroir (25 % du compte Alpaca restent libres pour la
poche IA quand elle partage le compte). Options : `--core-pct 0.3 --sweep BIL`
(`--core-pct 0 --sweep none` = ancien comportement).

Test 2007 → 2026 (univers sans biais, avant frais) :

| | Gain / an | Volatilité | Sharpe | Pire baisse | Bat le S&P (13 dates de départ) |
|---|---|---|---|---|---|
| S&P 500 | 10,6 % | 19,7 % | 0,54 | −55 % | — |
| Fonds sans trésorerie | 10,6 % | 13,1 % | 0,73 | −21 % | 0/13 |
| Fonds + socle 30 % + BIL | 11,7 % | 14,0 % | 0,76 | −28 % | 4/13 |
| Fonds + socle 40 % + BIL | 12,0 % | 14,5 % | 0,76 | −31 % | 8/13 |

Les autres stratégies de hedge funds testées (tendance multi-actifs façon CTA, rotation
sectorielle, parité de risque, vente de puts) réduisent le risque mais rapportent moins
sur la période : non retenues pour l'instant.

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

## Fonds d'investissement : fiche, frais, risque, actions (modules 16 & factsheet)

```powershell
python -m quantlab factsheet --profiles   # VL, brut vs net de frais, risque, activité, profils
python -m quantlab bonds                  # module 16 : obligations, duration, convexité, forwards
python -m quantlab options --symbol SPY   # module 15 : Black-Scholes, grecques, stratégies
python -m quantlab live --with-stocks     # ETF + BTC + actions screenées (mode du live)
```

- **Factsheet** : valeur de part base 100, frais 2 %/an + 20 % de performance au-dessus
  du plus haut historique (high-water mark), Sharpe, Sortino, Calmar, VaR/CVaR 95 %,
  bêta/alpha/corrélation vs S&P 500, rendements mensuels, rotation (turnover), durée de
  détention, profils de risque. Publiée sur le tableau de bord.
- **Actions individuelles dans le fonds** (`--with-stocks`, activé en live) : l'univers
  fixe (ETF + BTC) reste prioritaire ; s'y ajoutent les actions les plus liquides du
  screener TradingView, au plus la moitié des places, sans entrée à moins de 5 jours des
  résultats (risque de gap), et une seule ligne par émetteur (GOOG/GOOGL dédoublonnés
  par corrélation). Les backtests restent sur les ETF (pas de biais de sélection).
- **Grecques en continu** (poche IA) : Δ, Γ, ν, Θ, ρ de chaque position d'options et du
  portefeuille recalculées à chaque cycle (spot et vol implicite du moment), publiées
  dans `ai_greeks.json` et sur le tableau de bord, avec limites de risque (|Δ| ≤ 50 %
  de l'équité, |ν| ≤ 1 % par point de vol, Θ ≥ -0,2 %/jour) qui bloquent les nouveaux trades.

### Résultats des tests d'optimisation (oct. 2026, 13 dates de départ, 2016→2026)

| Variante | Rendement/an (médiane) | Sharpe | Pire baisse | Décision |
|---|---|---|---|---|
| Config live (risque 1 %) | 11,5 % | 0,72 | −21,5 % | conservée |
| Stop suiveur | 10,1 % | 0,78 | −21,9 % | non : rapporte moins, ne réduit pas les creux |
| Vol-targeting 12 % | 9,4 % | 0,53 | −33,7 % | non |
| + ETF obligataires (IEF, SHY, LQD, HYG, TIP) | 10,2 % | 0,69 | −21,3 % | non |
| Risque 0,5 %/trade (profil prudent) | 10,4 % | 0,84 | −15,0 % | meilleur risque ajusté — au choix du gérant |

Un seul backtest varie de ±1 %/an selon des micro-différences de données : seules les
conclusions stables sur toutes les dates de départ sont retenues. Net de frais 2/20, la
simulation rend ~6 %/an contre ~15 % pour le S&P 500 sur la période : un produit
commercialisable demanderait des frais plus bas ou un meilleur rendement.

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
  alpaca.py       # connecteur courtier Alpaca (REST, clés via env)
  mirror.py       # miroir du compte papier sur Alpaca (réconciliation)
  ai_trader.py    # Module 14 — poche IA (options, short, news, Groq, grecques en continu)
  pricing.py      # Module 15 — pricing d'options, grecques, stratégies (cours EFREI)
  bonds.py        # Module 16 — obligations, duration, convexité, forwards (cours EFREI)
  factsheet.py    # fiche du fonds : VL, frais, risque, activité, profils
  cli.py          # point d'entrée argparse
tests/test_all.py # 19 tests (déterministes, hors-ligne) dont absence de look-ahead
.github/workflows # CI pytest + cycle live quotidien automatisé
```

Points de rigueur du backtester : signal à la clôture J → exécution à l'open J+1
(pas de look-ahead, testé), stops/targets vérifiés en intraday avec gestion des
gaps, position sizing sur l'équité courante, commissions 5 bps par côté.
