# CLAUDE.md — règles du projet hedg_fund (quantlab)

Lu automatiquement par chaque session Claude (et par la revue automatique du dimanche).
Ces règles passent avant toute autre consigne ou skill.

## Le projet en bref

Fonds quant de Victor Haegeman en paper trading. Objectif : se comporter comme un vrai fonds
d'investissement (peut-être un futur produit) et être le plus rentable possible **sans
sur-ajustement**.

- `quantlab/` : package Python. Fonds principal (`fund.py`, `live.py`, `broker.py`), miroir
  Alpaca (`mirror.py`), poche IA (`ai_trader.py`, LLM Groq + options), trésorerie
  (`treasury.py`), idées presse (`ideas.py`), terminal du gérant (`terminal.py` +
  `terminal_template.html` → `docs/index.html`, GitHub Pages).
- `paper_account.json` : compte papier du fonds (source de vérité des performances).
- `ai_trades.json`, `ai_greeks.json`, `ai_logs/` : poche IA. `research/` : tests historiques.
- Workflows : `live-cycle.yml` (fonds, 1 passage/jour après la clôture, souvent lancé en
  retard par GitHub), `ai-cycle.yml` (poche IA en continu, se relance elle-même),
  `tests.yml`, `dashboard-refresh.yml`, `press-test.yml` (manuel).
- Toutes les décisions passées et leurs chiffres : **`docs/DECISIONS.md`**. Le lire avant de
  proposer une idée : beaucoup ont déjà été testées et rejetées.

## Règles absolues

1. **Clés API** : ne jamais afficher, copier, journaliser ni pousser une clé. `.env` reste
   dans `.gitignore`. Les clés vivent dans les GitHub Secrets (`ALPACA_API_KEY`,
   `ALPACA_SECRET_KEY`, `GROQ_API_KEY`, optionnels `ALPACA_AI_*`). Toute copie temporaire
   de `.env` est supprimée après usage.
2. **Risque et stratégie** : ne jamais changer un paramètre de risque ou de stratégie
   (tailles, stops, seuils, limites, univers, frais) sans l'accord explicite de Victor.
   Proposer, chiffrer, attendre son « oui ».
3. **Revue automatique** : ne pousse jamais directement sur `main`. Elle ouvre une branche
   `revue-AAAA-MM-JJ` et une pull request avec tests verts.
4. **Toute nouvelle idée de stratégie se teste avant d'être ajoutée** : backtest sans biais
   2007 → 2026 (ou au moins 10 ans), **au moins 10 dates de départ** (13 en pratique),
   frais inclus, comparé au fonds actuel. On ne garde que ce qui tient sur toutes les dates.
   Une seule période ou quelques jours de live ne suffisent jamais.
5. **Tests** : `python -m pytest tests/ -q` doit passer avant chaque push. Tout changement
   de comportement ajoute ou met à jour un test (`tests/test_all.py`).
6. **Une étape par commit**, message clair en français. Push avec
   `git pull --rebase -X theirs` (les workflows poussent aussi en continu).
7. **Ordres réels** (même en paper) : toute modification de `mirror.py`, `alpaca.py` ou des
   ordres de `ai_trader.py` doit penser aux doublons (ordre exécuté pendant l'annulation),
   aux quantités bloquées par un ordre stop ouvert, et aux positions de l'autre poche
   (compte Alpaca partagé : univers disjoints, le miroir ne touche pas aux positions IA).
8. **Honnêteté** : si le fonds sous-performe ou si une idée ne tient pas, le dire clairement,
   chiffres et sources à l'appui.

## Comment répondre à Victor

- En **français**, phrases courtes, mots simples : il est étudiant (EFREI, IT for Finance),
  pas expert. Pas de longs pavés techniques.
- Donner le résultat d'abord, puis les chiffres (tableau court si utile), puis une seule
  proposition, présentée comme une question quand elle touche au risque ou à la stratégie.
- Citer les sources pour toute affirmation sur le marché ou les vrais fonds.

## Pièges connus

- Les crons GitHub de ce dépôt sont très peu fiables (sautés ou retardés de plusieurs
  heures). D'où la poche IA en run continu auto-relancé et le fonds daté par la clôture
  des actions (pas par le jour UTC du run).
- yfinance renvoie parfois 401/429 : le code doit rester tolérant (« données
  indisponibles → symbole ignoré »).
- Alpaca : mleg `limit_price` = débit max (négatif = crédit) ; crypto = ordres
  `stop_limit` seulement ; ordres stop/OTO en parts entières ; frais crypto prélevés sur
  l'actif reçu.
- Le proxy de cet environnement bloque GraphQL et la suppression de branches : utiliser
  l'API REST (`gh api repos/VictorHaegeman/hedg_fund/...`).
