# Journal des décisions — hedg_fund

Une ligne par décision : ce qui a été décidé, le chiffre qui l'a justifié, où trouver le
détail. **À lire avant de proposer une idée** : beaucoup ont déjà été testées.

Sauf mention contraire, les backtests portent sur 2007 → 2026, 13 dates de départ, fonds
avec trésorerie, avant frais (README pour les tableaux complets). Référence actuelle du
fonds : **11,7 %/an, pire baisse −27,6 %, bat le S&P sur 4 des 13 départs**.

## Retenu

| Date | Décision | Pourquoi (chiffre) | Détail |
|---|---|---|---|
| 2026-10 | Garder la config live : risque 1 %/trade, 8 positions max | Variantes 6/10/12 positions ou risque différent : pas mieux sur toutes les dates (0,5 % = meilleur risque ajusté, au choix de Victor) | README « Résultats des tests d'optimisation » |
| 2026-10-26 | Trésorerie : socle SPY 30 % + BIL (bons du Trésor), coussin 1 %, bande 3 % | Pas de cash qui dort, comme les vrais fonds ; les stratégies restent prioritaires | `treasury.py`, README « Trésorerie » |
| 2026-10-26 | Frais par défaut = moyenne du marché HFR (1,33 % / 15,83 %) | 2/20 n'est plus la norme ; écart de performance nette trop grand | `factsheet.py` |
| 2026-10-28 | Poche IA : vente d'options (bull put, bear call, iron condor, covered call) + delta hedging des stratégies neutres | Ce que font les fonds d'options ; vente seulement si prime ≥ 1,05 × valeur théorique, crédit ≥ 20 % de la largeur, pas de résultats dans 45 j | `ai_trader.py`, README |
| 2026-10-28 | Options seulement « si vraiment intéressant » ; spreads préférés quand IV/HV ≤ 1 | Demande de Victor ; acheter des options chères n'est pas rentable | `ai_trader.py` |
| 2026-11-05 | IA : notes chiffrées par candidat, relance si réponse « par défaut » (0,58 partout) | L'IA restait bloquée en HOLD sans vraie analyse | `default_answer()` |
| 2026-11-05 | IA : stop adapté à la volatilité (≥ 1,5 × mouvement quotidien, 2–15 %), taille réduite pour garder 0,3 % de perte au stop | Stops de 3 % sur des actifs qui bougent de 4 %/jour sautaient sur du bruit | `spot_stop_and_size()` |
| 2026-11-05 | IA : au plus 2 cryptos ouvertes | 5/11 : UNI, ETH, XRP stoppées ensemble (−1 908 $), les cryptos bougent ensemble | `MAX_CRYPTO_OPEN` |
| 2026-11-05 | Vrais ordres stop chez Alpaca : fonds (stop GTC chaque soir + achat avec stop attaché OTO) et poche IA (stop GTC, crypto stop-limit) | Stops vendus le lendemain matin : 11,1 %/an ; vrais stops : 12,5 %/an ; sans stop le jour d'achat : 11,6 % | `mirror.py`, README « Vrais ordres stop » |
| 2026-11-05 | Poche IA en run continu qui se relance lui-même (30 min en séance, 3 h la nuit) | Les crons GitHub ne démarraient pas (0 à 2 cycles par séance au lieu de 13) | `ai-cycle.yml`, `chain_wait()` |
| 2026-11-05 | Courbe du fonds datée par la clôture des actions | 54 points sur 101 étaient décalés d'un jour (runs après minuit UTC) et comparés au mauvais S&P | `live.py` |
| 2026-11-05 | Presse = simple contexte pour l'IA (plus de bonus de classement) | Test 2 ans : voir « Rejeté » | `interest_score()` |

## Rejeté (ne pas reproposer sans élément nouveau)

| Date | Idée | Résultat | Détail |
|---|---|---|---|
| 2026-10 | Stop suiveur (`--trail`) | 9,2 %/an (10,1 % sur 2016→2026), ne réduit pas les creux | README |
| 2026-10 | Vol-targeting 12 % | 9,4 %/an, pire baisse −33,7 % | README |
| 2026-10 | Ajouter des ETF obligataires / internationaux (IEF, SHY, LQD, HYG, TIP, EFA, EEM, DBC) | 10,2 à 10,4 %/an, moins bien | README |
| 2026-10-28 | Filtre VIX (bloquer les achats en peur extrême) | 10,4 %/an, 0/13 : les meilleures entrées arrivent pendant la peur | README, `--vix-filter` (désactivé) |
| 2026-10-28 | Filtre crédit (HYG vs IEF) | 8,2 à 9,3 %/an | README |
| 2026-10-28 | Assurance puts S&P (toujours ou VIX bas) | 10,5 à 12,0 %, le gain disparaît avec des prix de puts réalistes | README |
| 2026-10 | Levier ×1,25 | ~14,2 %/an mais pire baisse −34 % : risque refusé | test approximatif |
| 2026-11-05 | Remonter le stop au prix d'achat après +3/6/10 % | 10,8 / 11,4 / 11,5 %/an, pas de baisse du risque ; ce n'est pas une pratique des fonds | conversation du 5/11 |
| 2026-11-05 | Faire tourner le fonds plusieurs fois par jour / acheter à la clôture du jour même | 10,4 %/an, pire baisse −31,2 % : acheter le lendemain à l'ouverture est meilleur | README « Vrais ordres stop » |
| 2026-11-05 | Poche presse automatique (acheter les bonnes nouvelles, shorter les mauvaises) | 2 ans, 487 k articles : −0,39 pt et −0,46 pt vs S&P à 5 j après frais, aucune variante robuste ; les +2,5 pts des 10 premiers jours = chance | `research/press_backtest.md` |
| 2026-11-04 | Dette française en direct (OAT) | Impossible chez Alpaca (seulement T-bills US et obligations d'entreprises US via Broker API) ; IGOV possible mais les ETF obligataires ont fait baisser le rendement | conversation du 4/11 |

## En attente d'une décision de Victor

- Monter le socle SPY à 40 % (11,9 %/an, 8/13, pire baisse −31 %).
- Tester l'ajout du baril (USO) au fonds.
- Supprimer la branche distante `terminal` (bloqué depuis cet environnement).
