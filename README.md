# Fondi vs ETF — pagina PortfolioSuite

Confronto tra i fondi attivi della lista fondi e l'ETF più economico della stessa categoria presente nella lista ETF.
Gratis: dati da Yahoo Finance e BCE, calcolo su GitHub Actions, nessun token, nessuna carta.

## File

| File | Cosa fa |
|---|---|
| `fondi-vs-etf.html` | La pagina. Legge `data/fondi_vs_etf.json`; se manca, mostra i dati demo con un avviso. |
| `data/lista_fondi.csv` | 1.839 ISIN di fondi. Colonne facoltative per correzioni a mano: nome, categoria, ter, classe, hedged. |
| `data/lista_etf.csv` | 141 ETF; 50 hanno una categoria di confronto. Un ETF senza categoria non viene usato. |
| `data/catalogo_portfoliosuite.csv` | Anagrafica già raccolta in PortfolioSuite (1.056 fondi, 134 ETF): nome completo, società, TER, classe Acc/Dist, valuta, segmento, classe istituzionale. Ha la precedenza su Yahoo. |
| `data/regole_categorie.csv` | Come la categoria Morningstar di un fondo diventa una categoria della pagina. |
| `scripts/fve_metrics.py` | Le formule (rendimenti, alfa/beta, tracking error, IR, finestre 12m, persistenza). |
| `scripts/fve_batch.py` | Il batch: anagrafica, prezzi, conversione in EUR, una classe per fondo, calcolo, JSON. |
| `tests/test_metrics.py` | 8 test su casi con risultato noto. Girano prima di ogni aggiornamento. |
| `.github/workflows/fondi-vs-etf.yml` | Calcolo completo il sabato; negli altri giorni gira solo finché l'anagrafica non è completa. |
| `data/anagrafica.csv` | Creato dal batch: ticker, nome, valuta, categoria e TER per ISIN. Colonna `ticker_manuale` per le correzioni. |
| `data/isin_non_trovati.csv` | Creato dal batch: ISIN senza dati, con il motivo. |
| `data/categorie_non_mappate.csv` | Creato dal batch: categorie dei fondi rimasti fuori perché senza ETF di confronto. |
| `data/fondi_vs_etf_demo.json` | Solo per provare la pagina: strumenti e numeri inventati. |

## Installazione nel repository PortfolioSuite

1. Copiare tutto nella radice del repository (la cartella `.github/workflows` si unisce a quella esistente).
2. Tab **Actions → Fondi vs ETF → Run workflow**.
3. La prima raccolta di 1.980 ISIN richiede più giri (circa 3 ore e mezza l'uno al massimo). I giri successivi partono da soli ogni giorno finché l'anagrafica non è completa, poi solo il sabato.
4. Aggiungere un link a `fondi-vs-etf.html` nel menu di PortfolioSuite.

## Collegamento con PortfolioSuite

- Il JSON contiene anche i campi che PortfolioSuite già legge (`aggiornato`, `categorie[].label`, per fondo `beta`, `te` in %, `ir`, `x3`, `x5` in punti, `win12` in %). Basta incollare l'indirizzo di `data/fondi_vs_etf.json` nel campo dati Fondi vs ETF di PortfolioSuite: la scelta del satellite (IR ≥ 0,3, TE 2–6%, beta 0,9–1,1) usa questi numeri.
- La pagina accetta `?emb=1&src=<url del JSON>` per stare dentro la scheda di PortfolioSuite e comunica la propria altezza.
- Priorità dei dati anagrafici: correzione a mano nelle liste > catalogo PortfolioSuite > Yahoo Finance. Per le categorie: Morningstar via Yahoo; se manca, il segmento PortfolioSuite solo per azionario USA, Europa, globale, emergenti e Italia (segnalato in pagina come "categoria approssimata").

## Regole anti-dato-inventato

- Un ISIN non trovato, senza storico, con prezzi fermi o con nome che non corrisponde **non entra nei calcoli**: finisce in `isin_non_trovati.csv` e in pagina.
- Correzioni: ticker giusto in `ticker_manuale` (anagrafica.csv); categoria, TER, classe o copertura nelle colonne di `lista_fondi.csv`. Le correzioni a mano vincono sempre.
- Un rendimento a N anni compare solo se lo storico copre tutti gli N anni.
- Un TER fuori dall'intervallo 0–5% viene scartato invece di essere usato.
- Se la BCE non risponde, la regressione usa rendimenti non in eccesso e la pagina lo dichiara.

## Prova locale

```
python tests/test_metrics.py
python scripts/fve_batch.py --demo     # rigenera i dati demo
python -m http.server                  # poi aprire http://localhost:8000/fondi-vs-etf.html
```
