"""
Fondi vs ETF — batch settimanale (GitHub Actions).

Uso:
  python scripts/fve_batch.py                  # dati reali (Yahoo Finance + BCE)
  python scripts/fve_batch.py --budget-min 200 # tempo massimo per completare l'anagrafica
  python scripts/fve_batch.py --demo           # dati SINTETICI per provare la pagina (marcati demo)

Input : data/lista_fondi.csv, data/lista_etf.csv, data/regole_categorie.csv,
        data/catalogo_portfoliosuite.csv (anagrafica già raccolta in PortfolioSuite: nomi, TER, classe, valuta, segmento)
Cache : data/anagrafica.csv (ticker, nome, valuta, categoria di origine, TER per ISIN; correzioni in ticker_manuale)
Output: data/fondi_vs_etf.json, data/isin_non_trovati.csv, data/categorie_non_mappate.csv
"""
from __future__ import annotations

import argparse
import hashlib
import csv
import datetime as dt
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import fve_metrics as M  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
F_FONDI = DATA / "lista_fondi.csv"
F_ETF = DATA / "lista_etf.csv"
F_REGOLE = DATA / "regole_categorie.csv"
F_REGOLE_NOMI = DATA / "regole_nomi.csv"
F_ANAG = DATA / "anagrafica.csv"
F_CAT = DATA / "catalogo_portfoliosuite.csv"
OUT_JSON = DATA / "fondi_vs_etf.json"
OUT_NF = DATA / "isin_non_trovati.csv"
OUT_NM = DATA / "categorie_non_mappate.csv"

CAMPI_ANAG = ["isin", "tipo", "ticker", "ticker_manuale", "nome_fonte", "valuta", "categoria_fonte",
              "ter_fonte", "stato", "somiglianza_nome", "ultimo_controllo"]
BORSE_ETF = ["GER", "MIL", "AMS", "PAR", "FRA", "STU", "MUN", "BER", "EBS", "LSE"]
PAROLE_VUOTE = {"fund", "funds", "fondo", "class", "classe", "acc", "dis", "dist", "eur", "usd", "hedged",
                "hdg", "sicav", "ucits", "etf", "the", "and", "plc", "lux", "shares", "share", "cap", "accumulating",
                "distributing", "unit", "units", "inc", "ishares", "amundi", "spdr", "xtrackers"}
VALUTE = {"eur", "usd", "gbp", "chf", "jpy", "sek", "nok", "dkk", "pln", "czk", "huf", "aud", "cad", "sgd", "hkd", "cnh", "rmb"}
SOGLIA_SOMIGLIANZA = 0.15
# Marchi da non pubblicare: sono salvati solo come impronta SHA-256 (primi 16 caratteri) della parola
# in minuscolo, così il nome non compare in chiaro in nessun file. Una parola del nome il cui hash è
# in questo elenco viene tolta da nomi e categorie prima di scrivere anagrafica, JSON e CSV.
MARCHI_NASCOSTI = {"a2661d000fde6f01", "8405abeec4a40aea", "b2e2212bcd7bd580", "0579d2f011648a45", "dbdbc8ce6333f81b"}
RINNOVO_ANAGRAFICA_GG = 90


def oggi() -> dt.date:
    return dt.date.today()


def data_riferimento() -> pd.Timestamp:
    """Ultimo fine mese chiuso."""
    return pd.Timestamp(oggi().replace(day=1)) - pd.offsets.MonthEnd(1)


# ------------------------------------------------------------------ pulizia nomi
def _nascosto(parola: str) -> bool:
    return hashlib.sha256(parola.lower().encode("utf-8")).hexdigest()[:16] in MARCHI_NASCOSTI


def pulisci(testo):
    """Toglie da un testo le parole dei marchi nascosti; applicata anche a liste e dizionari annidati."""
    if isinstance(testo, str):
        if not testo:
            return testo
        t = re.sub(r"[A-Za-z0-9]+", lambda m: "" if _nascosto(m.group(0)) else m.group(0), testo)
        return re.sub(r"\s{2,}", " ", t).strip(" -–")
    if isinstance(testo, list):
        return [pulisci(x) for x in testo]
    if isinstance(testo, dict):
        return {k: pulisci(v) for k, v in testo.items()}
    return testo


# ------------------------------------------------------------------ lettura file
def leggi_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [{k: (v or "").strip() for k, v in r.items()} for r in csv.DictReader(f)
                if r.get("isin", r.get("categoria", "")) and not (r.get("isin") or "").startswith("#")]


def num(x) -> float | None:
    try:
        return float(str(x).replace(",", ".")) if str(x).strip() else None
    except ValueError:
        return None


def leggi_anagrafica() -> dict[str, dict]:
    return {r["isin"]: r for r in leggi_csv(F_ANAG)}


def scrivi_anagrafica(anag: dict[str, dict]):
    with open(F_ANAG, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPI_ANAG)
        w.writeheader()
        for isin in sorted(anag):
            w.writerow({k: pulisci(anag[isin].get(k, "")) for k in CAMPI_ANAG})


# ------------------------------------------------------------------ regole categorie
def leggi_regole() -> list[tuple[str, list[list[str]], list[str]]]:
    regole = []
    for r in leggi_csv(F_REGOLE):
        inc = [[a.strip().lower() for a in t.split("|")] for t in r["includi"].split(";") if t.strip()]
        esc = [a.strip().lower() for a in r.get("escludi", "").split("|") if a.strip()]
        regole.append((r["categoria"], inc, esc))
    return regole


def leggi_regole_nomi() -> list[dict]:
    """Regole per ricavare la categoria dal nome del fondo, nell'ordine del file (vince la prima che corrisponde)."""
    out = []
    for i, r in enumerate(leggi_csv(F_REGOLE_NOMI), start=2):
        out.append({"riga": i, "categoria": r["categoria"], "inc": re.compile(r["includi"]),
                    "esc": re.compile(r["escludi"]) if r.get("escludi") else None,
                    "copertura": (r.get("copertura") or "").upper()})
    return out


def mappa_categoria(cat_fonte: str, regole) -> str | None:
    c = re.sub(r"^eaa fund\s+", "", cat_fonte.lower()).strip()
    if not c:
        return None
    for nome, inc, esc in regole:
        if all(any(a in c for a in alt) for alt in inc) and not any(e in c for e in esc):
            return nome
    return None


# frasi che non descrivono il fondo (ombrelli SICAV, sigle di classe) e confondono le regole sul nome
_RUMORE_NOMI = [r" blackrock global funds ", r" franklin templeton global funds plc ", r" robeco capital growth funds ",
                r" lux (bdh|bd|bh|b) "]


def testo_nome(*nomi: str) -> str:
    """Nomi del fondo normalizzati per le regole: minuscolo, solo lettere, cifre e '&', spazi singoli."""
    t = " " + re.sub(r"\s+", " ", re.sub(r"[^a-z0-9&]+", " ", " ".join(n for n in nomi if n).lower())).strip() + " "
    for r in _RUMORE_NOMI:
        t = re.sub(r, " lux " if "lux" in r else " ", t)
    return t


_ALTRA_COPERTURA = re.compile(r" (usd|chf|gbp|rmb|cnh|jpy|sgd|aud|hkd)[ -]?(h|h1|h2|hdg|hedged) | h (usd|chf|gbp) ")


def categoria_da_nome(testo: str, hedged: bool, regole_nomi: list[dict]) -> tuple[str, int] | tuple[None, None]:
    """Categoria dal nome; vince la prima regola che corrisponde. La copertura della classe deve corrispondere:
    EUR = classe coperta in euro, NO = classe non coperta, vuoto = indifferente."""
    eur_h = hedged and not _ALTRA_COPERTURA.search(testo)
    for r in regole_nomi:
        if not r["inc"].search(testo) or (r["esc"] and r["esc"].search(testo)):
            continue
        if (r["copertura"] == "EUR" and not eur_h) or (r["copertura"] == "NO" and hedged):
            continue
        return r["categoria"], r["riga"]
    return None, None


# ------------------------------------------------------------------ euristiche sul nome
def _norm(nome: str) -> str:
    return " " + re.sub(r"[^a-z0-9]+", " ", nome.lower()) + " "


def classe_da_nome(nome: str) -> str | None:
    n = _norm(nome)
    # prima le parole esplicite, poi le sigle di una lettera (meno affidabili)
    if re.search(r" (acc|accumulating|accumulation|capitalisation|capitalization) ", n):
        return "ACC"
    if re.search(r" (dist|dis|distributing|distribution|inc|qdis|mdis|qinc|minc) ", n):
        return "DIST"
    if re.search(r" (1c|2c|3c|4c|c) ", n):
        return "ACC"
    if re.search(r" (1d|2d|d) ", n):
        return "DIST"
    return None


def hedged_da_nome(nome: str) -> bool:
    """Classe con copertura valutaria. Oltre a 'hedged'/'hdg' riconosce le sigle di classe usate dalle società
    (HR Pictet, H1 Franklin, H2 Allianz, Bh/Bdh Capital Group, DH/BxH Robeco, LCH/NCH/LDH DWS, EURH, PfH, ' H ')."""
    n = _norm(nome)
    if re.search(r" (hedged|hdg|hedge|hr|h1|h2|bh|bdh|dh|bxh|lch|nch|ldh|ndh|ldhq|pfh|h) ", n):
        return True
    return bool(re.search(r"(eur|usd|chf|gbp)h(acc|inc|dis|ad)? ", n)) or bool(re.search(r" h(eur|usd|chf) ", n))


def nome_base(nome: str) -> str:
    """Nome del fondo senza la parte di classe: serve a riconoscere più classi dello stesso fondo."""
    toks = re.findall(r"[a-z0-9]+", nome.lower())
    tenuti = [t for t in toks if (len(t) > 2 or t.isdigit()) and t not in VALUTE and t not in
              {"acc", "dis", "dist", "inc", "hedged", "hdg", "cap", "class", "classe", "shares", "share", "unit", "units",
               "accumulating", "distributing", "accumulation", "distribution", "qdis", "mdis", "qinc", "minc"}]
    return " ".join(tenuti)


def token(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if len(t) >= 3 and t not in PAROLE_VUOTE}


def somiglianza(a: str, b: str) -> float:
    ta, tb = token(a), token(b)
    return len(ta & tb) / len(ta) if ta else 1.0


# ------------------------------------------------------------------ fonti reali
class Yahoo:
    def __init__(self):
        import yfinance as yf  # importato solo in modalità reale
        self.yf = yf
        self._fx: dict[str, pd.Series] = {}

    def _retry(self, fn, tent=3):
        for k in range(tent):
            try:
                return fn()
            except Exception:  # noqa: BLE001
                if k == tent - 1:
                    raise
                time.sleep(5 * (k + 1))

    def cerca(self, isin: str) -> list[dict]:
        try:
            return self._retry(lambda: self.yf.Search(isin, max_results=10, news_count=0).quotes) or []
        except Exception:  # noqa: BLE001
            return []

    def info(self, ticker: str) -> dict:
        """Nome, categoria (Morningstar via Yahoo) e TER in %. Campi assenti = stringa vuota."""
        t = self.yf.Ticker(ticker)
        out = {"nome": "", "categoria": "", "ter": None}
        try:
            fo = t.funds_data.fund_overview or {}
            out["categoria"] = fo.get("categoryName") or ""
            ops = t.funds_data.fund_operations
            if ops is not None and not ops.empty:
                row = [i for i in ops.index if "expense" in str(i).lower()]
                if row:
                    v = ops.loc[row[0]].iloc[0]
                    if v is not None and not pd.isna(v):
                        out["ter"] = float(v) * 100  # frazione → %
        except Exception:  # noqa: BLE001
            pass
        try:
            inf = t.info or {}
            out["nome"] = inf.get("longName") or inf.get("shortName") or ""
            out["categoria"] = out["categoria"] or inf.get("category") or ""
            if out["ter"] is None:
                if inf.get("netExpenseRatio") is not None:
                    out["ter"] = float(inf["netExpenseRatio"])          # già in %
                elif inf.get("annualReportExpenseRatio") is not None:
                    out["ter"] = float(inf["annualReportExpenseRatio"]) * 100
        except Exception:  # noqa: BLE001
            pass
        if out["ter"] is not None and not (0 <= out["ter"] <= 5):
            out["ter"] = None  # valore implausibile: meglio vuoto che sbagliato
        return out

    def storico(self, ticker: str) -> tuple[pd.Series, str | None, int]:
        """Chiusure rettificate giornaliere, valuta, numero di dividendi negli ultimi 3 anni."""
        t = self.yf.Ticker(ticker)
        h = self._retry(lambda: t.history(period="11y", interval="1d", auto_adjust=True, actions=True))
        if h is None or h.empty:
            return pd.Series(dtype=float), None, 0
        cur = (t.history_metadata or {}).get("currency")
        h.index = pd.to_datetime(h.index).tz_localize(None)
        div = 0
        if "Dividends" in h:
            d = h["Dividends"]
            div = int((d[d.index >= h.index[-1] - pd.Timedelta(days=3 * 365)] > 0).sum())
        return h["Close"], cur, div

    def fx(self, valuta: str) -> pd.Series:
        if valuta not in self._fx:
            s, _, _ = self.storico(f"EUR{valuta}=X")
            self._fx[valuta] = s
        return self._fx[valuta]

    def in_eur(self, s: pd.Series, valuta: str | None) -> tuple[pd.Series, str | None]:
        if valuta in (None, "", "EUR"):
            return s, None
        if valuta == "GBp":
            s, valuta = s / 100, "GBP"
        fx = self.fx(valuta)
        if fx.empty:
            return pd.Series(dtype=float), f"cambio EUR{valuta} non disponibile"
        fx = fx.reindex(s.index, method="ffill")
        return (s / fx).dropna(), f"convertito da {valuta} in EUR"


def tasso_privo_rischio() -> tuple[pd.Series | None, str]:
    """Euribor 1M media mensile (BCE, gratuito, senza chiave) → tasso mensile."""
    import requests
    from io import StringIO
    url = ("https://data-api.ecb.europa.eu/service/data/FM/M.U2.EUR.RT.MM.EURIBOR1MD_.HSTA"
           "?format=csvdata&startPeriod=2014-01")
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        s = pd.Series(df["OBS_VALUE"].astype(float).values,
                      index=pd.to_datetime(df["TIME_PERIOD"]) + pd.offsets.MonthEnd(0))
        return s / 100 / 12, "Euribor 1 mese, media mensile (BCE)"
    except Exception as e:  # noqa: BLE001
        return None, f"non disponibile ({type(e).__name__}): regressione su rendimenti non in eccesso"


def risolvi(y: Yahoo, isin: str, tipo: str, nome_lista: str, voce: dict) -> dict:
    """Trova ticker e anagrafica. Non inventa: se non trova o il nome non torna, lo segnala."""
    voce.update({"isin": isin, "tipo": tipo, "ultimo_controllo": oggi().isoformat()})
    ticker = voce.get("ticker_manuale")
    if not ticker:
        quotes = y.cerca(isin)
        if not quotes:
            voce.update(ticker="", stato="non_trovato")
            return voce
        voluto = "ETF" if tipo == "ETF" else "MUTUALFUND"
        cand = [q for q in quotes if q.get("quoteType") == voluto] or quotes
        if tipo == "ETF":
            cand.sort(key=lambda q: BORSE_ETF.index(q.get("exchange")) if q.get("exchange") in BORSE_ETF else 99)
        scelto = None
        for q in cand[:4]:
            try:
                s, cur, _ = y.storico(q["symbol"])
            except Exception:  # noqa: BLE001
                continue
            time.sleep(0.3)
            if len(s) < 120:
                continue
            if scelto is None or (tipo == "ETF" and cur == "EUR"):
                scelto = (q, cur)
            if tipo != "ETF" or cur == "EUR":
                break
        if scelto is None:
            voce.update(ticker="", stato="senza_storico")
            return voce
        q, cur = scelto
        ticker = q["symbol"]
        voce["nome_fonte"] = q.get("longname") or q.get("shortname") or ""
        voce["valuta"] = cur or ""
    inf = y.info(ticker)
    time.sleep(0.3)
    voce["ticker"] = ticker
    voce["nome_fonte"] = inf["nome"] or voce.get("nome_fonte", "")
    voce["categoria_fonte"] = inf["categoria"]
    voce["ter_fonte"] = f"{inf['ter']:.3f}" if inf["ter"] is not None else ""
    if voce.get("ticker_manuale"):
        voce["stato"] = "manuale"
    elif nome_lista:
        sim = somiglianza(nome_lista, voce["nome_fonte"])
        voce["somiglianza_nome"] = f"{sim:.2f}"
        voce["stato"] = "ok" if sim >= SOGLIA_SOMIGLIANZA else "nome_non_corrisponde"
    else:
        voce["stato"] = "ok"
    return voce


def da_aggiornare(voce: dict) -> bool:
    if not voce or not voce.get("stato"):
        return True
    if voce.get("ticker_manuale") and voce.get("stato") != "manuale":
        return True
    try:
        eta = (oggi() - dt.date.fromisoformat(voce.get("ultimo_controllo", "2000-01-01"))).days
    except ValueError:
        eta = 999
    if voce["stato"] in ("non_trovato", "senza_storico"):
        return eta >= 28
    return eta >= RINNOVO_ANAGRAFICA_GG


# ------------------------------------------------------------------ demo sintetica
def strumenti_demo(data_rif: pd.Timestamp):
    """Strumenti e serie SINTETICI, riproducibili. Servono solo a vedere la pagina: nessun dato reale."""
    rng = np.random.default_rng(42)
    idx = pd.date_range(data_rif - pd.offsets.MonthEnd(131), data_rif, freq="ME")
    cats = {"Azionario USA large cap": (0.011, 0.042), "Azionario Europa large cap": (0.006, 0.045),
            "Azionario globale": (0.008, 0.038), "Azionario emergenti": (0.004, 0.050),
            "Obbligazionario corporate euro": (0.0015, 0.012)}
    strumenti, serie = [], {}
    for ci, (cat, (mu, sd)) in enumerate(cats.items()):
        base = rng.normal(mu, sd, len(idx))
        code = ["USA", "EUR", "GLB", "EME", "BND"][ci]
        for k, (ter, cl, hd) in enumerate([(0.07, "ACC", False), (0.20, "ACC", False), (0.12, "ACC", True)][: 3 if ci in (0, 2) else 2]):
            isin = f"DEMO{code}ETF{k + 1:03d}"
            hedge_noise = rng.normal(0, 0.012, len(idx)) if hd else 0
            rr = base - ter / 1200 + rng.normal(0, 0.0005, len(idx)) + hedge_noise
            serie[isin] = pd.Series(100 * np.cumprod(1 + rr), index=idx)
            strumenti.append(dict(isin=isin, tipo="ETF", nome=f"ETF Demo {cat} {'ABC'[k]}" + (" EUR Hedged" if hd else ""),
                                  categoria=cat, ter=ter, classe=cl, hedged=hd, dividendi=0, ticker="", categoria_fonte="",
                                  ter_fonte=False))
        nf = 7 if ci < 3 else 5
        for i in range(nf):
            isin = f"DEMO{code}F{i + 1:03d}"
            nome = f"Fondo Demo {cat} {i + 1}" if not (i == 0 and ci in (0, 2)) else f"Fondo Demo {cat} Alfa"
            ter = round((1.6 if ci < 4 else 0.9) + 0.1 * i, 2)
            hd = (i == nf - 1 and ci in (0, 2))
            cl = "DIST" if (i == nf - 1 and ci == 1) else "ACC"
            beta = rng.uniform(0.85, 1.1)
            alfa = 0.0045 if "Alfa" in nome else rng.normal(0.0, 0.0012)
            rr = alfa + beta * base - ter / 1200 + rng.normal(0, 0.009, len(idx))
            s = pd.Series(100 * np.cumprod(1 + rr), index=idx)
            if i == nf - 2:
                s = s.iloc[-50:]
            serie[isin] = s
            strumenti.append(dict(isin=isin, tipo="FONDO", nome=nome, categoria=cat, ter=ter, classe=cl, hedged=hd,
                                  dividendi=4 if cl == "DIST" else 0, ticker="", categoria_fonte=f"(demo) {cat}",
                                  ter_fonte=True, classi_escluse=2 if i == 1 else 0))
    return strumenti, serie


# ------------------------------------------------------------------ deduplica classi
def deduplica(fondi: list[dict], serie_m: dict) -> list[dict]:
    """Una sola classe per fondo e categoria: ad accumulo, retail, non coperta, EUR, TER più basso, storico più lungo."""
    gruppi: dict[tuple, list[dict]] = {}
    for f in fondi:
        gruppi.setdefault((f["categoria"], nome_base(f.get("nome_dedup") or f["nome"]) or f["isin"]), []).append(f)
    tenuti = []
    for (_, _), g in gruppi.items():
        g.sort(key=lambda f: (f["classe"] != "ACC", f.get("istituzionale", False), f["hedged"], f.get("valuta_orig", "EUR") != "EUR",
                              f["ter"] if f["ter"] is not None else 99, -len(serie_m.get(f["isin"], []))))
        best = dict(g[0])
        best["classi_escluse"] = len(g) - 1
        best["solo_dist"] = all(x["classe"] == "DIST" for x in g)
        tenuti.append(best)
    return tenuti


# ------------------------------------------------------------------ costruzione JSON
def r6(x):
    return None if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else round(float(x), 6)


def affidabile(e: dict) -> bool:
    """Total return credibile: classe ad accumulo, oppure a distribuzione con dividendi presenti nella fonte."""
    return e["classe"] == "ACC" or (e["classe"] == "DIST" and e.get("dividendi", 0) > 0) or (e["classe"] is None and e.get("dividendi", 0) > 0)


def scegli_etf(etfs: list[dict], serie_m: dict, hedged: bool) -> dict | None:
    """Regola: stessa copertura valutaria del fondo, total return affidabile, TER più basso, storico più lungo."""
    con_dati = [e for e in etfs if not serie_m.get(e["isin"], pd.Series(dtype=float)).empty]
    if not con_dati:
        return None
    return sorted(con_dati, key=lambda e: (e["hedged"] != hedged, not affidabile(e),
                                           e["ter"] if e["ter"] is not None else 99, -len(serie_m[e["isin"]])))[0]


def note_etf(e: dict) -> list[str]:
    n = []
    if e["classe"] == "DIST" and not e.get("dividendi"):
        n.append("ETF a distribuzione senza dividendi nella fonte: il suo rendimento è sottostimato e il confronto favorisce i fondi")
    elif e["classe"] == "DIST":
        n.append("ETF a distribuzione: dividendi reinvestiti come da fonte")
    elif e["classe"] is None:
        n.append("classe dell'ETF non determinata")
    if e["ter"] is None:
        n.append("TER dell'ETF non disponibile")
    return n


def costruisci(strumenti, serie_m, flags, rf, rf_nota, data_rif, demo, extra_meta) -> dict:
    per_cat: dict[str, dict[str, list]] = {}
    for s in strumenti:
        if s.get("categoria"):
            per_cat.setdefault(s["categoria"], {"ETF": [], "FONDO": []})[s["tipo"]].append(s)

    categorie, globale = [], {str(a): {"n": 0, "battono": 0} for a in M.ORIZZONTI}
    mat_glob = [[0] * 4 for _ in range(4)]
    for cat in sorted(per_cat):
        etfs = per_cat[cat]["ETF"]
        fondi = [f for f in per_cat[cat]["FONDO"] if not serie_m.get(f["isin"], pd.Series(dtype=float)).empty]
        if not fondi:
            continue
        principale = scegli_etf(etfs, serie_m, False) or scegli_etf(etfs, serie_m, True)
        if principale is None:
            categorie.append({"nome": cat, "etf": None, "fondi": [], "n_fondi_senza_etf": len(fondi),
                              "note": ["nessun ETF della lista con dati: categoria non confrontabile"]})
            continue

        def etf_out(e):
            m = serie_m[e["isin"]]
            return {"isin": e["isin"], "nome": e["nome"], "ticker": e.get("ticker", ""), "ter": e["ter"],
                    "classe": e["classe"], "hedged": e["hedged"], "inizio": m.index[0].strftime("%Y-%m"),
                    "rend": {str(a): r6(M.rend_annualizzato(m, a, data_rif)) for a in M.ORIZZONTI},
                    "flags": flags.get(e["isin"], []) + note_etf(e)}

        rif_usati = {principale["isin"]: etf_out(principale)}
        note_cat = []
        if serie_m[principale["isin"]].index[0] > data_rif - pd.offsets.MonthEnd(120):
            note_cat.append(f"Storico ETF dal {rif_usati[principale['isin']]['inizio']}: i confronti più lunghi non sono calcolabili.")

        out_fondi, sintesi = [], {str(a): {"n": 0, "battono": 0} for a in M.ORIZZONTI}
        gruppi_pers: dict[str, dict[str, pd.Series]] = {}
        for f in fondi:
            m = serie_m[f["isin"]]
            rif = scegli_etf(etfs, serie_m, f["hedged"])
            if rif["isin"] not in rif_usati:
                rif_usati[rif["isin"]] = etf_out(rif)
            m_etf = serie_m[rif["isin"]]
            fl = list(flags.get(f["isin"], []))
            if f["classe"] == "DIST":
                fl.append("solo classe a distribuzione" + ("" if f.get("dividendi") else ": dividendi assenti nella fonte, rendimento sottostimato"))
            elif f["classe"] is None:
                fl.append("classe non determinata")
            if f["hedged"] != rif["hedged"]:
                fl.append("copertura valutaria diversa dall'ETF: differenze dovute anche al cambio")
            c = M.confronta(m, m_etf, rf, data_rif)
            reg = c["regressione"] or {}
            for a in M.ORIZZONTI:
                x = c["extra"][str(a)]
                if x is not None:
                    sintesi[str(a)]["n"] += 1
                    sintesi[str(a)]["battono"] += int(x > 0)
            if affidabile(f):
                gruppi_pers.setdefault(rif["isin"], {})[f["isin"]] = m
            out_fondi.append({
                "isin": f["isin"], "nome": f["nome"], "ticker": f.get("ticker", ""),
                "categoria_fonte": f.get("categoria_fonte", ""),
                "ter": r6(f["ter"]), "classe": f["classe"], "hedged": f["hedged"],
                "classi_escluse": f.get("classi_escluse", 0),
                "etf_rif": None if rif["isin"] == principale["isin"] else rif["isin"],
                "inizio": m.index[0].strftime("%Y-%m"),
                "rend": {k: r6(v) for k, v in c["rend"].items()},
                "extra": {k: r6(v) for k, v in c["extra"].items()},
                "alfa": r6(reg.get("alfa")), "alfa_t": r6(reg.get("alfa_t")), "beta": r6(reg.get("beta")),
                "r2": r6(reg.get("r2")), "te": r6(reg["te"] * 100) if reg.get("te") is not None else None, "ir": r6(reg.get("ir")),
                "mesi_regressione": reg.get("mesi"),
                "pct_batte_12m": r6(c["pct_batte_12m"]), "finestre_12m": c["finestre_12m"],
                # stessi dati nel formato letto da PortfolioSuite (punti percentuali / %)
                **{f"x{a}": (round(c["extra"][str(a)] * 100, 2) if c["extra"][str(a)] is not None else None) for a in M.ORIZZONTI},
                "win12": round(c["pct_batte_12m"] * 100, 1) if c["pct_batte_12m"] is not None else None,
                "societa": f.get("societa", ""), "categoria_approssimata": f.get("categoria_approssimata", False),
                "diff_costo": r6(f["ter"] - rif["ter"]) if (f["ter"] is not None and rif["ter"] is not None) else None,
                "rolling": c["rolling"], "flags": fl,
            })

        # persistenza: ranking solo tra fondi con lo stesso ETF di riferimento
        matrice, n_tr, date, rho, ultimo = [[0] * 4 for _ in range(4)], 0, [], [], {}
        for etf_isin, gruppo in gruppi_pers.items():
            p = M.persistenza(gruppo, serie_m[etf_isin], data_rif)
            for i in range(4):
                for j in range(4):
                    matrice[i][j] += p["matrice"][i][j]
            n_tr += p["n_transizioni"]
            date += p["date"]
            rho += [d["spearman"] for d in p["date"] if d["spearman"] is not None]
            if p["date"]:
                ultima = p["date"][0]["data"]
                for t in p["transizioni"]:
                    if t["formazione"] == ultima:
                        ultimo[t["isin"]] = [t["q_prima"], t["q_dopo"]]
        for fo in out_fondi:
            fo["quartili_ultimo"] = ultimo.get(fo["isin"])
        for i in range(4):
            for j in range(4):
                mat_glob[i][j] += matrice[i][j]
        for a in M.ORIZZONTI:
            globale[str(a)]["n"] += sintesi[str(a)]["n"]
            globale[str(a)]["battono"] += sintesi[str(a)]["battono"]
        out_fondi.sort(key=lambda f: (f["extra"]["5"] is None, -(f["extra"]["5"] or 0)))
        categorie.append({
            "nome": cat, "label": cat, "etf": rif_usati[principale["isin"]],
            "etf_altri_rif": [v for k, v in rif_usati.items() if k != principale["isin"]],
            "altri_etf": [{"isin": e["isin"], "nome": e["nome"], "ter": e["ter"]} for e in etfs if e["isin"] not in rif_usati],
            "fondi": out_fondi, "sintesi": sintesi,
            "persistenza": {"matrice": matrice, "n_transizioni": n_tr, "date": date,
                            "spearman_medio": float(np.mean(rho)) if rho else None},
            "note": note_cat})

    return {
        "aggiornato": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "meta": {
            "generato": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "data_riferimento": data_rif.strftime("%Y-%m-%d"),
            "demo": demo,
            "fonte_prezzi": "SINTETICI (demo)" if demo else "Yahoo Finance, chiusure rettificate per dividendi e split, convertite in EUR",
            "fonte_categorie": "demo" if demo else "categoria Morningstar riportata da Yahoo Finance, ricondotta alle categorie della pagina",
            "tasso_privo_rischio": rf_nota,
            **extra_meta,
        },
        "riepilogo": globale,
        "persistenza_globale": {"matrice": mat_glob, "n": sum(map(sum, mat_glob))},
        "categorie": categorie,
    }


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="usa dati sintetici")
    ap.add_argument("--budget-min", type=float, default=200, help="minuti massimi per completare l'anagrafica")
    a = ap.parse_args()
    t0 = time.time()
    data_rif = data_riferimento()
    flags: dict[str, list[str]] = {}
    serie_m: dict[str, pd.Series] = {}
    non_trovati: list[dict] = []
    extra_meta: dict = {}

    if a.demo:
        strumenti, serie_m = strumenti_demo(data_rif)
        rf, rf_nota = pd.Series(0.0015 / 12, index=pd.date_range("2014-01-31", data_rif, freq="ME")), "costante 0,15% annuo (demo)"
        fondi = deduplica([s for s in strumenti if s["tipo"] == "FONDO"], serie_m)
        for f in fondi:
            f["classi_escluse"] = next((s.get("classi_escluse", 0) for s in strumenti if s["isin"] == f["isin"]), 0)
        strumenti = [s for s in strumenti if s["tipo"] == "ETF"] + fondi
        extra_meta = {"n_isin_fondi": 31, "n_fondi_confrontati": len(fondi), "n_etf_lista": 13,
                      "anagrafica_completa": True, "n_anagrafica_mancante": 0}
    else:
        y = Yahoo()
        regole = leggi_regole()
        regole_nomi = leggi_regole_nomi()
        lista_f = {r["isin"]: r for r in leggi_csv(F_FONDI)}
        lista_e = {r["isin"]: r for r in leggi_csv(F_ETF)}
        anag = leggi_anagrafica()
        cat_etf = {r["categoria"] for r in lista_e.values() if r.get("categoria")}

        # 1) anagrafica, con budget di tempo: quello che non finisce oggi riparte al prossimo giro
        catal = {r["isin"]: r for r in leggi_csv(F_CAT)}
        nome_rif = lambda i, r: r.get("nome") or catal.get(i, {}).get("nome_completo", "")  # noqa: E731
        coda = [(i, "ETF", nome_rif(i, r)) for i, r in lista_e.items() if r.get("categoria")] + \
               [(i, "FONDO", nome_rif(i, r)) for i, r in lista_f.items()]
        mancanti = 0
        for k, (isin, tipo, nome) in enumerate(coda):
            if not da_aggiornare(anag.get(isin, {})):
                continue
            if time.time() - t0 > a.budget_min * 60:
                mancanti += 1
                continue
            anag[isin] = risolvi(y, isin, tipo, nome, anag.get(isin, {}))
            if k % 50 == 0:
                scrivi_anagrafica(anag)
                print(f"anagrafica {k}/{len(coda)}", flush=True)
        scrivi_anagrafica(anag)

        # 2) strumenti con categoria
        strumenti, non_mappate = [], Counter()
        for isin, tipo, _ in coda:
            v = anag.get(isin, {})
            riga = (lista_e if tipo == "ETF" else lista_f)[isin]
            motivo = None
            if not v.get("stato"):
                motivo = "anagrafica non ancora completata (riprende al prossimo aggiornamento)"
            elif v["stato"] == "non_trovato":
                motivo = "ISIN non trovato su Yahoo Finance"
            elif v["stato"] == "senza_storico":
                motivo = "trovato ma senza storico utilizzabile"
            elif v["stato"] == "nome_non_corrisponde":
                motivo = f"ticker {v.get('ticker')} trovato ma il nome '{v.get('nome_fonte')}' non corrisponde: verificare e scrivere ticker_manuale"
            ps = catal.get(isin, {})
            nome = (ps.get("nome_completo") or v.get("nome_fonte") or riga.get("nome") or isin) if tipo == "ETF" \
                else (riga.get("nome") or ps.get("nome_completo") or v.get("nome_fonte") or isin)
            hedged = riga["hedged"].upper() in ("SI", "SÌ", "YES", "1") if riga.get("hedged") \
                else hedged_da_nome(nome + " " + (v.get("nome_fonte") or ""))
            cat_approx, cat_fonte = False, v.get("categoria_fonte", "")
            if tipo == "ETF":
                cat = riga.get("categoria")
            else:
                # priorità: categoria scritta a mano > categoria Morningstar (Yahoo) > regole sul nome del fondo
                cat = riga.get("categoria") or mappa_categoria(cat_fonte, regole)
                if not cat and not cat_fonte:
                    testo = testo_nome(riga.get("nome", ""), ps.get("nome_completo", ""), v.get("nome_fonte", ""))
                    cat, n_regola = categoria_da_nome(testo, hedged, regole_nomi)
                    if cat:
                        cat_approx, cat_fonte = True, f"categoria dal nome del fondo (regole_nomi.csv, riga {n_regola})"
                if not motivo and cat and cat not in cat_etf:
                    non_mappate[cat] += 1
                    continue  # nessun ETF della lista in questa categoria: fuori perimetro, conteggiata a parte
                if not motivo and not cat:
                    if cat_fonte:
                        non_mappate[cat_fonte] += 1
                        continue  # categoria senza ETF confrontabile: fuori perimetro, conteggiata a parte
                    motivo = ("categoria non riconoscibile né dalla fonte né dal nome "
                              "(si può indicare a mano in lista_fondi.csv)")
            if motivo:
                non_trovati.append({"isin": isin, "nome": nome if nome != isin else "", "tipo": tipo, "categoria": cat or "", "motivo": motivo})
                continue
            # priorità: correzione a mano in lista > catalogo PortfolioSuite > Yahoo
            ter = num(riga.get("ter")) if riga.get("ter") else (num(ps.get("ter")) if ps.get("ter") else num(v.get("ter_fonte")))
            classe = (riga.get("classe") or "").upper() or ps.get("classe") or classe_da_nome(nome)
            strumenti.append(dict(isin=isin, tipo=tipo, nome=nome, categoria=cat, ter=ter, classe=classe or None,
                                  hedged=hedged, ticker=v["ticker"],
                                  categoria_fonte=cat_fonte,
                                  categoria_approssimata=cat_approx, societa=ps.get("societa", ""),
                                  nome_dedup=v.get("nome_fonte") or nome,  # stessa fonte per tutte le classi
                                  istituzionale=ps.get("istituzionale") == "SI",
                                  valuta_orig=ps.get("valuta") or v.get("valuta", ""), dividendi=0))

        # 3) prezzi
        rf, rf_nota = tasso_privo_rischio()
        for s in strumenti:
            try:
                px, cur, div = y.storico(s["ticker"])
                time.sleep(0.25)
            except Exception as e:  # noqa: BLE001
                non_trovati.append({"isin": s["isin"], "nome": s["nome"], "tipo": s["tipo"], "categoria": s["categoria"],
                                    "motivo": f"errore download ({type(e).__name__})"})
                continue
            s["dividendi"] = div
            if s["classe"] is None:
                s["classe"] = "DIST" if div > 0 else None
            px, nota_fx = y.in_eur(px, cur)
            m, fl = M.qualita(M.a_mensile(px, data_rif))
            if nota_fx:
                fl.append(nota_fx)
            if m.empty or m.index[-1] < data_rif - pd.offsets.MonthEnd(1):
                non_trovati.append({"isin": s["isin"], "nome": s["nome"], "tipo": s["tipo"], "categoria": s["categoria"],
                                    "motivo": "prezzi non aggiornati" if not m.empty else "nessun prezzo"})
                continue
            serie_m[s["isin"]], flags[s["isin"]] = m, fl

        etf = [s for s in strumenti if s["tipo"] == "ETF" and s["isin"] in serie_m]
        fondi_ok = [s for s in strumenti if s["tipo"] == "FONDO" and s["isin"] in serie_m]
        fondi = deduplica(fondi_ok, serie_m)
        strumenti = etf + fondi
        with open(OUT_NM, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["categoria_fonte", "n_classi"])
            w.writerows((pulisci(c), n) for c, n in non_mappate.most_common())
        extra_meta = {"n_isin_fondi": len(lista_f), "n_classi_con_dati": len(fondi_ok),
                      "n_fondi_confrontati": len(fondi), "n_etf_lista": len(lista_e),
                      "n_fuori_perimetro": sum(non_mappate.values()),
                      "anagrafica_completa": mancanti == 0, "n_anagrafica_mancante": mancanti}

    out = costruisci(strumenti, serie_m, flags, rf, rf_nota, data_rif, a.demo, extra_meta)
    out["non_trovati"] = non_trovati
    out = pulisci(out)
    non_trovati = out["non_trovati"]
    dest = OUT_JSON if not a.demo else DATA / "fondi_vs_etf_demo.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    if not a.demo:
        with open(OUT_NF, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["isin", "nome", "tipo", "categoria", "motivo"])
            w.writeheader()
            w.writerows(non_trovati)
    print(f"OK {dest.name}: {len(out['categorie'])} categorie, {len(serie_m)} serie, "
          f"{len(non_trovati)} senza dati, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
