"""
Fondi vs ETF — funzioni di calcolo pure (nessun accesso di rete).

Convenzioni
- Serie di prezzi: pandas.Series in EUR, indice = fine mese (Timestamp), total return
  (prezzo rettificato per dividendi/split quando la fonte lo fornisce).
- Rendimenti mensili semplici: r_t = P_t / P_{t-1} - 1.
- Tutte le metriche "fondo vs ETF" usano solo i mesi in comune.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

ORIZZONTI = (1, 3, 5, 10)          # anni
MIN_MESI_REGRESSIONE = 36
FINESTRA_REGRESSIONE = 120         # massimo 10 anni
SOGLIA_OUTLIER = 0.35              # |rendimento mensile| oltre cui si segnala


# ---------------------------------------------------------------- utilità serie
def a_mensile(prezzi_giornalieri: pd.Series, data_rif: pd.Timestamp) -> pd.Series:
    """Ultimo prezzo di ogni mese, fino alla data di riferimento (fine mese chiuso)."""
    s = prezzi_giornalieri.dropna()
    s = s[s > 0]
    if s.empty:
        return s
    s.index = pd.to_datetime(s.index).tz_localize(None) if getattr(s.index, "tz", None) else pd.to_datetime(s.index)
    m = s.resample("ME").last()
    return m[m.index <= data_rif]


def qualita(m: pd.Series) -> tuple[pd.Series, list[str]]:
    """Riempie al massimo 1 mese mancante isolato, segnala buchi e outlier."""
    flags = []
    if m.empty:
        return m, ["nessun dato"]
    full = m.reindex(pd.date_range(m.index[0], m.index[-1], freq="ME"))
    mancanti = int(full.isna().sum())
    if mancanti:
        filled = full.ffill(limit=1)
        residui = int(filled.isna().sum())
        flags.append(f"{mancanti} mesi senza prezzo" + (" (1 mese riempito col precedente)" if residui < mancanti else ""))
        full = filled.dropna()
    r = full.pct_change().dropna()
    out = r[r.abs() > SOGLIA_OUTLIER]
    if len(out):
        flags.append("rendimenti mensili anomali: " + ", ".join(f"{d:%Y-%m} {v:+.0%}" for d, v in out.items()))
    return full, flags


def rend_annualizzato(m: pd.Series, anni: int, data_rif: pd.Timestamp) -> float | None:
    """CAGR sugli ultimi `anni` anni terminanti a data_rif. None se lo storico non copre tutto il periodo."""
    inizio = data_rif - pd.offsets.MonthEnd(12 * anni)
    if m.empty or m.index[0] > inizio or data_rif not in m.index or inizio not in m.index:
        return None
    p0, p1 = m.loc[inizio], m.loc[data_rif]
    return (p1 / p0) ** (1 / anni) - 1


# ---------------------------------------------------------------- regressione
def regressione(rf_fondo: pd.Series, rf_etf: pd.Series, rf: pd.Series | None) -> dict | None:
    """
    OLS: (r_fondo - rf) = a + b (r_etf - rf) + e  sugli ultimi <=120 mesi in comune.
    Ritorna alfa annualizzato (a*12), t-stat di a (errori standard OLS), beta, R²,
    tracking error annualizzato e information ratio (media attiva*12 / TE).
    """
    df = pd.concat({"f": rf_fondo, "e": rf_etf}, axis=1).dropna().iloc[-FINESTRA_REGRESSIONE:]
    n = len(df)
    if n < MIN_MESI_REGRESSIONE:
        return None
    rfr = (rf.reindex(df.index).ffill().fillna(0.0) if rf is not None else pd.Series(0.0, index=df.index))
    y = (df["f"] - rfr).to_numpy()
    x = (df["e"] - rfr).to_numpy()
    X = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = coef
    res = y - X @ coef
    s2 = res @ res / (n - 2)
    cov = s2 * np.linalg.inv(X.T @ X)
    se_a = math.sqrt(cov[0, 0])
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (res @ res) / ss_tot if ss_tot > 0 else float("nan")
    att = (df["f"] - df["e"]).to_numpy()
    te = att.std(ddof=1) * math.sqrt(12)
    ir = (att.mean() * 12 / te) if te > 0 else None
    return {
        "alfa": a * 12,
        "alfa_t": a / se_a if se_a > 0 else None,
        "beta": b,
        "r2": r2,
        "te": te,
        "ir": ir,
        "mesi": n,
        "dal": df.index[0].strftime("%Y-%m"),
        "al": df.index[-1].strftime("%Y-%m"),
    }


# ---------------------------------------------------------------- finestre mobili
def rolling_12m(m_fondo: pd.Series, m_etf: pd.Series, mesi_max: int = FINESTRA_REGRESSIONE) -> pd.Series:
    """Extra-rendimento su 12 mesi mobili: R12(fondo) - R12(ETF), solo date in comune."""
    df = pd.concat({"f": m_fondo, "e": m_etf}, axis=1).dropna()
    if len(df) < 13:
        return pd.Series(dtype=float)
    r12 = df / df.shift(12) - 1
    diff = (r12["f"] - r12["e"]).dropna()
    return diff.iloc[-mesi_max:]


# ---------------------------------------------------------------- persistenza
def quartile(valori: dict[str, float]) -> dict[str, int]:
    """Quartile 1 = migliore. Con n fondi: q = ceil(4 * rango / n), rango 1 = valore più alto."""
    ordinati = sorted(valori.items(), key=lambda kv: kv[1], reverse=True)
    n = len(ordinati)
    return {k: math.ceil(4 * (i + 1) / n) for i, (k, _) in enumerate(ordinati)}


def spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3:
        return None
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def persistenza(serie_fondi: dict[str, pd.Series], m_etf: pd.Series, data_rif: pd.Timestamp,
                min_fondi: int = 4) -> dict:
    """
    Per ogni data di formazione d = T-36m, T-48m, ... (finché d-36m >= T-120m):
      extra-rendimento annualizzato vs ETF in [d-36m, d] → quartile "prima"
      extra-rendimento annualizzato vs ETF in [d, d+36m] → quartile "dopo"
    Servono almeno `min_fondi` fondi con storico completo su entrambe le finestre.
    Le date di formazione sono a passo 12 mesi: le finestre si sovrappongono (dichiarato in pagina).
    """
    def extra(m, a, b):
        if a not in m.index or b not in m.index or a not in m_etf.index or b not in m_etf.index:
            return None
        rf_ = (m.loc[b] / m.loc[a]) ** (1 / 3) - 1
        re_ = (m_etf.loc[b] / m_etf.loc[a]) ** (1 / 3) - 1
        return rf_ - re_

    matrice = [[0] * 4 for _ in range(4)]
    transizioni, date, rho = [], [], []
    limite = data_rif - pd.offsets.MonthEnd(120)
    k = 36
    while True:
        d = data_rif - pd.offsets.MonthEnd(k)
        inizio = d - pd.offsets.MonthEnd(36)
        fine = d + pd.offsets.MonthEnd(36)
        if inizio < limite:
            break
        pre, post = {}, {}
        for isin, m in serie_fondi.items():
            x, y = extra(m, inizio, d), extra(m, d, fine)
            if x is not None and y is not None:
                pre[isin], post[isin] = x, y
        if len(pre) >= min_fondi:
            qp, qd = quartile(pre), quartile(post)
            for isin in pre:
                matrice[qp[isin] - 1][qd[isin] - 1] += 1
                transizioni.append({"isin": isin, "formazione": d.strftime("%Y-%m"),
                                    "q_prima": qp[isin], "q_dopo": qd[isin]})
            keys = list(pre)
            r = spearman([pre[k_] for k_ in keys], [post[k_] for k_ in keys])
            date.append({"data": d.strftime("%Y-%m"), "n_fondi": len(pre), "spearman": r})
            if r is not None:
                rho.append(r)
        k += 12
    return {
        "matrice": matrice,
        "n_transizioni": len(transizioni),
        "date": date,
        "spearman_medio": float(np.mean(rho)) if rho else None,
        "transizioni": transizioni,
    }


# ---------------------------------------------------------------- confronto fondo/ETF
def confronta(m_fondo: pd.Series, m_etf: pd.Series, rf: pd.Series | None, data_rif: pd.Timestamp) -> dict:
    out = {"rend": {}, "extra": {}}
    for a in ORIZZONTI:
        rf_ = rend_annualizzato(m_fondo, a, data_rif)
        re_ = rend_annualizzato(m_etf, a, data_rif)
        out["rend"][str(a)] = rf_
        out["extra"][str(a)] = (rf_ - re_) if (rf_ is not None and re_ is not None) else None
    reg = regressione(m_fondo.pct_change(), m_etf.pct_change(), rf)
    out["regressione"] = reg
    roll = rolling_12m(m_fondo, m_etf)
    out["rolling"] = [[d.strftime("%Y-%m"), round(float(v), 5)] for d, v in roll.items()]
    out["finestre_12m"] = int(len(roll))
    out["pct_batte_12m"] = float((roll > 0).mean()) if len(roll) else None
    return out
