"""Test delle metriche su casi con risultato noto. Esegui: python tests/test_metrics.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import fve_metrics as M  # noqa: E402

T = pd.Timestamp("2026-10-31")
IDX = pd.date_range(T - pd.offsets.MonthEnd(131), T, freq="ME")


def prezzi(r):
    return pd.Series(100 * np.cumprod(1 + np.asarray(r)), index=IDX[-len(r):])


def test_cagr():
    m = prezzi(np.full(132, 0.01))
    assert abs(M.rend_annualizzato(m, 1, T) - (1.01 ** 12 - 1)) < 1e-12
    assert abs(M.rend_annualizzato(m, 10, T) - (1.01 ** 12 - 1)) < 1e-12
    corto = prezzi(np.full(40, 0.01))
    assert M.rend_annualizzato(corto, 3, T) is not None
    assert M.rend_annualizzato(corto, 5, T) is None  # storico insufficiente: niente numero


def test_regressione_alfa_beta_noti():
    rng = np.random.default_rng(0)
    e = rng.normal(0.006, 0.04, 132)
    f = 0.002 + 1.2 * e  # alfa 0,2%/mese, beta 1,2, nessun rumore
    reg = M.regressione(prezzi(f).pct_change(), prezzi(e).pct_change(), None)
    assert reg["mesi"] == 120
    assert abs(reg["beta"] - 1.2) < 1e-9
    assert abs(reg["alfa"] - 0.024) < 1e-9
    assert abs(reg["r2"] - 1) < 1e-9


def test_regressione_con_rf():
    """Con rf costante e beta 1.2: alfa di Jensen = a_grezzo - (1-beta)*rf = a + 0.2*rf."""
    rng = np.random.default_rng(1)
    e = rng.normal(0.006, 0.04, 132)
    f = 0.001 + 1.2 * e
    rf = pd.Series(0.002, index=IDX)
    reg = M.regressione(prezzi(f).pct_change(), prezzi(e).pct_change(), rf)
    assert abs(reg["alfa"] - (0.001 + 0.2 * 0.002) * 12) < 1e-9


def test_te_ir():
    rng = np.random.default_rng(2)
    e = rng.normal(0.006, 0.04, 132)
    att = rng.normal(0.001, 0.01, 132)
    reg = M.regressione(prezzi(e + att).pct_change(), prezzi(e).pct_change(), None)
    a = att[-120:]
    assert abs(reg["te"] - a.std(ddof=1) * np.sqrt(12)) < 1e-9
    assert abs(reg["ir"] - a.mean() * 12 / (a.std(ddof=1) * np.sqrt(12))) < 1e-9


def test_rolling_e_quota():
    e = np.full(132, 0.005)
    f = np.full(132, 0.006)
    roll = M.rolling_12m(prezzi(f), prezzi(e))
    assert len(roll) == 120
    assert abs(roll.iloc[-1] - (1.006 ** 12 - 1.005 ** 12)) < 1e-12
    c = M.confronta(prezzi(f), prezzi(e), None, T)
    assert c["pct_batte_12m"] == 1.0


def test_quartili():
    q = M.quartile({"a": 4, "b": 3, "c": 2, "d": 1})
    assert q == {"a": 1, "b": 2, "c": 3, "d": 4}
    q = M.quartile({str(i): i for i in range(8)})
    assert q["7"] == 1 and q["0"] == 4 and sorted(q.values()) == [1, 1, 2, 2, 3, 3, 4, 4]


def test_persistenza_perfetta():
    """Fondi con alfa costante e diverso: i quartili devono restare identici (diagonale)."""
    rng = np.random.default_rng(3)
    e = rng.normal(0.006, 0.04, 132)
    fondi = {f"F{k}": prezzi(e + 0.001 * k) for k in range(8)}
    p = M.persistenza(fondi, prezzi(e), T)
    mat = np.array(p["matrice"])
    assert mat.sum() == p["n_transizioni"] == 8 * len(p["date"])
    assert mat.trace() == mat.sum()
    assert len(p["date"]) == 5  # formazione a T-36, -48, -60, -72, -84 mesi
    assert abs(p["spearman_medio"] - 1) < 1e-12


def test_qualita_buchi_outlier():
    m = prezzi(np.r_[np.full(30, 0.01), 0.6, np.full(10, 0.01)])
    m = m.drop(m.index[5])
    full, flags = M.qualita(m)
    assert len(full) == len(m) + 1
    assert any("mesi senza prezzo" in f for f in flags)
    assert any("anomali" in f for f in flags)


if __name__ == "__main__":
    n = 0
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); n += 1
            print("ok", k)
    print(f"{n} test superati")
