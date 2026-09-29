import numpy as np
import pytest
from scipy.integrate import solve_ivp

from tcc_suspensao.cenarios import carregar_ensaios
from tcc_suspensao.estrada import Lombada, gerar_entrada
from tcc_suspensao.metricas import peso_we, peso_wk, ponderar, rainflow, rms
from tcc_suspensao.modelo import MeioCarro, matrizes_lineares
from tcc_suspensao.parametros import carregar_cenarios
from tcc_suspensao.simulacao import Ensaio, simular

CEN = carregar_cenarios()


def test_cenarios_carregam_e_estatica_fecha():
    for p in CEN.values():
        assert p.mc == pytest.approx(820.0)
        assert p.a + p.b == pytest.approx(p.L)
        assert p.Fz0_d + p.Fz0_t == pytest.approx(p.M * 9.81)
        assert p.curso_compressao + p.curso_extensao == pytest.approx(0.120)


def test_nao_linear_igual_ao_linear_sem_nao_linearidades():
    """Sem batentes, sem descolamento e sem transferencia de carga, o modelo
    nao linear deve reproduzir exatamente o espaco de estados X' = AX + BU."""
    p = CEN["original"]
    ent = gerar_entrada(Lombada(0.02, 1.5, 3.0), p.L, 5.0, [], 3.0)
    m = MeioCarro(p, ent, batentes=False, pneu_unilateral=False, transferencia_carga=False)
    A, B = matrizes_lineares(p)

    def f_lin(t, X):
        U = np.array([np.interp(t, ent.t, ent.zp[0]), np.interp(t, ent.t, ent.zp[1]),
                      np.interp(t, ent.t, ent.dzp[0]), np.interp(t, ent.t, ent.dzp[1])])
        return A @ X + B @ U

    kw = dict(t_eval=np.linspace(0, 3, 300), max_step=1e-3, rtol=1e-8, atol=1e-10)
    s1 = solve_ivp(m.rhs, (0, 3), np.zeros(8), **kw)
    s2 = solve_ivp(f_lin, (0, 3), np.zeros(8), **kw)
    assert np.max(np.abs(s1.y - s2.y)) < 1e-5


def test_frequencia_natural_massa_suspensa_plausivel():
    A, _ = matrizes_lineares(CEN["original"])
    f = np.abs(np.linalg.eigvals(A).imag) / (2 * np.pi)
    f = np.unique(np.round(f[f > 0.1], 3))
    assert 1.0 < f.min() < 2.0        # modo de bounce/pitch ~1-2 Hz
    assert 10 < f.max() < 20          # salto de roda ~10-20 Hz
    assert np.all(np.linalg.eigvals(A).real < 0)


def test_baixa_rigidez_menor_frequencia_e_vice_versa():
    f = {}
    for n, p in CEN.items():
        A, _ = matrizes_lineares(p)
        e = np.linalg.eigvals(A)
        f[n] = np.abs(e.imag[e.imag != 0]).min() / (2 * np.pi)
    assert f["alto"] < f["original"] < f["baixo"]


def test_frenagem_regime_permanente_igual_eq9():
    """Frenagem constante e longa: dFz dianteiro -> M*ax*h/L (eq. 9 do TCC)."""
    p = CEN["original"]
    ens = Ensaio("f", "", 8.0, 30.0, [[0.0, 8.0, -2.0]], {"tipo": "plana"})
    r = simular(p, ens, batentes=False)
    i = int(6.0 / r.dt)   # longe das bordas da suavizacao de ax
    ax = r.ax[i]
    esperado = -p.M * ax * p.h / p.L
    assert (r.Fz[0, i] - p.Fz0_d) == pytest.approx(esperado, rel=0.03)
    assert (r.Fz[1, i] - p.Fz0_t) == pytest.approx(-esperado, rel=0.03)


def test_menor_h_reduz_arfagem_em_frenagem_mesmos_k():
    """H1 isolada (mesmos k, c): pitch de regime cresce com h."""
    import dataclasses
    p0 = CEN["original"]
    ens = Ensaio("f", "", 6.0, 30.0, [[0.0, 6.0, -3.0]], {"tipo": "plana"})
    th = [np.max(np.abs(simular(dataclasses.replace(p0, h=h), ens, batentes=False).X[1]))
          for h in (0.70, 0.82, 0.95)]
    assert th[0] < th[1] < th[2]


def test_wk_ponderacao_senoide_1hz_e_8hz():
    fs = 1000.0
    t = np.arange(0, 20, 1 / fs)
    for f0, w in ((1.0, 0.885), (8.0, 0.494)):
        x = np.sin(2 * np.pi * f0 * t)
        assert rms(ponderar(x, fs, peso_wk)[2000:-2000]) / rms(x) == pytest.approx(w, rel=0.03)
    assert rms(ponderar(np.ones_like(t), fs, peso_wk)[2000:-2000]) < 0.05  # DC removido


def test_we_analitico_perto_de_1_em_baixa_frequencia():
    assert peso_we(np.array([1.0]))[0] == pytest.approx(0.89, abs=0.1)
    assert peso_we(np.array([20.0]))[0] < 0.1


def test_rainflow_exemplo_astm():
    ciclos = rainflow([-2, 1, -3, 5, -1, 3, -4, 4, -2], minimo=0)
    # exemplo classico da ASTM E1049: soma das contagens = 8 meios ciclos + ... conferir faixas
    total = sum(c for _, c in ciclos)
    assert total == pytest.approx(4.0)  # 8 reversoes -> 8 meios ciclos = 4
    assert max(r for r, _ in ciclos) == pytest.approx(9.0)


def test_baixo_bate_no_batente_em_lombada_e_original_nao():
    e = carregar_ensaios()["lombada_40kmh"]
    from tcc_suspensao.metricas import calcular_metricas
    b = calcular_metricas(simular(CEN["baixo"], e))
    o = calcular_metricas(simular(CEN["original"], e))
    assert b["batidas_compressao"] > 0 and o["batidas_compressao"] == 0
