"""Metricas de conforto (ISO 2631-1), seguranca e durabilidade."""
import numpy as np

from .parametros import G

# Fator de ponderacao Wk (vertical), ISO 2631-1 Tab. 3 (tercos de oitava).
_WK_F = np.array([0.4, 0.5, 0.63, 0.8, 1, 1.25, 1.6, 2, 2.5, 3.15, 4, 5, 6.3, 8, 10,
                  12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100])
_WK_W = np.array([0.313, 0.453, 0.626, 0.782, 0.885, 0.929, 0.932, 0.910, 0.872, 0.818,
                  0.750, 0.669, 0.582, 0.494, 0.422, 0.361, 0.306, 0.256, 0.212, 0.173,
                  0.140, 0.109, 0.0834, 0.0604, 0.0417])

K_ROT = 0.63   # ISO 2631-1 Tab. 4: multiplicador de rotacao (m/rad) no assento


def peso_wk(f):
    """|Wk(f)|: tabela ISO em log-log; abaixo de 0,4 Hz e acima de 100 Hz cai a 12 dB/oitava."""
    f = np.asarray(f, float)
    w = np.zeros_like(f)
    dentro = (f >= _WK_F[0]) & (f <= _WK_F[-1])
    w[dentro] = np.exp(np.interp(np.log(f[dentro]), np.log(_WK_F), np.log(_WK_W)))
    baixo = (f > 0) & (f < _WK_F[0])
    w[baixo] = _WK_W[0] * (f[baixo] / _WK_F[0]) ** 2
    alto = f > _WK_F[-1]
    w[alto] = _WK_W[-1] * (_WK_F[-1] / f[alto]) ** 2
    return w


def peso_we(f):
    """|We(f)| (rotacoes no assento), forma analitica do Anexo A da ISO 2631-1.

    Ht = (1 + s/w3) / (1 + s/(Q4 w4) + s^2/w4^2) com f3 = f4 = 1 Hz, Q4 = 0,63,
    multiplicado pelos filtros passa-banda Butterworth de 2a ordem (0,4 Hz e 100 Hz).
    """
    f = np.asarray(f, float)
    s = 1j * f
    Ht = (1 + s / 1.0) / (1 + s / (0.63 * 1.0) + s**2 / 1.0**2)
    f1, f2 = 0.4, 100.0
    Hh = s**2 / (s**2 + np.sqrt(2) * f1 * s + f1**2)
    Hl = f2**2 / (s**2 + np.sqrt(2) * f2 * s + f2**2)
    return np.abs(Ht * Hh * Hl)


def ponderar(sinal, fs, peso=peso_wk):
    """Aplica ponderacao em frequencia via FFT (com zero-padding)."""
    x = np.asarray(sinal, float)
    n = len(x)
    N = 1 << int(np.ceil(np.log2(2 * n)))
    X = np.fft.rfft(x, N)
    f = np.fft.rfftfreq(N, 1.0 / fs)
    return np.fft.irfft(X * peso(f), N)[:n]


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def classificar_conforto(a_rms):
    """Faixas indicativas da ISO 2631-1 (Anexo C), aceleracao ponderada total em m/s2."""
    if a_rms < 0.315:
        return "nao desconfortavel"
    if a_rms < 0.63:
        return "pouco desconfortavel"
    if a_rms < 1.0:
        return "razoavelmente desconfortavel"
    if a_rms < 1.6:
        return "desconfortavel"
    if a_rms < 2.5:
        return "muito desconfortavel"
    return "extremamente desconfortavel"


def rainflow(x, minimo=0.5):
    """Contagem rainflow (ASTM E1049, 3 pontos). Retorna lista de (faixa, contagem)."""
    x = np.asarray(x, float)
    x = x[np.r_[True, np.diff(x) != 0]]
    if len(x) < 3:
        return []
    d = np.sign(np.diff(x))
    idx = np.where(d[1:] != d[:-1])[0] + 1
    pts = np.concatenate(([x[0]], x[idx], [x[-1]]))
    pilha, ciclos = [], []
    for p in pts:
        pilha.append(p)
        while len(pilha) >= 3:
            X = abs(pilha[-1] - pilha[-2])
            Y = abs(pilha[-2] - pilha[-3])
            if X < Y:
                break
            if len(pilha) == 3:
                ciclos.append((Y, 0.5))
                pilha.pop(0)
            else:
                ciclos.append((Y, 1.0))
                del pilha[-3:-1]
    ciclos += [(abs(pilha[i + 1] - pilha[i]), 0.5) for i in range(len(pilha) - 1)]
    return [(r, c) for r, c in ciclos if r >= minimo]


def _episodios(ativo):
    """Numero de intervalos contiguos True."""
    a = np.asarray(ativo, bool)
    return int(np.sum(a[1:] & ~a[:-1]) + (1 if a[0] else 0))


def calcular_metricas(r, expoente_wohler=3.0):
    """Todas as metricas de um Resultado -> dict plano (unidades no nome)."""
    p, fs = r.p, 1.0 / r.dt
    m = {}

    # ---- Conforto (ISO 2631-1): vertical no banco + arfagem, ponderados ----
    aw_z = ponderar(r.a_banco, fs, peso_wk)
    aw_th = ponderar(r.acc[1], fs, peso_we)
    rms_z, rms_th = rms(aw_z), rms(aw_th)
    m["rms_vert_pond_ms2"] = rms_z
    m["rms_arfagem_pond_rads2"] = rms_th
    m["rms_total_pond_ms2"] = float(np.sqrt(rms_z**2 + (K_ROT * rms_th) ** 2))
    m["classe_conforto"] = classificar_conforto(m["rms_total_pond_ms2"])
    m["vdv_ms175"] = float((np.sum(aw_z**4) * r.dt) ** 0.25)
    m["fator_crista"] = float(np.max(np.abs(aw_z)) / rms_z) if rms_z > 0 else 0.0
    m["pico_acel_banco_ms2"] = float(np.max(np.abs(r.a_banco)))
    m["pico_acel_chassi_g"] = float(np.max(np.abs(r.acc[0])) / G)

    # ---- Seguranca ----
    m["pitch_max_graus"] = float(np.degrees(np.max(np.abs(r.X[1]))))
    m["vel_arfagem_max_grau_s"] = float(np.degrees(np.max(np.abs(r.X[5]))))
    for k, nome, Fz0 in ((0, "diant", p.Fz0_d), (1, "tras", p.Fz0_t)):
        dFz = r.Fz[k] - Fz0
        m[f"dFz_rms_{nome}_N"] = rms(dFz)
        m[f"dFz_rms_rel_{nome}"] = rms(dFz) / Fz0            # indice de "road holding"
        m[f"dFz_max_{nome}_N"] = float(np.max(np.abs(dFz)))
        m[f"Fz_min_rel_{nome}"] = float(np.min(r.Fz[k]) / Fz0)
        m[f"perda_contato_{nome}_pct"] = 100.0 * float(np.mean(r.Fz[k] <= 1e-6))
    m["perda_contato_pct"] = max(m["perda_contato_diant_pct"], m["perda_contato_tras_pct"])
    m["Fz_min_rel"] = min(m["Fz_min_rel_diant"], m["Fz_min_rel_tras"])
    ax_max = float(np.max(np.abs(r.ax)))
    m["transf_carga_teorica_N"] = p.M * ax_max * p.h / p.L    # eq. 9 do TCC
    m["transf_carga_simulada_N"] = float(np.max(np.abs(r.Fz[0] - p.Fz0_d)))

    # ---- Durabilidade ----
    pen_bump = np.maximum(0.0, -r.e - p.curso_compressao)       # (2, n)
    pen_reb = np.maximum(0.0, r.e - p.curso_extensao)
    m["batidas_compressao"] = sum(_episodios(pen_bump[k] > 0) for k in range(2))
    m["batidas_extensao"] = sum(_episodios(pen_reb[k] > 0) for k in range(2))
    m["penetracao_batente_max_mm"] = 1000.0 * float(max(pen_bump.max(), pen_reb.max()))
    m["F_batente_max_N"] = float(np.max(np.abs(r.F_bat)))
    m["energia_batente_J"] = float(np.sum(np.abs(r.F_bat * r.de)) * r.dt)
    m["curso_usado_compressao_mm"] = 1000.0 * float(max(0.0, np.max(-r.e)))
    m["curso_usado_extensao_mm"] = 1000.0 * float(max(0.0, np.max(r.e)))
    m["F_susp_max_N"] = float(np.max(np.abs(r.F_susp)))
    m["F_susp_rms_N"] = rms(r.F_susp)
    dano = 0.0
    for k in range(2):
        for faixa, cont in rainflow(r.F_susp[k]):
            dano += cont * (faixa / 2.0) ** expoente_wohler
    m["dano_fadiga_indice"] = float(dano)   # so faz sentido relativo (razao ao cenario original)
    return m
