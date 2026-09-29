"""Varredura continua da altura do CG (h) interpolando k, c e cursos entre os 3 cenarios.

Mostra a tendencia (nao apenas 3 pontos) de conforto, seguranca e durabilidade com h.
Fora do intervalo [0,76; 0,88] os parametros sao extrapolados linearmente (aproximacao).
"""
import dataclasses
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tcc_suspensao import graficos
from tcc_suspensao.cenarios import carregar_ensaios
from tcc_suspensao.metricas import calcular_metricas
from tcc_suspensao.parametros import RAIZ, carregar_cenarios, carregar_yaml, montar_parametros
from tcc_suspensao.simulacao import simular

ENSAIOS = ["lombada_40kmh", "buraco_50kmh", "frenagem_plana", "iso8608_classeC_60kmh"]
H_VALORES = np.round(np.arange(0.70, 0.9401, 0.02), 3)


def interp_extrap(x, xs, ys):
    """Interpolacao linear com extrapolacao linear nas pontas (xs crescente)."""
    xs, ys = np.asarray(xs), np.asarray(ys)
    if x <= xs[0]:
        return ys[0] + (x - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0])
    if x >= xs[-1]:
        return ys[-1] + (x - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    return float(np.interp(x, xs, ys))


def main():
    cfg = carregar_yaml()
    cens = sorted(cfg["cenarios"].values(), key=lambda c: c["h"])
    hs = [c["h"] for c in cens]
    chaves = ["fator_k", "fator_c", "curso_compressao_mm", "curso_extensao_mm"]
    ensaios = carregar_ensaios()
    linhas = []
    for h in H_VALORES:
        cen = {"h": float(h), **{k: interp_extrap(h, hs, [c[k] for c in cens]) for k in chaves}}
        cen["curso_compressao_mm"] = max(cen["curso_compressao_mm"], 5.0)
        cen["curso_extensao_mm"] = max(cen["curso_extensao_mm"], 5.0)
        cen["fator_c"] = max(cen["fator_c"], 0.3)
        cen["fator_k"] = max(cen["fator_k"], 0.3)
        p = montar_parametros(f"h={h}", cfg["veiculo"], cen)
        for en in ENSAIOS:
            m = calcular_metricas(simular(p, ensaios[en]))
            linhas.append({"h": h, "ensaio": en, **m})
        print(f"  h={h} ok", flush=True)
    df = pd.DataFrame(linhas)
    (RAIZ / "results" / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(RAIZ / "results" / "tables" / "varredura_altura.csv", index=False, encoding="utf-8-sig")
    fig = RAIZ / "results" / "figures"
    for m, lab in (("rms_total_pond_ms2", "Acel. RMS ponderada total (m/s²)"),
                   ("pitch_max_graus", "Arfagem máxima (°)"),
                   ("Fz_min_rel", "Fz mínima / Fz estática"),
                   ("F_batente_max_N", "Força máxima de batente (N)"),
                   ("dano_fadiga_indice", "Índice de dano de fadiga (N³)")):
        plt = graficos._plt()
        f, ax = plt.subplots(figsize=(8, 5))
        for en in ENSAIOS:
            d = df[df.ensaio == en]
            ax.plot(d.h, d[m], "-o", ms=3, label=en)
        for c in cens:
            ax.axvline(c["h"], color="gray", ls=":", lw=1)
        ax.set_xlabel("Altura do CG h (m)  [linhas: baixo / original / alto]")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        f.tight_layout()
        f.savefig(fig / f"varredura_altura_{m}.png", dpi=150)
        plt.close(f)


if __name__ == "__main__":
    main()
