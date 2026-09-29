"""Varredura de frequencia (pista senoidal a 50 km/h): conforto e aderencia vs frequencia.

Serve para a hipotese H3 (ΔFz perto da frequencia natural da massa suspensa).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tcc_suspensao import graficos
from tcc_suspensao.metricas import rms
from tcc_suspensao.parametros import RAIZ, carregar_cenarios
from tcc_suspensao.simulacao import Ensaio, simular

V = 50 / 3.6
AMP = 0.02
FREQS = np.round(np.geomspace(0.6, 16.0, 26), 3)


def main():
    cen = carregar_cenarios()
    linhas = []
    for f in FREQS:
        T = max(6.0, 10.0 / f)
        ens = Ensaio("senoidal", f"{f} Hz", T, V, [], {"tipo": "senoidal", "amplitude": AMP,
                                                       "comprimento_onda": V / f})
        for cn, p in cen.items():
            r = simular(p, ens)
            i0 = int(0.4 * len(r.t))  # descarta transiente
            linhas.append({
                "freq_hz": f, "cenario": cn,
                "acel_banco_rms": rms(r.a_banco[i0:]),
                "dFz_rms_rel_diant": rms(r.Fz[0, i0:] - p.Fz0_d) / p.Fz0_d,
                "dFz_max_rel": max(np.max(np.abs(r.Fz[0, i0:] - p.Fz0_d)) / p.Fz0_d,
                                   np.max(np.abs(r.Fz[1, i0:] - p.Fz0_t)) / p.Fz0_t),
                "pitch_max_graus": float(np.degrees(np.max(np.abs(r.X[1, i0:])))),
            })
        print(f"  {f} Hz ok", flush=True)
    df = pd.DataFrame(linhas)
    (RAIZ / "results" / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(RAIZ / "results" / "tables" / "varredura_frequencia.csv", index=False, encoding="utf-8-sig")
    fig = RAIZ / "results" / "figures"
    for m, lab in (("acel_banco_rms", "Acel. vertical RMS no banco (m/s²)"),
                   ("dFz_rms_rel_diant", "RMS(ΔFz)/Fz0 dianteiro"),
                   ("dFz_max_rel", "max |ΔFz|/Fz0 (1,0 = perda de contato)"),
                   ("pitch_max_graus", "Arfagem máxima (°)")):
        curvas = {c: df[df.cenario == c][m].to_numpy() for c in cen}
        graficos.plot_curvas(FREQS, curvas, "Frequência de excitação (Hz)", lab,
                             fig / f"varredura_freq_{m}.png", xlog=True,
                             titulo=f"Pista senoidal {AMP*1000:.0f} mm a 50 km/h")


if __name__ == "__main__":
    main()
