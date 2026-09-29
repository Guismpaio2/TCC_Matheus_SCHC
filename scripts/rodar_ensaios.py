"""Roda todos os ensaios x cenarios, salva tabelas (CSV) e figuras (PNG).

Uso:  python scripts/rodar_ensaios.py [--ensaios a b c]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tcc_suspensao import graficos
from tcc_suspensao.cenarios import carregar_ensaios
from tcc_suspensao.metricas import calcular_metricas
from tcc_suspensao.parametros import RAIZ, carregar_cenarios
from tcc_suspensao.simulacao import simular

METRICAS_BARRAS = {
    "rms_total_pond_ms2": "Aceleração RMS ponderada total (m/s²) - conforto",
    "vdv_ms175": "VDV (m/s^1,75) - conforto em choques",
    "pitch_max_graus": "Arfagem máxima (°)",
    "dFz_rms_rel_diant": "RMS(ΔFz)/Fz0 dianteiro - aderência",
    "Fz_min_rel": "Fz mínima / Fz estática",
    "F_batente_max_N": "Força máxima de batente (N) - durabilidade",
    "dano_fadiga_rel": "Dano de fadiga relativo ao cenário original",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensaios", nargs="*")
    args = ap.parse_args()

    cenarios, ensaios = carregar_cenarios(), carregar_ensaios()
    if args.ensaios:
        ensaios = {k: ensaios[k] for k in args.ensaios}
    tab, fig_dir = RAIZ / "results" / "tables", RAIZ / "results" / "figures"
    linhas = []
    for en, ensaio in ensaios.items():
        res = {}
        for cn, p in cenarios.items():
            print(f"  simulando {en} / {cn} ...", flush=True)
            res[cn] = simular(p, ensaio)
            linhas.append({"ensaio": en, "cenario": cn, **calcular_metricas(res[cn])})
        graficos.plot_ensaio(res, fig_dir / f"ensaio_{en}.png")

    df = pd.DataFrame(linhas)
    base = df[df.cenario == "original"].set_index("ensaio")["dano_fadiga_indice"]
    df["dano_fadiga_rel"] = df.apply(
        lambda r: r["dano_fadiga_indice"] / base[r["ensaio"]] if base[r["ensaio"]] > 0 else float("nan"), axis=1)
    tab.mkdir(parents=True, exist_ok=True)
    df.to_csv(tab / "metricas_completas.csv", index=False, encoding="utf-8-sig")

    # tabela de variacao percentual em relacao ao original
    num = df.select_dtypes("number").columns
    ref = df[df.cenario == "original"].set_index("ensaio")[num]
    var = df.copy()
    for i, r in df.iterrows():
        b = ref.loc[r["ensaio"]]
        var.loc[i, num] = [(r[c] - b[c]) / b[c] * 100 if b[c] else float("nan") for c in num]
    var.to_csv(tab / "variacao_percentual_vs_original.csv", index=False, encoding="utf-8-sig")

    for m, lab in METRICAS_BARRAS.items():
        graficos.plot_barras(df, m, lab, fig_dir / f"barras_{m}.png")
    pd.set_option("display.width", 200)
    print(df[["ensaio", "cenario", "rms_total_pond_ms2", "classe_conforto", "pitch_max_graus",
              "Fz_min_rel", "batidas_compressao", "F_batente_max_N", "dano_fadiga_rel"]].round(3).to_string())
    print(f"\nTabelas em {tab}\nFiguras em {fig_dir}")


if __name__ == "__main__":
    main()
