"""Figuras comparativas (Original x Alto x Baixo) para o TCC."""
from pathlib import Path

import matplotlib

import numpy as np

CORES = {"original": "#1f77b4", "alto": "#2ca02c", "baixo": "#d62728"}
ROTULO = {"original": "Original (h=0,82 m)", "alto": "Alto (h=0,88 m)", "baixo": "Baixo (h=0,76 m)"}


def _plt():
    import matplotlib.pyplot as plt
    return plt


def plot_ensaio(resultados, caminho=None, titulo=None):
    """Series temporais de um ensaio, um cenario por curva. `resultados` = {cenario: Resultado}."""
    plt = _plt()
    fig, ax = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    ref = next(iter(resultados.values()))
    ax[0, 0].plot(ref.t, 1000 * ref.zp[0], "k", label="roda dianteira")
    ax[0, 0].plot(ref.t, 1000 * ref.zp[1], "k--", label="roda traseira")
    ax[0, 0].set_ylabel("Perfil da pista (mm)")
    ax[0, 0].legend(loc="best", fontsize=8)
    for nome, r in resultados.items():
        c, lab, p = CORES.get(nome), ROTULO.get(nome, nome), r.p
        ax[0, 1].plot(r.t, r.a_banco, color=c, label=lab)
        ax[1, 0].plot(r.t, np.degrees(r.X[1]), color=c, label=lab)
        ax[1, 1].plot(r.t, 1000 * -r.e[0], color=c, label=lab)
        ax[1, 1].axhline(1000 * p.curso_compressao, color=c, ls=":", lw=1)
        ax[2, 0].plot(r.t, (r.Fz[0] - p.Fz0_d) / p.Fz0_d, color=c, label=lab)
        ax[2, 1].plot(r.t, (r.Fz[1] - p.Fz0_t) / p.Fz0_t, color=c, label=lab)
    ax[0, 1].set_ylabel("Acel. vertical no banco (m/s²)")
    ax[1, 0].set_ylabel("Arfagem θ (°) (+ = nariz p/ baixo)")
    ax[1, 1].set_ylabel("Compressão susp. dianteira (mm)\n(pontilhado = batente)")
    ax[2, 0].set_ylabel("ΔFz / Fz0 - eixo dianteiro")
    ax[2, 1].set_ylabel("ΔFz / Fz0 - eixo traseiro")
    for a in (ax[2, 0], ax[2, 1]):
        a.axhline(-1.0, color="k", ls=":", lw=1)  # -1 = perda total de contato
        a.set_xlabel("Tempo (s)")
    ax[1, 0].legend(fontsize=8)
    for a in ax.ravel():
        a.grid(alpha=0.3)
    fig.suptitle(titulo or ref.ensaio.descricao)
    fig.tight_layout()
    if caminho:
        Path(caminho).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(caminho, dpi=150)
        plt.close(fig)
    return fig


def plot_barras(df, metrica, ylabel, caminho=None):
    """Barras agrupadas de uma metrica: ensaios no eixo x, um cenario por cor.
    `df` tem colunas ensaio, cenario e a metrica."""
    plt = _plt()
    ensaios = list(dict.fromkeys(df["ensaio"]))
    cen = [c for c in CORES if c in set(df["cenario"])]
    w = 0.8 / len(cen)
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(ensaios)), 5))
    for i, c in enumerate(cen):
        v = [df[(df.ensaio == e) & (df.cenario == c)][metrica].iloc[0] for e in ensaios]
        ax.bar(np.arange(len(ensaios)) + i * w, v, w, color=CORES[c], label=ROTULO[c])
    ax.set_xticks(np.arange(len(ensaios)) + w * (len(cen) - 1) / 2)
    ax.set_xticklabels(ensaios, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if caminho:
        Path(caminho).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(caminho, dpi=150)
        plt.close(fig)
    return fig


def plot_curvas(x, curvas, xlabel, ylabel, caminho=None, titulo=None, xlog=False):
    """`curvas` = {cenario: y}."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 5))
    for nome, y in curvas.items():
        ax.plot(x, y, "-o", ms=3, color=CORES.get(nome), label=ROTULO.get(nome, nome))
    if xlog:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(titulo or "")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    if caminho:
        Path(caminho).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(caminho, dpi=150)
        plt.close(fig)
    return fig
