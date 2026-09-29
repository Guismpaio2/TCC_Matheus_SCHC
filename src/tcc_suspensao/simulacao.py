"""Integracao temporal (RK45 via scipy.integrate.solve_ivp) de um ensaio."""
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from .estrada import criar_perfil, gerar_entrada
from .modelo import MeioCarro


@dataclass
class Ensaio:
    nome: str
    descricao: str
    T: float
    v0: float                 # m/s
    acel: list = field(default_factory=list)
    pista: dict = field(default_factory=lambda: {"tipo": "plana"})
    _perfil: object = None

    def perfil(self):
        if self._perfil is None:
            self._perfil = criar_perfil(self.pista)
        return self._perfil


@dataclass
class Resultado:
    p: object
    ensaio: Ensaio
    t: np.ndarray
    X: np.ndarray            # (8, n)
    acc: np.ndarray          # (4, n) aceleracoes [zc'', th'', zrd'', zrt'']
    zp: np.ndarray           # (2, n) pista sob rodas [dianteira, traseira]
    ax: np.ndarray
    v: np.ndarray
    s: np.ndarray
    e: np.ndarray            # (2, n) extensao das suspensoes (negativo = comprimida)
    de: np.ndarray
    F_susp: np.ndarray       # (2, n) forca total (mola+amortecedor+batente)
    F_bat: np.ndarray        # (2, n) so batentes
    Fz: np.ndarray           # (2, n) forca vertical total nos pneus
    a_banco: np.ndarray      # aceleracao vertical no banco do motorista

    @property
    def dt(self):
        return self.t[1] - self.t[0]


def simular(p, ensaio, dt_saida=1e-3, max_step=1e-3, rtol=1e-6, atol=1e-9, **opcoes):
    """Simula `ensaio` para os parametros `p`. `opcoes` vai para MeioCarro."""
    ent = gerar_entrada(ensaio.perfil(), p.L, ensaio.v0, ensaio.acel, ensaio.T)
    modelo = MeioCarro(p, ent, **opcoes)
    t_eval = np.arange(0.0, ensaio.T, dt_saida)
    sol = solve_ivp(modelo.rhs, (0.0, ensaio.T), np.zeros(8), method="RK45",
                    t_eval=t_eval, max_step=max_step, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"solve_ivp falhou: {sol.message}")

    n = len(sol.t)
    acc = np.zeros((4, n))
    e = np.zeros((2, n))
    de = np.zeros((2, n))
    F = np.zeros((2, n))
    Fb = np.zeros((2, n))
    Fz = np.zeros((2, n))
    zp = np.zeros((2, n))
    ax = np.zeros(n)
    for i in range(n):
        _, a = modelo.calcular(sol.t[i], sol.y[:, i])
        acc[:, i] = a["acc"]
        e[:, i] = (a["ed"], a["et"])
        de[:, i] = (a["ded"], a["det"])
        F[:, i] = (a["Fd"], a["Ft"])
        Fb[:, i] = (a["Fbd"], a["Fbt"])
        Fz[:, i] = (a["Fz_d"], a["Fz_t"])
        zp[:, i] = (a["zpd"], a["zpt"])
        ax[i] = a["ax"]
    a_banco = acc[0] - p.x_banco * acc[1]   # zc'' - x_banco * theta''
    return Resultado(p, ensaio, sol.t, sol.y, acc, zp, ax,
                     np.interp(sol.t, ent.t, ent.v), np.interp(sol.t, ent.t, ent.s),
                     e, de, F, Fb, Fz, a_banco)
