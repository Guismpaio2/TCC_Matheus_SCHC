"""Perfis de pista e trajetoria longitudinal do veiculo."""
from dataclasses import dataclass

import numpy as np


def _suave(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


class Plana:
    def __call__(self, s):
        return np.zeros_like(np.asarray(s, dtype=float))


class Degrau:
    def __init__(self, altura, rampa=0.2, x0=0.0):
        self.altura, self.rampa, self.x0 = altura, rampa, x0

    def __call__(self, s):
        return self.altura * _suave((np.asarray(s, float) - self.x0) / self.rampa)


class Lombada:
    """Lombada em seno quadrado: 0 -> altura -> 0 ao longo de `comprimento`."""

    def __init__(self, altura, comprimento, x0=0.0):
        self.altura, self.comprimento, self.x0 = altura, comprimento, x0

    def __call__(self, s):
        u = (np.asarray(s, float) - self.x0) / self.comprimento
        return np.where((u >= 0) & (u <= 1), self.altura * np.sin(np.pi * u) ** 2, 0.0)


class Buraco:
    """Depressao de `comprimento` com rampas suaves de entrada e saida."""

    def __init__(self, profundidade, comprimento, rampa=0.1, x0=0.0):
        self.p, self.c, self.r, self.x0 = profundidade, comprimento, rampa, x0

    def __call__(self, s):
        s = np.asarray(s, float)
        entra = _suave((s - self.x0) / self.r)
        sai = _suave((s - (self.x0 + self.c - self.r)) / self.r)
        return -self.p * entra * (1.0 - sai)


class Senoidal:
    def __init__(self, amplitude, comprimento_onda, x0=0.0):
        self.A, self.lam, self.x0 = amplitude, comprimento_onda, x0

    def __call__(self, s):
        u = np.asarray(s, float) - self.x0
        env = _suave(u / self.lam)  # entrada suave ao longo de 1 comprimento de onda
        return np.where(u >= 0, self.A * env * np.sin(2 * np.pi * u / self.lam), 0.0)


# Gd(n0 = 0,1 ciclo/m) em m^3, valor central de cada classe da ISO 8608.
GD0 = {"A": 16e-6, "B": 64e-6, "C": 256e-6, "D": 1024e-6}


class ISO8608:
    """Pista aleatoria por soma de senos com PSD Gd(n) = Gd0 (n/0,1)^-2."""

    def __init__(self, classe="C", semente=42, comprimento=500.0, dx=0.02,
                 n_min=0.02, n_max=10.0, n_comp=800):
        rng = np.random.default_rng(semente)
        bordas = np.linspace(n_min, n_max, n_comp + 1)
        n = rng.uniform(bordas[:-1], bordas[1:])
        dn = np.diff(bordas)
        amp = np.sqrt(2 * GD0[classe] * (n / 0.1) ** -2 * dn)
        fase = rng.uniform(0, 2 * np.pi, n_comp)
        self.x = np.arange(0.0, comprimento, dx)
        z = np.zeros_like(self.x)
        for i in range(0, len(self.x), 4000):  # em blocos para limitar memoria
            xs = self.x[i:i + 4000, None]
            z[i:i + 4000] = (amp * np.cos(2 * np.pi * n * xs + fase)).sum(axis=1)
        self.z = z - z[0]

    def __call__(self, s):
        return np.interp(np.asarray(s, float), self.x, self.z, left=0.0)


def criar_perfil(cfg):
    cfg = dict(cfg)
    tipo = cfg.pop("tipo")
    return {"plana": Plana, "degrau": Degrau, "lombada": Lombada, "buraco": Buraco,
            "senoidal": Senoidal, "iso8608": ISO8608}[tipo](**cfg)


@dataclass
class Entrada:
    """Excitacoes ja amostradas em grade fina, usadas pelo integrador."""
    t: np.ndarray
    s: np.ndarray      # posicao do eixo dianteiro (m)
    v: np.ndarray      # velocidade longitudinal (m/s)
    ax: np.ndarray     # aceleracao longitudinal (m/s2)
    zp: np.ndarray     # (2, n) perfil sob roda dianteira e traseira
    dzp: np.ndarray    # (2, n) derivada temporal do perfil


def gerar_entrada(perfil, L, v0, segmentos, T, dt=5e-4, suavizacao=0.15):
    """Integra ax(t) -> v(t) -> s(t) e amostra a pista sob os dois eixos.

    A roda traseira ve o mesmo perfil atrasado (posicao s - L, ou seja L/v).
    ax e suavizada por media movel (`suavizacao` s) para evitar jerk infinito.
    """
    t = np.arange(0.0, T + dt, dt)
    ax = np.zeros_like(t)
    for t0, t1, a in segmentos:
        ax[(t >= t0) & (t < t1)] = a
    w = int(suavizacao / dt)
    if w > 1:
        ax = np.convolve(ax, np.ones(w) / w, mode="same")
    v = np.empty_like(t)
    v[0] = v0
    for i in range(1, len(t)):
        v[i] = v[i - 1] + ax[i - 1] * dt
        if v[i] <= 0.0:
            v[i] = 0.0
            ax[i] = max(ax[i], 0.0)
    s = np.concatenate(([0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * dt)))
    eps = 1e-3
    zp = np.vstack([perfil(s), perfil(s - L)])
    dz = np.vstack([(perfil(s + eps) - perfil(s - eps)) / (2 * eps) * v,
                    (perfil(s - L + eps) - perfil(s - L - eps)) / (2 * eps) * v])
    return Entrada(t, s, v, ax, zp, dz)
