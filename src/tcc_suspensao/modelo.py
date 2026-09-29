"""Half-car 4 GDL: equacoes de movimento (nao linear) e matrizes A, B linearizadas.

Estado X = [zc, theta, zrd, zrt, dzc, dtheta, dzrd, dzrt]  (desvios do equilibrio estatico).
Convencao do TCC (eq. 1-6): a frente esta em +a e theta > 0 abaixa a frente (nariz para baixo):
    e_d = zc - a*theta - zrd   (extensao da suspensao dianteira; e > 0 = estendida)
    e_t = zc + b*theta - zrt
    mc  zc''   = -(Fd + Ft)
    Iy  th''   =  Fd*a - Ft*b + M_lt
    mrd zrd''  =  Fd + dFz_d          mrt zrt'' = Ft + dFz_t
com F = ke*e + ce*de + F_batente e dFz = variacao da forca vertical do pneu.

Extensoes ao modelo linear do TCC (acordadas com o autor):
  * transferencia de carga longitudinal: momento M_lt = -M*ax*h sobre a massa suspensa
    (binario de inercia; em regime permanente gera dFz = M*ax*h/L, eq. 9 do TCC);
  * batentes de compressao/extensao (mola+amortecedor de contato);
  * pneu unilateral: Fz_total = Fz0 + kp*(zp - zr) + cp*(dzp - dzr) nao pode ser < 0.
"""
import numpy as np


def matrizes_lineares(p):
    """A (8x8) e B (8x4) do sistema linear, sem batentes nem descolamento do pneu.

    U = [zpd, zpt, dzpd, dzpt].  A e a matriz da eq. (8) do TCC.
    """
    mc, Iy, mrd, mrt = p.mc, p.Iy, p.m_rd, p.m_rt
    a, b = p.a, p.b
    K = np.array([
        [-(p.ked + p.ket), a * p.ked - b * p.ket, p.ked, p.ket],
        [a * p.ked - b * p.ket, -(a**2 * p.ked + b**2 * p.ket), -a * p.ked, b * p.ket],
        [p.ked, -a * p.ked, -(p.ked + p.kpd), 0.0],
        [p.ket, b * p.ket, 0.0, -(p.ket + p.kpt)],
    ])
    C = np.array([
        [-(p.ced + p.cet), a * p.ced - b * p.cet, p.ced, p.cet],
        [a * p.ced - b * p.cet, -(a**2 * p.ced + b**2 * p.cet), -a * p.ced, b * p.cet],
        [p.ced, -a * p.ced, -(p.ced + p.cpd), 0.0],
        [p.cet, b * p.cet, 0.0, -(p.cet + p.cpt)],
    ])
    Minv = np.diag([1 / mc, 1 / Iy, 1 / mrd, 1 / mrt])
    A = np.zeros((8, 8))
    A[:4, 4:] = np.eye(4)
    A[4:, :4] = Minv @ K
    A[4:, 4:] = Minv @ C
    B = np.zeros((8, 4))
    B[4:, 0] = Minv[:, 2] * p.kpd
    B[4:, 1] = Minv[:, 3] * p.kpt
    B[4:, 2] = Minv[:, 2] * p.cpd
    B[4:, 3] = Minv[:, 3] * p.cpt
    return A, B


def _forca_batente(p, e, de):
    """Forca de batente na convencao de F (positiva = tende a encurtar a suspensao)."""
    if e < -p.curso_compressao:          # batente de compressao
        return min(0.0, p.k_batente * (e + p.curso_compressao) + p.c_batente * de)
    if e > p.curso_extensao:             # batente de extensao
        return max(0.0, p.k_batente * (e - p.curso_extensao) + p.c_batente * de)
    return 0.0


class MeioCarro:
    """Modelo nao linear integravel com scipy.integrate.solve_ivp."""

    def __init__(self, p, entrada, batentes=True, pneu_unilateral=True, transferencia_carga=True):
        self.p = p
        self.ent = entrada
        self.batentes = batentes
        self.pneu_unilateral = pneu_unilateral
        self.transferencia_carga = transferencia_carga

    def _entrada_em(self, t):
        e = self.ent
        zp = (np.interp(t, e.t, e.zp[0]), np.interp(t, e.t, e.zp[1]))
        dzp = (np.interp(t, e.t, e.dzp[0]), np.interp(t, e.t, e.dzp[1]))
        return zp, dzp, np.interp(t, e.t, e.ax)

    def calcular(self, t, X):
        """Retorna (dX/dt, auxiliares) com as forcas do instante."""
        p = self.p
        zc, th, zd, zt, vc, vth, vd, vt = X
        (zpd, zpt), (dzpd, dzpt), ax = self._entrada_em(t)

        ed = zc - p.a * th - zd
        et = zc + p.b * th - zt
        ded = vc - p.a * vth - vd
        det = vc + p.b * vth - vt

        Fbd = _forca_batente(p, ed, ded) if self.batentes else 0.0
        Fbt = _forca_batente(p, et, det) if self.batentes else 0.0
        Fd = p.ked * ed + p.ced * ded + Fbd
        Ft = p.ket * et + p.cet * det + Fbt

        Fz_d = p.Fz0_d + p.kpd * (zpd - zd) + p.cpd * (dzpd - vd)
        Fz_t = p.Fz0_t + p.kpt * (zpt - zt) + p.cpt * (dzpt - vt)
        if self.pneu_unilateral:
            Fz_d, Fz_t = max(Fz_d, 0.0), max(Fz_t, 0.0)
        dFz_d, dFz_t = Fz_d - p.Fz0_d, Fz_t - p.Fz0_t

        M_lt = -p.M * ax * p.h if self.transferencia_carga else 0.0

        acc = np.array([
            -(Fd + Ft) / p.mc,
            (Fd * p.a - Ft * p.b + M_lt) / p.Iy,
            (Fd + dFz_d) / p.m_rd,
            (Ft + dFz_t) / p.m_rt,
        ])
        dX = np.concatenate([X[4:], acc])
        aux = dict(ed=ed, et=et, ded=ded, det=det, Fd=Fd, Ft=Ft, Fbd=Fbd, Fbt=Fbt,
                   Fz_d=Fz_d, Fz_t=Fz_t, acc=acc, ax=ax, zpd=zpd, zpt=zpt)
        return dX, aux

    def rhs(self, t, X):
        return self.calcular(t, X)[0]
