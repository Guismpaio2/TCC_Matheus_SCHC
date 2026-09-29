"""Parametros do veiculo e dos cenarios de altura do CG."""
from dataclasses import dataclass
from pathlib import Path

import yaml

G = 9.81
RAIZ = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Parametros:
    nome: str
    M: float
    m_rd: float
    m_rt: float
    L: float
    a: float          # CG -> eixo dianteiro
    b: float          # CG -> eixo traseiro
    h: float          # altura do CG ao solo
    Iy: float
    ked: float
    ket: float
    ced: float
    cet: float
    kpd: float
    kpt: float
    cpd: float
    cpt: float
    k_batente: float
    c_batente: float
    curso_compressao: float   # m, da posicao estatica ate o batente de compressao
    curso_extensao: float     # m, da posicao estatica ate o batente de extensao
    x_banco: float

    @property
    def mc(self):
        return self.M - self.m_rd - self.m_rt

    @property
    def Fz0_d(self):
        """Carga estatica no eixo dianteiro (N)."""
        return (self.mc * self.b / self.L + self.m_rd) * G

    @property
    def Fz0_t(self):
        return (self.mc * self.a / self.L + self.m_rt) * G


def montar_parametros(nome, veic, cen):
    L = veic["L"]
    a = (1.0 - veic["fracao_peso_dianteiro"]) * L
    b = L - a
    m_nao = veic["m_nao_suspensa_eixo"]
    mc = veic["M"] - 2 * m_nao
    ry = veic["raio_giracao_rel"] * L
    fk, fc = cen["fator_k"], cen["fator_c"]
    return Parametros(
        nome=nome, M=veic["M"], m_rd=m_nao, m_rt=m_nao, L=L, a=a, b=b,
        h=cen["h"], Iy=mc * ry**2,
        ked=veic["ke_d"] * fk, ket=veic["ke_t"] * fk,
        ced=veic["ce_d"] * fc, cet=veic["ce_t"] * fc,
        kpd=veic["kp_d"], kpt=veic["kp_t"], cpd=veic["cp_d"], cpt=veic["cp_t"],
        k_batente=veic["k_batente"], c_batente=veic["c_batente"],
        curso_compressao=cen["curso_compressao_mm"] / 1000.0,
        curso_extensao=cen["curso_extensao_mm"] / 1000.0,
        x_banco=veic["x_banco"],
    )


def carregar_yaml(caminho=None):
    caminho = Path(caminho) if caminho else RAIZ / "config" / "cenarios.yaml"
    with open(caminho, encoding="utf-8") as f:
        return yaml.safe_load(f)


def carregar_cenarios(caminho=None):
    """Retorna dict nome -> Parametros (original, alto, baixo)."""
    cfg = carregar_yaml(caminho)
    return {n: montar_parametros(n, cfg["veiculo"], c) for n, c in cfg["cenarios"].items()}
