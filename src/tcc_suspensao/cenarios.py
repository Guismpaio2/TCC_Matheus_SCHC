"""Leitura dos ensaios (config/ensaios.yaml) e execucao ensaios x cenarios."""
from pathlib import Path

import yaml

from .parametros import RAIZ, carregar_cenarios
from .simulacao import Ensaio, simular


def carregar_ensaios(caminho=None):
    caminho = Path(caminho) if caminho else RAIZ / "config" / "ensaios.yaml"
    with open(caminho, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["ensaios"]
    return {
        nome: Ensaio(nome=nome, descricao=c["descricao"], T=float(c["T"]),
                     v0=c["v0_kmh"] / 3.6, acel=c.get("acel", []), pista=c["pista"])
        for nome, c in cfg.items()
    }


def rodar(ensaios=None, cenarios=None, **opcoes):
    """Retorna {ensaio: {cenario: Resultado}}."""
    ensaios = ensaios or carregar_ensaios()
    cenarios = cenarios or carregar_cenarios()
    return {en: {cn: simular(p, e, **opcoes) for cn, p in cenarios.items()}
            for en, e in ensaios.items()}
