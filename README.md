# TCC Matheus - Half-car 4 GDL e altura do CG

Simula o impacto de rebaixar/elevar o veiculo (altura do CG) em **conforto**,
**seguranca** e **durabilidade**, em lombadas, buracos, degrau, senoidal,
pista aleatoria ISO 8608 e frenagem/aceleracao.
Base teorica: `Proj_Pesq_Matheus_Gaioso.pdf` e o modelo de Syed (2018), pasta `../half-car`.

## Setup (Windows / PowerShell)
    python -m venv env
    .\env\Scripts\Activate.ps1
    pip install -r requirements.txt
    pip install -e .
    pytest

No VS Code abra a pasta `TCC_Matheus` (o interpretador `env` ja esta em `.vscode/settings.json`).

## Uso
    python scripts/rodar_ensaios.py            # 10 ensaios x 3 cenarios -> results/
    python scripts/rodar_ensaios.py --ensaios lombada_40kmh frenagem_plana
    python scripts/varredura_frequencia.py     # conforto/aderencia vs frequencia (H3)
    python scripts/varredura_altura.py         # tendencia continua com h

Saidas: `results/tables/*.csv` (metricas completas, variacao % vs original) e
`results/figures/*.png`.

## Estrutura
| Caminho | Conteudo |
|---|---|
| `config/cenarios.yaml` | Veiculo (estimado) e os 3 cenarios: h, fatores k e c, cursos de batente |
| `config/ensaios.yaml` | Definicao dos ensaios (pista, velocidade, aceleracao) |
| `src/tcc_suspensao/modelo.py` | Equacoes do half-car, matrizes A e B (eq. 8 do TCC) |
| `src/tcc_suspensao/simulacao.py` | `solve_ivp` RK45 |
| `src/tcc_suspensao/estrada.py` | Degrau, lombada, buraco, senoidal, ISO 8608; atraso L/v na roda traseira |
| `src/tcc_suspensao/metricas.py` | ISO 2631-1 (Wk, We, RMS, VDV), ΔFz, batentes, rainflow |
| `tests/` | Validacao (linear x nao linear, eq. 9, autovalores, Wk, rainflow) |

## Como a altura do CG (h) entra no modelo
1. Transferencia de carga longitudinal: momento `M_lt = -M·ax·h` sobre a massa suspensa (regime permanente: ΔFz = M·ax·h/L, eq. 9).
2. ΔFz nos pneus resulta da simulacao (depende de h pela arfagem e das rigidezes).
3. Cursos ate os batentes (compressao/extensao) mudam com a altura de rodagem.
Alem disso, k e c equivalentes mudam por cenario (fatores arbitrarios do autor: 1,0 / 0,8 / 1,3).

## Extensoes nao lineares ao modelo do TCC (acordadas)
Batentes de compressao/extensao (mola+amortecedor de contato) e pneu unilateral (nao traciona; Fz >= 0).
Com `batentes=False, pneu_unilateral=False, transferencia_carga=False` o modelo coincide com o linear (testado).

## Metricas
- Conforto: aceleracao RMS ponderada (Wk vertical no banco incluindo arfagem, + rotacao We x 0,63), VDV, fator de crista, classe ISO.
- Seguranca: arfagem maxima, RMS(ΔFz)/Fz0, Fz minima/Fz0 e % de tempo sem contato, transferencia de carga.
- Durabilidade: numero de batidas no batente, penetracao, forca e energia de batente, forca maxima na suspensao, dano de fadiga relativo (rainflow + Miner, expoente 3).

## Suposicoes a revisar (todas [EST] no YAML)
massa nao suspensa (40 kg/eixo), 60% do peso na frente, r_y = 0,35 L (Iy = m_susp·r_y², sem variar com h),
k, c, kp e batentes tipicos de hatch, posicao do banco (0,30 m a frente do CG), cursos 70/50, 90/30, 40/80 mm,
dimensoes de lombada/buraco. O indice de dano so tem sentido em comparacao relativa (Wohler generico).
