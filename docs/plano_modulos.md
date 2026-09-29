# Modulos planejados (a preencher apos alinhamento):
# parametros : dataclass com M, L, h, Iy, a, b, ke, ce, kp, cp por cenario
# modelo     : matrizes A, B do espaco de estados 8x8 (half-car 4 GDL)
# estrada    : perfis de pista (degrau, lombada, buraco, senoidal, ISO 8608), com atraso L/v
# simulacao  : solve_ivp (RK45/Radau) + evento de fim de curso (batente)
# metricas   : RMS ISO 2631-1 (Wk), VDV, pitch, dFz, RMS load, folga/batente, dano
# cenarios   : original / alto / baixo
# graficos   : comparativos com matplotlib
