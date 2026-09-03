"""
Frenagem Automatica de Emergencia (Carro Autonomo) - Sistema Fuzzy (Mamdani)

Entradas:
    distancia  (0-100 m)  - distancia ate o obstaculo
    velocidade (0-100 km/h) - velocidade atual do veiculo
Saida:
    pressao (0-100 %) - pressao aplicada no freio

Este script contem DUAS versoes do sistema:
    1) sistema_centroid - a formulacao "padrao" (defuzzificacao por centroide)
    2) sistema_mom      - mesma base de regras e funcoes de pertinencia, mas
                           com defuzzificacao por "mean of maximum" (mom)

A versao 1) tem um problema conhecido: mesmo em casos extremos (obstaculo
a 0 m ou muito distante), a pressao de saida NUNCA atinge 0% ou 100%. Isso
nao e um bug de implementacao, e sim uma consequencia matematica de usar
funcoes de pertinencia triangulares (nao trapezoidais) combinadas com
defuzzificacao por centroide. A versao 2) resolve isso.
"""

import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def montar_sistema(metodo_defuzz):
    """
    Monta um sistema fuzzy completo (variaveis + regras) do zero.

    IMPORTANTE: cada chamada cria objetos Antecedent/Consequent novos e
    independentes. Isso e necessario porque ControlSystem guarda uma
    REFERENCIA aos objetos, nao uma copia -- se reaproveitassemos o mesmo
    objeto `pressao` para os dois sistemas, mudar seu defuzzify_method
    depois afetaria os dois sistemas ao mesmo tempo (foi exatamente o bug
    que apareceu na primeira versao deste script).
    """
    distancia = ctrl.Antecedent(np.arange(0, 101, 1), 'distancia')
    velocidade = ctrl.Antecedent(np.arange(0, 101, 1), 'velocidade')
    pressao = ctrl.Consequent(np.arange(0, 101, 1), 'pressao')
    pressao.defuzzify_method = metodo_defuzz

    # Funcoes de pertinencia (triangulares, conforme o enunciado)
    distancia['curta'] = fuzz.trimf(distancia.universe, [0, 0, 40])
    distancia['media']  = fuzz.trimf(distancia.universe, [20, 50, 80])
    distancia['longa']  = fuzz.trimf(distancia.universe, [60, 100, 100])

    velocidade['lenta']     = fuzz.trimf(velocidade.universe, [0, 0, 40])
    velocidade['moderada']  = fuzz.trimf(velocidade.universe, [20, 50, 80])
    velocidade['rapida']    = fuzz.trimf(velocidade.universe, [60, 100, 100])

    pressao['suave'] = fuzz.trimf(pressao.universe, [0, 0, 40])
    pressao['media'] = fuzz.trimf(pressao.universe, [20, 50, 80])
    pressao['forte'] = fuzz.trimf(pressao.universe, [60, 100, 100])

    # Regras de inferencia (Mamdani)
    #
    #   D \ V     LENTA      MODERADA    RAPIDA
    #   CURTA     FORTE      FORTE       FORTE
    #   MEDIA     SUAVE      MEDIA       FORTE
    #   LONGA     SUAVE      SUAVE       MEDIA
    #
    # Obstaculo perto -> frear forte independente da velocidade.
    r1 = ctrl.Rule(distancia['curta'], pressao['forte'])
    r2 = ctrl.Rule(distancia['media'] & velocidade['lenta'],     pressao['suave'])
    r3 = ctrl.Rule(distancia['media'] & velocidade['moderada'],  pressao['media'])
    r4 = ctrl.Rule(distancia['media'] & velocidade['rapida'],    pressao['forte'])
    r5 = ctrl.Rule(distancia['longa'] & velocidade['lenta'],     pressao['suave'])
    r6 = ctrl.Rule(distancia['longa'] & velocidade['moderada'],  pressao['suave'])
    r7 = ctrl.Rule(distancia['longa'] & velocidade['rapida'],    pressao['media'])

    sistema = ctrl.ControlSystem([r1, r2, r3, r4, r5, r6, r7])
    return distancia, velocidade, pressao, ctrl.ControlSystemSimulation(sistema)


# ------------------------------------------------------------
# Duas simulacoes independentes: centroide (padrao) e mom (correcao)
# ------------------------------------------------------------
distancia, velocidade, pressao, freio_centroid = montar_sistema('centroid')
_, _, pressao_mom, freio_mom = montar_sistema('mom')


def calcular_pressao(freio, dist_val, vel_val):
    """Recebe distancia (m) e velocidade (km/h) e retorna a pressao no freio (%)."""
    freio.input['distancia'] = dist_val
    freio.input['velocidade'] = vel_val
    freio.compute()
    return freio.output['pressao']


def varrer_espaco(freio, passo=2):
    """Varre todo o espaco de entrada e retorna a matriz de pressao resultante."""
    D = np.arange(0, 101, passo)
    V = np.arange(0, 101, passo)
    resultados = np.zeros((len(D), len(V)))
    for i, d in enumerate(D):
        for j, v in enumerate(V):
            resultados[i, j] = calcular_pressao(freio, float(d), float(v))
    return D, V, resultados


# ==============================================================================
# GRAFICOS
# ==============================================================================
def plotar_funcoes_pertinencia(caminho="funcoes_pertinencia.png"):
    """Reproduz o estilo do diagrama do enunciado: D, V e P lado a lado."""
    fig, eixos = plt.subplots(1, 3, figsize=(15, 4))
    for eixo, variavel, titulo in zip(
        eixos, [distancia, velocidade, pressao],
        ["Distância (D)", "Velocidade (V)", "Pressão no freio (P)"]
    ):
        for termo in variavel.terms:
            eixo.plot(variavel.universe, variavel[termo].mf, label=termo, linewidth=2)
        eixo.set_title(titulo)
        eixo.set_xlabel(variavel.label)
        eixo.set_ylabel("Grau de pertinência")
        eixo.legend()
        eixo.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(caminho, dpi=150)
    print(f"Gráfico salvo em: {caminho}")


def plotar_superficie(D, V, resultados, titulo, caminho):
    """Superfície 3D: pressão no freio em função de (distância, velocidade)."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (necessário para projection='3d')
    Dg, Vg = np.meshgrid(D, V, indexing="ij")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    superficie = ax.plot_surface(Dg, Vg, resultados, cmap="viridis", edgecolor="none")
    ax.set_xlabel("Distância (m)")
    ax.set_ylabel("Velocidade (km/h)")
    ax.set_zlabel("Pressão no freio (%)")
    ax.set_zlim(0, 100)
    ax.set_title(titulo)
    fig.colorbar(superficie, shrink=0.6, label="Pressão (%)")
    plt.tight_layout()
    plt.savefig(caminho, dpi=150)
    print(f"Gráfico salvo em: {caminho}")


def plotar_comparacao_min_max(res_centroid, res_mom, caminho="comparacao_defuzzificacao.png"):
    """Compara, lado a lado, os dois métodos de defuzzificação nos extremos."""
    fig, eixos = plt.subplots(1, 2, figsize=(12, 5))
    for eixo, dados, titulo in zip(
        eixos, [res_centroid, res_mom],
        ["Centroide (padrão) — nunca atinge 0/100%", "MOM (corrigido) — atinge 0/100%"]
    ):
        im = eixo.imshow(dados.T, origin="lower", extent=[0, 100, 0, 100],
                          cmap="RdYlGn_r", vmin=0, vmax=100, aspect="auto")
        eixo.set_title(titulo)
        eixo.set_xlabel("Distância (m)")
        eixo.set_ylabel("Velocidade (km/h)")
        fig.colorbar(im, ax=eixo, label="Pressão (%)")
    plt.tight_layout()
    plt.savefig(caminho, dpi=150)
    print(f"Gráfico salvo em: {caminho}")
    

# ==============================================================================
# EXECUCAO
# ==============================================================================
if __name__ == '__main__':
    casos = [
        (10, 20), (10, 90), (50, 50), (90, 10), (90, 90), (50, 90),
    ]

    # --- 1) CENTROIDE ---
    print("=== Casos de teste (defuzzificação por CENTROIDE - padrão) ===")
    for d, v in casos:
        p = calcular_pressao(freio_centroid, d, v)
        print(f"Distancia={d:>3} m | Velocidade={v:>3} km/h -> Pressao no freio = {p:6.2f} %")

    print("\n=== Varredura completa (min/max da saida) - CENTROIDE ===")
    D, V, res_centroid = varrer_espaco(freio_centroid)
    print(f"Pressao MINIMA = {res_centroid.min():.2f}%")
    print(f"Pressao MAXIMA = {res_centroid.max():.2f}%")
    print(f"Media geral    = {res_centroid.mean():.2f}%")

    # --- 2) MOM ---
    print("\n=== Casos de teste (defuzzificação por MOM - corrigido) ===")
    for d, v in casos:
        p = calcular_pressao(freio_mom, d, v)
        print(f"Distancia={d:>3} m | Velocidade={v:>3} km/h -> Pressao no freio = {p:6.2f} %")

    print("\n=== Varredura completa (min/max da saida) - MOM ===")
    _, _, res_mom = varrer_espaco(freio_mom)
    print(f"Pressao MINIMA = {res_mom.min():.2f}%")
    print(f"Pressao MAXIMA = {res_mom.max():.2f}%")
    print(f"Media geral    = {res_mom.mean():.2f}%")

    # Gráficos
    plotar_funcoes_pertinencia()
    plotar_superficie(D, V, res_centroid,
                      "Superfície de saída — defuzzificação por centroide",
                      "superficie_centroid.png")
    plotar_superficie(D, V, res_mom,
                      "Superfície de saída — defuzzificação por MOM (corrigido)",
                      "superficie_mom.png")
    plotar_comparacao_min_max(res_centroid, res_mom)