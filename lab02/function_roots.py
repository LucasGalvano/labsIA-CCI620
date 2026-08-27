"""
Raízes da função f(x, y, z) = x² + y² + z²
Resolvido com Algoritmo Genético (GA) e Enxame de Partículas (PSO)

Diferente do problema da mochila, aqui não existe uma lista de "itens": cada
indivíduo/partícula é um ponto (x, y, z) em R³, com x, y, z no intervalo
[-10, 10]. Como f(x,y,z) = x²+y²+z² só é zero quando x=y=z=0, buscamos
minimizar f, e a raiz encontrada é justamente o ponto (x, y, z) que faz
f(x,y,z) mais próximo de 0.

Esta versão também registra o histórico de convergência (melhor valor de
f encontrado a cada geração/iteração) e gera um gráfico comparando GA e
PSO, no mesmo espírito do slide "PSO - Convergência do enxame" da aula.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from pyeasyga import pyeasyga
import pyswarms as ps

LIMITE_INF, LIMITE_SUP = -10, 10


def f(x, y, z):
    return x**2 + y**2 + z**2


# ==============================================================================
# ALGORITMO GENÉTICO (GA)
# ==============================================================================
def resolver_com_ga():
    # seed_data não representa "itens" aqui — é só um placeholder de tamanho 3
    # (um para cada variável x, y, z) exigido pela biblioteca.
    seed_data = [None, None, None]

    ga = pyeasyga.GeneticAlgorithm(seed_data,
                                    population_size=50,
                                    generations=100,
                                    crossover_probability=0.9,
                                    mutation_probability=0.3,
                                    elitism=True,
                                    maximise_fitness=False)  # queremos MINIMIZAR f

    # Adaptação 1: indivíduo é um vetor de 3 números REAIS (x, y, z), e não
    # mais um vetor binário — precisamos criá-lo do zero.
    def my_create_individual(seed_data):
        return [random.uniform(LIMITE_INF, LIMITE_SUP) for _ in seed_data]
    ga.create_individual = my_create_individual

    # Adaptação 2: a fitness agora é o próprio valor da função objetivo.
    # Com maximise_fitness=False, o pyeasyga entende "menor é melhor".
    def aptidao(individual, seed_data):
        x, y, z = individual
        return f(x, y, z)
    ga.fitness_function = aptidao

    # Adaptação 3: crossover aritmético (blend) — faz sentido para genes
    # contínuos, diferente do corte-e-troca usado em vetores binários.
    def crossover(parent_1, parent_2):
        alpha = random.random()
        child_1 = [alpha * g1 + (1 - alpha) * g2 for g1, g2 in zip(parent_1, parent_2)]
        child_2 = [alpha * g2 + (1 - alpha) * g1 for g1, g2 in zip(parent_1, parent_2)]
        return child_1, child_2
    ga.crossover_function = crossover

    # Adaptação 4: mutação por perturbação gaussiana em torno do valor atual
    # (em vez de sortear um valor novo do zero), o que ajuda o GA a refinar
    # soluções já próximas do ótimo.
    def my_mutation(individual):
        indice = random.randrange(len(individual))
        novo_valor = individual[indice] + random.gauss(0, 1)
        individual[indice] = max(LIMITE_INF, min(LIMITE_SUP, novo_valor))
    ga.mutate_function = my_mutation

    # Seleção por torneio, mas agora "vence" quem tem MENOR fitness.
    def my_selection(population):
        competidores = random.sample(population, 3)
        competidores.sort(key=lambda ind: ind.fitness)
        return competidores[0]
    ga.selection_function = my_selection

    # Em vez de chamar ga.run() diretamente, replicamos o laço interno da
    # biblioteca (create_first_generation + create_next_generation em loop)
    # para conseguir capturar o melhor fitness a cada geração. O pyeasyga
    # não expõe um callback pronto para isso.
    ga.create_first_generation()
    historico = [ga.best_individual()[0]]
    for _ in range(1, ga.generations):
        ga.create_next_generation()
        historico.append(ga.best_individual()[0])

    melhor_fitness, melhor_individuo = ga.best_individual()
    return melhor_individuo, melhor_fitness, historico


# ==============================================================================
# ENXAME DE PARTÍCULAS (PSO)
# ==============================================================================
def resolver_com_pso():
    # Para este problema o PSO é usado no seu formato "nativo": otimização
    # contínua em R³, sem precisar de nenhuma adaptação discreta como no
    # knapsack (não há arredondamento nem penalidade de capacidade aqui).
    def aptidao(enxame):
        return np.array([f(x, y, z) for x, y, z in enxame])

    options = {'c1': 1.5, 'c2': 1.5, 'w': 0.7}
    bounds = (np.array([LIMITE_INF] * 3), np.array([LIMITE_SUP] * 3))

    pso = ps.single.GlobalBestPSO(n_particles=30,
                                   dimensions=3,
                                   options=options,
                                   bounds=bounds)

    melhor_custo, melhor_posicao = pso.optimize(aptidao, iters=100, verbose=False)

    # O pyswarms já guarda, internamente, o melhor custo global (gbest) a
    # cada iteração em pso.cost_history — não precisa reimplementar nada.
    historico = pso.cost_history
    return melhor_posicao.tolist(), melhor_custo, historico


# ==============================================================================
# GRÁFICO DE CONVERGÊNCIA
# ==============================================================================
def plotar_convergencia(historico_ga, historico_pso, caminho="convergencia_raizes.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(historico_ga, label="GA (melhor f encontrado)")
    plt.plot(historico_pso, label="PSO (melhor f encontrado)")
    plt.yscale("log")  # escala log: os valores caem várias ordens de grandeza
    plt.xlabel("Geração / Iteração")
    plt.ylabel("f(x, y, z) — melhor valor encontrado (escala log)")
    plt.title("Convergência: GA vs PSO — Raízes de f(x,y,z) = x²+y²+z²")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(caminho, dpi=150)
    print(f"\nGráfico salvo em: {caminho}")


# ==============================================================================
# EXECUÇÃO E COMPARAÇÃO
# ==============================================================================
if __name__ == "__main__":
    print("=== Algoritmo Genético (GA) ===")
    ponto_ga, valor_ga, hist_ga = resolver_com_ga()
    print(f"(x, y, z) = {ponto_ga}")
    print(f"f(x, y, z) = {valor_ga}\n")

    print("=== Enxame de Partículas (PSO) ===")
    ponto_pso, valor_pso, hist_pso = resolver_com_pso()
    print(f"(x, y, z) = {ponto_pso}")
    print(f"f(x, y, z) = {valor_pso}")

    plotar_convergencia(hist_ga, hist_pso)