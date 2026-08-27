"""
Problema da Mochila COM Repetições (Unbounded Knapsack)
Resolvido com Algoritmo Genético (GA) e Enxame de Partículas (PSO)

Cada item pode ser escolhido um número ilimitado de vezes (respeitando o peso
máximo). Diferente do exemplo "sem repetições" dado em aula, o indivíduo/
partícula aqui NÃO é mais um vetor binário (0/1) e sim um vetor de INTEIROS,
onde cada posição indica QUANTAS cópias daquele item foram escolhidas.

Esta versão também registra o histórico de convergência (melhor valor
encontrado a cada geração/iteração) e gera um gráfico comparando GA e PSO.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from pyeasyga import pyeasyga
import pyswarms as ps

# ==============================================================================
# DADOS DO PROBLEMA
# ==============================================================================
data = [{'cor': 'verde',   'valor': 4,  'peso': 12},
        {'cor': 'cinza',   'valor': 2,  'peso': 1},
        {'cor': 'amarelo', 'valor': 10, 'peso': 4},
        {'cor': 'laranja', 'valor': 1,  'peso': 1},
        {'cor': 'azul',    'valor': 2,  'peso': 2}]

CAPACIDADE = 15
# Quantas cópias de cada item cabem sozinhas na mochila (limite superior de busca)
MAX_COPIAS = [CAPACIDADE // caixa['peso'] for caixa in data]


def avalia_solucao(quantidades, data):
    """Calcula valor e peso totais para um vetor de quantidades."""
    dinheiro = sum(q * c['valor'] for q, c in zip(quantidades, data))
    peso = sum(q * c['peso'] for q, c in zip(quantidades, data))
    return dinheiro, peso


# ==============================================================================
# ALGORITMO GENÉTICO (GA)
# ==============================================================================
def resolver_com_ga():
    ga = pyeasyga.GeneticAlgorithm(data,
                                    population_size=30,
                                    generations=80,
                                    crossover_probability=0.9,
                                    mutation_probability=0.3,
                                    elitism=True,
                                    maximise_fitness=True)

    # Adaptação 1: indivíduo agora é um vetor de INTEIROS (quantidade por item),
    # não mais um vetor binário 0/1.
    def my_create_individual(data):
        return [random.randint(0, MAX_COPIAS[i]) for i in range(len(data))]
    ga.create_individual = my_create_individual

    # Adaptação 2: a fitness soma valor*quantidade e peso*quantidade.
    # Usamos penalidade suave (em vez de zerar tudo) quando estoura a
    # capacidade, pois com repetições a maioria dos indivíduos aleatórios
    # tende a ser inviável — um corte abrupto para 0 deixaria o GA "cego",
    # sem gradiente para melhorar.
    def aptidao(individual, data):
        dinheiro, peso = avalia_solucao(individual, data)
        if peso > CAPACIDADE:
            dinheiro -= (peso - CAPACIDADE) * 5
        return dinheiro
    ga.fitness_function = aptidao

    # Crossover de um ponto (mesma lógica do exemplo binário, funciona igual
    # para vetores de inteiros).
    def crossover(parent_1, parent_2):
        index = random.randrange(1, len(parent_1))
        child_1 = parent_1[:index] + parent_2[index:]
        child_2 = parent_2[:index] + parent_1[index:]
        return child_1, child_2
    ga.crossover_function = crossover

    # Adaptação 3: mutação sorteia uma NOVA quantidade para um gene, em vez de
    # apenas inverter um bit 0/1.
    def my_mutation(individual):
        indice = random.randrange(len(individual))
        individual[indice] = random.randint(0, MAX_COPIAS[indice])
    ga.mutate_function = my_mutation

    # Seleção por torneio: sorteia alguns indivíduos e retorna o mais apto.
    def my_selection(population):
        competidores = random.sample(population, 3)
        competidores.sort(key=lambda ind: ind.fitness, reverse=True)
        return competidores[0]
    ga.selection_function = my_selection

    # Reimplementamos o laço interno do run() (create_first_generation +
    # create_next_generation) para conseguir capturar o melhor fitness a
    # cada geração — o pyeasyga não expõe callback pronto para isso.
    ga.create_first_generation()
    historico = [ga.best_individual()[0]]
    for _ in range(1, ga.generations):
        ga.create_next_generation()
        historico.append(ga.best_individual()[0])

    _, melhor_individuo = ga.best_individual()
    dinheiro, peso = avalia_solucao(melhor_individuo, data)
    return melhor_individuo, dinheiro, peso, historico


# ==============================================================================
# ENXAME DE PARTÍCULAS (PSO)
# ==============================================================================
def resolver_com_pso():
    # Adaptação principal: o exemplo "sem repetições" usava BinaryPSO (cada
    # dimensão só podia ser 0 ou 1). Com repetições, cada dimensão precisa
    # representar "quantas cópias", então usamos PSO CONTÍNUO
    # (GlobalBestPSO) com limites por item, arredondando o valor de cada
    # partícula para o inteiro mais próximo dentro da função de aptidão.
    def aptidao(enxame, data):
        resultados = []
        for particula in enxame:
            quantidades = np.clip(np.round(particula).astype(int), 0, MAX_COPIAS)
            dinheiro, peso = avalia_solucao(quantidades, data)
            if peso > CAPACIDADE:
                dinheiro -= (peso - CAPACIDADE) * 5  # mesma penalidade do GA
            resultados.append(dinheiro)
        return -np.array(resultados)  # pyswarms minimiza -> negativo pra maximizar

    options = {'c1': 1.5, 'c2': 1.5, 'w': 0.7}
    bounds = (np.zeros(len(data)), np.array(MAX_COPIAS, dtype=float))

    pso = ps.single.GlobalBestPSO(n_particles=30,
                                   dimensions=len(data),
                                   options=options,
                                   bounds=bounds)

    _, melhor_particula = pso.optimize(aptidao, iters=100, data=data, verbose=False)
    melhor_individuo = np.clip(np.round(melhor_particula).astype(int), 0, MAX_COPIAS).tolist()
    dinheiro, peso = avalia_solucao(melhor_individuo, data)

    # pso.cost_history guarda o custo (negativo do valor) por iteração;
    # negamos de volta para exibir o "valor" (positivo) crescendo com o tempo.
    historico = [-c for c in pso.cost_history]
    return melhor_individuo, dinheiro, peso, historico


# ==============================================================================
# GRÁFICO DE CONVERGÊNCIA
# ==============================================================================
def plotar_convergencia(historico_ga, historico_pso, caminho="convergencia_mochila.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(historico_ga, label="GA (melhor valor encontrado)")
    plt.plot(historico_pso, label="PSO (melhor valor encontrado)")
    plt.xlabel("Geração / Iteração")
    plt.ylabel("Valor total da mochila")
    plt.title("Convergência: GA vs PSO — Mochila com Repetições")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(caminho, dpi=150)
    print(f"\nGráfico salvo em: {caminho}")


# ==============================================================================
# EXECUÇÃO E COMPARAÇÃO
# ==============================================================================
if __name__ == "__main__":
    print("=== Algoritmo Genético (GA) ===")
    individuo_ga, valor_ga, peso_ga, hist_ga = resolver_com_ga()
    print(f"Quantidades: {individuo_ga}")
    print(f"Valor total: {valor_ga} | Peso total: {peso_ga} kg (limite {CAPACIDADE} kg)\n")

    print("=== Enxame de Partículas (PSO) ===")
    individuo_pso, valor_pso, peso_pso, hist_pso = resolver_com_pso()
    print(f"Quantidades: {individuo_pso}")
    print(f"Valor total: {valor_pso} | Peso total: {peso_pso} kg (limite {CAPACIDADE} kg)")

    plotar_convergencia(hist_ga, hist_pso)