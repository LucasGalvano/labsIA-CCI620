import heapq
import osmnx as ox
import networkx as nx
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import folium
import random

# ==============================================================================
# DEFINIÇÃO DAS FUNÇÕES
# =============================================================================#

# 1. GEOCODIFICAÇÃO DE ENDEREÇOS
def obter_coordenadas(endereco: str):
    geolocator = Nominatim(user_agent="CCI620-Lab_A-Estrela")
    location = geolocator.geocode(endereco)
    if location is None:
        raise ValueError(f"Não foi possível encontrar as coordenadas para: {endereco}")
    return (location.latitude, location.longitude)


# 2. HEURÍSTICA ADMISSÍVEL (Distância Geodésica)
def heuristica_geodesica(u, v, graph):
    """
    Calcula a distância em linha reta (em metros) entre dois nós do grafo.
    Garante admissibilidade: nunca superestima a distância real por ruas.
    """
    p1 = (graph.nodes[u]['y'], graph.nodes[u]['x'])  # (lat, lon)
    p2 = (graph.nodes[v]['y'], graph.nodes[v]['x'])  # (lat, lon)

    valor_retornado = 0

    return geodesic(p1, p2).meters


# 2.1 HEURISTÍCA RETORNANDO ZERO
def heuristica_zero(u, v, graph):
    return 0


# 2.2 HEURÍSTICA ALEATÓRIA
def heuristica_aleatoria(u, v, graph):
    valor_gerado = random.uniform(0, 5000)
    print(valor_gerado)
    return valor_gerado


# 3. ALGORITMO A* IMPLEMENTADO MANUALMENTE
def a_star_search(graph, start_node, goal_node, heuristica= heuristica_zero, weight='length'):
    """
    Executa o algoritmo A* para encontrar o caminho de menor custo.
    Retorna o caminho (lista de nós) e estatísticas de busca.
    """
    frontier = []
    # Elemento na fila: (f_score, nó_atual)
    heapq.heappush(frontier, (0, start_node))

    came_from = {}
    g_score = {node: float('inf') for node in graph.nodes}
    g_score[start_node] = 0

    nodes_expanded = 0

    while frontier:
        current_f, current = heapq.heappop(frontier)
        nodes_expanded += 1

        if current == goal_node:
            # Reconstrução do caminho
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start_node)
            path.reverse()
            return path, nodes_expanded, g_score[goal_node]

        for neighbor in graph.neighbors(current):
            # No OSMnx (MultiDiGraph), pegamos a aresta de menor comprimento entre nós paralelos
            edge_data = min(graph.get_edge_data(current, neighbor).values(), key=lambda x: x.get(weight, 1))
            edge_weight = edge_data.get(weight, 1)

            tentative_g = g_score[current] + edge_weight

            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                h = heuristica(neighbor, goal_node, graph) # <- Mudei para o parametro heuristica
                f = tentative_g + h
                heapq.heappush(frontier, (f, neighbor))

    return None, nodes_expanded, float('inf')

# ==============================================================================
# EXECUÇÃO DO EXPERIMENTO
# ==============================================================================

if __name__ == "__main__":
    # Definição dos endereços de teste
    endereco_origem = f"Centro Universitario FEI, São Bernardo do Campo, SP, Brasil"
    endereco_destino = f"Praça Samuel Sabatini, São bernardo do Campo, SP, Brasil"

    print("0. Geocodificando endereços...")
    coord_origem = obter_coordenadas(endereco_origem)
    coord_destino = obter_coordenadas(endereco_destino)
    print(f"  • Origem:  {coord_origem}")
    print(f"  • Destino: {coord_destino}")


    print("Baixando grafo delimitado entre as duas coordenadas...")
    # Determina os extremos geográficos
    min_lat = min(coord_origem[0], coord_destino[0])
    max_lat = max(coord_origem[0], coord_destino[0])
    min_lon = min(coord_origem[1], coord_destino[1])
    max_lon = max(coord_origem[1], coord_destino[1])

    # Margem de segurança em graus (~0.02° ≈ 2,2 km ao redor da rota)
    margem = 0.02

    # No OSMnx v2.0+, a tupla de bbox é (west, south, east, north)
    Mybbox = (
      min_lon - margem,   # West
      min_lat - margem,  # South
      max_lon + margem,  # East
      max_lat + margem  # North
    )
    print(f"Area: {Mybbox}")

    # 'drive' para malha de carros, 'walk' para pedestres
    G = ox.graph_from_bbox(bbox=Mybbox, network_type='drive')

    # Atribui velocidades e tempos de trânsito se não existirem
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    print("[3/5] Mapeando nós mais próximos...")
    origem_node = ox.distance.nearest_nodes(G, X=coord_origem[1], Y=coord_origem[0])
    destino_node = ox.distance.nearest_nodes(G, X=coord_destino[1], Y=coord_destino[0])

    # =============
    # Mude o heuristica=... para o nome de alguma das funções de heuristica para ver os resultados diferentes
    # =============
    print("\n[4/5] Executando A*...")
    caminho, visitados, distancia_total = a_star_search(G, origem_node, destino_node, heuristica= heuristica_aleatoria, weight='length')

    if caminho:
        print(f"  ✓ Rota encontrada com sucesso!")
        print(f"  • Distância total: {distancia_total / 1000:.2f} km")
        print(f"  • Vértices no caminho: {len(caminho)}")
        print(f"  • Nós expandidos durante a busca: {visitados}")

        # Cálculo do tempo estimado
        tempo_total_seg = sum(
            min(G.get_edge_data(u, v).values(), key=lambda x: x.get('length', 1)).get('travel_time', 0)
            for u, v in zip(caminho[:-1], caminho[1:])
        )
        print(f"  • Tempo estimado de viagem: {tempo_total_seg / 60:.1f} minutos")

        print("\n[Gerando visualização interativa no mapa...")

# 1. Extrai as coordenadas (latitude, longitude) de cada nó do caminho
pontos_rota = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in caminho]

# 2. Cria o mapa base centralizado no início da rota
mapa = folium.Map(location=coord_origem, zoom_start=14, tiles="cartodbpositron")
folium.TileLayer(
    tiles="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
    attr="Google Maps",
    name="Google Maps",
    overlay=False,
    control=True
).add_to(mapa)

# 3. Desenha a linha da rota percorrida pelo A*
folium.PolyLine(
    locations=pontos_rota,
    color="#2b8cbe",
    weight=5,
    opacity=0.8,
    popup=f"Distância: {distancia_total / 1000:.2f} km"
).add_to(mapa)

# 4. Adiciona marcadores customizados para Origem e Destino
folium.Marker(
    location=coord_origem,
    popup="<b>Origem</b>",
    icon=folium.Icon(color="green", icon="play", prefix="fa")
).add_to(mapa)

folium.Marker(
    location=coord_destino,
    popup="<b>Destino</b>",
    icon=folium.Icon(color="red", icon="flag", prefix="fa")
).add_to(mapa)

# 5. Salva o arquivo HTML
mapa.save("rota_a_star.html")
print("  ✓ Mapa salvo como 'rota_a_star.html'. Abra em qualquer navegador.")