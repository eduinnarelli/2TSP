#%%
import sys
import math
import pickle
import random
from itertools import combinations
from tqdm import tqdm
import gurobipy as gp
from gurobipy import GRB

def shortest_cycle(n, edges):
    '''
    Função auxiliar que constrói o menor ciclo de um conjunto de arestas,
    em termos do número de vértices no ciclo.

    Args:
        n: nº de vértices.
        edges: lista de tuplas de arestas.

    Returns:
        Lista de vértices no menor ciclo, cada um conectado com o anterior
        e próximo da lista.
    '''

    unvisited = list(range(n))

    # Tamanho inicial tem um nó a mais, p/ forçar atualização
    cycle = range(n + 1) 

    while unvisited:  # 'True' enquanto pilha for não-vazia
        thiscycle = []
        neighbors = unvisited

        # Construir ciclo
        while neighbors:
            current = neighbors[0]
            thiscycle.append(current)
            unvisited.remove(current)
            # 'select(current, '*')' retorna todos vizinhos de 'current'
            neighbors = [
                j 
                for i, j in edges.select(current, '*')
                if j in unvisited
            ]

        # Atualizar menor ciclo, se preciso
        if len(cycle) > len(thiscycle):
            cycle = thiscycle

    return cycle


def subtour_elimination(model, where):
    '''
    Callback que, para uma solução ótima do K-TSP relaxado, verifica se essa
    solução viola restrições de eliminação de subciclo e, se sim, adiciona
    essas restrições ao modelo, que será re-otimizado.

    Args:
        model: o modelo associado a callback.
        where: indica da onde no processo de otimização a callback foi chamada.
    '''

    if where == GRB.Callback.MIPSOL:
        # Analisar cada rota t
        for t in range(model._K):

            # Criar lista de arestas na rota t selecionadas na solução
            x_sol = model.cbGetSolution(model._vars)
            edges_in_tour = gp.tuplelist(
                (i, j) 
                for i, j, k in model._vars.keys()
                if x_sol[i, j, k] > 0.5 and k == t
            )

            # Encontrar menor ciclo e verificar se viola restrição, i.e., se
            # não percorre todos os vértices, formando um subciclo
            cycle = shortest_cycle(model._n, edges_in_tour)
            if len(cycle) < n:

                # Adicionar restrições de eliminação de subciclo, para cada par 
                # de vértices do subciclo encontrado
                model.cbLazy(
                    gp.quicksum(
                        model._vars[i, j, t]
                        for i, j in combinations(cycle, 2)
                    ) <= len(cycle)-1
                )

def k_tsp_heuristic(K, n, dist):
    '''
    Função que define e resolve heurísticamente o modelo para o K-TSP, dada uma determinada 
    instância. Aqui, K-TSP generaliza o TSP e o 2-TSP para qualquer K, o que
    evita a implementação de modelos diferentes. A heurística consiste em resolver o 1-TSP
    K vezes, cada uma para um subgrafo da instância sem as arestas da solução anterior.

    Args:
        K: nº de caixeiros viajantes.
        n: nº de vértices do grafo.
        dist: dicionário de custo das arestas (i,j), i >= j.
    
    Returns:
        Dicionário da solução, contendo as K 'tours', 'objVal' e 'runTime'.
    '''

    # Criar cópia do dicionário de distâncias p/ evitar modificar original
    dist_copy = dist.copy()

    # Inicializar dicionário com solução
    sol = {
        'tours': {},
        'objVal': 0,
        'runTime': 0,
    }

    for t in range(K):

        # Resolver 1-TSP
        sol_1_tsp = k_tsp(1, n, dist_copy)

        # Incrementar custo e tempo na solução
        sol['objVal'] += sol_1_tsp['objVal']
        sol['runTime'] += sol_1_tsp['runTime']

        # Acoplar rota encontrada à solução
        sol['tours'][t] = sol_1_tsp['tours'][0]

        # Percorrer rota
        tour = sol['tours'][t]
        for i, j in zip(tour, tour[1:] + tour[:1]):

            # Emular remoção da aresta (i,j) ou (j,i), dependendo de como
            # estiver no dicionário. Atribuimos um custo infinito a essas
            # arestas para indicar ao modelo que elas não devem estar na 
            # próxima solução.
            if (i,j) in dist:
                dist_copy[i,j] = float('inf')
            else:
                dist_copy[j,i] = float('inf')          

    return sol
    
def k_tsp(K, n, dist):
    '''
    Função que define e resolve o modelo para o K-TSP, dada uma determinada 
    instância. Aqui, K-TSP generaliza o TSP e o 2-TSP para qualquer K, o que
    evita a implementação de modelos diferentes.

    Args:
        K: nº de caixeiros viajantes.
        n: nº de vértices do grafo.
        dist: dicionário de custo das arestas (i,j), i >= j.

    Returns:
        Dicionário da solução, contendo as K 'tours', 'objVal' e 'runTime'.
    '''

    # Inicializar ambiente
    env = gp.Env(empty = True)
    env.setParam('OutputFlag', 0)
    env.start()

    # Inicializar modelo
    model = gp.Model(name = str(K) + '-tsp', env = env)

    # Adaptar o dicionário de distâncias de acordo com a quantidade de 
    # caixeiros
    distK = {
        (i, j, k):  dist[i, j] 
                    for i in range(n) for j in range(i) for k in range(K)
    }

    # Criar variáveis
    vars = model.addVars(distK.keys(), obj=distK, vtype=GRB.BINARY, name='x')
    for i, j, k in vars.keys():
        vars[j, i, k] = vars[i, j, k]  # grafo não-orientado

    # Restrições de grau 2, p/ cada rota k
    model.addConstrs(
        (vars.sum(i, '*', k) == 2 for i in range(n) for k in range(K)), 
        name='deg-2'
    )

    # Restrições de disjunção entre arestas de diferentes rotas; se K = 1, tais 
    # restrições são redundantes e eliminadas pelo solver na etapa de 'presolve' 
    # da otimização
    model.addConstrs(
        (vars.sum(i, j, '*') <= 1 for i in range(n) for j in range(i)), 
        name='disj'
    )

    # Salvar alguns atributos no modelo para acessá-los facilmente na callback
    model._n = n
    model._K = K
    model._vars = vars

    # Otimizar modelo, indicando callback a ser chamada após a solução ótima do
    # modelo relaxado ser encontrada
    model.Params.lazyConstraints = 1
    model.setParam(GRB.param.TimeLimit, 1800)
    model.optimize(subtour_elimination)

    # Recuperar solução
    x_sol = model.getAttr('x', vars)
    edges_in_sol = gp.tuplelist(
        (i, j, k) for i, j, k in x_sol.keys()
        if x_sol[i, j, k] > 0.5
    )

    # Garantir que cada rota tenha tamanho n
    tours = {}
    for t in range(K):
        edges_in_tour = gp.tuplelist(
            (i,j) for i, j, k in edges_in_sol
            if k == t
        )
        tours[t] = shortest_cycle(n, edges_in_tour)
        assert len(tours[t]) == n

    # Retornar dicionário com solução
    return {
        'tours': tours,
        'objVal': model.objVal,
        'runTime': model.Runtime,
    }

def print_solution(K, n, sol): 
    '''
    Função que imprime solução no stdout.

    Args:
        K: nº de caixeiros viajantes.
        n: nº de vértices do grafo.
        sol: dicionário da solução, contendo 'tours', 'objVal' e 'runTime'. 
    '''

    print('')
    print('Vertices: %d' % n)
    for t in range(K):
        tour = sol['tours'][t]
        print(f'Optimal tour {t}: {tour}')
    print('Optimal cost: %g' % sol['objVal'])
    print('Runtime: %ss' % str(sol['runTime']))
    print('')


# Carregar instâncias salvas em 'fixed_instances.pkl'
with open("instances/fixed_instances.pkl", "rb") as fp:
    instances = pickle.load(fp)

dash = '===================='

# Salvar output em 'output.txt'
sys.stdout = open('output.txt', 'w')

for instance in tqdm(instances):
    n = instance['n']
    dist = instance['dist']

    # Resolver 1-TSP e 2-TSP de forma ótima
    for K in [1,2]:
        print(f'\n{dash} SOLUÇÃO DO {K}-TSP PARA N = {n} {dash}\n')
        sol = k_tsp(K, n, dist)
        print_solution(K, n, sol)

    # Resolver 2-TSP de forma heurística
    print(f'\n{dash} SOLUÇÃO HEURÍSTICA DO 2-TSP PARA N = {n} {dash}\n')
    sol = k_tsp_heuristic(2, n, dist)
    print_solution(K, n, sol)

sys.stdout.close()

#%%