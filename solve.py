#%%
import sys
import math
import pickle
import random
from itertools import combinations
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


def k_tsp(K, n, dist):
    '''
    Função que define e resolve o modelo para o K-TSP, dada uma determinada 
    instância. Aqui, K-TSP generaliza o TSP e o 2-TSP para qualquer K, o que
    evita a implementação de modelos diferentes.

    Args:
        K: nº de caixeiros viajantes.
        n: nº de vértices do grafo.
        dist: dicionário de custo das arestas (i,j).
    '''

    model = gp.Model(str(K) + '-tsp')

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

    # Imprimir solução
    print('')
    print('Vertices: %d' % n)
    for t in range(K):
        print(f'Optimal tour {t}: {tours[t]}')
    print('Optimal cost: %g' % model.objVal)
    print('Runtime: %ss' % str(model.Runtime))
    print('')


# Carregar instâncias salvas em 'fixed_instances.pkl'
with open("instances/fixed_instances.pkl", "rb") as fp:
    instances = pickle.load(fp)

dash = '===================='

# Executar 1-TSP e 2-TSP p/ todas as instâncias
for instance in instances:
    n = instance['n']
    dist = instance[ 'dist']

    for k in [1,2]:
        print(f'\n{dash} SOLUÇÃO DO {k}-TSP PARA N = {n} {dash}\n')
        k_tsp(k, n, dist)

#%%