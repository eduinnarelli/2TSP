import sys
import math
import pickle
import random
from itertools import combinations
import gurobipy as gp
from gurobipy import GRB

def shortest_cycle(n, edges):
    '''
    Função auxiliar para que constrói o menor ciclo de um conjunto de arestas,
    em termos do número de vértices no ciclo.

    Args:
        n: nº de vértices.
        edges: lista de tuplas de arestas.
    '''

    unvisited = list(range(n))

    # Tamanho inicial tem um nó a mais, p/ forçar atualização
    cycle = range(n+1) 

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
                j for i, j in edges.select(current, '*')
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
        # Analisar cada rota
        for k in range(model._K):

            # Criar lista de arestas selecionadas na solução
            vals = model.cbGetSolution(model._vars)
            selected = gp.tuplelist(
                (i, j) for i, j, k in model._vars.keys()
                if vals[i, j, k] > 0.5
            )

            # Encontrar menor ciclo e verificar se viola restrição, i.e., se
            # não percorre todos os vértices, formando um subciclo
            cycle = shortest_cycle(model._n, selected)
            if len(cycle) < n:

                # Adicionar restrições de eliminação de subciclo, para cada par 
                # de vértices do subciclo encontrado
                model.cbLazy(
                    gp.quicksum(
                        model._vars[i, j, k]
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

    m = gp.Model(str(K) + '-tsp')

    # Adapta o dicionário de distâncias de acordo com a quantidade de caixeiros
    distK = {}
    for i in range(n):
        for j in range(i):
            for k in range(K):
                distK[i, j, k] = dist[i, j]

    # Criar variáveis
    vars = m.addVars(distK.keys(), obj=distK, vtype=GRB.BINARY, name='x')
    for i, j, k in vars.keys():
        vars[j, i, k] = vars[i, j, k]  # grafo não-orientado

    # Restrições de grau 2, p/ cada rota k
    m.addConstrs(
        (vars.sum(i, '*', k) == 2 for i in range(n) for k in range(K)), 
        name='deg-2'
    )

    # Restrições de disjunção entre arestas de diferentes rotas
    m.addConstrs(
        (vars.sum(i, j, '*') <= 1 for i in range(n) for j in range(i)), 
        name='disj'
    )

    # Salvar alguns atributos no modelo para acessá-los facilmente na callback
    m._n = n
    m._K = K
    m._vars = vars

    # Otimizar modelo, indicando callback a ser chamada após a solução ótima do
    # modelo relaxado ser encontrada
    m.Params.lazyConstraints = 1
    m.setParam(GRB.param.TimeLimit, 1800)
    m.optimize(subtour_elimination)

    # Recuperar solução
    vals = m.getAttr('x', vars)
    selected = gp.tuplelist(
        (i, j) for i, j, k in vals.keys()
        if vals[i, j, k] > 0.5
    )
    # Garantir que rota tenha tamanho n
    tour = shortest_cycle(n, selected)
    assert len(tour) == n

    # Imprimir solução
    print('')
    print('Vertices: %d' % n)
    print('Optimal tour: %s' % str(tour))
    print('Optimal cost: %g' % m.objVal)
    print('Runtime: %ss' % str(m.Runtime))
    print('')

# Carregar instâncias em 'fixed_instances.pkl'
with open("instances/fixed_instances.pkl", "rb") as fp:
    instances = pickle.load(fp)

# Executa 1-TSP e 2-TSP p/ todas as instâncias
for instance in instances:
    n = instance['n']
    dist = instance[ 'dist']

    k_tsp(1, n, dist)
    k_tsp(2, n, dist)
