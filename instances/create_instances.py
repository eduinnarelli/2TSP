'''
Nesse módulo são geradas 5 instâncias para o problema 2-TSP. Renomeamos as 
instâncias que geramos e testamos (`fixed_instances.pkl`) para evitar que a 
execução desse código as sobreponha.
'''

import math
import pickle
import random

instances = []

# Variar quantidade de vértices
num_vertices_list = [20, 40, 60, 80, 100]

# Gerar uma instância para cada quantidade
for n in num_vertices_list:

  # Gerar n pontos aleatórios no intervalo [0,1]
  points = [(random.uniform(0, 1), random.uniform(0, 1)) for i in range(n)]

  # Dicionário de distâncias Euclidianas entre cada par de pontos
  dist = {
    (i, j): math.sqrt(sum((points[i][k]-points[j][k])**2 for k in range(2)))
            for i in range(n) for j in range(i)
  }

  instances.append({'n': n, 'dist': dist})

# Salvar instâncias em 'instances.pkl' 
with open('instances.pkl', 'wb') as output:
    pickle.dump(instances, output)
