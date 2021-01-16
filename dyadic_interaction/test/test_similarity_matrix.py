import os
from pyevolver.evolution import Evolution
from sklearn.metrics.pairwise import pairwise_distances
from dyadic_interaction.utils import linmap
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import RandomState
import matplotlib.pyplot as plt

def get_similarity_matrix(population):
    # population = linmap(population, (-1, 1), (0, 1))
    similarity = - pairwise_distances(population)    
    print(similarity)    
    print(similarity.shape)
    print("Min: {} Max: {}".format(np.min(similarity), np.max(similarity)))
    plt.imshow(similarity)
    plt.colorbar()
    plt.show()
    # max_v = np.full(len(population[0]), 1)
    # min_v = np.full(len(population[0]), -1)
    # print(max_v)
    # print(min_v)
    # print("dist = {}".format(- np.linalg.norm(max_v-min_v)))

def get_similarity_split(population):
    # population = linmap(population, (-1, 1), (0, 1))
    similarity = np.zeros((1, len(population)))
    for i,pair in enumerate(population):
        a,b = np.array_split(pair, 2)  
        similarity[0][i] = - np.linalg.norm(a-b)
    print(similarity)
    print(similarity.shape)
    print("Min: {} Max: {}".format(np.min(similarity), np.max(similarity)))    
    plt.imshow(similarity)
    plt.colorbar()
    plt.show()

def test_random_pairings():
    dir = "data/2n_rp-3_shannon-dd_neural_social_coll-edge/seed_001"
    evo_file = os.path.join(dir, "evo_2000.json")
    evo = Evolution.load_from_file(evo_file, folder_path=dir)   
    get_similarity_matrix(evo.population)

def test_split_genotypes():
    dir = "data/2n_shannon-dd_neural_social_coll-edge/seed_001"
    evo_file = os.path.join(dir, "evo_2000.json")
    evo = Evolution.load_from_file(evo_file, folder_path=dir)    
    get_similarity_matrix(evo.population)
    get_similarity_split(evo.population)
    
def test_random_genotypes():
    # gen_size = 20
    # pop_size = 96
    a = Evolution.get_random_genotype(RandomState(None), 20)        
    b = Evolution.get_random_genotype(RandomState(None), 20)        
    a = linmap(a, (-1, 1), (0, 1))
    b = linmap(b, (-1, 1), (0, 1))
    dist = np.linalg.norm(a-b)
    similarity = 1 - dist
    print(a)
    print(b)
    print(a-b)
    print(dist)
    print(similarity)


if __name__ == "__main__":
    # test_random_pairings()
    test_split_genotypes()
    # test_random_genotypes()