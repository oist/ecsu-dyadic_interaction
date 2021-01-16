import os
from pyevolver.evolution import Evolution
from dyadic_interaction.simulation import Simulation
import matplotlib.pyplot as plt
import numpy as np
from dyadic_interaction.utils import euclidean_distance
import json
from tqdm import tqdm

def read_data_from_file(file_path):
    with open(file_path) as f_in:
        data = json.load(f_in)
        perfomances = data['perfomances']
        distances = data['distances']
        return perfomances, distances

def write_data_to_file(perfomances, distances, file_path):
    with open(file_path, 'w') as f_out:
        data = {
            'perfomances': perfomances,
            'distances': distances
        }
        json.dump(data, f_out, indent=3)

def same_pairs():
    dir = "data/2n_shannon-dd_neural_social_coll-edge/seed_001"
    evo_file = os.path.join(dir, "evo_2000.json")
    sim_file = os.path.join(dir, "simulation.json")
    output_file = os.path.join(dir, "perf_dist.json")
    if os.path.exists(output_file):
        perfomances, distances = read_data_from_file(output_file)        
    else:
        evo = Evolution.load_from_file(evo_file, folder_path=dir)   
        sim = Simulation.load_from_file(sim_file)
        assert sim.num_random_pairings == 0
        pop_size = len(evo.population)
        perfomances = []
        distances = [] 
        for i in tqdm(range(pop_size)):
            perf = sim.run_simulation(evo.population, i)
            a,b = np.array_split(evo.population[i], 2)  
            perfomances.append(perf)
            distances.append(euclidean_distance(a, b))
        write_data_to_file(perfomances, distances, output_file)            
    plt.scatter(distances, perfomances)
    plt.show()

def random_pairs():
    dir = "data/2n_rp-3_shannon-dd_neural_social_coll-edge/seed_001"
    evo_file = os.path.join(dir, "evo_2000.json")
    sim_file = os.path.join(dir, "simulation.json")
    output_file = os.path.join(dir, "perf_dist.json")
    evo = Evolution.load_from_file(evo_file, folder_path=dir)   
    sim = Simulation.load_from_file(sim_file)
    sim.num_random_pairings = 0 # we build the pairs dynamically
    pop_size = len(evo.population)
    best_agent = evo.population[0]        
    if os.path.exists(output_file):
        perfomances, distances = read_data_from_file(output_file)        
    else:
        new_population_pairs = []
        perfomances = []
        distances = []
        for j in range(1, pop_size):
            b = evo.population[j]
            pair = np.concatenate([best_agent, b])
            new_population_pairs.append(pair)
            distances.append(euclidean_distance(best_agent, b))
        for i in tqdm(range(pop_size-1)):
            perf = sim.run_simulation(new_population_pairs, i)
            perfomances.append(perf)
        write_data_to_file(perfomances, distances, output_file)
    plt.scatter(distances, perfomances)
    plt.show()
        

if __name__ == "__main__":
    same_pairs()
    # random_pairs()