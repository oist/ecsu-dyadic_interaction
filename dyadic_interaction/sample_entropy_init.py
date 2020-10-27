from dyadic_interaction.simulation import Simulation
from dyadic_interaction import gen_structure
from pyevolver.evolution import Evolution
from numpy.random import RandomState
from pyevolver.json_numpy import NumpyListJsonEncoder
import json
import numpy as np
from tqdm import tqdm

def run_random_simulation():

    genotype_structure=gen_structure.DEFAULT_GEN_STRUCTURE(3)
    gen_size = gen_structure.get_genotype_size(genotype_structure)    

    sim = Simulation(
        entropy_type='shannon',
        genotype_structure=genotype_structure,       
    )

    num_runs = 1000
    num_trials = 4
    sim_duration = 2000
    num_data_points_per_agent_pair = sim_duration * num_trials
    num_data_points = num_data_points_per_agent_pair * num_runs # 8 million!
    all_distances = np.zeros(num_data_points)
    rs = RandomState(0)
    
    for r in tqdm(range(num_runs)):
        random_genotype = Evolution.get_random_genotype(rs, gen_size*2) # pairs of agents in a single genotype
        data_record = {}
        sim.compute_performance(random_genotype, data_record=data_record)
        concat_distances = np.concatenate(data_record['distance'])
        np.insert(all_distances, num_data_points_per_agent_pair*r, concat_distances)
        # json.dump(
        #     concat_distances,
        #     open('data/tmp.json', 'w'),
        #     indent=3,
        #     cls=NumpyListJsonEncoder
        # )
    return all_distances.std()
    

if __name__ == "__main__":
    std = run_random_simulation()
    print(std)