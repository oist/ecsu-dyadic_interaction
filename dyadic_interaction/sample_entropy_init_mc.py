from dyadic_interaction.simulation import Simulation
from dyadic_interaction import gen_structure
from pyevolver.evolution import Evolution
from numpy.random import RandomState
from pyevolver.json_numpy import NumpyListJsonEncoder
import json
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

num_cores = 40
num_runs = 1000
num_trials = 4
sim_duration = 2000

genotype_structure=gen_structure.DEFAULT_GEN_STRUCTURE(3)
gen_size = gen_structure.get_genotype_size(genotype_structure)    

num_data_points_per_agent_pair = sim_duration * num_trials
num_data_points = num_data_points_per_agent_pair * num_runs # 8 million!

all_distances = np.zeros(num_data_points)
rs = RandomState(0)

sim_array = [
    Simulation(
        entropy_type='shannon',
        genotype_structure=genotype_structure,       
    ) for _ in range(num_cores)
]

random_genotypes = [Evolution.get_random_genotype(rs, gen_size*2) for _ in range(num_runs)]

pbar = tqdm(total=num_runs) 

def run_one_core(r):
    data_record = {}
    sim_array[r%num_cores].compute_performance(random_genotypes[r], rand_seed=None, data_record=data_record)
    concat_distances = np.concatenate(data_record['distance'])
    np.insert(all_distances, num_data_points_per_agent_pair*r, concat_distances)
    pbar.update(1)
    # json.dump(
    #     concat_distances,
    #     open('data/tmp.json', 'w'),
    #     indent=3,
    #     cls=NumpyListJsonEncoder
    # )


def run_all_cores():
   
    Parallel(n_jobs=num_cores)( # prefer="threads" does not work
        delayed(run_one_core)(r) for r in range(num_runs)        
    )
          
    return all_distances.std()
    

if __name__ == "__main__":
    print(run_all_cores())