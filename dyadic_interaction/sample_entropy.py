from dyadic_interaction import gen_structure
from pyevolver.evolution import Evolution
from numpy.random import RandomState
from pyevolver.json_numpy import NumpyListJsonEncoder
import json
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

MAX_DISTANCE = 100.
DEFAULT_SAMPLE_ENTROPY_DISTANCE_STD = 12.936904060417206
DEFAULT_SAMPLE_ENTROPY_ANGLE_STD = 10
DEFAULT_SAMPLE_ENTROPY_NEURAL_STD = 0.5
MULTIPLY_FACTOR = 2 # this is to obtain the total runs being computed (in the hope of obtaining sufficient number of good runs)

num_trials = 4
sim_duration = 2000
seed = 0
num_neurons = 3

def compute_std_from_random_runs(num_cores, num_good_runs, entropy_target_value):

    from dyadic_interaction.simulation import Simulation

    assert entropy_target_value in ['neural', 'distance', 'angle']

    genotype_structure=gen_structure.DEFAULT_GEN_STRUCTURE(num_neurons)
    gen_size = gen_structure.get_genotype_size(genotype_structure)    

    num_data_points_per_agent_pair = sim_duration * num_trials
    num_data_points = num_data_points_per_agent_pair * num_good_runs # 8 million!
    if entropy_target_value == 'distance':
        num_all_runs = num_good_runs * MULTIPLY_FACTOR
    else:
        num_all_runs = num_good_runs

    all_distances = np.zeros(num_data_points)
    rs = RandomState(seed)

    sim_array = [
        Simulation(
            entropy_type='shannon-dd',
            genotype_structure=genotype_structure,       
        ) for _ in range(num_cores)
    ]

    random_genotypes = [Evolution.get_random_genotype(rs, gen_size*2) for _ in range(num_all_runs)]

    def run_one_core(r):
        data_record = {}
        sim_array[r%num_cores].compute_performance(random_genotypes[r], data_record=data_record)        
        if entropy_target_value == 'neural': 
            concat_outputs = np.concatenate([
                data_record['brain_output'][t][a]
                for t in range(4)
                for a in range(2)
            ])
            concat_outputs = np.concatenate([concat_outputs[:,c] for c in range(num_neurons)])
            return concat_outputs
        elif entropy_target_value == 'distance': 
            concat_distances = np.concatenate(data_record['distance'])
            if any(concat_distances > MAX_DISTANCE):
                return None
            return concat_distances
        else:
            assert entropy_target_value == 'angle'
            concat_angles = np.concatenate([
                data_record['angle'][t][a]
                for t in range(4)
                for a in range(2)
            ])
            return concat_angles


    run_distances = Parallel(n_jobs=num_cores)( # prefer="threads" does not work
        delayed(run_one_core)(r) for r in tqdm(range(num_all_runs))        
    )
    good_run_distances = [run for run in run_distances if run is not None]
    print("Number of good runs: {}".format(len(good_run_distances)))
    assert len(good_run_distances) >= num_good_runs
    all_distances = np.concatenate(good_run_distances[:num_good_runs]) # take only the first 1000 good runs

    # json.dump(
    #     all_distances,
    #     open('data/tmp_distances.json', 'w'),
    #     indent=3,
    #     cls=NumpyListJsonEncoder
    # )
          
    std = all_distances.std()
    print(std)
    

if __name__ == "__main__":
    # compute_std_from_random_runs(num_cores = 40, num_good_runs = 1000, entropy_target_value='distance')
    # compute_std_from_random_runs(num_cores = 5, num_good_runs = 10, entropy_target_value='angle')
    compute_std_from_random_runs(num_cores = 5, num_good_runs = 10, entropy_target_value='neural')