"""
TODO: Missing module docstring
"""

import os
import sys
from joblib import Parallel, delayed
from dyadic_interaction import gen_structure
from dyadic_interaction.simulation import Simulation
from pyevolver.evolution import Evolution
import numpy as np
from numpy.random import RandomState
from pyevolver import utils
import argparse
from pytictoc import TicToc

def run_experiment(seed, folder_path, num_cores):

    random_seed = seed

    population_size = 96
    max_generation = 500
    trial_duration = 200
    genotype_structure = gen_structure.DEFAULT_GEN_STRUCTURE
    genotype_size = gen_structure.get_genotype_size(genotype_structure)
    

    utils.make_dir_if_not_exists(folder_path)

    sim = Simulation(
        genotype_structure=genotype_structure,
        agent_body_radius=4,
        agents_pair_initial_distance=20,
        agent_sensors_divergence_angle=np.radians(45),  # angle between sensors and axes of symmetry
        brain_step_size=0.1,
        trial_duration=trial_duration,  # the brain would iterate trial_duration/brain_step_size number of time
        num_cores=num_cores
    )

    sim_config_json = os.path.join(folder_path, 'simulation.json')
    sim.save_to_file(sim_config_json)

    evo = Evolution(
        random_seed=random_seed,
        population_size=population_size,
        genotype_size=genotype_size*2, # two agents per genotype
        evaluation_function=sim.evaluate,
        fitness_normalization_mode='NONE', # 'NONE', FPS', 'RANK', 'SIGMA' -> NO NORMALIZATION
        selection_mode='UNIFORM', # 'UNIFORM', 'RWS', 'SUS'
        reproduce_from_elite=True,
        reproduction_mode='GENETIC_ALGORITHM',  # 'HILL_CLIMBING',  'GENETIC_ALGORITHM'
        mutation_variance=0.1, # mutation noice with variance 0.1
        elitist_fraction=0.04, # elite fraction of the top 4% solutions
        mating_fraction=0.96, # the 
        crossover_probability=0.1,
        crossover_mode='UNIFORM',
        crossover_points= None, #genotype_structure['crossover_points'],
        folder_path=folder_path,
        max_generation=max_generation,
        termination_function=None,
        checkpoint_interval=np.ceil(max_generation/100),
    )
    evo.run()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Compare a simulation between single and multi-processing'
    )

    parser.add_argument('--dir', type=str, default='./data/tmp', help='Output directory')
    parser.add_argument('--cores', type=int, default=4, help='Number of cores')    
    parser.add_argument('--seed', type=int, default=0, help='Random seed')    

    args = parser.parse_args()

    t = TicToc()
    t.tic()

    run_experiment(args.seed, args.dir, args.cores)

    print('Ellapsed time: {}'.format(t.tocvalue()))
