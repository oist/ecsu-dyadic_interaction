"""
TODO: Missing module docstring
"""

import os
import sys
from joblib import Parallel, delayed
from dyadic_interaction import gen_structure
from dyadic_interaction.simulation import Simulation
from dyadic_interaction import transfer_entropy
from pyevolver.evolution import Evolution
import numpy as np
from numpy.random import RandomState
from pyevolver import utils
import argparse
from pytictoc import TicToc

def run_experiment(seed, entropy_type, entropy_target_value, folder_path, num_cores, performance_objective,
    num_neurons=2, population_size=96, max_generation=500, trial_duration=200): ##

    random_seed = seed

    genotype_structure = gen_structure.DEFAULT_GEN_STRUCTURE(num_neurons)
    genotype_size = gen_structure.get_genotype_size(genotype_structure)

    utils.make_dir_if_not_exists(folder_path)

    sim = Simulation(
        entropy_type = entropy_type,
        # entropy_target_value = entropy_target_value, ##
        genotype_structure = genotype_structure,
        agent_body_radius = 4,
        agents_pair_initial_distance = 20,
        agent_sensors_divergence_angle = np.radians(45),  # angle between sensors and axes of symmetry
        brain_step_size = 0.1,
        trial_duration = trial_duration,  # the brain would iterate trial_duration/brain_step_size number of time
        num_cores = num_cores     
    )

    sim_config_json = os.path.join(folder_path, 'simulation.json')
    sim.save_to_file(sim_config_json)

    evo = Evolution(
        random_seed=random_seed,
        population_size=population_size,
        genotype_size=genotype_size*2, # two agents per genotype
        evaluation_function=sim.evaluate,
        performance_objective=performance_objective,
        fitness_normalization_mode='NONE', # 'NONE', FPS', 'RANK', 'SIGMA' -> NO NORMALIZATION
        selection_mode='UNIFORM', # 'UNIFORM', 'RWS', 'SUS'
        reproduce_from_elite=True,
        reproduction_mode='GENETIC_ALGORITHM',  # 'HILL_CLIMBING',  'GENETIC_ALGORITHM'
        mutation_variance=0.1, # mutation noice with variance 0.1
        elitist_fraction=0.04, # elite fraction of the top 4% solutions
        mating_fraction=0.96, # the remaining mating fraction
        crossover_probability=0.1,
        crossover_mode='UNIFORM',
        crossover_points= None, #genotype_structure['crossover_points'],
        folder_path=folder_path,
        max_generation=max_generation,
        termination_function=None,
        checkpoint_interval=np.ceil(max_generation/100),
    )
    evo.run()

    if entropy_type == 'transfer':
        # shutdown JVM
        transfer_entropy.shutdown_JVM() 


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Compare a simulation between single and multi-processing'
    )

    parser.add_argument('--seed', type=int, default=0, help='Random seed')     
    parser.add_argument('--entropy', choices=['shannon', 'transfer', 'sample_entropy'], default='shannon', help='Type of entropy measure to use')   ##
    parser.add_argument('--entropy_target_value', choices=['output_neuron', 'agents_distance'], default='output_neuron', help='Type of value to be used to calculate entropy')   ##
    parser.add_argument('--dir', type=str, default='./data/tmp', help='Output directory')
    parser.add_argument('--cores', type=int, default=4, help='Number of cores')        
    parser.add_argument('--num_neurons', type=int, default=2, help='Number of neurons in agent')    
    parser.add_argument('--popsize', type=int, default=96, help='Population size')    
    parser.add_argument('--num_gen', type=int, default=500, help='Number of generations')    
    parser.add_argument('--trial_duration', type=int, default=200, help='Trial duration')    
    parser.add_argument('--perf_obj', default='MAX', help='Fitness normalization mode') # 'MAX', 'MIN', 'ZERO', 'ABS_MAX' or float value

    args = parser.parse_args()

    t = TicToc()
    t.tic()

    run_experiment(args.seed, args.entropy, args.entropy_target_value, 
        args.dir, args.cores, args.perf_obj,
        args.num_neurons, args.popsize, args.num_gen, args.trial_duration) ##

    print('Ellapsed time: {}'.format(t.tocvalue()))
