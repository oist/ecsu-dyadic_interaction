"""
TODO: Missing module docstring
"""

import os
from dyadic_interaction import gen_structure
from dyadic_interaction.simulation import Simulation
from dyadic_interaction import transfer_entropy
from dyadic_interaction import utils
from pyevolver.evolution import Evolution
import numpy as np

import argparse
from pytictoc import TicToc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Compare a simulation between single and multi-processing'
    )

    parser.add_argument('--seed', type=int, default=0, help='Random seed')     
    parser.add_argument('--entropy', choices=['shannon', 'transfer', 'sample'], default='shannon', help='Type of entropy measure to use')
    parser.add_argument('--entropy_target_value', choices=['neural_outputs', 'agents_distance'], default='neural_outputs', help='Type of value to be used to calculate entropy')   ##
    parser.add_argument('--collision_type', choices=['none', 'overlapping', 'edge_bounded'], default='overlapping', help='Type of collison')
    parser.add_argument('--dir', type=str, default=None, help='Output directory')
    parser.add_argument('--cores', type=int, default=4, help='Number of cores')        
    parser.add_argument('--num_neurons', type=int, default=2, help='Number of neurons in agent')    
    parser.add_argument('--popsize', type=int, default=96, help='Population size')    
    parser.add_argument('--num_gen', type=int, default=500, help='Number of generations')    
    parser.add_argument('--trial_duration', type=int, default=200, help='Trial duration')    
    parser.add_argument('--perf_obj', default='MAX', help='Fitness normalization mode') # 'MAX', 'MIN', 'ZERO', 'ABS_MAX' or float value

    args = parser.parse_args()

    t = TicToc()
    t.tic()

    genotype_structure = gen_structure.DEFAULT_GEN_STRUCTURE(args.num_neurons)
    genotype_size = gen_structure.get_genotype_size(genotype_structure)

    if args.dir is not None:
        utils.make_dir_if_not_exists(args.dir)

    sim = Simulation(
        entropy_type = args.entropy,
        entropy_target_value = args.entropy_target_value,
        collision_type=args.collision_type,
        genotype_structure = genotype_structure,
        agent_body_radius = 4,
        agents_pair_initial_distance = 20,
        agent_sensors_divergence_angle = np.radians(45),  # angle between sensors and axes of symmetry
        brain_step_size = 0.1,
        trial_duration = args.trial_duration,  # the brain would iterate trial_duration/brain_step_size number of time
        num_cores = args.cores     
    )

    if args.dir is not None:
        sim_config_json = os.path.join(args.dir, 'simulation.json')
        sim.save_to_file(sim_config_json)

    evo = Evolution(
        random_seed=args.seed,
        population_size=args.popsize,
        genotype_size=genotype_size*2, # two agents per genotype
        evaluation_function=sim.evaluate,
        performance_objective=args.perf_obj,
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
        folder_path=args.dir,
        max_generation=args.num_gen,
        termination_function=None,
        checkpoint_interval=np.ceil(args.num_gen/100),
    )
    evo.run()

    if args.entropy == 'transfer':
        # shutdown JVM
        transfer_entropy.shutdown_JVM() 

    print('Ellapsed time: {}'.format(t.tocvalue()))
