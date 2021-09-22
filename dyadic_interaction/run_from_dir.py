import os
from numpy.random import RandomState
from dyadic_interaction.simulation import Simulation
from dyadic_interaction import utils
from dyadic_interaction.visual import Visualization
from dyadic_interaction.plot_results import plot_results

def run_simulation_from_dir(dir, generation, genotype_idx, **kwargs):    
    ''' 
    utitity function to get data from a simulation
    '''

    random_pos_angle = kwargs.get('random_pos_angle', None)
    entropy_type = kwargs.get('entropy_type', None)
    entropy_target_value = kwargs.get('entropy_target_value', None)
    concatenate = kwargs.get('concatenate', None)
    collision_type = kwargs.get('collision_type', None)
    ghost_index = kwargs.get('ghost_index', None)
    initial_distance = kwargs.get('initial_distance', None)
    isolation = kwargs.get('isolation', None)
    write_data = kwargs.get('write_data', None)


    func_arguments = locals()
    from pyevolver.evolution import Evolution
    evo_files = [f for f in os.listdir(dir) if f.startswith('evo_')]
    assert len(evo_files)>0, "Can't find evo files in dir {}".format(dir)
    file_num_zfill = len(evo_files[0].split('_')[1].split('.')[0])
    if generation is None:
        # assumes last generation
        evo_files = sorted([f for f in os.listdir(dir) if f.startswith('evo')])
        evo_json_filepath = os.path.join(dir, evo_files[-1])
    else:
        generation = str(generation).zfill(file_num_zfill)
        evo_json_filepath = os.path.join(dir, 'evo_{}.json'.format(generation))
    sim_json_filepath = os.path.join(dir, 'simulation.json')    
    sim = Simulation.load_from_file(sim_json_filepath)
    evo = Evolution.load_from_file(evo_json_filepath, folder_path=dir)

    if initial_distance is not None:
        print("Forcing initial distance to: {}".format(initial_distance))
        sim.agents_pair_initial_distance = initial_distance
        sim.set_initial_positions_angles()

    if random_pos_angle:
        print("Randomizing positions and angles")
        random_state = RandomState()
        sim.set_initial_positions_angles(random_state)

    if entropy_type is not None:
        sim.entropy_type = entropy_type
        print("Forcing entropy type: {}".format(sim.entropy_type))

    if entropy_target_value is not None:
        sim.entropy_target_value = entropy_target_value
        print("Forcing entropy target value: {}".format(sim.entropy_target_value))

    if concatenate is not None:
        sim.concatenate = concatenate == 'on'
        print("Forcing concatenation: {}".format(sim.concatenate))

    if collision_type is not None:
        sim.collision_type = collision_type
        sim.init_agents_pair()
        print("Forcing collision_type: {}".format(sim.collision_type))

    if isolation is not None:
        sim.isolation = isolation
        print("Forcing isolation to: {}".format(isolation))
    
    data_record_list = []
    genotype_idx_unsorted = evo.population_sorted_indexes[genotype_idx]
    random_seed = evo.pop_eval_random_seeds[genotype_idx_unsorted] 

    if ghost_index is not None:
        assert ghost_index in [0,1], 'ghost_index must be 0 or 1'        
        # get original results without ghost condition and no random
        func_arguments['ghost_index'] = None
        func_arguments['random_pos_angle'] = False
        func_arguments['initial_distance'] = None
        func_arguments['write_data'] = None
        _, _, original_data_record_list = run_simulation_from_dir(**func_arguments) 
        perf = sim.run_simulation(evo.population_unsorted, genotype_idx_unsorted, 
            random_seed, data_record_list, ghost_index=ghost_index, 
            original_data_record_list=original_data_record_list)
        print("Overall Performance recomputed (non-ghost agent only): {}".format(perf))
    else:                
        perf = sim.run_simulation(evo.population_unsorted, genotype_idx_unsorted, 
            random_seed, data_record_list)
        if genotype_idx==0:
            original_perfomance = evo.best_performances[-1]
            print("Original Performance: {}".format(original_perfomance))
        print("Overall Performance recomputed: {}".format(perf))

    if write_data:        
        for s, data_record in enumerate(data_record_list,1):
            if len(data_record_list)>1:                
                outdir = os.path.join(dir, 'data' , 'sim_{}'.format(s))
            else:
                outdir = os.path.join(dir, 'data')
            utils.make_dir_if_not_exists(outdir)
            for t in range(sim.num_trials):
                for k,v in data_record.items():
                    if v is dict: 
                        # summary
                        if t==0: # only once                            
                            outfile = os.path.join(outdir, '{}.json'.format(k))
                            utils.save_json_numpy_data(v, outfile)
                    elif len(v)!=sim.num_trials:
                        # genotype/phenotype
                        outfile = os.path.join(outdir, '{}.json'.format(k))
                        utils.save_json_numpy_data(v, outfile)
                    elif len(v[0])==2:
                        # data for each agent
                        for a in range(2):
                            outfile = os.path.join(outdir, '{}_{}_{}.json'.format(k,t+1,a+1))
                            utils.save_json_numpy_data(v[t][a], outfile)
                    else:
                        # single data for both agent (e.g., distance)
                        outfile = os.path.join(outdir, '{}_{}.json'.format(k,t+1))
                        utils.save_json_numpy_data(v[t], outfile)                      

    return evo, sim, data_record_list

def get_argparse():
    import argparse

    parser = argparse.ArgumentParser(
        description='Rerun simulation'
    )

    parser.add_argument('--dir', type=str, help='Directory path')
    parser.add_argument('--generation', type=int, help='number of generation to load')
    parser.add_argument('--genotype_idx', type=int, default=0, help='Index of agent in population to load')    
    parser.add_argument('--random_pos_angle', action='store_true', help='Whether to randomize initial pos and angle')
    parser.add_argument('--entropy_type', type=str, choices=['shannon', 'transfer', 'sample'], default=None, help='Whether to change the entropy_type')
    parser.add_argument('--entropy_target_value', type=str, default=None, help='To change the entropy_target_value')    
    parser.add_argument('--concatenate', choices=['on', 'off'], default=None, help='To change the concatenation')
    parser.add_argument('--collision_type', choices=['none', 'overlapping', 'edge'], default=None, help='To change the type of collison')
    parser.add_argument('--initial_distance', type=int, default=None, help='Initial distance (must be >=0 or else it will be set as in simulation default)')    
    parser.add_argument('--ghost_index', type=int, default=None, help='Ghost index (must be 0 or 1 or else ghost condition will not be enabled)')    
    parser.add_argument('--isolation', action='store_true', default=None, help='Whether simulation runs on single agents (as if second agent does not exits) or two agents')
    parser.add_argument('--write_data', action='store_true', help='Whether to output data (same directory as input)')

    # additional args (viz, plot)
    parser.add_argument('--plot', action='store_true', help='Plot')
    parser.add_argument('--sim_num', type=int, default=1, help='Index of agent in population to load')
    parser.add_argument('--viz', action='store_true', help='Visualize trial')    
    parser.add_argument('--trial', type=int, choices=[1,2,3,4], help='Visualize Trial (1-index)')    
    parser.add_argument('--plot_trial_num', type=int, choices=[1,2,3,4], help='Plot Trial index (1-index)')    

    return parser

if __name__ == "__main__":
    parser = get_argparse()
    args = parser.parse_args()
    evo, sim, data_record_list = run_simulation_from_dir(**vars(args))

    sim_index = args.sim_num - 1

    for s in range(len(data_record_list)):
        sim_performance = data_record_list[s]['summary']['performance_sim']
        trial_performances = data_record_list[s]['summary']['performance_trials']
        marker = ' <--' if s == sim_index else ''
        print("Sim #{} performance: {}{}".format(s+1, sim_performance, marker))
        print("  Trials performances: {}".format(trial_performances))

    if args.plot:
        trial_index = args.trial - 1 if args.trial is not None else 'all'

        data_record = data_record_list[sim_index]    
        plot_results(evo, sim, data_record, trial_index)

    if args.viz:
        trial_index = args.trial - 1 if args.trial is not None else 1 # np.argmax(trials_perfs)
        
        vis = Visualization(sim)

        data_record = data_record_list[sim_index]    
        vis.start_simulation_from_data(trial_index, data_record)


