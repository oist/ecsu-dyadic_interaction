"""
TODO: Missing module docstring
"""

from dyadic_interaction.simulation import run_simulation_from_dir
from dyadic_interaction.visual import Visualization


if __name__ == "__main__":

    from dyadic_interaction.simulation import get_argparse, run_simulation_from_dir
    
    parser = get_argparse()
    parser.add_argument('--sim_num', type=int, default=1, help='Index of agent in population to load')
    parser.add_argument('--trial_num', type=int, default=1, choices=[1,2,3,4], help='Trial index')    
    args = parser.parse_args()

    args_dict = vars(args)
    trial_index = args_dict['trial_num'] - 1
    sim_index = args_dict['sim_num'] - 1
    del args_dict['trial_num']
    del args_dict['sim_num']

    evo, sim, data_record_list = run_simulation_from_dir(**args_dict)

    vis = Visualization(sim)

    for s in range(len(data_record_list)):
        sim_performance = data_record_list[s]['summary']['performance_sim']
        trial_performances = data_record_list[s]['summary']['performance_trials']
        marker = ' <--' if s == sim_index else ''
        print("Sim #{} performance: {}{}".format(s+1, sim_performance, marker))
        print("  Trials performances: {}".format(trial_performances))

    data_record = data_record_list[sim_index]    
    vis.start_simulation_from_data(trial_index, data_record)


    