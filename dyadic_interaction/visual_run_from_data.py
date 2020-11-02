
import argparse
from dyadic_interaction.simulation import obtain_trial_data
from dyadic_interaction.visual import Visualization


if __name__ == "__main__":

    from dyadic_interaction.simulation import get_argparse, obtain_trial_data
    
    parser = get_argparse()
    parser.add_argument('--trial', type=int, default=0, help='Trial index')    
    args = parser.parse_args()

    args_dict = vars(args)
    trial_num = args_dict['trial']
    del args_dict['trial']

    evo, sim, data_record = obtain_trial_data(**args_dict)

    vis = Visualization(sim)
    vis.start_simulation_from_data(trial_num, data_record)


    