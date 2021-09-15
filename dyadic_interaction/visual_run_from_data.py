
import argparse
from dyadic_interaction.simulation import obtain_trial_data
from dyadic_interaction.visual import Visualization


if __name__ == "__main__":

    from dyadic_interaction.simulation import get_argparse, obtain_trial_data
    
    parser = get_argparse()
    parser.add_argument('--trial', type=int, default=1, choices=[1,2,3,4], help='Trial index')    
    parser.add_argument('--video_path', type=str, default=None, help='Output video')    
    args = parser.parse_args()

    args_dict = vars(args)
    trial_index = args_dict['trial'] - 1    
    del args_dict['trial']

    video_path = None
    if 'video_path' in args_dict:
        video_path = args_dict['video_path']
        del args_dict['video_path']

    evo, sim, data_record = obtain_trial_data(**args_dict)

    vis = Visualization(sim, video_path=video_path)
    vis.start_simulation_from_data(trial_index, data_record)


    