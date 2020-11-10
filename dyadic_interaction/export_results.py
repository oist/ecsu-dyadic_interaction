import os
import json
import matplotlib.pyplot as plt
from scipy import stats
import sys
import pandas as pd 
from dyadic_interaction.analyze_results import get_last_entropies_runs

def export_results(base_dir):        
    dirs = sorted(
        [ 
            dir_name for dir_name in os.listdir(base_dir) 
            if os.path.isdir(os.path.join(base_dir, dir_name)) 
        ]
    )
    data_dict = None
    for exp_name in dirs:
        seed_entropy = get_last_entropies_runs(os.path.join(base_dir,exp_name), plot=False)
        if data_dict is None:
            seeds = list(seed_entropy.keys())
            data_dict = {'seeds': seeds}
        data_dict[exp_name] = list(seed_entropy.values())
    df = pd.DataFrame(data_dict) 
    df.set_index('seeds', inplace=True)
    print(df)
    df.to_csv(os.path.join(base_dir, 'all_results.csv'))



if __name__ == "__main__":
    assert len(sys.argv)==2, "You need to specify the directory with the various runs to analyze"    
    base_dir = sys.argv[1]
    export_results(base_dir)