import os
import json
import matplotlib.pyplot as plt
from scipy import stats

def analyze_histo_entropy():
    # base_dir = 'data/histo_entropy'
    base_dir = 'data/transfer_entropy/min'
    exp_dirs = sorted(os.listdir(base_dir))
    best_exp_performance = []
    for exp in exp_dirs:
        evo_file = os.path.join(base_dir, exp, 'evo_500.json')
        with open(evo_file) as f_in:
            exp_evo_data = json.load(f_in)
            gen_best_perf = exp_evo_data['best_performances']
            
            # make sure it's monotonic increasing(otherwise there is a bug)
            # assert all(gen_best_perf[i] <= gen_best_perf[i+1] for i in range(len(gen_best_perf)-1))
            
            last_best_performance = gen_best_perf[-1]
            print('{} {:.3f}'.format(exp, last_best_performance))
            best_exp_performance.append(last_best_performance)
    print(stats.describe(best_exp_performance))
    plt.plot(best_exp_performance, label='Best')
    plt.show()

if __name__ == "__main__":
    analyze_histo_entropy()