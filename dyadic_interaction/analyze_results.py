import os
import json
import matplotlib.pyplot as plt

def analyze_histo_entropy():
    base_dir = 'data/histo-entropy'
    exp_dirs = sorted(os.listdir(base_dir))
    best_exp_performance = []
    for exp in exp_dirs:
        evo_file = os.path.join(base_dir, exp, 'evo_500.json')
        with open(evo_file) as f_in:
            exp_evo_data = json.load(f_in)
            last_best_performance = exp_evo_data['best_performances'][-1]
            print('{} {:.3f}'.format(exp, last_best_performance))
            best_exp_performance.append(last_best_performance)
    plt.plot(best_exp_performance, label='Best')
    plt.show()

if __name__ == "__main__":
    analyze_histo_entropy()