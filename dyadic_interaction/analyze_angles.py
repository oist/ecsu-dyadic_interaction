import os
from dyadic_interaction.simulation import obtain_trial_data
from dyadic_interaction import spinning_agents
import numpy as np
from dyadic_interaction import entropy as ent
import json
from pyevolver.json_numpy import NumpyListJsonEncoder
import matplotlib.pyplot as plt
from dyadic_interaction.plot_results import plot_behavior


def analyze_angle_entropy(entropy_type):
    print('Analyzing behavioral complexity of entropy ' + entropy_type)
    df = np.genfromtxt('data/angles_{}.csv'.format(entropy_type), delimiter=';')
    metrics = dict()

    metrics['perm_ent'] = np.apply_along_axis(ent.perm_entropy, 1, df, normalize=True)  # Permutation entropy
    metrics['spect_ent'] = np.apply_along_axis(ent.spectral_entropy, 1, df, sf=10, method='fft', normalize=True)  # Spectral entropy
    metrics['svd_ent'] = np.apply_along_axis(ent.svd_entropy, 1, df, normalize=True)  # Singular value decomposition
    metrics['app_ent'] = np.apply_along_axis(ent.app_entropy, 1, df)  # Approximate entropy
    metrics['samp_ent'] = np.apply_along_axis(ent.sample_entropy, 1, df)  # Sample entropy

    metrics['petrosian'] = np.apply_along_axis(ent.petrosian_fd, 1, df)  # Petrosian fractal dimension
    metrics['katz'] = np.apply_along_axis(ent.katz_fd, 1, df)  # Katz fractal dimension
    metrics['higuchi'] = np.apply_along_axis(ent.higuchi_fd, 1, df, kmax=10)  # Higuchi fractal dimension
    metrics['dfa'] = np.apply_along_axis(ent.detrended_fluctuation, 1, df)  # Detrended fluctuation analysis

    print('Permutation entropy: ' + str(np.mean(metrics['perm_ent'])))
    print('Spectral entropy: ' + str(np.nanmean(metrics['spect_ent'])))
    print('SVD entropy: ' + str(np.mean(metrics['svd_ent'])))
    print('Approximate entropy: ' + str(np.mean(metrics['app_ent'])))
    print('Sample entropy: ' + str(np.mean(metrics['samp_ent'])))
    print('Petrosian fractal dimension: ' + str(np.mean(metrics['petrosian'])))
    print('Katz fractal dimension: ' + str(np.mean(metrics['katz'])))
    print('Higuchi fractal dimension: ' + str(np.mean(metrics['higuchi'])))
    print('Detrended fluctuation analysis: ' + str(np.mean(metrics['dfa'])))

    # # for the 2 agents in spinning agents separately:
    # for k, v in metrics.items():
    #     print(k)
    #     print(np.nanmean(v[:4]))
    #     print(np.nanmean(v[4:]))

    return metrics


def analyze_file_angles(entropy_type):
    base_dir = 'data/{}_entropy/MAX'.format(entropy_type)
    exp_dirs = sorted(os.listdir(base_dir))
    all_angles = []
    for exp in exp_dirs:
        exp_file = os.path.join(base_dir, exp)
        _, _, trial_data = obtain_trial_data(exp_file, 500, 0,
                                             force_random=False, invert_sim_type=False)
        for a in range(2):
            for t in range(4):
                angle = trial_data['agent_angle'][t][a]
                angles_diff = np.diff(angle)
                all_angles.append(angles_diff)

                # plt.plot(angles_diff)
                # ax = plt.gca()
                # ax.get_yaxis().get_major_formatter().set_useOffset(False)
                # plot_behavior(trial_data)

    return np.array(all_angles)  # (N_runs x 2a x 4t) by 1999


def analyze_circle_angles():
    data_record = spinning_agents.get_spinning_agents_data()
    all_angles = []

    for a in range(2):
        for t in range(4):
            angle = data_record['agent_angle'][t][a]
            angles_diff = np.diff(angle)
            all_angles.append(angles_diff)

            # plt.plot(angles_diff)
            # ax = plt.gca()
            # ax.get_yaxis().get_major_formatter().set_useOffset(False)
            # plt.show()

    np.savetxt('data/angles_spinning.csv', np.array(all_angles), delimiter=';')
    # print('Agent angles')
    # print(np.degrees(angle[:10]))
    
    # pos = data_record['agent_pos'][0][1]
    # print('Agent pos')
    # print(pos[:10])

    # pos_diff = pos[1:]-pos[:-1]
    # mov_angle = np.arctan(pos_diff[:,1]/pos_diff[:,0]) # 1 less element wrt to angles and pos
    # print('mov_angle')
    # print(np.degrees(mov_angle[:10]))

    # plt.plot(mov_angle, label='mov_angle')
    # plt.show()

    # mov_angle_diff = mov_angle - angle[:-1]
    # print('mov_angle_diff')
    # print(mov_angle_diff[:100])


if __name__ == "__main__":
    # analyze_circle_angles()
    spinning_complexity = analyze_angle_entropy('spinning')

    # histo_angles = analyze_file_angles('histo')
    # np.savetxt('data/angles_histo.csv', histo_angles, delimiter=';')
    #
    # transfer_angles = analyze_file_angles('transfer')
    # np.savetxt('data/angles_transfer.csv', transfer_angles, delimiter=';')

    # beh_complexity_h = analyze_angle_entropy('histo')
    # with open('data/histo_beh_metrics.json', 'w') as f:
    #     json.dump(beh_complexity_h, f, cls=NumpyListJsonEncoder)
    #
    # beh_complexity_te = analyze_angle_entropy('transfer')
    # with open('data/te_beh_metrics.json', 'w') as f:
    #     json.dump(beh_complexity_te, f, cls=NumpyListJsonEncoder)
