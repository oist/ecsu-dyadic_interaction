import math
import numpy as np
import pandas as pd
from numpy.random import RandomState
from matplotlib import pyplot as plt
from dyadic_interaction.utils import add_noise
from dyadic_interaction.neural_transfer_entropy import get_transfer_entropy, shutdown_JVM
from dyadic_interaction.neural_shannon_entropy import get_norm_entropy, BINS


def generate_correlated_data(num_data_points, cov, delay, rs):
    source = rs.normal(0, 1, num_data_points)
    col1 = source[:-delay] * cov
    col2 = rs.normal(0, 1, num_data_points - delay) * (1 - cov)
    summed = col1 + col2
    destination = np.concatenate((np.zeros(delay), summed))
    return np.column_stack((source, destination))


""" Tests with simulated data"""


def test_neural_entropy_random(num_experiments, num_data_points, distribution='uniform'):
    """
    Simulate uncorrelated random arrays.
    :param num_experiments: how many simulations to run
    :param num_data_points: how many data points per time series
    :param distribution: normal or uniform
    """
    transfer_entropy = []
    norm_entropy = []
    rs = RandomState(0)
    if distribution == 'normal':
        brain_output = rs.normal(0, 1, (num_experiments, num_data_points, 2))
        for i in range(num_experiments):
            norm_entropy.append(get_norm_entropy(brain_output[i, :, :], min_v=-3., max_v=3.))
            transfer_entropy.append(get_transfer_entropy(brain_output[i, :, :], min_v=-3., max_v=3.))
    else:
        brain_output = rs.rand(num_experiments, num_data_points, 2)
        for i in range(num_experiments):
            norm_entropy.append(get_norm_entropy(brain_output[i, :, :]))
            transfer_entropy.append(get_transfer_entropy(brain_output[i, :, :]))
    print("Simulated {} experiments of {} data points".format(num_experiments, num_data_points))
    print("Transfer Entropy on random {} data: {}".format(distribution, transfer_entropy))
    print("Shannon Entropy on random {} data: {}".format(distribution, norm_entropy))
    # plt.plot(brain_output)
    # plt.show()
    return norm_entropy, transfer_entropy


def test_neural_entropy_single(num_experiments, num_data_points):
    # Constant arrays of the same value (single bin)
    transfer_entropy = []
    norm_entropy = []
    brain_output = np.ones((num_data_points, 2))
    for i in range(num_experiments):
        rs = RandomState(1)
        brain_output = add_noise(brain_output, rs, noise_level=1e-8)
        norm_entropy.append(get_norm_entropy(brain_output))
        transfer_entropy.append(get_transfer_entropy(brain_output))
    print("Transfer Entropy on 1D constant data: {}".format(transfer_entropy))
    print("Shannon Entropy on 1D constant data: {}".format(norm_entropy))
    # plt.plot(brain_output)
    # plt.show()
    return norm_entropy, transfer_entropy


def test_neural_entropy_constant(num_experiments, num_data_points):
    # Correlated and constant arrays
    transfer_entropy = []
    norm_entropy = []
    source = np.ones(num_data_points)
    destination = np.ones(num_data_points) / 2.
    brain_output = np.column_stack((source, destination))
    for _ in range(num_experiments):
        rs = RandomState(1)
        brain_output = add_noise(brain_output, rs, noise_level=1e-8)  # does rs keep going?
        norm_entropy.append(get_norm_entropy(brain_output))
        transfer_entropy.append(get_transfer_entropy(brain_output))
    print("Transfer Entropy on 2D constant data: {}".format(transfer_entropy))
    print("Shannon Entropy on 2D constant data: {}".format(norm_entropy))
    # plt.plot(brain_output)
    # plt.show()
    return norm_entropy, transfer_entropy


def test_neural_entropy_uniform(num_experiments, scramble=True):
    # Uniform filling of all the bins
    transfer_entropy = []
    norm_entropy = []
    data_per_bin = 1
    num_data_points = data_per_bin * BINS * BINS
    brain_output = np.zeros((num_data_points, 2))
    row = 0
    for i in range(BINS):
        for j in range(BINS):
            for _ in range(data_per_bin):
                brain_output[row, :] = [i / 100 + 0.0001, j / 100 + 0.0001]
                row += 1
    if scramble:
        rs = RandomState(1)
        for _ in range(num_experiments):
            rs.shuffle(brain_output)
            norm_entropy.append(get_norm_entropy(brain_output))
            transfer_entropy.append(get_transfer_entropy(brain_output))
    else:
        for _ in range(num_experiments):
            norm_entropy.append(get_norm_entropy(brain_output))
            transfer_entropy.append(get_transfer_entropy(brain_output))

    print("Transfer Entropy on uniform bin data: {}".format(transfer_entropy))
    print("Shannon Entropy on uniform bin data: {}".format(norm_entropy))
    # plt.plot(brain_output)
    # plt.show()
    return norm_entropy, transfer_entropy


def test_neural_entropy_correlated(num_experiments, num_data_points, cov=0.99, delay=1):
    # One series random, the other correlated with the first at some delay
    transfer_entropy = []
    norm_entropy = []
    corr_expected = cov / (1 * math.sqrt(cov ** 2 + (1 - cov) ** 2))
    entropy_expected = -0.5 * math.log(1 - corr_expected ** 2)
    rs = RandomState(0)
    for _ in range(num_experiments):
        brain_output = generate_correlated_data(num_data_points, cov, delay, rs)
        norm_entropy.append(get_norm_entropy(brain_output, min_v=-3., max_v=3.))
        transfer_entropy.append(get_transfer_entropy(brain_output, delay, log=True,
                                                     min_v=-3., max_v=3.))

    # transfer_entropy, local_te = get_transfer_entropy(brain_output, delay, local=True)
    # local_te = np.array(local_te)
    # plt.plot(brain_output)
    # plt.show()
    # plt.plot(local_te[0])
    # plt.plot(local_te[1])
    # plt.show()

    print("Transfer Entropy on correlated data ({} data points, covariance {}, delay {}): {}\n"
          "Expected TE: {}".format(num_data_points, cov, delay, transfer_entropy,
                                   entropy_expected))
    print("Shannon Entropy on correlated data: {}".format(norm_entropy))
    return norm_entropy, transfer_entropy


def test_coupled_oscillators(num_experiments):
    from dyadic_interaction.dynamical_systems import spring_mass_system
    transfer_entropy = []
    norm_entropy = []
    rs = RandomState(0)
    for _ in range(num_experiments):
        spring_data = spring_mass_system(masses=rs.uniform(1.0, 10.0, 2),
                                         constants=rs.uniform(1.0, 50.0, 2),
                                         lengths=rs.uniform(0.1, 5.0, 2))
        pos = np.column_stack((spring_data[:, 0], spring_data[:, 2]))
        # transfer_entropy, local_te = get_transfer_entropy(pos, local=True)
        norm_entropy.append(get_norm_entropy(pos, min_v=pos.min(), max_v=pos.max()))
        transfer_entropy.append(get_transfer_entropy(pos, min_v=pos.min(), max_v=pos.max()))

        print("Transfer Entropy of spring positions: {}".format(transfer_entropy))
        print("Shannon Entropy of spring positions: {}".format(norm_entropy))
        # plt.plot(pos)
        # plt.show()
        # vel = np.column_stack((spring_data[:, 1], spring_data[:, 3]))
        # transfer_entropy = get_transfer_entropy(vel, log=True)
        # norm_entropy = get_norm_entropy(vel)
        # print("Transfer Entropy of spring velocities: {}".format(transfer_entropy))
        # print("Shannon Entropy of spring velocities: {}".format(norm_entropy))
        # plt.plot(vel)
        # plt.show()
        # plt.plot(local_te[0])
        # plt.plot(local_te[1])
        # plt.show()
    return norm_entropy, transfer_entropy


def analyze_sample_brain():
    import json
    with open('dyadic_interaction/tmp_brains.json') as f:
        data = json.load(f)
    df = np.array(data)
    t1_a1 = df[0][0]
    t1_a2 = df[0][1]
    te1 = get_transfer_entropy(t1_a1, log=True)
    te2 = get_transfer_entropy(t1_a2, log=True)
    print('TE agent1: {}'.format(te1))
    print('TE agent2: {}'.format(te2))
    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(t1_a1[150:, 0])
    ax = fig.add_subplot(2, 2, 2)
    ax.plot(t1_a1[150:, 1])
    ax = fig.add_subplot(2, 2, 3)
    ax.plot(t1_a2[150:, 0])
    ax = fig.add_subplot(2, 2, 4)
    ax.plot(t1_a2[150:, 1])
    plt.show()


def analyze_brain_freqs():
    import json
    from scipy import fftpack
    with open('dyadic_interaction/tmp_brains.json') as f:
        data = json.load(f)
    df = np.array(data)
    sampling_rate = 10

    t1_a1 = df[0][0]
    t1_a2 = df[0][1]

    t1_a1_x1 = fftpack.fft(t1_a1[:, 0])
    t1_a1_x2 = fftpack.fft(t1_a1[:, 1])
    t1_a2_x1 = fftpack.fft(t1_a2[:, 0])
    t1_a2_x2 = fftpack.fft(t1_a2[:, 1])
    t1_a1_freqs1 = fftpack.fftfreq(len(t1_a1[:, 0])) * sampling_rate
    t1_a1_freqs2 = fftpack.fftfreq(len(t1_a1[:, 1])) * sampling_rate
    t1_a2_freqs1 = fftpack.fftfreq(len(t1_a2[:, 0])) * sampling_rate
    t1_a2_freqs2 = fftpack.fftfreq(len(t1_a2[:, 1])) * sampling_rate

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].stem(t1_a1_freqs1, np.abs(t1_a1_x1))
    axs[0, 1].stem(t1_a1_freqs2, np.abs(t1_a1_x2))
    axs[1, 0].stem(t1_a2_freqs1, np.abs(t1_a2_x1))
    axs[1, 1].stem(t1_a2_freqs2, np.abs(t1_a2_x2))
    for _, subplot in np.ndenumerate(axs):
        subplot.set_xlabel('Frequency [Hz]')
        subplot.set_ylabel('Spectrum Magnitude')
        subplot.set_xlim(0.1, sampling_rate / 2)
        subplot.set_ylim(-5, 20)

    plt.show()


if __name__ == "__main__":
    entropies = dict()
    num_exp = 100
    data_size = 2000
    binned = True
    if binned:
        suffix = 'd'
    else:
        suffix = 'c'
    entropies['sh_random-uniform'], entropies['te{}_random-uniform'.format(suffix)] = \
        test_neural_entropy_random(num_exp, data_size, 'uniform')
    entropies['sh_random-normal'], entropies['te{}_random-normal'.format(suffix)] = \
        test_neural_entropy_random(num_exp, data_size, 'normal')
    entropies['sh_corr-constant'], entropies['te{}_corr-constant'.format(suffix)] = \
        test_neural_entropy_constant(num_exp, data_size)
    entropies['sh_singleton'], entropies['te{}_singleton'.format(suffix)] = \
        test_neural_entropy_single(num_exp, data_size)
    entropies['sh_uniform-unscrambled'], entropies['te{}_uniform-unscrambled'.format(suffix)] = \
        test_neural_entropy_uniform(num_exp, False)
    entropies['sh_uniform-scrambled'], entropies['te{}_uniform-scrambled'.format(suffix)] = \
        test_neural_entropy_uniform(num_exp)
    entropies['sh_corr-01-1'], entropies['te{}_corr-01-1'.format(suffix)] = \
        test_neural_entropy_correlated(num_exp, data_size, 0.1, 1)
    entropies['sh_corr-04-1'], entropies['te{}_corr-04-1'.format(suffix)] = \
        test_neural_entropy_correlated(num_exp, data_size, 0.4, 1)
    entropies['sh_corr-09-1'], entropies['te{}_corr-09-1'.format(suffix)] = \
        test_neural_entropy_correlated(num_exp, data_size, 0.9, 1)
    entropies['sh_corr-04-2'], entropies['te{}_corr-04-2'.format(suffix)] = \
        test_neural_entropy_correlated(num_exp, data_size, 0.4, 2)
    entropies['sh_mass-spring'], entropies['te{}_mass-spring'.format(suffix)] = \
        test_coupled_oscillators(num_exp)

    df = pd.DataFrame.from_dict(entropies)
    with open('data/entropies_sim_{}.csv'.format(suffix), 'w') as f:
        df.to_csv(f, ';', index=False)
    # analyze_sample_brain()
    # analyze_brain_freqs()
    shutdown_JVM()
