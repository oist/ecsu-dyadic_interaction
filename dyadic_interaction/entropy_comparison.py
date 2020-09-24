import math
import numpy as np
from numpy.random import RandomState
from matplotlib import pyplot as plt
from dyadic_interaction.utils import add_noise
from dyadic_interaction.neural_transfer_entropy import get_transfer_entropy, shutdown_JVM
from dyadic_interaction.neural_histo_entropy import get_norm_entropy, BINS


def generate_correlated_data(num_data_points, cov, delay):
    rs = RandomState(0)
    source = rs.normal(0, 1, num_data_points)
    col1 = source[:-delay] * cov
    col2 = rs.normal(0, 1, num_data_points - delay) * (1 - cov)
    summed = col1 + col2
    destination = np.concatenate((np.zeros(delay), summed))
    return np.column_stack((source, destination))


""" Tests with simulated data"""


def test_neural_entropy_random(num_data_points, distribution='uniform'):
    # Uncorrelated and random array
    rs = RandomState(0)
    if distribution == 'normal':
        brain_output = rs.normal(0, 1, (num_data_points, 2))
    else:
        brain_output = rs.rand(num_data_points, 2)
    transfer_entropy = get_transfer_entropy(brain_output)
    norm_entropy = get_norm_entropy(brain_output)
    print("Simulated {} data points".format(num_data_points))
    print("Transfer Entropy on random {} data: {}".format(distribution, transfer_entropy))
    print("Shannon Entropy on random {} data: {}".format(distribution, norm_entropy))
    plt.plot(brain_output)
    plt.show()


def test_neural_entropy_single():
    # Constant arrays of the same value (single bin)
    num_data_points = BINS * BINS
    brain_output = np.ones((num_data_points, 2))
    rs = RandomState(1)
    brain_output = add_noise(brain_output, rs, noise_level=1e-8)
    transfer_entropy = get_transfer_entropy(brain_output)
    norm_entropy = get_norm_entropy(brain_output)
    print("Transfer Entropy on 1D constant data: {}".format(transfer_entropy))
    print("Shannon Entropy on 1D constant data: {}".format(norm_entropy))
    plt.plot(brain_output)
    plt.show()


def test_neural_entropy_constant(num_data_points):
    # Correlated and constant arrays
    source = np.ones(num_data_points)
    destination = np.ones(num_data_points) * 2
    brain_output = np.column_stack((source, destination))
    rs = RandomState(1)
    brain_output = add_noise(brain_output, rs, noise_level=1e-8)
    transfer_entropy = get_transfer_entropy(brain_output)
    norm_entropy = get_norm_entropy(brain_output)
    print("Transfer Entropy on 2D constant data: {}".format(transfer_entropy))
    print("Shannon Entropy on 2D constant data: {}".format(norm_entropy))
    plt.plot(brain_output)
    plt.show()


def test_neural_entropy_uniform(scramble=True):
    # Uniform filling of all the bins
    data_per_bin = 10
    num_data_points = data_per_bin * BINS * BINS
    brain_output = np.zeros((num_data_points, 2))
    row = 0
    for i in range(BINS):
        for j in range(BINS):
            for _ in range(data_per_bin):
                brain_output[row, :] = [i / 100 + 0.0001, j / 100 + 0.0001]
                row += 1
    if scramble:
        rs = RandomState(0)
        rs.shuffle(brain_output)
    transfer_entropy = get_transfer_entropy(brain_output)
    norm_entropy = get_norm_entropy(brain_output)
    print("Transfer Entropy on uniform data: {}".format(transfer_entropy))
    print("Shannon Entropy on uniform data: {}".format(norm_entropy))
    plt.plot(brain_output)
    plt.show()


def test_neural_entropy_correlated(num_data_points, cov=0.99, delay=1):
    # One series random, the other correlated with the first at some delay
    corr_expected = cov / (1 * math.sqrt(cov ** 2 + (1 - cov) ** 2))
    entropy_expected = -0.5 * math.log(1 - corr_expected ** 2)
    brain_output = generate_correlated_data(num_data_points, cov, delay)
    transfer_entropy, local_te = get_transfer_entropy(brain_output, delay, local=True)
    local_te = np.array(local_te)
    print("Transfer Entropy on correlated data ({} data points, covariance {}, delay {}): {}\n"
          "Expected TE: {}".format(num_data_points, cov, delay, transfer_entropy,
                                   entropy_expected))
    plt.plot(brain_output)
    plt.show()
    plt.plot(local_te[0])
    plt.plot(local_te[1])
    plt.show()
    norm_entropy = get_norm_entropy(brain_output)
    print("Shannon Entropy on correlated data: {}".format(norm_entropy))


def test_coupled_oscillators():
    from dyadic_interaction.dynamical_systems import spring_mass_system
    spring_data = spring_mass_system()
    pos = np.column_stack((spring_data[:, 0], spring_data[:, 2]))
    vel = np.column_stack((spring_data[:, 1], spring_data[:, 3]))
    transfer_entropy, local_te = get_transfer_entropy(pos, local=True)
    norm_entropy = get_norm_entropy(pos)
    print("Transfer Entropy of spring positions: {}".format(transfer_entropy))
    print("Shannon Entropy of spring positions: {}".format(norm_entropy))
    plt.plot(pos)
    plt.show()
    transfer_entropy = get_transfer_entropy(vel)
    norm_entropy = get_norm_entropy(vel)
    print("Transfer Entropy of spring velocities: {}".format(transfer_entropy))
    print("Shannon Entropy of spring velocities: {}".format(norm_entropy))
    plt.plot(vel)
    plt.show()
    plt.plot(local_te[0])
    plt.plot(local_te[1])
    plt.show()


def analyze_sample_brain():
    import json
    with open('dyadic_interaction/tmp_brains.json') as f:
        data = json.load(f)
    df = np.array(data)
    t1_a1 = df[0][0]
    t1_a2 = df[0][1]
    te1 = get_transfer_entropy(t1_a1)
    te2 = get_transfer_entropy(t1_a2)
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
    data_size = 1000
    test_neural_entropy_random(data_size, 'uniform')
    test_neural_entropy_random(data_size, 'normal')
    test_neural_entropy_constant(data_size)
    test_neural_entropy_single()
    test_neural_entropy_uniform(False)
    test_neural_entropy_uniform()
    test_neural_entropy_correlated(data_size, 0.1, 1)
    test_neural_entropy_correlated(data_size, 0.4, 1)
    test_neural_entropy_correlated(data_size, 0.9, 1)
    test_neural_entropy_correlated(data_size, 0.4, 2)
    test_coupled_oscillators()

    # analyze_sample_brain()
    # analyze_brain_freqs()
    shutdown_JVM()
