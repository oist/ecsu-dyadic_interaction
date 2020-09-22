"""
Transfer entropy on continuous data using Kraskov estimators
for various types of time series relationships.
"""
import jpype
import math
import os
import numpy as np
from numpy.random import RandomState
from matplotlib import pyplot as plt
from dyadic_interaction.utils import add_noise

DEST_HISTORY = 1
SOURCE_HISTORY = 1
DELAY = 1

infodynamics_dir = './infodynamics'
jarLocation = os.path.join(infodynamics_dir, "infodynamics.jar")
jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)


# Shut down JVM
def shutdown_JVM():
    jpype.shutdownJVM()


def initialize_calc(calc, delay=DELAY):
    # Normalise the individual variables
    # calc.setProperty("NORMALISE", "true")
    calc.setProperty("DELAY", str(delay))
    calc.setProperty("k", "4")
    # no stochastic noise for reproducibility,
    # see https://github.com/jlizier/jidt/wiki/FAQs#
    # why-are-my-results-from-a-kraskov-stoegbauer-grassberger-estimator-stochastic
    calc.setProperty("NOISE_LEVEL_TO_ADD", "0")
    calc.initialise()


def get_transfer_entropy(brain_output, delay=1, reciprocal=True, log=False):
    calcClass = jpype.JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
    calc = calcClass()
    initialize_calc(calc, delay)
    source = brain_output[:, 0]
    destination = brain_output[:, 1]
    calc.setObservations(source, destination)
    te_src_dst = calc.computeAverageLocalOfObservations()
    if log:
        print('te_src_dst: {}'.format(te_src_dst))
    if not reciprocal:
        return te_src_dst
    calc.initialise()  # Re-initialise leaving the parameters the same
    calc.setObservations(destination, source)
    te_dst_src = calc.computeAverageLocalOfObservations()
    if log:
        print('te_dst_src: {}'.format(te_dst_src))
    return np.mean([te_src_dst, te_dst_src])


""" Tests with simulated data"""


def test_neural_entropy_random(num_data_points, distribution='uniform'):
    # Uncorrelated and random array
    rs = RandomState(0)
    if distribution == 'normal':
        brain_output = rs.normal(0, 1, (num_data_points, 2))
    else:
        brain_output = rs.rand(num_data_points, 2)
    transfer_entropy = get_transfer_entropy(brain_output)
    print("Transfer Entropy on random data ({} data points): {}".format(num_data_points, transfer_entropy))
    plt.plot(brain_output)
    plt.show()


def test_neural_entropy_constant(num_data_points):
    # Uncorrelated and constant array
    source = np.ones(num_data_points)
    destination = np.ones(num_data_points) * 2
    brain_output = np.column_stack((source, destination))
    rs = RandomState(1)
    brain_output = add_noise(brain_output, rs, noise_level=1e-8)
    transfer_entropy = get_transfer_entropy(brain_output)
    print("Transfer Entropy on 2D constant data: {}".format(transfer_entropy))
    plt.plot(brain_output)
    plt.show()


def test_neural_entropy_single():
    bins = 100
    num_data_points = bins * bins
    brain_output = np.ones((num_data_points, 2))
    rs = RandomState(1)
    brain_output = add_noise(brain_output, rs, noise_level=1e-8)
    transfer_entropy = get_transfer_entropy(brain_output)
    print("Transfer Entropy on 1D constant data: {}".format(transfer_entropy))
    plt.plot(brain_output)
    plt.show()


def generate_correlated_data(num_data_points, cov, delay):
    rs = RandomState(0)
    source = rs.normal(0, 1, num_data_points)
    col1 = source[:-delay] * cov
    col2 = rs.normal(0, 1, num_data_points - delay) * (1 - cov)
    summed = col1 + col2
    destination = np.concatenate((np.zeros(delay), summed))
    return np.column_stack((source, destination))


def test_neural_entropy_correlated(num_data_points, cov=0.99, delay=1):
    corr_expected = cov / (1 * math.sqrt(cov ** 2 + (1 - cov) ** 2))
    entropy_expected = -0.5 * math.log(1 - corr_expected ** 2)
    brain_output = generate_correlated_data(num_data_points, cov, delay)
    transfer_entropy = get_transfer_entropy(brain_output, delay)
    print("Transfer Entropy on random data ({} data points, covariance {}, delay {}): {}\n"
          "Expected TE: {}".format(num_data_points, cov, delay, transfer_entropy,
                                   entropy_expected))
    plt.plot(brain_output)
    plt.show()


def test_neural_entropy_uniform(scramble=True):
    bins = 100
    data_per_bin = 10
    num_data_points = data_per_bin * bins * bins
    brain_output = np.zeros((num_data_points, 2))
    row = 0
    for i in range(bins):
        for j in range(bins):
            for _ in range(data_per_bin):
                brain_output[row, :] = [i / 100 + 0.0001, j / 100 + 0.0001]
                row += 1
    if scramble:
        rs = RandomState(0)
        rs.shuffle(brain_output)
    transfer_entropy = get_transfer_entropy(brain_output)
    print("Transfer Entropy on uniform data: {}".format(transfer_entropy))
    plt.plot(brain_output)
    plt.show()


def analyze_sample_brain():
    import json
    with open('data/tmp_brains.json') as f:
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


if __name__ == "__main__":
    # data_size = 1000
    # test_neural_entropy_random(data_size, 'uniform')
    # test_neural_entropy_random(data_size, 'normal')
    # test_neural_entropy_constant(data_size)
    # test_neural_entropy_single()
    # test_neural_entropy_correlated(data_size, 0.1, 1)
    # test_neural_entropy_correlated(data_size, 0.4, 1)
    # test_neural_entropy_correlated(data_size, 0.9, 1)
    # test_neural_entropy_correlated(data_size, 0.4, 2)
    # test_neural_entropy_uniform(False)
    # test_neural_entropy_uniform()
    analyze_sample_brain()
    shutdown_JVM()
