import os
import numpy as np
from numpy.random import RandomState
import jpype


HISTORY_LENGTH = 2

infodynamics_dir = './infodynamics'
jarLocation = os.path.join(infodynamics_dir, "infodynamics.jar")
jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)
teCalcClass = jpype.JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov

def initialize_teCalc(calc):
    calc.setProperty("NORMALISE", "true")  # Normalise the individual variables
    calc.initialise(HISTORY_LENGTH)  # Use history length 1 (Schreiber k=1)
    calc.setProperty("k", "4")  # Use Kraskov parameter K=4 for 4 nearest points


def get_transfer_entropy(brain_output, reciprocal=True):
    calc = teCalcClass()
    initialize_teCalc(calc)
    source = brain_output[:,0]
    destination = brain_output[:,1]
    calc.setObservations(source, destination)    
    te_src_dst = calc.computeAverageLocalOfObservations()
    # print('te_src_dst: {}'.format(te_src_dst))
    if not reciprocal:
        return te_src_dst
    calc.initialise()  # Re-initialise leaving the parameters the same
    calc.setObservations(destination, source)    
    te_dst_src = calc.computeAverageLocalOfObservations()
    # print('te_dst_src: {}'.format(te_dst_src))
    return np.mean([te_src_dst, te_dst_src])

def test_neural_entropy_random():    
    bins = 100
    num_data_points = 10 * bins * bins
    brain_output = np.random.rand(num_data_points, 2)
    transfer_entropy = get_transfer_entropy(brain_output)
    print("Transfer Entropy on random data ({} data points): {}".format(num_data_points, transfer_entropy))
    
def test_neural_entropy_single():   
    bins = 100
    num_data_points = bins * bins 
    brain_output = np.ones((num_data_points, 2))
    transfer_entropy = get_transfer_entropy(brain_output)
    print("Transfer Entropy on singleton data: {}".format(transfer_entropy))

def test_neural_entropy_uniform():
    bins = 100
    data_per_bin = 10
    num_data_points = data_per_bin * bins * bins
    brain_output = np.zeros((num_data_points, 2))    
    row = 0
    for i in range(bins):
        for j in range(bins):
            for _ in range(data_per_bin):
                brain_output[row,:] = [i/100+0.0001,j/100+0.0001]
                row += 1
    # print(brain_output)
    transfer_entropy = get_transfer_entropy(brain_output)
    print("Transfer Entropy on uniform data: {}".format(transfer_entropy))

def test_neural_entropy_reciprocal():
    num_data_points = 100
    rs = RandomState(0)
    brain_output = rs.rand(num_data_points, 2)
    print(brain_output[1:5,:])
    transfer_entropy = get_transfer_entropy(brain_output)
    print("Transfer Entropy on random data ({} data points) A B: {}".format(num_data_points, transfer_entropy))
    # swapping columns
    brain_output[:,[0,1]] = brain_output[:,[1,0]]
    transfer_entropy = get_transfer_entropy(brain_output)
    print("Transfer Entropy on random data ({} data points) B A: {}".format(num_data_points, transfer_entropy))

if __name__ == "__main__":
    # test_neural_entropy_random()
    # test_neural_entropy_single()
    # test_neural_entropy_uniform()
    test_neural_entropy_reciprocal()