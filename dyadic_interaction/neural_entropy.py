import numpy as np

BINS = 100

def get_norm_entropy(brain_output):
    num_data_points = len(brain_output)
    histo, _, _ = np.histogram2d(
        brain_output[:,0], 
        brain_output[:,1],
        bins=[BINS,BINS],
        range=[[0.,1.],[0.,1.]],    
    )    
    # print(histo)
    with np.errstate(divide='ignore'):
        histo_prob = histo / num_data_points    
        histo_neg_prob = np.negative(histo_prob)
        hist_log_prob = np.log2(histo_prob)
    with np.errstate(invalid='ignore'):
        histo_neg_prob_log_prob = np.multiply(histo_neg_prob, hist_log_prob)
    entropy = np.nansum(histo_neg_prob_log_prob)
    norm_entropy = entropy / np.log2(BINS*BINS)
    return norm_entropy


def test_neural_entropy_random(num_data_points = 2000):    
    brain_output = np.random.rand(num_data_points, 2)
    norm_entropy = get_norm_entropy(brain_output)
    print("Norm entropy on random data: {}".format(norm_entropy))
    
def test_neural_entropy_single(num_data_points = 2000):    
    brain_output = np.ones((num_data_points, 2))
    norm_entropy = get_norm_entropy(brain_output)
    print("Norm entropy on singleton data: {}".format(norm_entropy))

def test_neural_entropy_uniform(num_data_points = 20000):    
    # num_data_points = 100
    # BINS = 10
    brain_output = np.zeros((num_data_points, 2))
    data_per_bin = int(num_data_points / (BINS*BINS))
    # print("data_per_bin: {}".format(data_per_bin))
    assert data_per_bin > 0
    row = 0
    for i in range(BINS):
        for j in range(BINS):
            for _ in range(data_per_bin):
                brain_output[row,:] = [i/100,j/100]
                row += 1
    # print(brain_output)
    norm_entropy = get_norm_entropy(brain_output)
    print("Norm entropy on uniform data: {}".format(norm_entropy))


if __name__ == "__main__":
    test_neural_entropy_random()
    test_neural_entropy_single()
    test_neural_entropy_uniform(20000)