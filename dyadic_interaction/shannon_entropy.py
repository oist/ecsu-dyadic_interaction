import numpy as np

BINS = 100

def get_shannon_entropy_1d(data_1d, min_v=0., max_v=100.):
    num_data_points = len(data_1d)
    histo, _ = np.histogram(
        data_1d,
        bins=BINS,
        range=[min_v, max_v]
    )

    # print(histo)
    with np.errstate(divide='ignore'):
        histo_prob = histo / num_data_points
        histo_neg_prob = np.negative(histo_prob)
        hist_log_prob = np.log2(histo_prob)
    with np.errstate(invalid='ignore'):
        histo_neg_prob_log_prob = np.multiply(histo_neg_prob, hist_log_prob)
    entropy = np.nansum(histo_neg_prob_log_prob)

    distance_shannon_entropy = entropy / np.log2(BINS)
    return distance_shannon_entropy


def get_shannon_entropy_2d(brain_output, min_v=0., max_v=1.):
    num_data_points = len(brain_output)
    histo, _, _ = np.histogram2d(
        brain_output[:, 0],
        brain_output[:, 1],
        bins=[BINS, BINS],
        range=[[min_v, max_v], [min_v, max_v]],
    )
    # print(histo)
    with np.errstate(divide='ignore'):
        histo_prob = histo / num_data_points
        histo_neg_prob = np.negative(histo_prob)
        hist_log_prob = np.log2(histo_prob)
    with np.errstate(invalid='ignore'):
        histo_neg_prob_log_prob = np.multiply(histo_neg_prob, hist_log_prob)
    entropy = np.nansum(histo_neg_prob_log_prob)
    norm_entropy = entropy / np.log2(BINS * BINS)
    return norm_entropy

def get_shannon_entropy_dd(data, min_v=0., max_v=1.):
    num_data_points = len(data)
    dimensions = data.shape[1]
    histo, _ = np.histogramdd(
        data,
        bins = BINS,
        range =[(min_v, max_v)] * dimensions,
    )
    # print(histo)
    with np.errstate(divide='ignore'):
        histo_prob = histo / num_data_points
        histo_neg_prob = np.negative(histo_prob)
        hist_log_prob = np.log2(histo_prob)
    with np.errstate(invalid='ignore'):
        histo_neg_prob_log_prob = np.multiply(histo_neg_prob, hist_log_prob)
    entropy = np.nansum(histo_neg_prob_log_prob)
    norm_entropy = entropy / np.log2(BINS ** dimensions)
    return norm_entropy

def test_1d():
    a = np.array([1.1, 2.2, 3.3, 4.4])
    print(a)
    distance_entropy = get_shannon_entropy_1d(a)
    print(distance_entropy)

def test_dd():
    a = np.array([
        [.0, .0, .0]
    ])
    print(a.shape[1])
    norm_entropy = get_shannon_entropy_dd(a)
    print(norm_entropy)

def test_2d():
    a = np.array([[.1, .2, .3, .4],[.1, .2, .3, .4]])
    a = np.transpose(a)
    print(a)
    norm_entropy = get_shannon_entropy_2d(a)
    print(norm_entropy)

if __name__ == "__main__":
    test_1d()    
    # test_2d()
    # test_dd()