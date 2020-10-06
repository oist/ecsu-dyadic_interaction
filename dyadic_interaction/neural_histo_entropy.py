import numpy as np

BINS = 100


def get_norm_entropy(brain_output, min_v=0., max_v=1.):
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
