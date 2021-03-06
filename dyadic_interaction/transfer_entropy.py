"""
Transfer entropy on continuous data using Kraskov estimators
for various types of time series relationships.
"""
import jpype
import os
import numpy as np
from dyadic_interaction.utils import discretize

DEST_HISTORY = 1
SOURCE_HISTORY = 1
DELAY = 1
BINS = 100

infodynamics_dir = './infodynamics'
jarLocation = os.path.join(infodynamics_dir, "infodynamics.jar")
jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)


# Shut down JVM
def shutdown_JVM():
    jpype.shutdownJVM()


def initialize_calc(calc, delay=DELAY):
    # Normalise the individual variables
    calc.setProperty("NORMALISE", "true")
    calc.setProperty("DELAY", str(delay))
    calc.setProperty("k", "4")
    # no stochastic noise for reproducibility,
    # see https://github.com/jlizier/jidt/wiki/FAQs#
    # why-are-my-results-from-a-kraskov-stoegbauer-grassberger-estimator-stochastic
    calc.setProperty("NOISE_LEVEL_TO_ADD", "0")
    calc.initialise()


def get_transfer_entropy(brain_output, delay=1, reciprocal=True, log=False,
                         local=False, binning=True, min_v=0., max_v=1.):
    """
    Calculate transfer entropy from 2D time series.
    :param brain_output: time series numpy array
    :param delay: lag between source and destination
    :param reciprocal: whether to calculate average TE in both directions
    :param log: whether to print intermediate results
    :param local: whether to calculate local entropy values
    """
    if binning:
        calcClass = jpype.JPackage("infodynamics.measures.discrete").TransferEntropyCalculatorDiscrete
        source = discretize(brain_output[:, 0], BINS, min_v, max_v).tolist()
        destination = discretize(brain_output[:, 1], BINS, min_v, max_v).tolist()
        calc = calcClass(BINS,1)
        calc.initialise()
        calc.addObservations(source, destination)            
    else:
        calcClass = jpype.JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
        source = brain_output[:, 0]
        destination = brain_output[:, 1]
        calc = calcClass()
        initialize_calc(calc, delay)
        calc.setObservations(source, destination)        
    
    te_src_dst = calc.computeAverageLocalOfObservations()
    if binning:
        te_src_dst = te_src_dst / np.log2(BINS)

    if log:
        print('te_src_dst: {}'.format(te_src_dst))
    local_te = []
    if local:
        te_src_dst_local = calc.computeLocalOfPreviousObservations()
        local_te.append(te_src_dst_local)
    if not reciprocal:
        return te_src_dst
    
    calc.initialise()  # Re-initialise leaving the parameters the same
    if binning:
        calc.addObservations(destination, source)
    else:
        calc.setObservations(destination, source)
    te_dst_src = calc.computeAverageLocalOfObservations()
    if binning:
        te_dst_src = te_dst_src / np.log2(BINS)
    avg_te = np.mean([te_src_dst, te_dst_src])
    if log:
        print('te_dst_src: {}'.format(te_dst_src))
    if local:
        te_dst_src_local = calc.computeLocalOfPreviousObservations()
        local_te.append(te_dst_src_local)
        return avg_te, local_te
    return avg_te
