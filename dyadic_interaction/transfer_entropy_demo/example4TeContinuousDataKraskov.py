"""
Java Information Dynamics Toolkit (JIDT)
Copyright (C) 2012, Joseph T. Lizier
Example 4 - Transfer entropy on continuous data using Kraskov estimators
Simple transfer entropy (TE) calculation on continuous-valued data using the Kraskov-estimator TE calculator.
"""
import jpype
import random
import math
import os
import numpy as np

infodynamics_dir = './infodynamics'

# Change location of jar to match yours (we assume script is called from demos/python):
jarLocation = os.path.join(infodynamics_dir, "infodynamics.jar")
if not (os.path.isfile(jarLocation)):
    exit("infodynamics.jar not found (expected at " +
         os.path.abspath(jarLocation) +
         ") - are you running from demos/python?")
# Start the JVM (add the "-Xmx" option with say 1024M if you get crashes due to not enough memory space)
jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)

def random_data():
    """a) Example with random data"""
    # Generate some random normalised data.    
    numObservations = 1000
    covariance = 0.4
    # Source array of random normals:
    sourceArray = [random.normalvariate(0, 1) for r in range(numObservations)]
    # Destination array of random normals with partial correlation to previous value of sourceArray
    destArray = [0] + [sum(pair) for pair in
                    zip([covariance*y for y in sourceArray[0:numObservations-1]],
                        [(1-covariance)*y for y in [random.normalvariate(0, 1) for
                                                    r in range(numObservations-1)]])]
    # Uncorrelated source array:
    sourceArray2 = [random.normalvariate(0, 1) for r in range(numObservations)]
    # Create a TE calculator and run it:
    teCalcClass = jpype.JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
    teCalc = teCalcClass()
    teCalc.setProperty("NORMALISE", "true")  # Normalise the individual variables
    teCalc.initialise(1)  # Use history length 1 (Schreiber k=1)
    teCalc.setProperty("k", "4")  # Use Kraskov parameter K=4 for 4 nearest points
    # Perform calculation with correlated source:
    teCalc.setObservations(jpype.JArray(jpype.JDouble, 1)(sourceArray), jpype.JArray(jpype.JDouble, 1)(destArray))
    result = teCalc.computeAverageLocalOfObservations()
    # Note that the calculation is a random variable (because the generated
    #  data is a set of random variables) - the result will be of the order
    #  of what we expect, but not exactly equal to it; in fact, there will
    #  be a large variance around it.
    # Expected correlation is expected covariance / product of expected standard deviations:
    #  (where square of destArray standard dev is sum of squares of std devs of
    #  underlying distributions)
    corr_expected = covariance / (1 * math.sqrt(covariance**2 + (1-covariance)**2))
    print("TE result %.4f nats; expected to be close to %.4f nats for these correlated Gaussians" %
        (result, -0.5 * math.log(1-corr_expected**2)))
    # Perform calculation with uncorrelated source:
    teCalc.initialise()  # Initialise leaving the parameters the same
    teCalc.setObservations(jpype.JArray(jpype.JDouble, 1)(sourceArray2), jpype.JArray(jpype.JDouble, 1)(destArray))
    result2 = teCalc.computeAverageLocalOfObservations()
    print("TE result %.4f nats; expected to be close to 0 nats for these uncorrelated Gaussians" % result2)


def from_file():
    """b) Example with actual data"""
    # 0. Load/prepare the data:
    demo_data_file = os.path.join(infodynamics_dir, "data/2coupledRandomCols-1.txt")
    data = np.genfromtxt(demo_data_file, dtype=np.float)
    source = data[:, 0]
    destination = data[:, 1]

    # 1. Construct the calculator:
    calcClass = jpype.JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
    calc = calcClass()
    # 2. Set any properties to non-default values:
    calc.setProperty("k_HISTORY", "2")
    # 3. Initialise the calculator for (re-)use:
    calc.initialise()
    # 4. Supply the sample data:
    calc.setObservations(source, destination)
    # 5. Compute the estimate:
    result = calc.computeAverageLocalOfObservations()

    print("TE_Kraskov (KSG)(col_0 -> col_1) = %.4f nats" % result)


if __name__ == "__main__":

    # random.seed(0)

    random_data()

    from_file()

    # Shut down JVM
    jpype.shutdownJVM()
