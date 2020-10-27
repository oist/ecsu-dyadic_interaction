import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from dyadic_interaction.entropy import sample_entropy

if __name__ == "__main__":
    import csv
    file_path = 'data/dist_entropy/PositionsT1_0.csv'
    from numpy import genfromtxt
    my_data = genfromtxt(file_path, delimiter=',')
    agent_1_xy = my_data[:,0:2]
    agent_2_xy = my_data[:,2:]
    print(agent_1_xy[:5,:])
    print(agent_2_xy[:5,:])
    
    distances = norm(agent_1_xy - agent_2_xy, axis=1)            

    # print(distances)
    # plt.plot(distances)
    # plt.show()

    min_value = np.min([agent_1_xy, agent_2_xy])
    max_value = np.max([agent_1_xy, agent_2_xy])

    # prob = np.histogram(distances, bins=100, range=(min_value, max_value))[0] / len(distances)
    # entrp = np.nansum(-1 * prob * np.log(prob)) / np.log(100)    

    entrp = sample_entropy(distances)

    print(entrp)
