import os
import matplotlib.pyplot as plt
from dyadic_interaction import simulation
from dyadic_interaction.simulation import Simulation
from dyadic_interaction import gen_structure
from dyadic_interaction import utils
from dyadic_interaction import spinning_agents
import numpy as np
from numpy.random import RandomState
from pyevolver.evolution import Evolution

def analyze_circle_angles():
    data_record = spinning_agents.get_spinning_agents_data()    
    
    angle = data_record['agent_angle'][0][1]    
    angles_diff = np.diff(angle)
    print(angles_diff[:10])

    plt.plot(angles_diff)
    ax = plt.gca()
    ax.get_yaxis().get_major_formatter().set_useOffset(False)
    plt.show()

    # print('Agent angles')
    # print(np.degrees(angle[:10]))
    
    # pos = data_record['agent_pos'][0][1]
    # print('Agent pos')
    # print(pos[:10])

    # pos_diff = pos[1:]-pos[:-1]
    # mov_angle = np.arctan(pos_diff[:,1]/pos_diff[:,0]) # 1 less element wrt to angles and pos
    # print('mov_angle')
    # print(np.degrees(mov_angle[:10]))

    # plt.plot(mov_angle, label='mov_angle')
    # plt.show()

    # mov_angle_diff = mov_angle - angle[:-1]
    # print('mov_angle_diff')
    # print(mov_angle_diff[:100])



if __name__ == "__main__":
    analyze_circle_angles()

