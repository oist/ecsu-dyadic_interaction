import numpy as np
from dyadic_interaction import gen_structure
from dyadic_interaction.simulation import Simulation
from dataclasses import asdict
from pyevolver.json_numpy import NumpyListJsonEncoder
from dyadic_interaction.agent_network import AgentNetwork
from dyadic_interaction.agent_body import AgentBody
from dyadic_interaction import utils
import json

'''
Check the performance are reproducible with the same Simulation object
(in consecutive runs)
'''

agent_pair_genome = \
    np.array([ 
        0.09762701,  0.43037873,  0.20552675,  0.08976637, -0.1526904 ,
        0.29178823, -0.12482558,  0.783546  ,  0.92732552, -0.23311696,
        0.58345008,  0.05778984,  0.13608912,  0.85119328, -0.85792788,
       -0.8257414 , -0.95956321,  0.66523969,  0.5563135 ,  0.7400243 ,
        0.95723668,  0.59831713, -0.07704128,  0.56105835, -0.76345115,
        0.27984204, -0.71329343,  0.88933783,  0.04369664, -0.17067612,
       -0.47088878,  0.54846738, -0.08769934,  0.1368679 , -0.9624204 ,
        0.23527099,  0.22419145,  0.23386799,  0.88749616,  0.3636406 
    ])

def test(entropy_type):
    sim = Simulation(
        entropy_type = entropy_type,
        genotype_structure = gen_structure.DEFAULT_GEN_STRUCTURE,
        agent_body_radius = 4,
        agents_pair_initial_distance = 20,
        agent_sensors_divergence_angle = np.radians(45),  # angle between sensors and axes of symmetry
        brain_step_size = 0.1,
        trial_duration = 20,  # the brain would iterate trial_duration/brain_step_size number of time
        num_cores = 1
    )    

    data_record = {}
    perf = sim.compute_performance(agent_pair_genome, data_record = data_record)
    print('Performance: {}'.format(perf))
    t = 0
    # trial_data_record = {k:v[trial_index] for k,v in data_record.items()}
    utils.save_numpy_data(data_record['position'][t], 'data/positions_new.json')
    utils.save_numpy_data(data_record['position'][t], 'data/brain_output_new.json')



if __name__ == "__main__":
    test('shannon')
    # for entropy_type in ['shannon','transfer']:
    #     print(entropy_type)
    #     test(entropy_type)
    #     print()