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
    np.array(      [
         -0.7562370205971931,
         -0.780304551233475,
         0.9814727096214281,
         0.5146751791050619,
         -0.7216033919480229,
         0.9515233481643174,
         0.17877538640830895,
         -0.1724482239533077,
         0.25747111408791695,
         -0.5191985279442142,
         -0.2752507560221079,
         -0.18006493467646623,
         -0.4127344451831818,
         -0.10509666964599175,
         0.3630909527831917,
         -0.8290791318495379,
         0.8765249811234751,
         0.4795130240089538,
         0.4924513410793277,
         0.5425757087064366,
         -0.8118738326524774,
         0.15425480821649531,
         -0.4129169555901072,
         0.018891165557032904,
         0.046841091024842435,
         0.28229852983953724,
         0.3734892853885499,
         0.2803370838924146,
         0.012069762709188046,
         -0.9855760690165177,
         0.5088166947529534,
         0.42092777694431915,
         0.9893834513017314,
         -0.4239067770881425,
         -0.5205278341676729,
         -0.4633839563397559,
         -0.47349365900065266,
         0.7803947889566571,
         0.461392585471145,
         0.35688559458014874
      ])

def test_data(entropy_type):

    sim = Simulation(
        entropy_type = entropy_type,
        genotype_structure = gen_structure.DEFAULT_GEN_STRUCTURE,
        trial_duration = 20,  # the brain would iterate trial_duration/brain_step_size number of time
        num_cores = 1
    )    

    data_record = {}
    perf = sim.compute_performance(agent_pair_genome, data_record = data_record)
    print('Performance: {}'.format(perf))
    trial_index = 1
    # trial_data_record = {k:v[trial_index] for k,v in data_record.items()}
    utils.save_numpy_data(data_record['position'][trial_index], 'data/positions_new.json')
    utils.save_numpy_data(data_record['brain_output'][trial_index], 'data/brain_output_new.json')

def test_visual(entropy_type):
    from pyevolver.evolution import Evolution
    from numpy.random import RandomState
    from dyadic_interaction.visual import Visualization

    default_gen_structure = gen_structure.DEFAULT_GEN_STRUCTURE    

    sim = Simulation(
        entropy_type = entropy_type,
        genotype_structure = default_gen_structure,
        trial_duration = 20,  # the brain would iterate trial_duration/brain_step_size number of time
        num_cores = 1
    )    

    # gen_size = gen_structure.get_genotype_size(gen_structure.DEFAULT_GEN_STRUCTURE)
    # agent_pair_genome = Evolution.get_random_genotype(RandomState(None), gen_size*2)

    data_record = {}
    sim.compute_performance(agent_pair_genome, data_record = data_record)

    vis = Visualization(sim)
    trial_index = 1
    vis.start_simulation_from_data(trial_index, data_record)

    # print(agent_pair_genome.tolist())


if __name__ == "__main__":
    test_data('shannon')
    # test_visual('shannon')
    
    # for entropy_type in ['shannon','transfer']:
    #     print(entropy_type)
    #     test(entropy_type)
    #     print()