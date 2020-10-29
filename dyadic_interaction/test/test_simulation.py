import numpy as np
from dyadic_interaction import gen_structure
from dyadic_interaction.simulation import Simulation
from dyadic_interaction import utils
from dyadic_interaction.agent_network import AgentNetwork
import json

GEORGINA_GEN_STRUCTURE = gen_structure.load_genotype_structure('config/genotype_structure_2n.json')

'''
Check the performance are reproducible with the same Simulation object
(in consecutive runs)
'''

agent_pair_genome = \
    np.array([
        -0.138300787995370766392255745813599787652492523193359375,
        -0.54189722127169248633293818784295581281185150146484375, 
        0.108851114935846926545792712204274721443653106689453125, 
        0.187434893864179608069520099888904951512813568115234375, 
        0.493528557346037566322394241069559939205646514892578125, 
        -0.55181291667760057340075263709877617657184600830078125, 
        -0.473328126528107129278311049347394146025180816650390625, 
        -0.99822365115097688725853686264599673449993133544921875, 
        -0.8256929085903224052600535287638194859027862548828125, 
        0.56657910538214506512844081953517161309719085693359375, 
        -0.652729413114339518386941563221625983715057373046875, 
        0.72665004096520802168157615597010590136051177978515625, 
        0.99688422931032250051686105507542379200458526611328125, 
        0.061756354358896654865862529959485982544720172882080078125, 
        -0.08621485681672678980103086132658063434064388275146484375, 
        -0.41516933494953989214337752855499275028705596923828125, 
        0.699334661902925258658569873659871518611907958984375, 
        -0.177208966483298258065559593887883238494396209716796875, 
        0.414718057732613443189251256626448594033718109130859375, 
        0.128516169746153485764494917020783759653568267822265625, 
        -0.7365894463833930760898738299147225916385650634765625, 
        0.204744068126590461620395444697351194918155670166015625, 
        -0.9826875435977762140993263528798706829547882080078125, 
        -0.055494983290108101936510820451076142489910125732421875, 
        0.929585033995432130637937007122673094272613525390625, 
        0.4077434591931672347442372483783401548862457275390625, 
        -0.1615989361533980428475842927582561969757080078125, 
        -0.53938596894061541231479850466712377965450286865234375, 
        0.33618676342015973990129396042902953922748565673828125, 
        -0.2552392951729693937323872887645848095417022705078125, 
        0.10466919667055431253377406619620160199701786041259765625, 
        -0.83751681388398402372530426873709075152873992919921875, 
        0.94113732708186537134764648726559244096279144287109375, 
        0.0955303849910250912902398567894124425947666168212890625, 
        0.62740679163165247178568506569718010723590850830078125, 
        -0.0349203898759716524491381051120697520673274993896484375, 
        0.07684178405376286546957231848864466883242130279541015625, 
        -0.59421889727346000054097885367809794843196868896484375, 
        0.912250638675411007483262437744997441768646240234375, 
        0.267757338521746157677938526830985210835933685302734375,
    ])

def test_genotype(index):
    genotype = np.array_split(agent_pair_genome, 2)[index]
    agent_net = AgentNetwork(
        gen_structure.get_num_brain_neurons(GEORGINA_GEN_STRUCTURE),
        0.01,
        GEORGINA_GEN_STRUCTURE,
    )
    phenotype = np.zeros(len(genotype))
    agent_net.genotype_to_phenotype(genotype, phenotype)
    print(json.dumps(phenotype.tolist(), indent=3))
    agent_net.brain.states = np.array([0., 0.])
    agent_net.brain.compute_output()    
    print('brain output: {}'.format(agent_net.brain.output))    
    motor_outputs = agent_net.compute_motor_outputs()
    print('motor output: {}'.format(motor_outputs))

def test_data(entropy_type):

    sim = Simulation(
        entropy_type=entropy_type,
        genotype_structure=GEORGINA_GEN_STRUCTURE,
        num_cores=1
    )

    data_record = {}
    perf = sim.compute_performance(agent_pair_genome, data_record=data_record)
    print('Performance: {}'.format(perf))
    for t in range(4):
    # t = 0    
        for k,v in data_record.items():
            for a in range(2):
                utils.save_numpy_data(v[t][a], 'data/test/{}_{}_{}.json'.format(k,t+1,a+1))


def test_visual(entropy_type):
    from dyadic_interaction.visual import Visualization

    sim = Simulation(
        entropy_type=entropy_type,
        genotype_structure=GEORGINA_GEN_STRUCTURE,
        num_cores=1
    )

    data_record = {}
    perf = sim.compute_performance(agent_pair_genome, data_record=data_record)
    print('Performance: {}'.format(perf))

    vis = Visualization(sim)
    trial_index = 1
    vis.start_simulation_from_data(trial_index, data_record)

def test_plot(entropy_type):
    from dyadic_interaction import plot_results

    sim = Simulation(
        entropy_type=entropy_type,
        genotype_structure=GEORGINA_GEN_STRUCTURE,
        num_cores=1
    )

    data_record = {}
    perf = sim.compute_performance(agent_pair_genome, data_record=data_record)
    print('Performance: {}'.format(perf))

    plot_results.plot_behavior(data_record)


if __name__ == "__main__":
    # test_genotype(0)
    test_data('shannon')
    # test_visual('shannon')
    # test_plot('shannon')
