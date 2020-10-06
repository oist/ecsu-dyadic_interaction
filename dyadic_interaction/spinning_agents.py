import numpy as np
from dyadic_interaction.simulation import Simulation
from dyadic_interaction import gen_structure
from dyadic_interaction import utils


def get_spinning_agents_data():
    genotype_structure=gen_structure.DEFAULT_GEN_STRUCTURE

    sim = Simulation(
        entropy_type = 'shannon',
        genotype_structure = genotype_structure,
        trial_duration = 200,  # the brain would iterate trial_duration/brain_step_size number of time
    )

    # set network manually
    for a in range(2):
        agent_net = sim.agents_pair_net[a]

        # set agent network manually
        agent_net.sensor_gains = np.zeros((2))
        agent_net.sensor_biases = np.zeros((2))
        agent_net.sensor_weights = np.zeros((2,2))
        agent_net.motor_gains = np.ones((3))
        agent_net.motor_biases = np.zeros((3))
        agent_net.motor_weights = \
            np.full((3,2),-np.inf) if a==0 \
            else np.array([
                [0.4, 0.4],
                [-np.inf, -np.inf], # set emitter weights to zero so that output is zero 
                [1., 1.]
            ])
        agent_net.brain.taus = np.ones((2))
        agent_net.brain.gains = np.ones((2))
        agent_net.brain.biases = np.ones((2))
        agent_net.brain.weights = np.ones((2,2))

    data_record = {}
    
    random_seed = utils.random_int()

    perf = sim.compute_performance(rnd_seed=random_seed, data_record=data_record)
    print("random perf: {}".format(perf))

    # from dyadic_interaction.visual import Visualization
    # vis = Visualization(sim)
    # vis.start_simulation_from_data(trial_index=0, data_record=data_record)

    from dyadic_interaction import plot_results
    plot_results.plot_behavior(data_record)
    plot_results.plot_activity(data_record)
    plot_results.plot_inputs(data_record)
    plot_results.plot_motor_output(data_record)
    plot_results.plot_emitters(data_record)

    return data_record

if __name__ == "__main__":
    get_spinning_agents_data()