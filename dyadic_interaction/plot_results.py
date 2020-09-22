"""
TODO: Missing module docstring
"""

import os
import matplotlib.pyplot as plt
from dyadic_interaction.simulation import Simulation
from dyadic_interaction import gen_structure
from dyadic_interaction import utils
import numpy as np
from numpy.random import RandomState
from pyevolver.evolution import Evolution


def plot_performances(evo):
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle("Agent Performances")

    plt.plot(evo.best_performances, label='Best')
    plt.plot(evo.avg_performances, label='Avg')
    plt.plot(evo.worst_performances, label='Worst')
    # plt.ylim(0, 1)
    plt.legend()
    plt.show()


def plot_behavior(trial_data):    
    agent_pos = [
        [x[0].transpose(), x[1].transpose()] 
        for x in trial_data['agent_pos']
    ]
    num_trials = len(agent_pos)
    # print("agent_pos shape: {}".format(agent_pos[0].shape))
    num_cols = num_trials
    fig = plt.figure(figsize=(12, 5))
    fig.suptitle("Agents behavior")
    for t in range(num_trials):
        ax = fig.add_subplot(1, num_cols, t + 1)
        for a in range(2):
            ax.scatter(agent_pos[t][a][0][0],
                    agent_pos[t][a][1][0], label='Tracker start',
                    color='orange')
            ax.scatter(agent_pos[t][a][0][-1],
                    agent_pos[t][a][1][-1], label='Tracker stop')
        for a in range(2):
            ax.plot(agent_pos[t][a][0],
                    agent_pos[t][a][1], 
                    label='Tracker position')
        # ax.set_xlim(0, trial_data['env_width'])
        # ax.set_ylim(0, trial_data['env_height'])
        ax.set_aspect('equal')
        handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper right')
    # plt.legend()
    plt.show()


def plot_activity(trial_data):
    num_trials = len(trial_data['agent_pos'])
    num_cols = num_trials
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle("Brain activity")
    for t in range(num_trials):
        ax = fig.add_subplot(1, num_cols, t + 1)
        for a in range(2):
            brain_output = trial_data['brain_output'][t][a]
            ax.plot(brain_output[:, 0], label='Output of n1 agent {}'.format(a))
            ax.plot(brain_output[:, 1], label='Output of n2 agent {}'.format(a))
        # trial_data['brain_state'][t]
        # trial_data['derivatives'][t]
    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper right')
    plt.legend()
    plt.show()


def plot_inputs(trial_data):
    num_trials = len(trial_data['agent_pos'])
    num_cols = num_trials
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle("Sensor inputs")
    for t in range(num_trials):
        ax = fig.add_subplot(1, num_cols, t + 1)
        for a in range(2):
            ax.plot(trial_data['sensors_input'][t][a][:, 0], label='Eye input to s1')
            ax.plot(trial_data['sensors_input'][t][a][:, 1], label='Eye input to s2')
            # ax.plot(trial_data['brain_input'][t][a][:, 0], label='Input to n1')
            # ax.plot(trial_data['brain_input'][t][a][:, 1], label='Input to n2')
    plt.legend()
    plt.show()


def plot_motor_output(trial_data):
    num_trials = len(trial_data['agent_pos'])
    num_cols = num_trials
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle("Wheels")
    for t in range(num_trials):
        ax = fig.add_subplot(1, num_cols, t + 1)
        for a in range(2):
            ax.plot(trial_data['wheels'][t][a][:, 0], label='Wheel 1 agent {}'.format(a))
            ax.plot(trial_data['wheels'][t][a][:, 1], label='Wheel 2 agent {}'.format(a))
            # ax.plot(trial_data['wheels'][t][a][:, 1], label='Emitter agent {}'.format(a))
            # ax.plot(trial_data['wheels'][t][a][:, 2], label='Wheel 2 agent {}'.format(a))
    plt.legend()
    plt.show()

def plot_emitters(trial_data):
    num_trials = len(trial_data['agent_pos'])
    num_cols = num_trials
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle("Emitters")
    for t in range(num_trials):
        ax = fig.add_subplot(1, num_cols, t + 1)
        for a in range(2):
            ax.plot(trial_data['emitter'][t][a], label='Emitter agent {}'.format(a))
    plt.legend()
    plt.show()


def plot_simultation_results(dir, generation, genotype_index, force_random=False):
    sim_json_filepath = os.path.join(dir, 'simulation.json')
    evo_json_filepath = os.path.join(dir, 'evo_{}.json'.format(generation))
    sim = Simulation.load_from_file(sim_json_filepath)
    evo = Evolution.load_from_file(evo_json_filepath, folder_path=dir)
    genotype = evo.population[genotype_index]

    if force_random:
        rs = RandomState()
        sim.set_initial_positions_angles(rs)
        random_seed = rs.randint(10000)
    else:
        random_seed = evo.pop_eval_random_seed[genotype_index]
        
    trial_data = {}
    perf = sim.compute_performance(genotype, random_seed, trial_data)
    print("Best performance recomputed: {}".format(perf))

    plot_performances(evo)
    plot_behavior(trial_data)
    plot_activity(trial_data)
    plot_inputs(trial_data)
    plot_motor_output(trial_data)
    plot_emitters(trial_data)
    plot_motor_output(trial_data)
    # utils.save_numpy_data(trial_data['brain_output'], 'data/tmp_brains.json')    


def plot_random_simulation_results():

    genotype_structure = gen_structure.DEFAULT_GEN_STRUCTURE
    gen_size = gen_structure.get_genotype_size(genotype_structure)
    random_genotype = Evolution.get_random_genotype(RandomState(None), gen_size*2) # pairs of agents in a single genotype

    sim = Simulation(
        genotype_structure=genotype_structure,
        agent_body_radius=4,
        agents_pair_initial_distance=20,
        agent_sensors_divergence_angle=np.radians(45),  # angle between sensors and axes of symmetry
        brain_step_size=0.1,
        trial_duration=200,  # the brain would iterate trial_duration/brain_step_size number of time
        num_cores=1
    )
    
    random_seed = np.random.randint(10000)

    trial_data = {}
    perf = sim.compute_performance(random_genotype, random_seed, trial_data)
    print("random perf: {}".format(perf))        

if __name__ == "__main__":
    import argparse

    # default_dir = 'data/histo_entropy/dyadic_exp_096'
    default_dir = 'data/transfer_entropy/max/dyadic_exp_018'    # 5?, 10, 18(max)
    default_gen = '500'
    default_index = 0
    default_random = False

    # plot_simultation_results()
    # plot_random_simulation_results()

    parser = argparse.ArgumentParser(
        description='Plot results'
    )

    parser.add_argument('--dir', type=str, default=default_dir, help='Directory path')     
    parser.add_argument('--gen', type=str, default=default_gen, help='number of genration (str)')    
    parser.add_argument('--index', type=int, default=default_index, help='Index of agent in population')
    parser.add_argument('--random', type=bool, default=default_random, help='If to randomize result')

    args = parser.parse_args()
    plot_simultation_results(args.dir, args.gen, args.index, args.random)
