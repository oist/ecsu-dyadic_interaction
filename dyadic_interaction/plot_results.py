"""
TODO: Missing module docstring
"""

import os
import matplotlib.pyplot as plt
from dyadic_interaction import simulation
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


def plot_behavior(data_record):    
    agent_pos = [
        [x[0].transpose(), x[1].transpose()] 
        for x in data_record['agent_pos']
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
        # ax.set_xlim(0, data_record['env_width'])
        # ax.set_ylim(0, data_record['env_height'])
        ax.set_aspect('equal')
        handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper right')
    # plt.legend()
    plt.show()


def plot_angles(data_record):    
    angle_data = np.mod(np.degrees(data_record['agent_angle']), 360.)
    num_trials = len(angle_data)
    num_cols = num_trials
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle("Angles")
    for t in range(num_trials):
        ax = fig.add_subplot(1, num_cols, t + 1)
        for a in range(2):
            ax.plot(angle_data[t][a], label='Angle agent {}'.format(a))
    # ax = fig.add_subplot(1, 1, 1)
    # for a in range(2):
    #     ax.plot(angle_data[0][a], label='Angle agent {}'.format(a))
    plt.legend()
    plt.show()

def plot_norm_x(data_record):    
    pos_data = data_record['agent_pos']
    num_trials = len(pos_data)    
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle("X Pos")
    # num_cols = num_trials
    # for t in range(num_trials):
    #     ax = fig.add_subplot(1, num_cols, t + 1)
    #     for a in range(2):
    #         ax.plot(x_data[t][a], label='Angle agent {}'.format(a))
    ax = fig.add_subplot(1, 1, 1)
    for a in range(2):
        x_data = pos_data[0][a][:,0]
        x_data = (x_data - x_data.min()) / (x_data.max() - x_data.min())
        ax.plot(x_data, label='X agent {}'.format(a))
    plt.legend()
    plt.show()

def plot_activity_scatter(data_record):
    num_trials = len(data_record['agent_pos'])
    num_cols = num_trials
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle("Brain activity")
    for t in range(num_trials):       
        for a in range(2):
            ax = fig.add_subplot(2, num_cols, (a*num_trials)+t+1)
            brain_output = data_record['brain_output'][t][a]
            ax.scatter(brain_output[0][0], brain_output[0][1], label='Tracker start', color='orange')            
            ax.plot(brain_output[:, 0], brain_output[:, 1], label='Output of n1 agent {}'.format(a))            
        # data_record['brain_state'][t]
        # data_record['derivatives'][t]
    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper right')
    plt.legend()
    plt.show()

def plot_activity(data_record):
    num_trials = len(data_record['agent_pos'])
    num_cols = num_trials
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle("Brain activity")
    for t in range(num_trials):
        ax = fig.add_subplot(1, num_cols, t + 1)
        for a in range(2):
            brain_output = data_record['brain_output'][t][a]
            ax.plot(brain_output[:, 0], label='Output of n1 agent {}'.format(a))
            ax.plot(brain_output[:, 1], label='Output of n2 agent {}'.format(a))
        # data_record['brain_state'][t]
        # data_record['derivatives'][t]
    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper right')
    plt.legend()
    plt.show()


def plot_inputs(data_record):
    num_trials = len(data_record['agent_pos'])
    num_cols = num_trials
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle("Inputs")
    for t in range(num_trials):
        ax = fig.add_subplot(1, num_cols, t + 1)
        for a in range(2):
            # ax.plot(data_record['signal_strength'][t][a][:, 0], label='Signal strength to s1 agent {}'.format(a))
            # ax.plot(data_record['signal_strength'][t][a][:, 1], label='Signal strength to s2 agent {}'.format(a))
            ax.plot(data_record['brain_input'][t][a][:, 0], label='Brain Input to n1 agent {}'.format(a))
            ax.plot(data_record['brain_input'][t][a][:, 1], label='Brain Input to n2 agent {}'.format(a))
    plt.legend()
    plt.show()

def plot_signal(data_record):
    num_trials = len(data_record['agent_pos'])
    num_cols = num_trials
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle("Signal Strength")
    for t in range(num_trials):
        for a in range(2):
            ax = fig.add_subplot(2, num_cols, (a*num_trials)+t+1)
            ax.plot(data_record['signal_strength'][t][a][:, 0], label='Signal strength to s1 agent {}'.format(a))
            ax.plot(data_record['signal_strength'][t][a][:, 1], label='Signal strength to s2 agent {}'.format(a))
    plt.legend()
    plt.show()


def plot_motor_output(data_record):
    num_trials = len(data_record['agent_pos'])
    num_cols = num_trials
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle("Wheels")
    for t in range(num_trials):
        ax = fig.add_subplot(1, num_cols, t + 1)
        for a in range(2):
            ax.plot(data_record['wheels'][t][a][:, 0], label='Wheel 1 agent {}'.format(a))
            ax.plot(data_record['wheels'][t][a][:, 1], label='Wheel 2 agent {}'.format(a))
            # ax.plot(data_record['wheels'][t][a][:, 1], label='Emitter agent {}'.format(a))
            # ax.plot(data_record['wheels'][t][a][:, 2], label='Wheel 2 agent {}'.format(a))
    plt.legend()
    plt.show()

def plot_emitters(data_record):
    num_trials = len(data_record['agent_pos'])
    num_cols = num_trials
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle("Emitters")
    for t in range(num_trials):
        ax = fig.add_subplot(1, num_cols, t + 1)
        for a in range(2):
            ax.plot(data_record['emitter'][t][a], label='Emitter agent {}'.format(a))
    plt.legend()
    plt.show()


def plot_simultation_results(dir, num_generation, genotype_index, force_random=False, invert_sim_type=False):

    evo, sim, data_record = simulation.obtain_trial_data(
        dir, num_generation, genotype_index, 
        force_random, invert_sim_type
    )

    plot_performances(evo)
    plot_activity_scatter(data_record)
    # plot_angles(data_record)
    plot_norm_x(data_record)
    plot_behavior(data_record)
    plot_activity(data_record)    
    plot_signal(data_record)
    plot_inputs(data_record)
    plot_motor_output(data_record)
    plot_emitters(data_record)
    plot_motor_output(data_record)
    # utils.save_numpy_data(data_record['brain_output'], 'data/tmp_brains.json')    


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
    
    random_seed = utils.random_int()

    data_record = {}
    perf = sim.compute_performance(random_genotype, random_seed, data_record)
    print("random perf: {}".format(perf))        

if __name__ == "__main__":
    import argparse

    # default_dir = 'data/histo_entropy/dyadic_exp_096'
    default_dir = 'data/transfer_entropy_test/tmp1'    # 5?, 10, 18(max)
    default_gen = 20
    default_index = 0
    default_random = False
    default_invert = False

    # plot_simultation_results()
    # plot_random_simulation_results()

    parser = argparse.ArgumentParser(
        description='Plot results'
    )

    parser.add_argument('--dir', type=str, default=default_dir, help='Directory path')     
    parser.add_argument('--gen', type=int, default=default_gen, help='number of genration')    
    parser.add_argument('--index', type=int, default=default_index, help='Index of agent in population')
    parser.add_argument('--random', type=bool, default=default_random, help='Whether to randomize result')
    parser.add_argument('--invert', type=bool, default=default_invert, help='Whther to invert the simulation type (histo <-> transfer)')

    args = parser.parse_args()
    plot_simultation_results(args.dir, args.gen, args.index, args.random, args.invert)
