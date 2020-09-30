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


def plot_behavior(data_record, to_plot='all'):
    agent_pos = [
        [x[0].transpose(), x[1].transpose()]
        for x in data_record['agent_pos']
    ]
    num_trials = len(agent_pos)
    # print("agent_pos shape: {}".format(agent_pos[0].shape))
    num_cols = num_trials
    if to_plot == 'all':
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
    else:
        fig = plt.figure(figsize=(6, 5))
        fig.suptitle("Agents behavior")
        for a in range(2):
            plt.scatter(agent_pos[int(to_plot) - 1][a][0][0],
                        agent_pos[int(to_plot) - 1][a][1][0], label='Tracker start',
                        color='orange')
            plt.scatter(agent_pos[int(to_plot) - 1][a][0][-1],
                        agent_pos[int(to_plot) - 1][a][1][-1], label='Tracker stop')
        for a in range(2):
            plt.plot(agent_pos[int(to_plot) - 1][a][0],
                     agent_pos[int(to_plot) - 1][a][1],
                     label='Tracker position')
        plt.xlim((-4, 31))
        plt.ylim((-15, 20))
    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper right')
    # plt.legend()
    # plt.show()
    # plt.savefig('plots/shannon_beh.eps', format='eps')
    plt.savefig('plots/transfer_beh.eps', format='eps')


def plot_activity(data_record, to_plot):
    num_trials = len(data_record['agent_pos'])
    num_cols = num_trials
    if to_plot == 'all':
        fig = plt.figure(figsize=(10, 6))
        fig.suptitle("Brain activity")
        for t in range(num_trials):
            for a in range(2):
                ax = fig.add_subplot(2, num_cols, (a * num_trials) + t + 1)
                brain_output = data_record['brain_output'][t][a]
                ax.plot(brain_output[:, 0], label='Output of n1 agent {}'.format(a))
                ax.plot(brain_output[:, 1], label='Output of n2 agent {}'.format(a))
            # data_record['brain_state'][t]
            # data_record['derivatives'][t]
    else:
        # fig = plt.figure(figsize=(10, 6))
        # fig.suptitle("Brain activity")
        # for a in range(2):
        #     brain_output = data_record['brain_output'][int(to_plot)][a]
        #     ax = fig.add_subplot(1, 2, a + 1)
        #     ax.set_title('Agent {}'.format(a))
        #     ax.plot(brain_output[1000:1500, 0], label='N1 output')
        #     ax.plot(brain_output[1000:1500, 1], label='N2 output')
        #     handles, labels = ax.get_legend_handles_labels()
        # transfer entropy
        # fig = plt.figure()
        # ax1 = fig.add_axes([0.12, 0.7, 0.8, 0.15],
        #                    xticklabels=[], xticks=[],
        #                    ylim=(0.653, 0.655))
        # ax2 = fig.add_axes([0.12, 0.5, 0.8, 0.15],
        #                    xticklabels=[], xticks=[],
        #                    ylim=(0.61, 0.6125))
        # ax3 = fig.add_axes([0.12, 0.3, 0.8, 0.15],
        #                    xticklabels=[], xticks=[],
        #                    ylim=(0.9976, 0.9978))
        # ax4 = fig.add_axes([0.12, 0.1, 0.8, 0.15],
        #                    # xticks=np.arange(1000, 1500, step=100),
        #                    xticklabels=np.arange(900, 1600, step=100),
        #                    ylim=(0.00011, 0.00014))
        #
        # brain_output1 = data_record['brain_output'][int(to_plot)][0]
        # brain_output2 = data_record['brain_output'][int(to_plot)][1]
        # ax1.plot(brain_output1[1000:1500, 0])
        # ax1.set_title('Agent 1, N1 output', y=0.9)
        # ax1.spines["top"].set_visible(False)
        # ax1.spines["right"].set_visible(False)
        # ax1.spines["bottom"].set_visible(False)
        #
        # ax2.plot(brain_output1[1000:1500, 1])
        # ax2.set_title('Agent 1, N2 output', y=0.9)
        # ax2.spines["top"].set_visible(False)
        # ax2.spines["right"].set_visible(False)
        # ax2.spines["bottom"].set_visible(False)
        #
        # ax3.plot(brain_output2[1000:1500, 0])
        # ax3.set_title('Agent 2, N1 output', y=0.9)
        # ax3.spines["top"].set_visible(False)
        # ax3.spines["right"].set_visible(False)
        # ax3.spines["bottom"].set_visible(False)
        #
        # ax4.plot(brain_output2[1000:1500, 1])
        # ax4.set_title('Agent 2, N2 output', y=0.9)
        # ax4.spines["top"].set_visible(False)
        # ax4.spines["right"].set_visible(False)
        # ax4.spines["bottom"].set_visible(True)
        # # handles, labels = ax2.get_legend_handles_labels()
        # # fig.legend(handles, labels, loc='upper right')

        # shannon entropy
        fig = plt.figure()
        ax1 = fig.add_axes([0.12, 0.7, 0.8, 0.15],
                           xticklabels=[], xticks=[],
                           ylim=(-0.2, 1.2))
        ax2 = fig.add_axes([0.12, 0.5, 0.8, 0.15],
                           xticklabels=[], xticks=[],
                           ylim=(-0.2, 1.2))
        ax3 = fig.add_axes([0.12, 0.3, 0.8, 0.15],
                           xticklabels=[], xticks=[],
                           ylim=(-0.2, 1.2))
        ax4 = fig.add_axes([0.12, 0.1, 0.8, 0.15],
                           ylim=(-0.2, 1.2))

        brain_output1 = data_record['brain_output'][int(to_plot)][0]
        brain_output2 = data_record['brain_output'][int(to_plot)][1]
        ax1.plot(brain_output1[1000:1500, 0])
        ax1.set_title('Agent 1, N1 output', y=0.9)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.spines["bottom"].set_visible(False)

        ax2.plot(brain_output1[1000:1500, 1])
        ax2.set_title('Agent 1, N2 output', y=0.9)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.spines["bottom"].set_visible(False)

        ax3.plot(brain_output2[1000:1500, 0])
        ax3.set_title('Agent 2, N1 output', y=0.9)
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        ax3.spines["bottom"].set_visible(False)

        ax4.plot(brain_output2[1000:1500, 1])
        ax4.set_title('Agent 2, N2 output', y=0.9)
        ax4.spines["top"].set_visible(False)
        ax4.spines["right"].set_visible(False)
        ax4.spines["bottom"].set_visible(True)

    # plt.show()
    plt.savefig('plots/shannon_activity.eps', format='eps')
    # plt.savefig('plots/transfer_activity.eps', format='eps')


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

    # plot_performances(evo)
    # plot_behavior(data_record, '1')
    plot_activity(data_record, '1')
    # plot_inputs(data_record)
    # plot_motor_output(data_record)
    # plot_emitters(data_record)
    # utils.save_numpy_data(data_record['brain_output'], 'data/tmp_brains.json')    


def plot_random_simulation_results():
    genotype_structure = gen_structure.DEFAULT_GEN_STRUCTURE
    gen_size = gen_structure.get_genotype_size(genotype_structure)
    random_genotype = Evolution.get_random_genotype(RandomState(None),
                                                    gen_size * 2)  # pairs of agents in a single genotype

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
    default_dir = 'data/transfer_entropy_test/tmp1'  # 5?, 10, 18(max)
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
    parser.add_argument('--invert', type=bool, default=default_invert,
                        help='Whether to invert the simulation type (histo <-> transfer)')

    args = parser.parse_args()
    plot_simultation_results(args.dir, args.gen, args.index, args.random, args.invert)
