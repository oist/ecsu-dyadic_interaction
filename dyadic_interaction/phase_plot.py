from dyadic_interaction.agent_network import AgentNetwork
from dyadic_interaction.run_from_dir import run_simulation_from_dir
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation


def plot_phase_space_2N(ctrnn_brain, states_series):
    """ Plot the phase portrait
    We'll use matplotlib quiver function, which wants as arguments the grid of x and y coordinates,
    and the derivatives of these coordinates.
    In the plot we see the locations of stable and unstable equilibria,
    and can eyeball the trajectories that the system will take through
    the state space by following the arrows.
    """
    # Define the sample space (plotting ranges)
    ymin = np.amin(states_series)
    ymax = np.amax(states_series)
    # ymin = -10
    # ymax = 10
    y1 = np.linspace(ymin, ymax, 30)
    y2 = np.linspace(ymin, ymax, 30)
    Y1, Y2 = np.meshgrid(y1, y2)
    dim_y = y1.shape[0]

    # calculate the state space derivatives across our sample space
    changes_y1 = np.zeros([dim_y, dim_y])
    changes_y2 = np.zeros([dim_y, dim_y])

    brain_input = np.zeros(ctrnn_brain.num_neurons)

    def compute_derivatives(states):
        return ctrnn_brain.step_size * \
            np.multiply(
                1 / ctrnn_brain.taus,
                - states + np.dot(ctrnn_brain.output, ctrnn_brain.weights) + brain_input
            )

    for i in range(dim_y):
        for j in range(dim_y):
            states = np.array([Y1[i, j], Y2[i, j]])
            dy_dt = compute_derivatives(states)
            changes_y1[i,j], changes_y2[i,j] = dy_dt

    plt.figure(figsize=(10,6))
    plt.quiver(Y1, Y2, changes_y1, changes_y2, color='b', alpha=.75)

    plt.plot(states_series[:, 0], states_series[:, 1], color='r')
    plt.plot(states_series[0][0], states_series[0][1], marker='x', color='orange', zorder=1)
    plt.xlabel('y1', fontsize=14)
    plt.ylabel('y2', fontsize=14)
    plt.title('Phase portrait and a single trajectory', fontsize=16)
    plt.show()

def plot_phase_space_3N(ctrnn_brain, states_series, render_animation=False):
    """ Plot the phase portrait
    We'll use matplotlib quiver function, which wants as arguments the grid of x and y coordinates,
    and the derivatives of these coordinates.
    In the plot we see the locations of stable and unstable equilibria,
    and can eyeball the trajectories that the system will take through
    the state space by following the arrows.
    """
    # Define the sample space (plotting ranges)
    ymin = np.amin(states_series)
    ymax = np.amax(states_series)
    # ymin = -10
    # ymax = 10
    y1 = np.linspace(ymin, ymax, 10)
    y2 = np.linspace(ymin, ymax, 10)
    y3 = np.linspace(ymin, ymax, 10)
    Y1, Y2, Y3 = np.meshgrid(y1, y2, y3)
    
    dim_y = y1.shape[0]

    # calculate the state space derivatives across our sample space
    changes_y1 = np.zeros([dim_y, dim_y, dim_y])
    changes_y2 = np.zeros([dim_y, dim_y, dim_y])
    changes_y3 = np.zeros([dim_y, dim_y, dim_y])

    brain_input = np.zeros(ctrnn_brain.num_neurons)

    def compute_derivatives(states):
        return ctrnn_brain.step_size * \
            np.multiply(
                1 / ctrnn_brain.taus,
                - states + np.dot(ctrnn_brain.output, ctrnn_brain.weights) + brain_input
            )

    for i in range(dim_y):
        for j in range(dim_y):
            for k in range(dim_y):
                states = np.array([Y1[i,j,k], Y2[i,j,k], Y3[i,j,k]])
                dy_dt = compute_derivatives(states)
                changes_y1[i,j,k], changes_y2[i,j,k], changes_y3[i,j,k] = dy_dt

    # ax = plt.figure().add_subplot(projection='3d')
    fig = plt.figure()
    ax = Axes3D(fig)
    # ax.quiver(Y1, Y2, Y3, changes_y1, changes_y2, changes_y3, color='b', alpha=.75)
    
    def init():        
        ax.plot(states_series[:,0], states_series[:,1], states_series[:,2], color='r')
        ax.plot(states_series[0,0], states_series[0,1], states_series[0,2], marker='x', color='r')
        ax.set_title('Phase portrait and a single trajectory', fontsize=16)
        return fig,
    
    def animate(i):
        ax.view_init(elev=10., azim=i)
        return fig,    
    
    if render_animation:
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=360, interval=20, blit=True)

        anim.save('state_plot_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    else:
        init()
        plt.show()

def plot_phase_traj_3N(states_multi_series, render_animation=False):
    # Define the sample space (plotting ranges)
    fig = plt.figure()
    ax = Axes3D(fig)
    # ax.quiver(Y1, Y2, Y3, changes_y1, changes_y2, changes_y3, color='b', alpha=.75)
    
    def init():      
        for i in range(len(states_multi_series)):
            states_series = states_multi_series[i]
            ax.plot(states_series[:,0], states_series[:,1], states_series[:,2])
            # ax.plot(states_series[0,0], states_series[0,1], states_series[0,2], marker='x', color='r')
        ax.set_title('Phase portrait and a single trajectory', fontsize=16)
        return fig,
    
    def animate(i):
        ax.view_init(elev=10., azim=i)
        return fig,    
    
    if render_animation:
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=360, interval=20, blit=True)

        anim.save('state_plot_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    else:
        init()
        plt.show()

def plot_2n(dir, agent_idx, trial_idx):
    evo, sim, data_record_list = run_simulation_from_dir(dir)
    genotype_evo = evo.population[0]
    genotype_sim = np.array(sim.genotype_population[0])
    assert np.all(genotype_evo==genotype_sim)
    agent_ctrnn = sim.agents_pair_net[agent_idx].brain
    
    brain_state_trial = np.array(data_record_list[0]['brain_state'][trial_idx][agent_idx])
    # shape: (2000, 2)

    plot_phase_space_2N(agent_ctrnn, brain_state_trial)    

def plot_3n_iso(dir, agent_idx, trial_idx, render_animation):    
    evo, sim, data_record_list = run_simulation_from_dir(dir)
    genotype_evo = evo.population[0]
    genotype_sim = np.array(sim.genotype_population[0])
    # genotypes_agent = np.array_split(genotype_sim, 2)[0] # paired genotype
    assert np.all(genotype_evo==genotype_sim)
    agent_ctrnn = sim.agents_pair_net[agent_idx].brain
    
    brain_state_trial = np.array(data_record_list[0]['brain_state'][trial_idx][agent_idx])
    num_steps, num_neurons = brain_state_trial.shape # (2000, 3)

    def compute_brain_states(init_brain_states):
        brain_state_trial_recomputed = np.zeros_like(brain_state_trial)
        agent = sim.agents_pair_net[0]
        
        signal_strength = np.zeros(2)
        agent_brain = agent.brain

        agent.init_params(
            brain_states = init_brain_states
        )
        
        for i in range(num_steps):
            brain_state_trial_recomputed[i] = agent_brain.states
            # signal_strength = np.random.random_sample(2) * 5
            agent.compute_brain_input(signal_strength)        
            agent_brain.euler_step()

        return brain_state_trial_recomputed

    states_multi_series = np.array(
        [
            compute_brain_states(init_brain_states)
            for init_brain_states in [
                np.array([0.,0.,0.]),
                np.array([2.,2.,2.]),
                np.array([3.,3.,3.]),
                np.array([4.,4.,4.]),
                np.array([2.,1.,1.]),
            ]
        ]
    )

    # plot_phase_space_3N(agent_ctrnn, brain_state_trial, render_animation)    
    # plot_phase_space_3N(agent_ctrnn, brain_state_trial_recomputed, render_animation)    
    plot_phase_traj_3N(states_multi_series, render_animation)



if __name__ == "__main__":
    plot_3n_iso(
        dir = 'data/frontiers_paper_new/3n_rp-0_shannon-dd_neural_iso_coll-edge/seed_005',
        agent_idx = 0,
        trial_idx = 0,
        render_animation=False
    )
    # plot_3n_iso(
    #     dir = 'data/frontiers_paper_new/3n_rp-0_shannon-dd_neural_social_coll-edge/seed_001',
    #     agent_idx = 0,
    #     trial_idx = 0,
    #     render_animation=False
    # )    
    # plot_2n(
    #     dir = 'data/frontiers_paper_new/2n_rp-0_shannon-dd_neural_social_coll-edge/seed_004',
    #     agent_idx = 0s,
    #     trial_idx = 0
    # )
    # plot_2n(
    #     dir = 'data/frontiers_paper_new/2n_rp-0_shannon-dd_neural_iso_coll-edge/seed_006',
    #     agent_idx = 0,
    #     trial_idx = 0
    # )