"""
TODO: Missing module docstring
"""

import os
import numpy as np
from numpy import pi as pi
import pyevolver.utils
from dyadic_interaction.agent_body import AgentBody
from dyadic_interaction.agent_network import AgentNetwork
from dyadic_interaction import gen_structure
from dyadic_interaction.neural_entropy import get_norm_entropy
from dataclasses import dataclass, field, asdict, fields
from typing import Dict, Tuple, List
import json
from pyevolver.json_numpy import NumpyListJsonEncoder
from pyevolver.timing import Timing
from pyevolver import utils
from numpy.random import RandomState
from joblib import Parallel, delayed
import multiprocessing


@dataclass
class Simulation:
    genotype_structure: Dict = field(default_factory=lambda:gen_structure.DEFAULT_GEN_STRUCTURE)
    num_brain_neurons: int = None  # initialized in __post_init__
    agent_body_radius: int = 4
    agents_pair_initial_distance: int = 20
    agent_sensors_divergence_angle: float = np.radians(45)  # angle between sensors and axes of symmetry
    brain_step_size: float = 0.1
    num_trials: int = 4 # hard coded
    trial_duration: int = 200
    num_cores: int = 1
    timeit: bool = False

    def __post_init__(self):        

        self.num_brain_neurons = gen_structure.get_num_brain_neurons(self.genotype_structure)
        self.num_data_points = int(self.trial_duration / self.brain_step_size)

        self.init_agents_pair()
        self.set_initial_positions_angles()

        self.timing = Timing(self.timeit)


    def init_agents_pair(self):
        self.agents_pair_net = []
        self.agents_pair_body = []
        for _ in range(2):
            self.agents_pair_net.append(
                AgentNetwork(
                    self.num_brain_neurons,
                    self.brain_step_size,
                    self.genotype_structure,
                )
            )
            self.agents_pair_body.append(
                AgentBody(
                    self.agent_body_radius,
                    self.agent_sensors_divergence_angle,
                    timeit = self.timeit
                )
            )

    def set_initial_positions_angles(self, random_state=None):
        
        if random_state:
            self.agents_pair_start_angle_trials = pi * random_state.uniform(0, 2, (self.num_trials,2))
        else:            
            # first agent always points right
            # second agent at points right, up, left, down in each trial respectively
            self.agents_pair_start_angle_trials = [
                [0., 0.],
                [0., pi/2],
                [0., pi],
                [0., 3*pi/2],
            ]

        # first agent positioned at (0,0)
        # second agent 20 units away from first, along its facing direction 
        # (right, up, left, down) if not random
        self.agents_pair_start_pos_trials = [
            [
                np.array([0.,0.]), 
                self.agents_pair_initial_distance * \
                    np.array(
                        [
                            np.cos(self.agents_pair_start_angle_trials[i][1]),
                            np.sin(self.agents_pair_start_angle_trials[i][1])
                        ]
                    )
            ]
            for i in range(self.num_trials)
        ]

    def set_agents_pos_angle(self, trial_index):
        for a in range(2):
            agent_pos = self.agents_pair_start_pos_trials[trial_index][a]
            agent_angle = self.agents_pair_start_angle_trials[trial_index][a]
            self.agents_pair_body[a].set_position_and_angle(agent_pos, agent_angle)

    def save_to_file(self, file_path):
        with open(file_path, 'w') as f_out:
            obj_dict = asdict(self)
            json.dump(obj_dict, f_out, indent=3, cls=NumpyListJsonEncoder)

    @staticmethod
    def load_from_file(file_path, **kwargs):
        with open(file_path) as f_in:
            obj_dict = json.load(f_in)

        if kwargs:
            obj_dict.update(kwargs)

        sim = Simulation(**obj_dict)
        gen_structure.process_genotype_structure(sim.genotype_structure)
        return sim


    def compute_performance(self, genotypes_pair, data_record=None):
        '''
        Main function to compute performace
        
        :param np.ndarray genotypes_pair: sequence with two genotypes (one after the other)
        '''

        tim = self.timing.init_tictoc()

        genotypes_split = np.array_split(genotypes_pair, 2)

        for a in range(2):
            self.agents_pair_net[a].genotype_to_phenotype(genotypes_split[a])
            
        self.timing.add_time('SIM_1_genotype_to_phenotype', tim)

        # initialize agents brain output for computing entropy
        # data from all trials are put together to make a single histogram
        agents_pair_brain_output_trials = [np.zeros((self.num_data_points * self.num_trials, 2)) for _ in range(2)]

        if data_record is not None:            
            data_record['agent_pos'] = [[None,None] for _ in range(self.num_trials)]
            data_record['agent_angle'] = [[None,None] for _ in range(self.num_trials)]
            data_record['sensors_input'] = [[None,None] for _ in range(self.num_trials)]
            data_record['brain_input'] = [[None,None] for _ in range(self.num_trials)]
            data_record['brain_state'] = [[None,None] for _ in range(self.num_trials)]
            data_record['derivatives'] = [[None,None] for _ in range(self.num_trials)]
            data_record['brain_output'] = [[None,None] for _ in range(self.num_trials)]
            data_record['wheels'] = [[None,None] for _ in range(self.num_trials)]
            data_record['emitter'] = [[None,None] for _ in range(self.num_trials)]

        self.timing.add_time('SIM_2_init_data', tim)

        data_point_i = 0        

        # do experiment
        for t in range(self.num_trials):

            prev_delta_xy_agents = [np.array([0.,0.]), np.array([0.,0.])]
            prev_angle_agents = [None, None]

            self.set_agents_pos_angle(t)
            
            for a in range(2):
                # set initial state in specified range and compute output
                self.agents_pair_net[a].brain.states = np.array([0., 0.]) # stats initialized with zeros
                self.timing.add_time('SIM_3_reinitialize_states', tim)
            
                # compute output
                self.agents_pair_net[a].brain.compute_output()
                self.timing.add_time('SIM_4_randomize_statess_and_compute_output', tim)

                if data_record is not None:                    
                    data_record['agent_pos'][t][a] = np.zeros((self.num_data_points, 2))
                    data_record['agent_angle'][t][a] = np.zeros(self.num_data_points)
                    data_record['sensors_input'][t][a] = np.zeros((self.num_data_points, 2))
                    data_record['brain_input'][t][a] = np.zeros((self.num_data_points, self.num_brain_neurons))
                    data_record['brain_state'][t][a] = np.zeros((self.num_data_points, self.num_brain_neurons))
                    data_record['derivatives'][t][a] = np.zeros((self.num_data_points, self.num_brain_neurons))
                    data_record['brain_output'][t][a] = np.zeros((self.num_data_points, self.num_brain_neurons))
                    data_record['wheels'][t][a] = np.zeros((self.num_data_points, 2))
                    data_record['emitter'][t][a] = np.zeros(self.num_data_points)
                self.timing.add_time('SIM_5_init_trial_data', tim)

            for i in range(self.num_data_points):

                # 1) Add random noise to emitter position
                # TODO: check if this is necessary
                
                for a in range(2):

                    agent_net = self.agents_pair_net[a]
                    agent_body = self.agents_pair_body[a]

                    # 2) Agent sees
                    b = 1-a
                    sensor_inputs = agent_body.get_sensor_inputs(
                        self.agents_pair_body[b].position,
                        self.agents_pair_net[b].motors_outputs[1] # index 1:   EMITTER
                    )
                    self.timing.add_time('SIM_6a_get_sensor_input', tim)

                    # 3-dim vector: strength of emitter from the two sensors                
                    agent_net.compute_brain_input(sensor_inputs)
                    self.timing.add_time('SIM_6b_compute_brain_input', tim)

                    # 4) Update agent's neural system
                    agent_net.brain.euler_step()  # this sets agent.brain.output (2-dim vector)
                    self.timing.add_time('SIM_7_euler_step', tim)

                    # 5) Agent updates wheels displacement
                    motor_outputs = agent_net.compute_motor_outputs()
                    agent_body.wheels = np.take(motor_outputs, [0,2]) # index 0,2: MOTORS
                    emitter_strength = motor_outputs[1]               # index 1:   EMITTER
                    self.timing.add_time('SIM_8_compute_motor_outputs', tim)

                    # 6) Save data
                    agents_pair_brain_output_trials[a][data_point_i,:] = agent_net.brain.output
                    if data_record is not None:
                        data_record['agent_pos'][t][a][i] = agent_body.position
                        data_record['agent_angle'][t][a][i] = agent_body.angle
                        data_record['sensors_input'][t][a][i] = sensor_inputs
                        data_record['brain_input'][t][a][i] = agent_net.brain.input
                        data_record['brain_state'][t][a][i] = agent_net.brain.states
                        data_record['derivatives'][t][a][i] = agent_net.brain.dy_dt
                        data_record['brain_output'][t][a][i] = agent_net.brain.output
                        data_record['wheels'][t][a][i] = agent_body.wheels
                        data_record['emitter'][t][a][i] = emitter_strength
                    self.timing.add_time('SIM_9_save_data', tim)
                
                delta_xy_agents = [None, None]
                angle_agents = [None, None]
                for a in range(2):
                    # 7) Move one step
                    # TODO: check if the agents didn't go too far from one another
                    b = 1-a
                    delta_xy_agents[a], angle_agents[a] =  self.agents_pair_body[a].move_one_step(
                        prev_delta_xy_agents[b],
                        prev_angle_agents[b]
                    )                
                prev_delta_xy_agents = delta_xy_agents
                prev_angle_agents = angle_agents

                data_point_i += 1

                self.timing.add_time('SIM_10_move_one_step', tim)


        assert data_point_i == self.num_data_points * self.num_trials

        # calculate performance        
        # TODO
        performance_agent_A, performance_agent_B = (
            get_norm_entropy(agents_pair_brain_output_trials[a])
            for a in range(2)
        )
        avg_performance = np.mean([performance_agent_A, performance_agent_B])

        self.timing.add_time('SIM_11_compute_performace', tim)

        return avg_performance

    '''
    POPULATION EVALUATION FUNCTION
    '''
    def evaluate(self, population, random_seeds):        
        population_size = len(population)
        assert population_size == len(random_seeds)
        # we are not using random seeeds because behaviour of agents is fully deterministic (with not randomality)
        if self.num_cores > 1:
            # run parallel job
            assert population_size % self.num_cores == 0, \
                "Population size ({}) must be a multiple of num_cores ({})".format(
                    population_size, self.num_cores)

            sim_array = [Simulation(**asdict(self)) for _ in range(self.num_cores)]
            performances = Parallel(n_jobs=self.num_cores)( # prefer="threads" does not work
                delayed(sim_array[i%self.num_cores].compute_performance)(genotypes_pair) \
                for i, (genotypes_pair) in enumerate(population)
            )

        else:
            performances = [
                self.compute_performance(genotypes_pair)
                for genotypes_pair in population
            ]
        return performances


