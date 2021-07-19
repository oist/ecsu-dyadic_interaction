"""
TODO: Missing module docstring
"""

import os
import numpy as np
from numpy import pi as pi
from dyadic_interaction.agent_body import AgentBody
from dyadic_interaction.agent_network import AgentNetwork
from dyadic_interaction import gen_structure
from dyadic_interaction.shannon_entropy import get_shannon_entropy_dd_simplified, get_shannon_entropy_1d
from dyadic_interaction.transfer_entropy import get_transfer_entropy
from dyadic_interaction.entropy.entropy import _numba_sampen
from dyadic_interaction.sample_entropy import DEFAULT_SAMPLE_ENTROPY_DISTANCE_STD, DEFAULT_SAMPLE_ENTROPY_ANGLE_STD, DEFAULT_SAMPLE_ENTROPY_NEURAL_STD
from dyadic_interaction import utils
from dyadic_interaction.utils import assert_string_in_values
from dataclasses import dataclass, field, asdict, fields
from typing import Dict, Tuple, List
import json
from pyevolver.json_numpy import NumpyListJsonEncoder
from pyevolver.timing import Timing
from numpy.random import RandomState
from numpy.linalg import norm
from joblib import Parallel, delayed
from copy import deepcopy

@dataclass
class Simulation:
    num_random_pairings: int = 0 
        # N==0 -> (agents are evolved in pairs: a genotype contains a pair of agents) 
        # N>0  -> each agent will go though a simulation with N other agents (randomly chosen)
    entropy_type: str = 'shannon-dd' # 'shannon-1d', 'shannon-dd', 'transfer', 'sample'
    entropy_target_value: str = 'neural' # 'neural', 'distance', 'angle'
    concatenate: bool = True # whether to concatenate values in entropy_target_value
    isolation: bool = False # whether to run simulation on a single agent (as if second agent does not exits)
    genotype_structure: Dict = field(default_factory=lambda:gen_structure.DEFAULT_GEN_STRUCTURE(2))
    num_brain_neurons: int = None  # initialized in __post_init__
    collision_type: str = 'overlapping' # 'none', 'overlapping', 'edge'
    agent_body_radius: int = 4
    agents_pair_initial_distance: int = 20
    agent_sensors_divergence_angle: float = np.radians(45)  # angle between sensors and axes of symmetry
    brain_step_size: float = 0.1
    num_trials: int = 4 # hard coded
    trial_duration: int = 200
    num_cores: int = 1
    data_noise_level: float = 1e-8
    timeit: bool = False

    def __post_init__(self):          

        self.num_brain_neurons = gen_structure.get_num_brain_neurons(self.genotype_structure)
        self.num_data_points = int(self.trial_duration / self.brain_step_size)

        self.init_agents_pair()
        self.set_initial_positions_angles()

        if self.isolation:
            # if we run agents in isolation we want to ignore collisions
            self.collision_type = 'none'

        self.timing = Timing(self.timeit)        

        self.__check_params__()

    def __check_params__(self):
        assert self.num_random_pairings >= 0, \
            "Number of pairing must be >= 0  (0 if a genotype already contains a pair of agents)"

        assert_string_in_values(self.collision_type, 'collision_type', ['none', 'overlapping', 'edge'])
        assert_string_in_values(self.entropy_type, 'entropy_type', ['shannon-1d', 'shannon-dd', 'transfer', 'sample'])
        assert_string_in_values(self.entropy_target_value, 'entropy_target_value', ['neural', 'distance', 'angle'])

        if self.entropy_type in ['shannon-1d', 'shannon-dd']:
            accepted_entropy_target_values = ['neural', 'distance', 'angle']
            assert self.entropy_target_value in accepted_entropy_target_values, \
                "Shannon entropy currently works only when entropy_target_value in {}".format(accepted_entropy_target_values)

        if self.entropy_type == 'transfer':
            assert self.entropy_target_value == 'neural' and self.num_brain_neurons == 2, \
                'Transfer entropy currently works only on two dimensional data (i.e., 2 neural outputs per agent)'

        if self.entropy_target_value == 'angle':
            assert self.entropy_type in ['shannon-1d','sample'], \
                "entropy on angle works only for entropy_type in ['shannon-1d','sample']"

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
                    collision_type=self.collision_type,
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
        if random_state:
            # reinitialized the angle because it was used for positioning
            # we don't want the second agent to necessarily face outwards
            self.agents_pair_start_angle_trials = pi * random_state.uniform(0, 2, (self.num_trials,2))

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
        gen_structure.check_genotype_structure(sim.genotype_structure)
        return sim        

    def set_agents_genotype_phenotype(self):
        '''
        Split genotype and set phenotype of the two agents
        :param np.ndarray genotypes_pair: sequence with two genotypes (one after the other)
        '''
        
        phenotypes = [None,None]
        if self.num_random_pairings == 0:
            genotypes_pair = self.genotype_population[self.genotype_index]
            genotypes_split = np.array_split(genotypes_pair, 2)                
        else:
            genotypes_split = [
                self.genotype_population[self.genotype_index], 
                self.genotype_population[self.rand_agent_indexes[self.sim_index]], 
            ]
        if self.data_record is not None:
            self.data_record['genotypes'] = genotypes_split
            phenotypes = [{},{}]
            self.data_record['phenotypes'] = phenotypes
        for a in range(2):
            self.agents_pair_net[a].genotype_to_phenotype(
                genotypes_split[a], phenotype_dict=phenotypes[a])

    def init_values_for_computing_entropy(self):

        if self.entropy_target_value == 'neural':
            # initialize agents brain output of all trial for computing entropy
            # list of list (4 trials x 2 agents) each containing array (num_data_points,num_brain_neurons)
            self.values_for_computing_entropy = [
                [
                    np.zeros((self.num_data_points, self.num_brain_neurons)) 
                    for _ in range(2)
                ] for _ in range(self.num_trials)
            ]
        elif self.entropy_target_value == 'distance':
            # distance (1-d data) per trial            
            # entropy is computed based on distances
            # 4 list (one per trial) with the agent distances
            self.values_for_computing_entropy = [
                np.zeros((self.num_data_points,1))
                for _ in range(self.num_trials)
            ]
        else:
            # angle: (1-d data) per trial per agent
            assert self.entropy_target_value == 'angle'
            self.values_for_computing_entropy = [
                [
                    np.zeros((self.num_data_points,1))
                    for _ in range(2)
                ] for _ in range(self.num_trials)
            ]

    def init_data_record(self):
        if self.data_record is  None:                       
            return            
        self.data_record['position'] = [[None,None] for _ in range(self.num_trials)]
        self.data_record['distance'] = [None for _ in range(self.num_trials)]
        self.data_record['angle'] = [[None,None] for _ in range(self.num_trials)]
        self.data_record['collision'] = [[None,None] for _ in range(self.num_trials)]
        self.data_record['delta_xy'] = [[None,None] for _ in range(self.num_trials)]
        self.data_record['signal_strength'] = [[None,None] for _ in range(self.num_trials)]
        self.data_record['brain_input'] = [[None,None] for _ in range(self.num_trials)]
        self.data_record['brain_state'] = [[None,None] for _ in range(self.num_trials)]
        self.data_record['derivatives'] = [[None,None] for _ in range(self.num_trials)]
        self.data_record['brain_output'] = [[None,None] for _ in range(self.num_trials)]
        self.data_record['wheels'] = [[None,None] for _ in range(self.num_trials)]
        self.data_record['emitter'] = [[None,None] for _ in range(self.num_trials)]                
        self.timing.add_time('SIM_init_data', self.tim)

    def init_data_record_trial(self, t):
        if self.data_record is None:            
            return
        self.data_record['distance'][t] = np.zeros(self.num_data_points)
        for a in range(2):
            if self.ghost_index == a:
                # copy all ghost agent's values from original_data_record
                if t == 0:
                    for k in self.data_record:
                        self.data_record[k] = deepcopy(self.original_data_record[k])
            else:
                self.data_record['position'][t][a] = np.zeros((self.num_data_points, 2))
                self.data_record['angle'][t][a] = np.zeros(self.num_data_points)
                self.data_record['collision'][t][a] = np.zeros(self.num_data_points)
                self.data_record['delta_xy'][t][a] = np.zeros((self.num_data_points, 2))
                self.data_record['signal_strength'][t][a] = np.zeros((self.num_data_points, 2))
                self.data_record['brain_input'][t][a] = np.zeros((self.num_data_points, self.num_brain_neurons))
                self.data_record['brain_state'][t][a] = np.zeros((self.num_data_points, self.num_brain_neurons))
                self.data_record['derivatives'][t][a] = np.zeros((self.num_data_points, self.num_brain_neurons))
                self.data_record['brain_output'][t][a] = np.zeros((self.num_data_points, self.num_brain_neurons))
                self.data_record['wheels'][t][a] = np.zeros((self.num_data_points, 2))
                self.data_record['emitter'][t][a] = np.zeros(self.num_data_points)
        self.timing.add_time('SIM_init_trial_data', self.tim)            

    def save_data_record(self, t, i):
        if self.data_record is None: 
            return
        self.data_record['distance'][t][i] = self.get_agents_distance()
        for a in range(2):    
            if self.ghost_index == a:                    
                continue # do not save data for ghost: already saved in init_data_trial
            agent_net = self.agents_pair_net[a]
            agent_body = self.agents_pair_body[a]
            self.data_record['position'][t][a][i] = agent_body.position
            self.data_record['angle'][t][a][i] = agent_body.angle
            self.data_record['collision'][t][a][i] = 1 if agent_body.flag_collision else 0
            self.data_record['delta_xy'][t][a][i] = self.prev_delta_xy_agents[a]                                        
            self.data_record['wheels'][t][a][i] = agent_body.wheels
            self.data_record['emitter'][t][a][i] = self.emitter_agents[a]
            self.data_record['signal_strength'][t][a][i] = self.signal_strength_agents[a]
            self.data_record['brain_input'][t][a][i] = agent_net.brain.input                    
            self.data_record['brain_state'][t][a][i] = agent_net.brain.states
            self.data_record['derivatives'][t][a][i] = agent_net.brain.dy_dt
            self.data_record['brain_output'][t][a][i] = agent_net.brain.output
        self.timing.add_time('SIM_save_data', self.tim)                            

    def compute_signal_strength_agents(self):
        for a in [x for x in range(2) if x != self.ghost_index]:    
            if self.isolation and a==1:
                self.signal_strength_agents[a] = 0
            else:
                b = 1 - a
                # signal_strength = np.array([0.,0.])  # if we want to mimic zero signal strength
                self.signal_strength_agents[a] = self.agents_pair_body[a].get_signal_strength(
                    self.agents_pair_body[b].position,
                    self.emitter_agents[b]
                )
        self.timing.add_time('SIM_get_signal_strength', self.tim)

    def update_wheels_emitter_agents(self, t,i):
        for a in range(2):
            if a == self.ghost_index:
                self.emitter_agents[a] = self.original_data_record['emitter'][t][a][i]
            if self.isolation and a==1:
                self.emitter_agents[a] = 0
            else:
                motor_outputs = self.agents_pair_net[a].compute_motor_outputs()
                self.agents_pair_body[a].wheels = np.take(motor_outputs, [0,2]) # index 0,2: MOTORS  
                self.emitter_agents[a] = motor_outputs[1] # index 1: EMITTER
        self.timing.add_time('SIM_compute_motors_emitter', self.tim)

    def get_agents_distance(self):
        return self.agents_pair_body[0].get_dist_centers(self.agents_pair_body[1].position)

    def store_values_for_entropy(self, t,i):
        if self.entropy_target_value == 'neural': #neural outputs 
            for a in [x for x in range(2) if x != self.ghost_index]:
                self.values_for_computing_entropy[t][a][i] = self.agents_pair_net[a].brain.output  
        elif self.entropy_target_value == 'angle': # angle
            for a in [x for x in range(2) if x != self.ghost_index]:
                self.values_for_computing_entropy[t][a][i] = self.agents_pair_body[a].angle
        else: # distance
            self.values_for_computing_entropy[t][i] = self.get_agents_distance()
                                
    def prepare_agents_for_trial(self, t):
        for a in range(2):
            agent_net = self.agents_pair_net[a]
            agent_body = self.agents_pair_body[a]
            # reset params that are due to change during the experiment
            agent_body.init_params(
                wheels = np.zeros(2),
                flag_collision = False
            )
            # set initial states to zeros
            agent_net.init_params(
                brain_states = np.zeros(self.num_brain_neurons),
            )
            agent_pos = np.copy(self.agents_pair_start_pos_trials[t][a])
            agent_angle = self.agents_pair_start_angle_trials[t][a]
            agent_body.set_position_and_angle(agent_pos, agent_angle)
            # compute output
            agent_net.brain.compute_output()        
        # compute motor outpus    
        self.update_wheels_emitter_agents(t, 0)                          
        # compute signal streng

        self.store_values_for_entropy(t,0) #

        self.timing.add_time('SIM_prepare_agents_for_trials', self.tim)     

    def compute_brain_input_agents(self):                
        for a in [x for x in range(2) if x != self.ghost_index]:    
            if self.isolation and a==1:
                continue
            self.agents_pair_net[a].compute_brain_input(self.signal_strength_agents[a])
        self.timing.add_time('SIM_compute_brain_input', self.tim)

    def compute_brain_euler_step_agents(self):          
        for a in [x for x in range(2) if x != self.ghost_index]:              
            if self.isolation and a==1:
                continue
            self.agents_pair_net[a].brain.euler_step()  # this sets agent.brain.output (2-dim vector)
            self.timing.add_time('SIM_euler_step', self.tim)

    def move_one_step_agents(self, t, i):
        delta_xy_agents = [None, None]
        angle_agents = [None, None]
        for a in range(2):                
            if self.ghost_index == a:
                # for ghost agent we need to retrieve position, delta_xy, and angle from data
                self.agents_pair_body[a].position = self.original_data_record['position'][t][a][i] 
                delta_xy_agents[a] = self.original_data_record['delta_xy'][t][a][i]                        
                angle_agents[a] = self.original_data_record['angle'][t][a][i]
            else:                                                
                # TODO: check if the agents didn't go too far from one another
                b = 1 - a
                delta_xy_agents[a], angle_agents[a] =  self.agents_pair_body[a].move_one_step(
                    self.prev_delta_xy_agents[b],
                    self.prev_angle_agents[b]
                )                                           
        self.prev_delta_xy_agents = delta_xy_agents
        self.prev_angle_agents = angle_agents
        self.timing.add_time('SIM_move_one_step', self.tim)  

    def compute_performance(self, t):
        performance_agent_AB = []
        if self.entropy_type=='transfer':
            # it only applies to neural_outputs (with 2 neurons)
            # add random noise to data before calculating transfer entropy
            for a in range(2):
                if self.ghost_index == a:
                    continue
                if self.isolation and a==1:
                    continue                    
                
                if self.concatenate:
                    all_values_for_computing_entropy = np.concatenate([
                        self.values_for_computing_entropy[t][a]
                        for t in range(self.num_trials)
                    ])
                else:
                    all_values_for_computing_entropy = self.values_for_computing_entropy[t][a]
                
                all_values_for_computing_entropy = utils.add_noise(
                    all_values_for_computing_entropy, 
                    self.random_state, 
                    noise_level=self.data_noise_level
                )

                # calculate performance        
                # TODO: understand what happens if reciprocal=False
                performance_agent_AB.append(
                    get_transfer_entropy(all_values_for_computing_entropy, binning=True) 
                )

        elif self.entropy_type in ['shannon-1d', 'shannon-dd']:
            # shannon-1d, shannon-dd
            if self.entropy_target_value == 'distance':
                if self.concatenate:
                    all_values_for_computing_entropy = np.concatenate([
                        self.values_for_computing_entropy
                    ])
                else:
                    all_values_for_computing_entropy = self.values_for_computing_entropy[t]
                min_v, max_v= 0., 100.
                performance_agent_AB = [
                    get_shannon_entropy_dd_simplified(
                        all_values_for_computing_entropy, min_v, max_v)
                ]
            if self.entropy_target_value == 'angle':
                # angle (apply modulo angle of 2*pi)
                # min_v, max_v= 0., 2*np.pi
                min_v, max_v= -np.pi/4, np.pi/4
                for a in range(2):
                    if self.ghost_index == a:
                        continue
                    if self.isolation and a==1:
                        continue
                    if self.concatenate:
                        all_values_for_computing_entropy = np.concatenate([
                            self.values_for_computing_entropy[t][a]
                            for t in range(self.num_trials)
                        ])
                    else:
                        all_values_for_computing_entropy = self.values_for_computing_entropy[t][a]
                    # all_values_for_computing_entropy = all_values_for_computing_entropy % 2*np.pi
                    all_values_for_computing_entropy = all_values_for_computing_entropy.flatten()
                    all_values_for_computing_entropy = np.diff(all_values_for_computing_entropy)
                    performance_agent_AB.append(
                        get_shannon_entropy_1d(all_values_for_computing_entropy, min_v, max_v)
                    )
            else: # neural
                min_v, max_v= 0., 1.
                for a in range(2):
                    if self.ghost_index == a:
                        continue
                    if self.isolation and a==1:
                        continue
                    if self.concatenate:
                        all_values_for_computing_entropy = np.concatenate([
                            self.values_for_computing_entropy[t][a]
                            for t in range(self.num_trials)
                        ])
                    else:
                        all_values_for_computing_entropy = self.values_for_computing_entropy[t][a]

                    if self.entropy_type == 'shannon-dd':
                        performance_agent_AB.append(
                            get_shannon_entropy_dd_simplified(all_values_for_computing_entropy, min_v, max_v)
                        )
                    else:
                        # shannon-1d
                        for c in range(self.num_brain_neurons):
                            column_values = all_values_for_computing_entropy[:,c]
                            performance_agent_AB.append(
                                get_shannon_entropy_1d(column_values, min_v, max_v)
                            )            
        else:
            # sample entropy
            # only applies to 1d data
            if self.entropy_target_value == 'neural':
                for a in range(2):
                    if self.ghost_index == a:
                        continue
                    if self.isolation and a==1:
                        continue
                    if self.concatenate:
                        all_values_for_computing_entropy = np.concatenate([
                            self.values_for_computing_entropy[t][a]
                            for t in range(self.num_trials)
                        ])
                    else:
                        all_values_for_computing_entropy = self.values_for_computing_entropy[t][a]

                    for c in range(self.num_brain_neurons):
                        column_values = all_values_for_computing_entropy[:,c]
                        mean = column_values.mean()
                        std = column_values.std()
                        normalize_values = (column_values - mean) / std
                        performance_agent_AB.append(
                            _numba_sampen(normalize_values, order=2, r=(0.2 * DEFAULT_SAMPLE_ENTROPY_NEURAL_STD)) 
                        )        
            elif self.entropy_target_value == 'distance':
                if self.concatenate:
                    all_values_for_computing_entropy = np.concatenate([
                        self.values_for_computing_entropy
                    ])
                else:
                    all_values_for_computing_entropy = self.values_for_computing_entropy[t]                    
                mean = all_values_for_computing_entropy.mean()
                std = all_values_for_computing_entropy.std()
                normalize_values = (all_values_for_computing_entropy - mean) / std
                performance_agent_AB = [
                    _numba_sampen(normalize_values.flatten(), order=2, 
                        r=(0.2 * DEFAULT_SAMPLE_ENTROPY_DISTANCE_STD)) 
                ]
            else: 
                assert self.entropy_target_value == 'angle'
                for a in range(2):
                    if self.ghost_index == a:
                        continue
                    if self.isolation and a==1:
                        continue
                    if self.concatenate:
                        all_values_for_computing_entropy = np.concatenate([
                            self.values_for_computing_entropy[t][a]
                            for t in range(self.num_trials)
                        ])
                    else:
                        all_values_for_computing_entropy = self.values_for_computing_entropy[t][a]
                    all_values_for_computing_entropy = np.diff(all_values_for_computing_entropy)
                    mean = all_values_for_computing_entropy.mean()
                    std = all_values_for_computing_entropy.std()
                    normalize_values = (all_values_for_computing_entropy - mean) / std
                    performance_agent_AB.append(
                        _numba_sampen(normalize_values.flatten(), order=2, r=(0.2 * DEFAULT_SAMPLE_ENTROPY_ANGLE_STD)) 
                    )      
        return performance_agent_AB                                           

    #################
    # MAIN FUNCTION
    #################
    def run_simulation(self, genotype_population=None, genotype_index=None,
        rnd_seed=0, data_record_list=None, ghost_index=None, original_data_record_list=None):
        '''
        Main function to compute shannon/transfer/sample entropy performace        
        '''

        self.tim = self.timing.init_tictoc()

        self.genotype_population = genotype_population
        self.genotype_index = genotype_index
        self.random_state = RandomState(rnd_seed)
        self.rand_agent_indexes = []
        self.ghost_index = ghost_index

        # fill rand_agent_indexes with n indexes i
        while len(self.rand_agent_indexes) != self.num_random_pairings:
            next_rand_index = self.random_state.randint(len(self.genotype_population))
            if next_rand_index != self.genotype_index:
                self.rand_agent_indexes.append(next_rand_index)

        num_simulations = max(1, self.num_random_pairings)        
        sim_performances = []

        for self.sim_index in range(num_simulations):

            self.data_record = None 
            if data_record_list is not None: 
                self.data_record = {}
                data_record_list.append(self.data_record)
            self.original_data_record = None if original_data_record_list is None else original_data_record_list[self.sim_index]
            self.values_for_computing_entropy = [] # initialized in init_values_for_computing_entropy

            if self.genotype_population is not None:            
                self.set_agents_genotype_phenotype()    
                self.timing.add_time('SIM_init_agent_phenotypes', self.tim)    

            trial_performances = []
            self.signal_strength_agents = [None, None]
            self.emitter_agents = [None, None]
            self.prev_delta_xy_agents, self.prev_angle_agents = None, None # pylint: disable=W0612
            self.init_values_for_computing_entropy()

            # INITIALIZE DATA RECORD
            self.init_data_record()        

            # EXPERIMENT START
            for t in range(self.num_trials):

                # SETUP AGENTS FOR TRIAL
                self.prepare_agents_for_trial(t)            
                
                # initialize prev_delta_xy with zeros (zero dispacement)
                self.prev_delta_xy_agents = [np.array([0.,0.]), np.array([0.,0.])]            
                # initialize prev_angle as initial angle of each agent
                self.prev_angle_agents = [self.agents_pair_body[a].angle for a in range(2)]                        
                
                # INIT DATA for TRIAL
                self.init_data_record_trial(t)           

                self.save_data_record(t, 0)

                # TRIAL START
                for i in range(1, self.num_data_points):                

                    # 1) Agent senses strength of emitter from the two sensors
                    self.compute_signal_strength_agents() # deletece dist_centers

                    # 2) compute brain input
                    self.compute_brain_input_agents()

                    # 3) Update agent's neural system
                    self.compute_brain_euler_step_agents()

                    # 4) Agent updates wheels and  emitter
                    self.update_wheels_emitter_agents(t,i)                            

                    # 5) Move one step  agents
                    self.move_one_step_agents(t, i)

                    # 6) Store the values for computing entropy
                    self.store_values_for_entropy(t,i)  # deletece dist_centers

                    self.save_data_record(t, i)             

                # TRIAL END

                if self.concatenate and t!=self.num_trials-1:
                    # do not compute performance until the last trial
                    continue

                performance_agent_AB = self.compute_performance(t)

                if self.num_random_pairings==0:
                    # when agent sare evolved in pairs the 
                    # performance is the mean between the two agents
                    agents_perf = np.mean(performance_agent_AB)
                else:
                    # otherwise it's the performance of the first agent
                    agents_perf = np.mean(performance_agent_AB[0])

                # appending mean performance between two agents in trial_performances
                trial_performances.append(agents_perf)

                self.timing.add_time('SIM_compute_performace', self.tim)

            # SIMULATION END

            # returning mean performances between all trials
            sim_perf = np.mean(trial_performances)
            sim_performances.append(sim_perf)       
            if self.data_record:
                self.data_record['summary'] = {
                    'rand_agent_indexes': self.rand_agent_indexes,
                    'performance_trials': trial_performances,
                    'performance_sim': sim_perf
                }
        
        return np.mean(sim_performances)

    '''
    POPULATION EVALUATION FUNCTION
    '''
    def evaluate(self, population, random_seeds):                
        population_size = len(population)
        assert population_size == len(random_seeds)

        if self.num_cores > 1:
            # run parallel job
            sim_array = [Simulation(**asdict(self)) for _ in range(self.num_cores)]
            performances = Parallel(n_jobs=self.num_cores)( # prefer="threads" does not work
                delayed(sim_array[i%self.num_cores].run_simulation)(population, i, rnd_seed) \
                for i, (_, rnd_seed) in enumerate(zip(population, random_seeds))
            )
        else:
            # single core
            performances = [
                self.run_simulation(population, i, rnd_seed)
                for i, (_, rnd_seed) in enumerate(zip(population, random_seeds))
             ]

        return performances

def run_simulation_from_dir(dir, generation, genotype_idx, 
    random_pos_angle=False, entropy_type=None, entropy_target_value=None,
    concatenate=None, collision_type=None, ghost_index=None, initial_distance=None,
    isolation=False, write_data=None):    
    ''' 
    utitity function to get data from a simulation
    '''
    func_arguments = locals()
    from pyevolver.evolution import Evolution
    evo_files = [f for f in os.listdir(dir) if f.startswith('evo_')]
    assert len(evo_files)>0, "Can't find evo files in dir {}".format(dir)
    file_num_zfill = len(evo_files[0].split('_')[1].split('.')[0])
    if generation is None:
        # assumes last generation
        evo_files = sorted([f for f in os.listdir(dir) if f.startswith('evo')])
        evo_json_filepath = os.path.join(dir, evo_files[-1])
    else:
        generation = str(generation).zfill(file_num_zfill)
        evo_json_filepath = os.path.join(dir, 'evo_{}.json'.format(generation))
    sim_json_filepath = os.path.join(dir, 'simulation.json')    
    sim = Simulation.load_from_file(sim_json_filepath)
    evo = Evolution.load_from_file(evo_json_filepath, folder_path=dir)

    if initial_distance is not None:
        print("Forcing initial distance to: {}".format(initial_distance))
        sim.agents_pair_initial_distance = initial_distance
        sim.set_initial_positions_angles()

    if random_pos_angle:
        print("Randomizing positions and angles")
        random_state = RandomState()
        sim.set_initial_positions_angles(random_state)

    if entropy_type is not None:
        sim.entropy_type = entropy_type
        print("Forcing entropy type: {}".format(sim.entropy_type))

    if entropy_target_value is not None:
        sim.entropy_target_value = entropy_target_value
        print("Forcing entropy target value: {}".format(sim.entropy_target_value))

    if concatenate is not None:
        sim.concatenate = concatenate == 'on'
        print("Forcing concatenation: {}".format(sim.concatenate))

    if collision_type is not None:
        sim.collision_type = collision_type
        sim.init_agents_pair()
        print("Forcing collision_type: {}".format(sim.collision_type))

    if isolation != sim.isolation:
        sim.isolation = isolation
        print("Forcing isolation to: {}".format(isolation))
    
    data_record_list = []
    genotype_idx_unsorted = evo.population_sorted_indexes[genotype_idx]
    random_seed = evo.pop_eval_random_seeds[genotype_idx_unsorted] 

    if ghost_index is not None:
        assert ghost_index in [0,1], 'ghost_index must be 0 or 1'        
        # get original results without ghost condition and no random
        func_arguments['ghost_index'] = None
        func_arguments['random_pos_angle'] = False
        func_arguments['initial_distance'] = None
        func_arguments['write_data'] = None
        _, _, original_data_record_list = run_simulation_from_dir(**func_arguments) 
        perf = sim.run_simulation(evo.population_unsorted, genotype_idx_unsorted, 
            random_seed, data_record_list, ghost_index=ghost_index, 
            original_data_record_list=original_data_record_list)
        print("Overall Performance recomputed (non-ghost agent only): {}".format(perf))
    else:                
        perf = sim.run_simulation(evo.population_unsorted, genotype_idx_unsorted, 
            random_seed, data_record_list)
        print("Overall Performance recomputed: {}".format(perf))

    if write_data:        
        for s, data_record in enumerate(data_record_list,1):
            if len(data_record_list)>1:                
                outdir = os.path.join(dir, 'data' , 'sim_{}'.format(s))
            else:
                outdir = os.path.join(dir, 'data')
            utils.make_dir_if_not_exists(outdir)
            for t in range(sim.num_trials):
                for k,v in data_record.items():
                    if v is dict: 
                        # summary
                        if t==0: # only once                            
                            outfile = os.path.join(outdir, '{}.json'.format(k))
                            utils.save_json_numpy_data(v, outfile)
                    elif len(v)!=sim.num_trials:
                        # genotype/phenotype
                        outfile = os.path.join(outdir, '{}.json'.format(k))
                        utils.save_json_numpy_data(v, outfile)
                    elif len(v[0])==2:
                        # data for each agent
                        for a in range(2):
                            outfile = os.path.join(outdir, '{}_{}_{}.json'.format(k,t+1,a+1))
                            utils.save_json_numpy_data(v[t][a], outfile)
                    else:
                        # single data for both agent (e.g., distance)
                        outfile = os.path.join(outdir, '{}_{}.json'.format(k,t+1))
                        utils.save_json_numpy_data(v[t], outfile)                      

    return evo, sim, data_record_list

def get_argparse():
    import argparse

    parser = argparse.ArgumentParser(
        description='Rerun simulation'
    )

    parser.add_argument('--dir', type=str, help='Directory path')
    parser.add_argument('--generation', type=int, help='number of generation to load')
    parser.add_argument('--genotype_idx', type=int, default=0, help='Index of agent in population to load')    
    parser.add_argument('--random_pos_angle', action='store_true', help='Whether to randomize initial pos and angle')
    parser.add_argument('--entropy_type', type=str, choices=['shannon', 'transfer', 'sample'], default=None, help='Whether to change the entropy_type')
    parser.add_argument('--entropy_target_value', type=str, default=None, help='To change the entropy_target_value')    
    parser.add_argument('--concatenate', choices=['on', 'off'], default=None, help='To change the concatenation')
    parser.add_argument('--collision_type', choices=['none', 'overlapping', 'edge'], default=None, help='To change the type of collison')
    parser.add_argument('--initial_distance', type=int, default=None, help='Initial distance (must be >=0 or else it will be set as in simulation default)')    
    parser.add_argument('--ghost_index', type=int, default=None, help='Ghost index (must be 0 or 1 or else ghost condition will not be enabled)')    
    parser.add_argument('--isolation', action='store_true', help='Whether simulation runs on single agents (as if second agent does not exits) or two agents')
    parser.add_argument('--write_data', action='store_true', help='Whether to output data (same directory as input)')

    return parser

if __name__ == "__main__":
    parser = get_argparse()
    args = parser.parse_args()
    evo, _, data_record_list = run_simulation_from_dir(**vars(args))
