"""
TODO: Missing module docstring
"""

import os
import numpy as np
from numpy import pi as pi
from numpy.random import RandomState
import pygame
from dyadic_interaction.agent_body import AgentBody
from dyadic_interaction.simulation import Simulation
from dyadic_interaction import gen_structure
from pyevolver.evolution import Evolution
from pyevolver.json_numpy import NumpyListJsonEncoder
import json

MAX_CANVAS_SIZE = 500
ZOOM_FACTOR = 3
REFRESH_RATE = 60

CANVAS_CENTER = np.array([MAX_CANVAS_SIZE/2, MAX_CANVAS_SIZE/2])
SHIFT_CENTER_TO_FIRST_AGENT = True

black = (0, 0, 0)
white = (255, 255, 255)
sensor_color = (255, 255, 0)
emitter_color = (200, 50, 20)
emitter_radius = 1
sensor_radius = 2 * ZOOM_FACTOR

kb_motor_increment = 0.2

class Visualization:

    def __init__(self, simulation=None):

        self.simulation = simulation or Simulation()

        self.agents_pair_body = self.simulation.agents_pair_body
        self.agents_pair_net = self.simulation.agents_pair_net

        # self.agent_body_radius = simulation.agent_body_radius
        # self.agent_sensors_divergence_angle = simulation.agent_sensors_divergence_angle
        # self.agent_position = None
        # self.agent_angle = None
        # self.emitter_position = None

        # self.agent = AgentBody(
        #     agent_body_radius=self.agent_body_radius,
        #     agent_sensors_divergence_angle=self.agent_sensors_divergence_angle
        # )

        pygame.init()
        self.main_surface = pygame.display.set_mode((MAX_CANVAS_SIZE, MAX_CANVAS_SIZE))


    def draw_agent(self, a_index, center_shift):

        agent = self.agents_pair_body[a_index]

        angent_center_pos = ZOOM_FACTOR * agent.position - ZOOM_FACTOR * center_shift + CANVAS_CENTER

        radius = ZOOM_FACTOR * agent.agent_body_radius

        pygame.draw.circle(self.main_surface, white, angent_center_pos, radius, width=1)

        for sp in agent.get_abs_sorsors_pos():
            sp = ZOOM_FACTOR * sp - ZOOM_FACTOR * center_shift + CANVAS_CENTER
            pygame.draw.circle(self.main_surface, sensor_color, sp, sensor_radius)


    def start_simulation_with_keyboard(self, trial_index):
        running = True

        self.simulation.set_agents_pos_angle(trial_index)

        clock = pygame.time.Clock()

        self.key_motors_increase_velocity = {                        
            pygame.K_q: (0, (-kb_motor_increment, 0)),  # Q - first agent
            pygame.K_w: (0, (+kb_motor_increment, 0)),  # W - first agent
            pygame.K_i: (0, (0, -kb_motor_increment)),  # I - first agent
            pygame.K_o: (0, (0, +kb_motor_increment)),  # O - first agent
            pygame.K_a: (1, (-kb_motor_increment, 0)),  # A - second agent
            pygame.K_s: (1, (+kb_motor_increment, 0)),  # S - second agent
            pygame.K_k: (1, (0, -kb_motor_increment)),  # K - second agent
            pygame.K_l: (1, (0, +kb_motor_increment)),  # L - second agent
        }

        prev_delta_xy_agents = [np.array([0.,0.]), np.array([0.,0.])]
        prev_angle_agents = [None, None]
        sensor_inputs = [None, None]

        while running:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                # elif event.type == pygame.MOUSEBUTTONDOWN:
                #     sensor_inputs = self.agent.get_sensor_inputs(self.emitter_position)
                #     print('sensor inputs: {}'.format(sensor_inputs))
                elif event.type == pygame.KEYDOWN:
                    agent_index, wheel_update = self.key_motors_increase_velocity.get(event.key, (None,None))
                    if wheel_update:
                        agent = self.agents_pair_body[agent_index]
                        agent.wheels += wheel_update
                        print("New wheels values {}: {}".format(agent_index+1, agent.wheels))

            # update emitter position based on mouse
            # mouse_pos = pygame.mouse.get_pos()
            # self.emitter_position = np.array([mouse_pos[0], self.env_height - mouse_pos[1]])  # bottom-up flip

            # reset canvas
            self.main_surface.fill(black)

            center_shift = - self.agents_pair_body[0].position if SHIFT_CENTER_TO_FIRST_AGENT else 0

            # draw agents
            for a in range(2):                
                self.draw_agent(a, center_shift)

            for a in range(2):
                agent = self.agents_pair_body[a]
                b = 1-a
                sensor_inputs[a] = agent.get_sensor_inputs(
                    self.agents_pair_body[b].position,
                    self.agents_pair_net[b].motors_outputs[1] # index 1:   EMITTER
                )

            # final traformations
            self.final_tranform_main_surface()
            pygame.display.update()

            # next step
            delta_xy_agents = [None, None]
            angle_agents = [None, None]
            for a in range(2):
                b = 1-a
                delta_xy_agents[a], angle_agents[a] = self.agents_pair_body[a].move_one_step(
                    prev_delta_xy_agents[b],
                    prev_angle_agents[b]
                )
            prev_delta_xy_agents = delta_xy_agents
            prev_angle_agents = angle_agents

            clock.tick(REFRESH_RATE)
    
    def start_simulation_from_data(self, trial_index, trial_data):
        running = True

        clock = pygame.time.Clock()
        
        duration = self.simulation.num_data_points        
        print("Duration: {}".format(duration))

        agent_pair_pos = trial_data['agent_pos'][trial_index]
        agent_pair_angle = trial_data['agent_angle'][trial_index]

        i = 0

        while running and i<duration:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return

            # reset canvas
            self.main_surface.fill(black)

            center_shift = agent_pair_pos[0][i] if SHIFT_CENTER_TO_FIRST_AGENT else 0

            # draw agents
            for a in range(2):
                agent = self.agents_pair_body[a]                 
                agent.set_position_and_angle(agent_pair_pos[a][i], agent_pair_angle[a][i])
                self.draw_agent(a, center_shift)

            # final traformations
            self.final_tranform_main_surface()
            pygame.display.update()

            clock.tick(REFRESH_RATE)

            i += 1


    def final_tranform_main_surface(self):
        '''
        final transformations:
        - shift coordinates to conventional x=0, y=0 in bottom left corner
        - zoom...
        '''
        self.main_surface.blit(pygame.transform.flip(self.main_surface, False, True), dest=(0, 0))


def draw_line(surface, x1y1, theta, length):
    x2y2 = (
        int(x1y1[0] + length * np.cos(theta)),
        int(x1y1[1] + length * np.sin(theta))
    )
    pygame.draw.line(surface, white, x1y1, x2y2, width=1)

def run_with_keyboard(trial_index):
    vis = Visualization()
    vis.start_simulation_with_keyboard(trial_index)

def run_from_data():
    working_dir = 'data/histo-entropy/dyadic_exp_006'
    generation = '500'
    trial_index = 0
    genotype_index = 0
    sim_json_filepath = os.path.join(working_dir, 'simulation.json')
    evo_json_filepath = os.path.join(working_dir, 'evo_{}.json'.format(generation))
    sim = Simulation.load_from_file(sim_json_filepath)
    evo = Evolution.load_from_file(evo_json_filepath, folder_path=working_dir)
    genotype = evo.population[genotype_index]    

    force_random = False
    if force_random:
        random_seed = np.random.randint(10000)
    else:
        random_seed = evo.pop_eval_random_seed[genotype_index]
    
    trial_data = {}
    perf = sim.compute_performance(genotype, random_seed, trial_data)

    print("perf: {}".format(perf))
    # print("start pos: {}".format(trial_data['agent_pos'][0][0]))

    vis = Visualization(sim)
    vis.start_simulation_from_data(trial_index, trial_data)

def run_random_agent():
    genotype_structure=gen_structure.DEFAULT_GEN_STRUCTURE
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

    trial_index = 0
    trial_data = {}
    random_seed = np.random.randint(10000)
    perf = sim.compute_performance(random_genotype, random_seed, trial_data)
    print("random perf: {}".format(perf))

    vis = Visualization(sim)
    vis.start_simulation_from_data(trial_index, trial_data)

if __name__ == "__main__":
    # run_with_keyboard(trial_index=1)
    run_from_data()
    # run_random_agent()
