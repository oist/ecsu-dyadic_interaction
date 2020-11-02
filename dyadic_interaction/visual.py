"""
TODO: Missing module docstring
"""

import os
import numpy as np
from numpy import pi as pi
from numpy.random import RandomState
import pygame
from dyadic_interaction.agent_body import AgentBody
from dyadic_interaction import simulation
from dyadic_interaction.simulation import Simulation
from dyadic_interaction import gen_structure
from dyadic_interaction import utils
from pyevolver.evolution import Evolution
from pyevolver.json_numpy import NumpyListJsonEncoder
import json

MAX_CANVAS_SIZE = 500
ZOOM_FACTOR = 4
REFRESH_RATE = 40

CANVAS_CENTER = np.array([MAX_CANVAS_SIZE/2, MAX_CANVAS_SIZE/2])
SHIFT_CENTER_TO_FIRST_AGENT = False

black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)
blue = (0, 0, 255)
agents_color = [red, blue]
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

        pygame.draw.circle(self.main_surface, agents_color[a_index], angent_center_pos, radius, width=0)

        for sp in agent.get_abs_sensors_pos():
            sp = ZOOM_FACTOR * sp - ZOOM_FACTOR * center_shift + CANVAS_CENTER
            pygame.draw.circle(self.main_surface, sensor_color, sp, sensor_radius)


    def start_simulation_with_keyboard(self, trial_index):
        running = True

        self.simulation.prepare_agents_for_trial(trial_index)

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
        signal_strength = [None, None]

        while running:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                # elif event.type == pygame.MOUSEBUTTONDOWN:
                #     signal_strength = self.agent.get_signal_strength(self.emitter_position)
                #     print('sensor inputs: {}'.format(signal_strength))
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
                signal_strength[a] = agent.get_signal_strength(
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
    
    def start_simulation_from_data(self, trial_index, data_record):
        running = True

        clock = pygame.time.Clock()
        
        duration = self.simulation.num_data_points        
        # print("Duration: {}".format(duration))

        agent_pair_pos = data_record['position'][trial_index]
        agent_pair_angle = data_record['angle'][trial_index]

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

def run_random_agents():
    genotype_structure=gen_structure.DEFAULT_GEN_STRUCTURE(2)
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
    data_record = {}
    random_seed = utils.random_int()

    perf = sim.compute_performance(random_genotype, random_seed, data_record)
    print("random perf: {}".format(perf))

    vis = Visualization(sim)
    vis.start_simulation_from_data(trial_index, data_record)

if __name__ == "__main__":
    # run_with_keyboard(trial_index=1)
    run_random_agents()
