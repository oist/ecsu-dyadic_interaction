"""
TODO: Missing module docstring
"""

from dataclasses import dataclass, field
import numpy as np
from numpy import pi as pi
from numpy.linalg import norm
import trianglesolver
from pyevolver.utils import linmap
from dyadic_interaction.utils import modulo_radians
from pyevolver.timing import Timing
from dyadic_interaction.utils import random_string


@dataclass
class AgentBody:
    """
    This is a class that implements agents in the simulation, i.e. their virtual physical instantiation
    in the environment.
    """
    agent_body_radius: int = 4
    # angle between each sensor and axes of symmetry (angle of agent),
    # ideally between 0 (sensors overlap) and pi/2 (sensors on left and right)
    agent_sensors_divergence_angle: float = pi/4
    name: str = field(default_factory=lambda: random_string)
    position: list = None
    angle: float = None
    wheels: np.ndarray = field(default_factory=lambda: np.array([0., 0.]))  # wheel displacement at next step
    flag_collision: float = False
    timeit: bool = False

    def __post_init__(self):
        assert 0 < self.agent_sensors_divergence_angle <= pi / 2
        self.timing = Timing(self.timeit)        

    def init_params(self, wheels, flag_collision = False):
        self.wheels = wheels
        self.flag_collision = flag_collision            

    def set_position_and_angle(self, agent_position, agent_angle):
        assert all(type(x) in [float, np.float64] for x in agent_position), "Agent's position must be a pair of float"
        assert type(agent_angle) in [float, np.float64], "Agent's angle must be a float"

        self.position = agent_position # absolute position
        self.angle = agent_angle
        self.__update_sensors_pos()

    def __update_sensors_pos(self):
        # sensors position are relative to center of the agents
        self.sensors_angle = [
            modulo_radians(self.angle + self.agent_sensors_divergence_angle),  # left sensor
            modulo_radians(self.angle - self.agent_sensors_divergence_angle)  # right sensor
        ]
        self.sensors_pos = [
            self.agent_body_radius * np.array([np.cos(angle), np.sin(angle)])
            for angle in self.sensors_angle
        ]

    def get_abs_sensors_pos(self):
        return [self.position + ep for ep in self.sensors_pos]


    def get_signal_strength(self, emitter_position, emitter_strenght):
        """
        The sensor input to the agent
        :param emitter_position: absolute position of the emitter
        :param emitter_strenght: strenght of the emitter signal
        :return:
        """

        t = self.timing.init_tictoc()

        signal_strengths = [0, 0]
        # print()
        # print("Emitter position: {}".format(emitter_position))
        self.timing.add_time('AB2-GVI_emitter_pos', t)

        dist_centers = max(norm(self.position - emitter_position), 2 * self.agent_body_radius)

        for i, sp in enumerate(self.get_abs_sensors_pos()):
            self.timing.add_time('AB2-GVI_emitter_translated_angle', t)
            # print("SENSOR POSITION {}: {}".format(i+1, sp))
            self.timing.add_time('AB2-GVI_check_in_vision', t)
            # TODO: check the following
            dist_sensor_emitter = max(norm(sp - emitter_position),  self.agent_body_radius)           
            N = dist_sensor_emitter / self.agent_body_radius
            Is = emitter_strenght / np.power(N, 2)
            self.flag_collision = dist_centers <= 2*self.agent_body_radius # collision detection
            pow_D_centers = np.power(dist_centers,2)
            pow_Radius = np.power(self.agent_body_radius,2)
            pow_Dsen = np.power(dist_sensor_emitter,2)
            A = (pow_D_centers - pow_Radius)/pow_Dsen                        
            Dsh = 0 if A >= 1 else dist_sensor_emitter * (1 - A)
            AttenuationFactor = (-0.1125 * Dsh) + 1
            TotalSignal = Is * AttenuationFactor
            # print("emitter signal to sensor {}: {}".format(i+1, TotalSignal))
            signal_strengths[i] = TotalSignal            

            self.timing.add_time('AB2-GVI_compute_inputs', t)
        return signal_strengths

    def get_delta_xy(self):
        avg_displacement = np.mean(self.wheels)
        delta_xy = avg_displacement * np.array([np.cos(self.angle), np.sin(self.angle)])
        return delta_xy

    # move the agent of one step
    # see equation 6 in http://rossum.sourceforge.net/papers/DiffSteer/#d6
    def move_one_step(self, other_delta_xy, other_angle):

        if self.flag_collision:
            # if self.flag_collision:
            #     print("Collision!")
            self.position += other_delta_xy
            if self.angle != other_angle:
                self.angle = other_angle
                self.__update_sensors_pos()
            return other_delta_xy, other_angle

        wheel_diff = self.wheels[1] - self.wheels[0]  # right - left
        delta_angle = wheel_diff / self.agent_body_radius
        self.angle += delta_angle
        delta_xy = self.get_delta_xy()
        self.position += delta_xy

        if delta_angle:
            self.__update_sensors_pos()

        return delta_xy, self.angle
