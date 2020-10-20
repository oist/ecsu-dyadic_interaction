import numpy as np
from numpy.linalg import norm
from dyadic_interaction.utils import modulo_radians
from numpy import pi

def get_abs_sensors_pos(self):
    return [self.position + ep for ep in self.sensors_pos]

def get_signal_strength(agent_position, agent_angle, emitter_position, emitter_strenght):
    """
    The sensor input to the agent
    :param emitter_position: absolute position of the emitter
    :param emitter_strenght: strenght of the emitter signal
    :return:
    """

    agent_body_radius = 4
    visual_inputs = [0, 0]
    agent_sensors_divergence_angle = pi/4

    sensors_angle = [
        modulo_radians(agent_angle + agent_sensors_divergence_angle),  # left sensor
        modulo_radians(agent_angle - agent_sensors_divergence_angle)  # right sensor
    ]
    
    sensors_pos = [
        agent_body_radius * np.array([np.cos(angle), np.sin(angle)])
        for angle in sensors_angle
    ]

    abs_sensors_pos = [agent_position + ep for ep in sensors_pos]

    flag_collision = False


    for i, sp in enumerate(abs_sensors_pos):
        dist_sensor_emitter = norm(sp - emitter_position)            
        N = dist_sensor_emitter / agent_body_radius
        Is = emitter_strenght / np.power(N - 1, 2)
        dist_centers = norm(agent_position - emitter_position)
        flag_collision = dist_centers <= 2*agent_body_radius # collision detection
        pow_D_centers = np.power(dist_centers,2)
        pow_Radius = np.power(agent_body_radius,2)
        pow_Dsen = np.power(dist_sensor_emitter,2)
        A = (pow_D_centers - pow_Radius)/pow_Dsen                        
        Dsh = 0 if A >= 1 else dist_sensor_emitter * (1 - A)
        AttenuationFactor = (-0.1125 * Dsh) + 1
        TotalSignal = Is * AttenuationFactor
        visual_inputs[i] = TotalSignal            

    return visual_inputs

if __name__ == "__main__":
    signal_strenght = get_signal_strength(
        agent_position = np.array([14.79713344,5.6109246]),  #np.array([0.,0.]), 
        agent_angle = 26.11226905,         
        emitter_position = np.array([19.8530914, 11.74030021]), #np.array([20., 0.]), 
        emitter_strenght = 4.435263092 #3.9530477287491896
    )
    print(signal_strenght)