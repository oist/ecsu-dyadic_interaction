"""
TODO: Missing module docstring
"""

import numpy as np
from scipy.special import expit # pylint: disable-msg=E0611
from pyevolver.ctrnn import BrainCTRNN
from pyevolver.utils import linmap
from pyevolver.evolution import MIN_SEARCH_VALUE, MAX_SEARCH_VALUE

# range of each site in the genotype (pyevolver)
EVOLVE_GENE_RANGE = (MIN_SEARCH_VALUE, MAX_SEARCH_VALUE)

class AgentNetwork:
    """
    Agents' have brains, that can be ctrnn but could also be a different kind of network.
    They also have a particular anatomy and a connection to external input and output.
    The anatomy is set up by the parameters defined in the genotype_structure: they
    specify the number of particular sensors and effectors the agent has and the number
    of connections each sensor has to other neurons, as well as the weight and gene ranges.
    This class defines how the brain and other parts of the agent's body interact.
    TODO: update documentation above
    """

    def __init__(self, num_brain_neurons, brain_step_size, genotype_structure,
        brain_state_range=None, random_seed=None, genotype=None):

        get_param_range = lambda param: \
            genotype_structure[param]['range'] \
                if 'range' in genotype_structure[param] else None

        get_param_default = lambda param: \
            genotype_structure[param]['default'] \
                if 'default' in genotype_structure[param] else None

        '''
        Initialize brain with params in genotype structure
        '''
        brain_params = {}
        for p,n in zip(['taus', 'gains', 'biases'],['tau', 'gain', 'bias']):
            neural_p = 'neural_{}'.format(p)
            p_range = get_param_range(neural_p)
            if p_range:
                p_range_name = '{}_range'.format(n) # without the s
                brain_params[p_range_name] = p_range
            else:
                brain_params[p] = get_param_default(neural_p)


        self.brain = BrainCTRNN(
            random_seed=random_seed,
            num_neurons=num_brain_neurons,
            step_size=brain_step_size,
            states=np.array([0.,0.]), # states are initialized with zeros
            **brain_params
        )

        # these will be set in genotype_to_phenotype()
        self.sensor_gains = None
        self.sensor_biases = None
        self.sensor_weights = None
        self.motor_gains = None
        self.motor_biases = None
        self.motor_weights = None
        
        # IMPORTANT: 
        # this needs to be initialized in the simulation
        self.motors_outputs = None 

        self.genotype_structure = genotype_structure

        if genotype:
            self.genotype_to_phenotype(genotype)

    def init_params(self, brain_states, motors_outputs):
        self.brain.states = brain_states
        self.motors_outputs = motors_outputs

    def genotype_to_phenotype(self, genotype):
        '''
        map genotype to brain values (self.brain) and sensor/motor (self)
        '''
        for k, val in self.genotype_structure.items():
            if k == 'crossover_points':
                continue
            if k.startswith('neural'):
                brain_field = k.split('_')[1]  
                # 1 neural_taus -> (1,2) self.brain.taus
                # 1 neural_biases -> (1,2) self.brain.biases
                # 1 neural_gains -> (1,2) self.brain.gains
                # 4 neural_weights -> (2,2) self.brain.weights
                if 'indexes' in val:
                    gene_values = np.array([genotype[i] for i in val['indexes']])
                    if k == 'neural_weights':
                        gene_values = gene_values.reshape(self.brain.num_neurons, -1)
                    else:
                        # biases, gains, weights
                        # same values for all neurons
                        gene_values = np.tile(gene_values, self.brain.num_neurons) 
                    setattr(
                        self.brain, brain_field,
                        linmap(gene_values, EVOLVE_GENE_RANGE, val['range'])
                    )
                else:
                    default_val_copy = np.copy(val['default'])
                    setattr(self.brain, brain_field, default_val_copy)
            else:  # sensor*, motor*
                # using same fields as in genotype_structure
                if 'indexes' in val:
                    gene_values = np.array([genotype[i] for i in val['indexes']])
                    if k == 'sensor_weights':
                        gene_values = gene_values.reshape(2, -1) # num_sensor == 2
                    elif k == 'motor_weights':
                        gene_values = gene_values.reshape(3, -1) # num_motors (including emitter) == 3
                    else:
                        num_units = 2 if k.split('_')[0]=='sensor' else 3 # 3 motors
                        gene_values = np.tile(gene_values, num_units) # same tau/bias values for all sensors/motors
                    setattr(
                        self, k,
                        linmap(gene_values, EVOLVE_GENE_RANGE, val['range'])
                    )
                else:
                    default_val_copy = np.copy(val['default'])
                    setattr(self, k, default_val_copy)

    def compute_brain_input(self, signal_strength):
        sensor_outputs = np.multiply(self.sensor_gains, expit(signal_strength + self.sensor_biases))  # [o1, o2]
        self.brain.input = np.dot(sensor_outputs, self.sensor_weights)  # [1,2]·[2,2] = [1,2] two dimensional array

    def compute_motor_outputs(self):
        self.motors_outputs = np.multiply(
            self.motor_gains,
            expit(np.dot(self.motor_weights, self.brain.output) + self.motor_biases)
        )
        return self.motors_outputs
