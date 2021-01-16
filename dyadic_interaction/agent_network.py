"""
TODO: Missing module docstring
"""

import numpy as np
from scipy.special import expit # pylint: disable-msg=E0611
from pyevolver.ctrnn import BrainCTRNN
from dyadic_interaction.utils import linmap
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

    def __init__(self, num_brain_neurons, brain_step_size, genotype_structure, genotype=None):

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

    def init_params(self, brain_states):
        self.brain.states = brain_states

    def genotype_to_phenotype(self, genotype, phenotype_list=None, phenotype_dict=None):
        '''
        map genotype to brain values (self.brain) and sensor/motor (self)
        '''
        i = 0
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
                    gene_values = np.take(genotype, val['indexes'])
                    if k == 'neural_weights':
                        gene_values = gene_values.reshape(self.brain.num_neurons, -1)
                    else:
                        # biases, gains, weights
                        # same values for all neurons
                        gene_values = np.tile(gene_values, self.brain.num_neurons) 
                    phenotype_value = linmap(gene_values, EVOLVE_GENE_RANGE, val['range'])                                        
                else:
                    phenotype_value = np.array(val['default'])
                setattr(self.brain, brain_field, phenotype_value)  
                if phenotype_dict is not None:
                    phenotype_dict[k] = phenotype_value        
            else:  # sensor*, motor*
                # using same fields as in genotype_structure
                if 'indexes' in val:
                    gene_values = np.take(genotype, val['indexes'])
                    if k == 'sensor_weights':
                        gene_values = gene_values.reshape(2, -1) # the matrix will have 2 rows (number of sensors)
                    elif k == 'motor_weights':
                        gene_values = gene_values.reshape(self.brain.num_neurons, -1)
                    else:
                        num_units = 2 if k.split('_')[0]=='sensor' else 3 # 3 motors
                        gene_values = np.tile(gene_values, num_units) # same tau/bias values for all sensors/motors
                    phenotype_value = linmap(gene_values, EVOLVE_GENE_RANGE, val['range'])                    
                else:
                    phenotype_value = np.array(val['default'])                    
                setattr(self, k, phenotype_value)
                if phenotype_dict is not None:
                    phenotype_dict[k] = phenotype_value
            if phenotype_list is not None and 'indexes' in val:
                if type(phenotype_value) == np.ndarray:
                    if k.endswith('_weights'):
                        phenotype_list[i:i+phenotype_value.size] = phenotype_value.flatten()
                        i += phenotype_value.size
                    else:
                        phenotype_list[i] = phenotype_value[0] # tiled value, take only one
                        i += 1
                else:
                    phenotype_list[i] = phenotype_value
                    i += 1
    
    def compute_brain_input(self, signal_strength):
        # let n be the number of neurons in the brain
        # sensor_output shape is (2, ): [O1, O2]        
        # sensor_weights shape is (2, n): [[W11,W12, ..., W1n],[W21,W22, ..., W2n]]
        # np.dot(sensor_output, sensor_weights) returns a vector of shape (n,): (2,)·(2,n) = (n,)
        # [O1·W11+O2·W21, O1·W12+O2·W22, ..., O1·W1n+O2·W2n]        
        sensor_outputs = np.multiply(self.sensor_gains, expit(signal_strength + self.sensor_biases))  # [o1, o2]
        self.brain.input = np.dot(sensor_outputs, self.sensor_weights)          

    def compute_motor_outputs(self):
        # let n be the number of neurons in the brain
        # brain_outputs shape is (n, ): [O1, O2, ..., On]        
        # motor_weights shape is (num_neurons=n, 3): [[W11,W12,W13],[W21,W22,W23],...,[Wn1,Wn2,Wn3]]
        # np.dot(brain_output, motor_weights) returns a vector of shape (3,): (n,)·(n,3) = (3,)
        # [O1·W11 + O2·W21 + ... + On·Wn1, O1·W12 + O2·W22 + ... + On·Wn2, O1·W13 + O2·W23 + ... + On·Wn3]
        self.motors_outputs = np.multiply(
            self.motor_gains,
            expit(np.dot(self.brain.output, self.motor_weights) + self.motor_biases)
        )
        return self.motors_outputs

def test_random_genotype():
    from dyadic_interaction import gen_structure
    from pyevolver.evolution import Evolution
    from numpy.random import RandomState
    default_gen_structure = gen_structure.DEFAULT_GEN_STRUCTURE(2)
    gen_size = gen_structure.get_genotype_size(default_gen_structure)
    num_brain_neurons = gen_structure.get_num_brain_neurons(default_gen_structure)
    print('Gen size of agent: {}'.format(gen_size))
    print('Num brain neurons: {}'.format(num_brain_neurons))
    random_genotype = Evolution.get_random_genotype(RandomState(None), gen_size)        
    agent_net = AgentNetwork(
        num_brain_neurons,
        brain_step_size=0.1,
        genotype_structure=default_gen_structure,
        genotype = random_genotype
    )
    agent_net.brain.states = np.array([0., 0.])
    agent_net.brain.compute_output()    
    print('brain output: {}'.format(agent_net.brain.output))    
    motor_outputs = agent_net.compute_motor_outputs()
    print('motor output: {}'.format(motor_outputs))

if __name__ == "__main__":
    test_random_genotype()