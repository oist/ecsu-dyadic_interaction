
import argparse
from dyadic_interaction.simulation import obtain_trial_data
from dyadic_interaction.visual import Visualization

def run_default():
    evo, sim, data_record = obtain_trial_data(
        dir='data/shannon_entropy/MAX/dyadic_exp_001', 
        num_generation=500, 
        genotype_index=0, 
        force_random=True, 
        invert_sim_type=False, 
        initial_distance=None,
        ghost_index=0
    )

    vis = Visualization(sim)
    vis.start_simulation_from_data(0, data_record)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compare a simulation between single and multi-processing'
    )

    parser.add_argument('--dir', type=str, help='Output directory')
    parser.add_argument('--generation', type=int, help='Generation index to replay')        
    parser.add_argument('--genotype', type=int, help='Genotype index (0 is the best performing one)')    
    parser.add_argument('--trial', type=int, help='Trial index')    
    parser.add_argument('--random', action='store_true', help='Force random initialization')    
    parser.add_argument('--invert', action='store_true', help='Whether to invert the simulation type (shannon <-> transfer)')    
    parser.add_argument('--distance', type=int, default=-1, help='Initial distance (must be >=0 or else it will be set as in simulation default)')    
    parser.add_argument('--ghost', type=int, default=-1, help='Ghost index (must be 0 or 1 or else ghost condition will not be enabled)')    

    args = parser.parse_args()

    evo, sim, data_record = obtain_trial_data(
        dir=args.dir, 
        num_generation=args.generation, 
        genotype_index=args.genotype, 
        force_random=args.random, 
        invert_sim_type=args.invert, 
        initial_distance=args.distance if args.distance>=0 else None,
        ghost_index=args.ghost if args.ghost in (0,1) else None        
    )
    
    vis = Visualization(sim)
    vis.start_simulation_from_data(args.trial, data_record)



    