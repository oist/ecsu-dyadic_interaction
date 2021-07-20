# Dyadic Interaction

Evolutionary simulation of 2 agents maximizing neural complexity

## How to guide

### Main
Runs evolutionary code and the simulation. To get the full list of option run:

`python -m dyadic_interaction.main --help`

If you provide an output directory it will generate:
- a list of of files `evo_xxx.json` containing the parameters of `pyevolver.Evolution` object at generation `xxx` (including the genotype of the agents' population, and the fitness).
- a file `simulation.json` with the parameters of the `dol.Simulation` that the agents go throughout evolution.

### Rerun simulation
Assuming you have a specific dir with outputs from the `dyadic_interaction.main` code (see above), you can rerun the simulation of a given generation and a given agent in the population. 

For instance,

`python -m dyadic_interaction.simulation --dir <dirpath>`

Will run the simulation of the last saved generation and the best agent in the simulation and

`python -m dyadic_interaction.simulation --dir <dirpath> --write_data` will create a subfolder `data` in the same directory with all the data from the simulation. 

## Front. Neurorobot., 11 June 2021 paper
The [paper](https://www.frontiersin.org/articles/10.3389/fnbot.2021.634085/full) __Shrunken Social Brains? A Minimal Model of the Role of Social Interaction in Neural Complexity__ by Georgina Montserrat Reséndiz-Benhumea, Ekaterina Sangati, Federico Sangati, Soheil Keshmiri and Tom Froese is based on this code.

### Steps to reproduce the results:
1. Install `python 3.7.3`
2. Clone repository and checkout version tag `1.0.0`
   - `git clone https://github.com/oist/ecsu-dyadic_interaction`
   - `cd ecsu-dyadic_interaction`
   - `git checkout 1.0.0`
3. Create and activate python virtual environment, and upgrade pip
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
   - `python -m pip install --upgrade pip`
4. Build required libraries
   - `pip install -r requirements.txt`
5. If you want to **run the simulations on a cluster**, execute the following 3 scritps in the `slurm` directory :
   - `sbatch array_2n_shannon-dd_iso.slurm` (2 neurons, isolated condition)
   - `sbatch array_2n_shannon-dd_social.slurm` (2 neurons, social condition)
   - `sbatch array_3n_shannon-dd_iso.slurm` (3 neurons, isolated condition)
   - `sbatch array_3n_shannon-dd_social.slurm` (3 neurons, social condition)

   Each sbatch file will run an array of 20 simulations (20 random seeds from `1` to `20`). The output directories are respectively: 
   - `2n_shannon-dd_neural_iso_coll-edge`
   - `2n_shannon-dd_neural_social_coll-edge`
   - `3n_shannon-dd_neural_iso_coll-edge`
   - `3n_shannon-dd_neural_social_coll-edge`
   
   Our code has been run on 128 `AMD Epyc` CPUs nodes [cluster at OIST](https://groups.oist.jp/scs/deigo) running `CentOS 8`.
6. Alternatively, if you want to **run the simulation on a personal computer**: execute the `python3` command included in any slurm file above, setting `seed` and `output directory` appropriately.
