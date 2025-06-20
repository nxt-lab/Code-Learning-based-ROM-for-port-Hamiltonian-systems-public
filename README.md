# Structure- and Stability-Preserving Learning-based Reduced Ordered Modeling (ROM) of Port-Hamiltonian systems
Code for a paper (under review) on structure- and stability-preserving learning-based model order reduction (ROM) for port-Hamiltonian systems. The code is written in Python and Julia. This code provides an example of ROM of Toda lattice systems.

## Installation and dependencies
- Julia code: All dependencies of for Julia exactly as specified in Manifest.toml. Use `Pkg.instantiate()` in `TodaLattice.jl` or `instantiate` in the package management prompt to install all dependencies.
- Python code: All dependencies of for python are listed in `requirements.txt`. Run `pip install -r requirements.txt` to install the dependencies.

## Data generation
To generate synthetic data for trainning and validation, run the Julia script `TodaLattice.jl`. 
It will generate two data files `TodaLat_data_test.npz` and `TodaLat_data_train.npz`.

## Neural network ROM models
First, obtain the training and validation data for the Toda lattice system by executing the Julia code (see above).
Then run `NeuralToda_smsFNN.py` to learn and validate the ROM from the data.
