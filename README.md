# Code-Learning-based-ROM-for-port-Hamiltonian-systems-public
Public code for paper on structure- and stability-preserving learning-based model order reduction for port-Hamiltonian systems. Code is written in Python and Julia for simulation and data generation; respectively. This code provides an example for reduced-order modelling (ROM) of Toda lattice systems.

## Julia code
All dependencies of for Julia exactly as specified in Manifest.toml. Use Pkg.instantiate() in TodaLattice.jl to install all dependencies.

## Python code
All dependencies of for python are listed in requirements.text. Run pip install -r requirements.txt to install the dependencies.

## Data generation
To generate synthetic data for trainning and validation, run julia files TodaLattice.jl folder. 
It will generate TodaLat_data_test.npz and TodaLat_data_train.npz.

## Neural network ROM models
To execute ROM model, get the synthetic data.
Then run NeuralToda_smsFNN.py in NeuralPHs folder to use our proposed ROM.