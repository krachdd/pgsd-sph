# Parallelized General Simulation Data 

Parallel General Simulation Data Format for SPH Solver. 

Draft.
Current version: pgsd-3.2.0


A subproject to the HOOMD SPH Code to keep the [General Simulation Data Packages form Glotzer Group](https://github.com/glotzerlab/gsd) up to date. 

## Build 

For the `conda` VENV that is requried for the project see main `hoomd-sph3` project.

```bash
# PGSD Package
mkdir build ; cd build 
CC=mpicc CXX=mpicxx cmake .. 
# or better since no dependencies on buggy compiler wrappers
CC=/usr/bin/mpicc CXX=/usr/bin/mpicxx cmake ..
make
```

## Main Modifications GSD->PGSD

1. Header and index written by main rank, but information is gathered from all ranks. 
2. No collection of all particle data on the main rank. Incremental peak RAM needed at main rank.
3. No sorting of the particles.  
4. This (3) results in unstructured information per timestep. Meaning also fields that did not change, e.g. paritcle ID, are rewritten to disk. For each timestep
5. Results in a (significatly) bigger `.gsd` file.


## Developer

- [David Krach](https://www.mib.uni-stuttgart.de/institute/team/Krach/) E-mail: [david.krach@mib.uni-stuttgart.de](mailto:david.krach@mib.uni-stuttgart.de)
