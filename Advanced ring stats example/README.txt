This is an application of the ring stats calculation that uses a data pipeline

REQUIREMENTS AND RECOMMENDATIONS:
  - Requires a LAMMPS dump file to be used as input. This needs to be the .lammpstrj format
  - Requires parallelization in steps. This code is all CPU-based and can be quite computationally intense
  - It is useful to set up a script to sequentially run a series of jobs

DEPENDENCIES:
  - Python 3.11
  - Numpy < 2.0
  - Wheel, Cython, panel
  - pyqt5, pyqt6, pyvistaqt, pyvista
  - ase, ripser, py3Dmol, ase.io
  - homcloud 4.8.0
  - h5py
  - os, io, sys, glob, tarfile
  - mpi4py
  -matplotlib


FILES ARE SUBJECT TO SUBSTANTIAL CHANGE OR REPLACEMENT AS THIS PROCESS IS BEING BUILT
