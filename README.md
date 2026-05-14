# READ ME
Contains the important python scripts used to post-process LAMMPS trajectories used in my Master's and PhD projects. The code provided will be specific to structural studies of water, but can be adapted for more general cases.

## File Converting Scripts
1. LAMMPS_to_XYZ_converter.py
2. lammpstrj_to_xyz_V3.py
3. lmp_to_xyz.py

The first two scripts require a `dump.lammpstrj` file as input. The last script only requires a single frame of the trajectory file, in the form of `structure_file.lmp`. The first script reads in the LAMMPS trajectory and returns the entire trajectory in an XYZ format. The second script, for the most part, does the exact same thing as the first script, however it produces a modified XYZ trajectory where the cell parameters are stored in the comment line, and the atom symbols are instead numbers representing atom type. The last script does the same job as the first script, but only takes in a single frame to return a single XYZ file.

## Bond Analysis Scripts
1. bond_map.py
2. BondAnalysis.py
3. output_to_parquet.py
4. fft_parquet_HOH.py
5. fft_sum.py

These scripts were to serve as a data pipeline of a sort. This did not end up working very well, and this code is not very efficient. For optimal results, it is best to use `bond_map.py` first. This script requires a LAMMPS input structure (`structure.lmp`) to run. It will return a bond map of the original input structure. *This is best suited for crystalline input structures*.

The next part of this data pipeline is the `BondAnalysis.py` script. **This script has more or less been hard-coded for analysis of ice systems, however can be used as a template for similar analysis**. There are two types of calculations performed: bond length calculations and bond angle calculations (OH1 and OH2 lengths, HOH and OOO angles). The script requires the bond map produced in the previous step, as well as a folder storing the XYZ files required for the analysis. It is expected that the input data covers a notable amount of simulation time (>= 1ns). The script outputs a series of text files specific to the type of calculation. ***The output data is a written time series for EVERY relevant bond/pair in the supercell***.

The remaining scripts are used to study vibrational spectra of ice. The first part of this additional analysis is to use `output_to_parquet.py`. This script takes in the HOH, OH1 and OH2 time series files from the last step and converts them to parquet files. These files are column-oriented, so this allows for the individual bond/pair series to occupy its own column in the file, and can be parsed easily.

Vibrational spectra are produced by performing a FFT on the input time series data. The `fft_parquet_HOH.py` script has the capability of doing this. The result is the vibrational spectra produced by individual water molecules. Because of my own skill issue, I could not figure out iterative reading/processing/writing of parquet files, so the output of these tasks are HDF5 files.

To observe the ensemble average (and to see an actual clear spectrum), the individual molecule components are summed together with the `fft_sum.py` script. These read in the HDF5 files from the last step, sum together all of the columns in the input file, and return a text file with the resultant vibrational spectrum.

**KNOWN PROBLEMS AND FUTURE CHANGES:**
- The `BondAnalysis.py` script is quite slow. This requires a prohibitively long calculation. This will eventually be fixed with MPI parallelization

## Persistent Homology Analysis Scripts
1. ring_stats.py
2. full_ring_stats.py
3. FSDP.py

Persistent homology calculations are employed using the Homcloud Python library. The first 2 scripts do essentially the same job. The only difference is the amount of data used in later ring calculations. The `ring_stats.py` script allows one to use a rectangular or polygon mask to analyze a small region of a persistence diagram, while `full_ring_stats.py` performs the calculations using all features from the persistence diagram. Both scripts require a `.pdgm` input file, and both scripts will perform the same calculations. Current ring statistic calculations are:

- Number of atoms per ring
- Pair type that kills each ring
- Ring size
- Degree of ring flatness
- Accumulative persistence function

These two scripts will output two text files: one containing the information of the APF, and the other is a space-delimited file containing the output of the other calculations, as well as the birth and death index of any ring used in the calculation.

The `FSDP.py` script requires a CSV file that contains the ring sizes for each type of ring seen in the persistence diagram (4-membered ring, 5-membered ring, etc). The script outputs a figure of the FSDP containing the individual loop contributions.

**KNOWN PROBLEMS AND FUTURE CHANGES:**
- Parallelization has been implemented in the `ring_stats.py` script, but it has not been benchmarked to suggest any specific amount of resources given a specific input data size.
- The `ring_stats.py` and `full_ring_stats.py` scripts will eventually be merged. I envision a user-defined arg that one can change so that the script knows the extent of the persistence diagram to be studied (rectangular mask, polygon mask, full PD).
- The `ring_stats.py` and `full_ring_stats.py` scripts will be updated so that they can take an XYZ file as input instead of the more prohibitive `.pdgm` files.
- The Homcloud library has been known to contain memory leaks in different versions. This may cause any of the ring statistic scripts to break. There are try/except cases implemented in these scripts to be able to avoid the Homcloud problems that result in OOM issues, but this may need to be modified in the future if similar issues arise.
- Ring statistics script will be updated to output a series of text files that contain information on the birth, death, and lifetime distributions.
- A script will be developed to further process some of the raw output of the original ring statistics calculations.

A more advanced example of using the ring statistics scripts is currently in production. A script named `Ring_PCA_analysis.py` is to be used to perform PCA and regression analysis between two different data sets of ring statistics.

## Miscellaneous Scripts
1. supercell_subdividor_GPU_boosted.py

This is a GPU-oriented script that will perform local density calculations on larger supercells. Should only need an XYZ file, or something related, for this to run. An interesting use case would be to observe local fluctuations of other physical properties.

## Dependencies
All of these calculations can be performed with a single Python environment. Listed below are the requirements for a Python/3.10 environment.
- ase/3.10.0
- fastparquet/2026.3.0
- h5py/3.16.0
- matplotlib/3.10.0
- numpy/1.26.4
- pandas/2.0.0
- pyarrow/24.0.0
- pyqt5/5.15.11
- pyqt6/6.11.0
- pyvista/0.48.1
- pyvistaqt/0.11.4
- scipy/1.14.0
- scikit-learn/1.7.2
- wheel/0.47.0
- homcloud/4.2.0
- shapely/2.0.0
- seaborn/0.13.2
- mpi4py

Homcloud can be a finnicky library and typically requires older versions of Numpy. As a result, there are often issues with version compatibility or the environment not being able to properly build the wheel for Homcloud. The environment that I have made in this way works, but specific library versions can be changed.
