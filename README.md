# Ice_Project
Contains the important python scripts used to post-process LAMMPS trajectories used in my Master's project


LAMMPS_to_XYZ_converter.py will take in a lammpstrj trajectory file from LAMMPS simulations and will convert each timestep in the original trajectory to XYZ format. This can be uploaded to software that requires or accepts an XYZ trajectory format. Alternatively, the trajectory can be externally sliced into individual timesteps.


BondAnalysis.py performs bond length and bond angle analysis of solid water systems. Of particular interest are the OH bond lengths, the OOO angle, and the HOH angle. This script requires a structure file in the modified XYZ format made by lammpstrj_to_xyz.py. This script will produce a file similar to a lammps dump file where all of the temporal bonding information will be logged for every molecule within the supercell. This will likely produce large files.


supercell_subdividor_GPU_boosted.py reads in a structure file in XYZ format and subdivide the cell into equal subdivisions using the supercell dimensions. This is a template script and can be used for any kind of local structural analysis. An excellent example would be to use this script in conjunction with the bond angle analysis to look at local fluctuations of these properties. This script can also be used for local density calculations. GPU support has been added to make the script faster when using very large supercells


ring_stats.py employs the Homcloud library to perform topological data analysis of ice structures. The input structure needs to be in XYZ format. Homcloud will calculate the various ring features that emerge through 1D persistent homology calculations. This script will take the homcloud output and process it further. The output of the script are statistics about the population of atoms within the rings, the ring size, and the ring flatness. Homcloud seems to be an inherently slow library, so this calculation requires parallelization in order for calculations to be performed at a reasonable pace. Change the filtration parameter alpha if there are issues with feature lifetime in the homcloud calculations. A larger value will usually speed up the calculation at the expense of the accuracy of the output


bond_map.py will read a lammps compatable structure file and produce a text file with all of the bonding information. Namely, it stores the atom IDs for every bond in the structure


output_to_parquet.py is a script that will take the large space delimited text files made by the bond analysis code and convert it to a parquet file. These files are very helpful for column-oriented data and is able to be read in chunks when handling large data sets.
