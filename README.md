# Ice_Project
Contains the important python scripts used to post-process LAMMPS trajectories used in my Master's project

###########################################################################################################################################################################################

lammpstrj_to_xyz.py will take in a lammpstrj trajectory file from LAMMPS simulations and will convert each timestep in the original trajectory to XYZ format. This can be uploaded to software that requires or accepts an XYZ trajectory format. Alternatively, the trajectory can be externally sliced into individual timesteps

For now, this script is very basic and may break in liquid MD simulations where atoms are likely to wrap around PBCs. This only unscales the coordinates, and does not unwrap them. Future versions may include this feature. This function will take in the cell dimensions from the lammpstrj header lines and put them in the comment line of XYZ format

###########################################################################################################################################################################################

BondAnalysis.py performs bond length and bond angle analysis of solid water systems. Of particular interest are the OH bond lengths, the OOO angle, and the HOH angle. This script requires a structure file in XYZ format. 

The code will only handle "one at a time" calculations as the cell dimensions need to be changed. Future versions of this code will handle the search radius for OH and HOH analysis better. I also wish to merge this with lammpstrj_to_xyz.py so that the supercell dimensions are grabbed from the header before converting a timestep to XYZ format

###########################################################################################################################################################################################

supercell_subdividor_GPU_boosted.py reads in a structure file in XYZ format and subdivide the cell into equal subdivisions using the supercell dimensions. 

This is a template script and can be used forany kind of local structural analysis. An excellent example would be to use this script in conjunction with the bond angle analysis to look at local fluctuations of these properties. This script can also be used for local density calculations. GPU support has been added to make the script faster when using very large supercells

###########################################################################################################################################################################################

ring_stats.py employs the Homcloud library to perform topological data analysis of ice structures. The input structure needs to be in XYZ format. Homcloud will calculate the various ring features that emerge through 1D persistent homology calculations. This script will take the homcloud output and process it further. The output of the script are statistics about the population of atoms within the rings, the ring size, and the ring flatness.

Homcloud seems to be an inherently slow library, so this calculation requires parallelization in order for calculations to be performed at a reasonable pace. Change the filtration parameter alpha if there are issues with feature lifetime in the homcloud calculations. A larger value will usually speed up the calculation at the expense of the accuracy of the output

###########################################################################################################################################################################################
