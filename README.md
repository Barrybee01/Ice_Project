# Ice_Project
Contains the important python scripts used to post-process LAMMPS trajectories used in my Master's project

##############################################################################################################################################################################################

lammpstrj_to_xyz.py will take in a lammpstrj trajectory file from LAMMPS simulations and will convert each timestep in the original trajectory to XYZ format. This can be uploaded to software that requires or accepts an XYZ trajectory format. Alternatively, the trajectory can be externally sliced into individual timesteps

For now, this script is very basic and may break in liquid MD simulations where atoms are likely to wrap around PBCs. This only unscales the coordinates, and does not unwrap them. Future versions may include this feature.

##############################################################################################################################################################################################
