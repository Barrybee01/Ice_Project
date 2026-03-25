import os
import sys

converter = "/home/rielly/scratch/HDA_VHDA_PDs/1h_to_HDA_to_VHDA_fullringstats/22d5kbar/P3/Full_PD/Comprehensive_ringstats"
sys.path.append(converter)

from LAMMPSTRJ_to_XYZ_converter import convert_to_xyz

dump_path = "/home/rielly/scratch/HDA_VHDA_PDs/1h_to_HDA_to_VHDA_fullringstats/22d5kbar/P3/Full_PD/Comprehensive_ringstats"
dump_filename = "dump.lammpstrj"
dump_fullpath = os.path.join(dump_path, dump_filename)

xyz_filename = "dump_trj.xyz"
xyz_fullpath = os.path.join(converter, xyz_filename)

convert_to_xyz(dump_fullpath, xyz_fullpath)
print(f"Converted trajectory saved to: {xyz_fullpath}")

split_folder = os.path.join(converter, "Split_xyz_trj")
os.makedirs(split_folder, exist_ok=True)

lines_per_timestep = 18434
timestep_interval = 100

with open(xyz_fullpath, "r") as infile:
    timestep_value = 100

    while True:
        block = [infile.readline() for _ in range(lines_per_timestep)]

        if not block[0]:
            break

        output_filename = f"timestep_{timestep_value}.xyz"
        output_fullpath = os.path.join(split_folder, output_filename)

        with open(output_fullpath, "w") as outfile:
            outfile.writelines(block)

        timestep_value += timestep_interval

print(f"Split xyz files saved in: {split_folder}")