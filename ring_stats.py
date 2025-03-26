import ase
import ase.io
import homcloud.interface as hc
import homcloud.interface.exceptions
import numpy as np
import pandas as pd
import scipy
from mpi4py import MPI

# MPI Initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Process rank
size = comm.Get_size()  # Total number of processes

pd1 = hc.PDList("35kbar_H2O_weighted.pdgm").dth_diagram(1)

R_H = 0.775
R_O = 0.175

birth_min, birth_max = 0.3, 0.75
death_min, death_max = 1.1, 4.0
max_atoms = 30
batch_size = 1

def fit_plane(points):
    A = np.c_[points[:, 0], points[:, 1], np.ones(points.shape[0])]
    b = points[:, 2]
    coeff, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return coeff  # a, b, c

def calculate_flatness(points, plane_coeff):
    a, b, c = plane_coeff
    distances = []
    denominator = np.sqrt(a**2 + b**2 + 1)

    for point in points:
        x, y, z = point
        distance = abs(a * x + b * y + c - z) / denominator
        distances.append(distance)

    total_distance = np.sum(distances)
    degree_of_flatness = total_distance / len(points)
    return degree_of_flatness

# Filter relevant pairs
relevant_pairs = [pair for pair in pd1.pairs()
                  if birth_min <= pair.birth <= birth_max and death_min <= pair.death <= death_max]

# Distribute workload among processes
total_pairs = len(relevant_pairs)
pairs_per_rank = total_pairs // size
start_idx = rank * pairs_per_rank
end_idx = (rank + 1) * pairs_per_rank if rank != size - 1 else total_pairs
my_pairs = relevant_pairs[start_idx:end_idx]

# Output storage for each process
output_data = []

for pair in my_pairs:
    try:
        stable_volume = pair.stable_volume(0.1)
        boundary_points = stable_volume.boundary_points()
        num_atoms = len(boundary_points)

        if num_atoms <= max_atoms:
            # Determine pair type
            death_scale = pair.death
            death_symbols = pair.death_position_symbols
            if death_symbols.count('H') == 3:  # HH pair
                ring_size = 2 * (np.sqrt(R_H**2 + death_scale))
                pair_type = "HH"
            elif death_symbols.count('H') == 2 and death_symbols.count('O') == 1:  # OH pair
                ring_size = np.sqrt(R_H**2 + death_scale) + np.sqrt(R_O**2 + death_scale)
                pair_type = "OH"
            else:
                continue  # Skip invalid pair types

            boundary_points = np.array(boundary_points)
            if boundary_points.shape[1] == 3:
                plane_coeff = fit_plane(boundary_points)
                flatness = calculate_flatness(boundary_points, plane_coeff)
                output_data.append((num_atoms, ring_size, flatness, pair_type))
    except (TypeError, AssertionError, homcloud.interface.exceptions.VolumeNotFound):
        continue

# Gather results from all processes
all_data = comm.gather(output_data, root=0)

# Process 0 writes the output to a file
if rank == 0:
    with open("ring_stats.txt", "w") as f:
        for proc_data in all_data:
            for atom_count, ring_size, flatness, pair_type in proc_data:
                f.write(f"{atom_count},{ring_size:.6f},{flatness:.6f},{pair_type}\n")
    print(f"Finished processing. Results saved to 'ring_stats.txt'.")