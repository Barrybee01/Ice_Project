import ase
import ase.io
import homcloud.interface as hc
import homcloud.interface.exceptions
import numpy as np
import pandas as pd
import scipy
from mpi4py import MPI
from shapely.geometry import Point, Polygon

# MPI Initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Process rank
size = comm.Get_size()  # Total number of processes

R_H = 0.775
R_O = 0.175

amorph_ice = ase.io.read('infile.xyz')
weights = np.array([R_O**2 if atom == "O" else R_H**2 for atom in amorph_ice.get_chemical_symbols()]) #weighting is not explicitly required

pd1 = hc.PDList.from_alpha_filtration(amorph_ice.get_positions(),vertex_symbols=amorph_ice.get_chemical_symbols(),
        weight=weights, save_boundary_map=True, save_phtrees=True, save_to='file_name.pdgm').dth_diagram(1)

print('The persistence diagram has been made') #for log file

max_atoms = 30

print(f'The maximum number of atoms accepted in a ring is {max_atoms}')

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

def compute_apf(pairs):
    apf_data = []

    for pair in pairs:
        b = pair.birth
        d = pair.death
        m = 0.5 * (b + d)
        persistence = d - b
        apf_data.append((m, persistence))

    # Sort by midpoint time
    apf_data.sort(key=lambda x: x[0])

    times = []
    apf_values = []
    running_sum = 0.0

    for m, persistence in apf_data:
        running_sum += persistence
        times.append(m)
        apf_values.append(running_sum)

    return np.array(times), np.array(apf_values)

# Use the full persistence diagram instead of filtering by a masked region
relevant_pairs = list(pd1.pairs())
print(f'The total number of pairs in the full persistence diagram is {len(relevant_pairs)}')

# Distribute workload among processes
total_pairs = len(relevant_pairs)
print(f'The total number of accepted rings is {total_pairs}')
pairs_per_rank = total_pairs // size
start_idx = rank * pairs_per_rank
end_idx = (rank + 1) * pairs_per_rank if rank != size - 1 else total_pairs
my_pairs = relevant_pairs[start_idx:end_idx]

# Output storage for each process
output_data = []

for pair in my_pairs:
    try:
        stable_volume = pair.stable_volume(1e-4)
        boundary_points = stable_volume.boundary_points()
        num_atoms = len(boundary_points)

        if num_atoms <= max_atoms:
            # Determine pair type
            death_scale = pair.death
            birth_scale = pair.birth
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
                output_data.append((num_atoms, ring_size, flatness, pair_type, birth_scale, death_scale))
    except (TypeError, AssertionError, homcloud.interface.exceptions.VolumeNotFound):
        continue

# Gather results from all processes
all_data = comm.gather(output_data, root=0)

# Process 0 writes the output to a file
if rank == 0:
    with open("Persistent_island_ring_stats.txt", "w") as f:
        for proc_data in all_data:
            for atom_count, ring_size, flatness, pair_type, birth_scale, death_scale in proc_data:
                f.write(f"{atom_count},{ring_size:.6f},{flatness:.6f},{pair_type},{birth_scale:.6f},{death_scale:.6f}\n")
    print(f"Finished processing. Results saved to 'Persistent_island_ring_stats.txt'.")

    # Compute and save APF for the same accepted characteristic region
    apf_times, apf_values = compute_apf(relevant_pairs)

    with open("APF_output.txt", "w") as f_apf:
        for t, apf in zip(apf_times, apf_values):
            f_apf.write(f"{t:.6f},{apf:.6f}\n")

    print("APF results saved to 'APF_output.txt'.")
