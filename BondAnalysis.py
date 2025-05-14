import os
import numpy as np
import time

bond_topology_file = "oh_bond_topology.txt"
split_steps_folder = "split_steps"
log_file_path = "Bond_Analysis_Log.txt"
file_list_path = "Identified_Files_List.txt"

start_time = time.time()

# Count number of atoms (lines) in topology file, skipping header
with open(bond_topology_file, "r") as topo_file:
    next(topo_file)  # skip header
    num_atoms = sum(1 for _ in topo_file)

with open(log_file_path, "a") as log_file:
    log_file.write(f"Number of atoms (Oxygen atoms counted) in bond topology: {num_atoms}\n")

# Identify all split_step files
split_files = sorted(
    (f for f in os.listdir(split_steps_folder) if f.startswith("split_step_")),
    key=lambda x: int(x.split("_")[-1])
)
num_frames = len(split_files)

with open(log_file_path, "a") as log_file:
    log_file.write(f"Number of split step files identified: {num_frames}\n")

with open(file_list_path, "w") as list_file:
    for filename in split_files:
        list_file.write(f"{filename}\n")

def lattice_parameters_to_box_matrix(a, b, c, alpha_deg, beta_deg, gamma_deg):
    alpha = np.radians(alpha_deg)
    beta = np.radians(beta_deg)
    gamma = np.radians(gamma_deg)

    v_x = a
    v_y = b * np.cos(gamma)
    v_z = c * np.cos(beta)

    v_yz = b * np.sin(gamma)
    v_zy = (c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma))) / np.sin(gamma)
    v_zz = np.sqrt(c**2 - v_z**2 - v_zy**2)

    return np.array([
        [v_x, 0.0, 0.0],
        [v_y, v_yz, 0.0],
        [v_z, v_zy, v_zz]
    ])

def calculate_distance(coord1, coord2, box_matrix):
    delta = coord1 - coord2
    inv_box = np.linalg.inv(box_matrix)
    fractional = inv_box @ delta
    fractional -= np.round(fractional)
    corrected = box_matrix @ fractional
    return corrected


def read_xyz_file(xyz_file_path):
    with open(xyz_file_path, 'r') as f:
        lines = f.readlines()

    num_atoms = int(lines[0].strip())
    lattice_parts = list(map(float, lines[1].strip().split()))
    if len(lattice_parts) != 6:
        raise ValueError("Expected 6 lattice parameters: a, b, c, alpha, beta, gamma")
    a, b, c, alpha, beta, gamma = lattice_parts
    box_matrix = lattice_parameters_to_box_matrix(a, b, c, alpha, beta, gamma)

    atom_coords = {}
    for line in lines[2:2 + num_atoms]:
        parts = line.split()
        atom_id = int(parts[0])
        coord = np.array([float(parts[2]), float(parts[3]), float(parts[4])])
        atom_coords[atom_id] = coord

    return atom_coords, box_matrix

def read_bond_topology(topology_path):
    triplets = []
    with open(topology_path, 'r') as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                triplets.append(tuple(map(int, parts)))
    return np.array(triplets, dtype=int)

def analyze_frame(atom_coords, bond_triplets):
    n = len(bond_triplets)
    hoh_angles = np.empty(n)
    oh1_lengths = np.empty(n)
    oh2_lengths = np.empty(n)

    for i, triplet in enumerate(bond_triplets):
        o_id, h1_id, h2_id = triplet

        if o_id in atom_coords and h1_id in atom_coords and h2_id in atom_coords:
            o = atom_coords[o_id]
            h1 = atom_coords[h1_id]
            h2 = atom_coords[h2_id]

            v1 = calculate_distance(h1, o, box)
            v2 = calculate_distance(h2, o, box)

            oh1 = np.linalg.norm(v1)
            oh2 = np.linalg.norm(v2)
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

            hoh_angles[i] = angle
            oh1_lengths[i] = oh1
            oh2_lengths[i] = oh2
        else:
            hoh_angles[i] = np.nan
            oh1_lengths[i] = np.nan
            oh2_lengths[i] = np.nan

    return hoh_angles, oh1_lengths, oh2_lengths


bond_triplets = read_bond_topology(bond_topology_file)
num_mols = len(bond_triplets)

header = "timestep " + " ".join(f"mol{j}" for j in range(num_mols)) + "\n"

with open("HOH.txt", "w") as hoh_file, \
     open("OH1.txt", "w") as oh1_file, \
     open("OH2.txt", "w") as oh2_file:

    hoh_file.write(header)
    oh1_file.write(header)
    oh2_file.write(header)

    for i, filename in enumerate(split_files):
        file_path = os.path.join(split_steps_folder, filename)
        atom_coords, _ = read_xyz_file(file_path)

        hoh_vals, oh1_vals, oh2_vals = analyze_frame(atom_coords, bond_triplets)

        hoh_line = f"{i} " + " ".join(f"{val:.5f}" for val in hoh_vals) + "\n"
        oh1_line = f"{i} " + " ".join(f"{val:.5f}" for val in oh1_vals) + "\n"
        oh2_line = f"{i} " + " ".join(f"{val:.5f}" for val in oh2_vals) + "\n"

        hoh_file.write(hoh_line)
        oh1_file.write(oh1_line)
        oh2_file.write(oh2_line)

        with open(log_file_path, "a") as log_file:
            log_file.write(f"Processed file: {filename}\n")

end_time = time.time()
elapsed = end_time - start_time

with open(log_file_path, "a") as log_file:
    log_file.write(f"Total processing time: {elapsed:.2f} seconds\n")
