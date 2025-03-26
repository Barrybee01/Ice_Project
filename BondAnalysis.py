import numpy as np

def read_xyz(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        num_atoms = int(lines[0])
        atom_data = lines[2:2+num_atoms]
        atom_types = []
        coordinates = []
        for line in atom_data:
            parts = line.split()
            atom_types.append(parts[0])
            coordinates.append(list(map(float, parts[1:4])))
        return atom_types, np.array(coordinates)

def calculate_distance(coord1, coord2, box=None):
    diff = coord1 - coord2
    if box is not None:
        diff -= np.round(diff / box) * box  # corrects for PBCs that affect boundary molecules
    return np.linalg.norm(diff)

def calculate_angle(vec1, vec2):
    cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

def analyze_ooo_angles(atom_types, coordinates, cutoff_radius, box=None):
    oxygen_indices = [i for i, atom in enumerate(atom_types) if atom == 'O']
    results = []

    for i in oxygen_indices:
        neighbors = []
        for j in oxygen_indices:
            if i != j:
                dist = calculate_distance(coordinates[i], coordinates[j], box)
                if dist <= cutoff_radius:
                    neighbors.append((j, coordinates[j]))

        # Ensure triplets are unique and no duplicate angles are calculated
        for k in range(len(neighbors)):
            for l in range(k + 1, len(neighbors)):
                if neighbors[k][0] < neighbors[l][0]:  # Ensure uniqueness by ordering the pair
                    vec1 = neighbors[k][1] - coordinates[i]
                    vec2 = neighbors[l][1] - coordinates[i]
                    if box is not None:
                        vec1 -= np.round(vec1 / box) * box
                        vec2 -= np.round(vec2 / box) * box
                    angle = calculate_angle(vec1, vec2)
                    results.append((i, angle))

    with open('ooo_bond_angles.txt', 'w') as file:
        for atom_id, angle in results:
            file.write(f"{atom_id} {angle:.2f}\n")

    print(f"Total OOO angles calculated: {len(results)}")

def analyze_oh_bond_lengths_and_hoh_angles(atom_types, coordinates, oh_cutoff_radius, hoh_cutoff_radius, box=None):
    """
    Analyzes O-H bond lengths and H-O-H bond angles and writes them to files.
    """
    oxygen_indices = [i for i, atom in enumerate(atom_types) if atom == 'O']
    hydrogen_indices = [i for i, atom in enumerate(atom_types) if atom == 'H']
    bond_lengths = []
    hoh_angles = []

    for i in oxygen_indices:
        neighbors = []
        for j in hydrogen_indices:
            dist = calculate_distance(coordinates[i], coordinates[j], box)
            if dist <= oh_cutoff_radius:
                neighbors.append((j, dist))
        
        if len(neighbors) == 2:
            neighbors.sort(key=lambda x: x[1])  # Sort by distance
            h1_id, h1_dist = neighbors[0]
            h2_id, h2_dist = neighbors[1]
            bond_lengths.append((i, h1_dist, h2_dist))

            vec1 = coordinates[h1_id] - coordinates[i]
            vec2 = coordinates[h2_id] - coordinates[i]
            if box is not None:
                vec1 -= np.round(vec1 / box) * box
                vec2 -= np.round(vec2 / box) * box
            angle = calculate_angle(vec1, vec2)
            hoh_angles.append((i, angle))  # Store oxygen ID with angle
        elif len(neighbors) != 2:
            print(f"Skipping Oxygen {i} (found {len(neighbors)} hydrogens)")

    with open('oh_bond_lengths.txt', 'w') as file:
        for atom_id, h1_dist, h2_dist in bond_lengths:
            file.write(f"{atom_id} {h1_dist:.3f} {h2_dist:.3f}\n")
    
    with open('hoh_angles.txt', 'w') as file:
        for oxygen_id, angle in hoh_angles:
            file.write(f"{oxygen_id} {angle:.2f}\n")
    
    print(f"Total valid molecules: {len(bond_lengths)}")
    print(f"Total HOH angles calculated: {len(hoh_angles)}")



if __name__ == "__main__":
    ocentric_file = "18kbar only O.xyz"
    h2o_file = "18kbar.xyz"
    box_dimensions = np.array([51.369529, 58.843018, 54.508783])  #Maybe feed it lmp structure to convert to xyz or some shit
    ocentric_atom_types, ocentric_coordinates = read_xyz(ocentric_file)
    analyze_ooo_angles(ocentric_atom_types, ocentric_coordinates, cutoff_radius=2.75, box=box_dimensions)
    h2o_atom_types, h2o_coordinates = read_xyz(h2o_file)
    analyze_oh_bond_lengths_and_hoh_angles(
        h2o_atom_types,
        h2o_coordinates,
        oh_cutoff_radius=1.2,
        hoh_cutoff_radius=1.2,
        box=box_dimensions
    )