import numpy as np
from collections import defaultdict

def read_lammps_structure(filename):
    """Read LAMMPS structure file and return atom information"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Find relevant sections
    atoms_start = None
    masses = {}
    box = None
    
    for i, line in enumerate(lines):
        if "xlo xhi" in line:
            xlo, xhi = map(float, line.split()[:2])
            ylo, yhi = map(float, lines[i+1].split()[:2])
            zlo, zhi = map(float, lines[i+2].split()[:2])
            box = np.array([[xlo, xhi], [ylo, yhi], [zlo, zhi]])
        elif "Masses" in line:
            j = i + 2
            while lines[j].strip() and not lines[j].startswith('Atoms'):
                parts = lines[j].split()
                masses[int(parts[0])] = float(parts[1])
                j += 1
        elif "Atoms" in line:
            atoms_start = i + 2
    
    # Read atom data
    atoms = []
    for line in lines[atoms_start:]:
        if not line.strip():
            continue
        parts = line.split()
        atom_id = int(parts[0])
        atom_type = int(parts[1])
        x, y, z = map(float, parts[2:5])
        atoms.append({
            'id': atom_id,
            'type': atom_type,
            'pos': np.array([x, y, z]),
            'mass': masses[atom_type]
        })
    
    return atoms, box

def find_oh_bonds(atoms, box, cutoff=1.15):
    """Find O-H bonds within cutoff distance considering periodic boundaries"""
    # Separate O and H atoms
    oxygen = [a for a in atoms if np.isclose(a['mass'], 15.999)]
    hydrogen = [a for a in atoms if np.isclose(a['mass'], 1.008)]
    
    # Build bond map
    bond_map = defaultdict(list)
    box_size = np.array([box[i][1] - box[i][0] for i in range(3)])
    
    for o in oxygen:
        for h in hydrogen:
            # Calculate minimum image distance
            delta = o['pos'] - h['pos']
            delta -= np.round(delta / box_size) * box_size
            distance = np.linalg.norm(delta)
            
            if distance < cutoff:
                bond_map[o['id']].append(h['id'])
    
    return bond_map

def write_bond_topology(bond_map, output_file):
    """Write bond topology to output file"""
    with open(output_file, 'w') as f:
        f.write("# O_id    H1_id    H2_id\n")
        for o_id, h_ids in bond_map.items():
            if len(h_ids) == 2:  # Only write if O has exactly 2 H bonds
                f.write(f"{o_id:8d}{h_ids[0]:8d}{h_ids[1]:8d}\n")
            elif len(h_ids) > 2:
                print(f"Warning: Oxygen {o_id} has more than 2 H bonds ({len(h_ids)})")
            else:
                print(f"Warning: Oxygen {o_id} has less than 2 H bonds ({len(h_ids)})")

def main():
    input_file = "iceIh_96_4x4x4.lmp"  # Change to your input filename
    output_file = "oh_bond_topology.txt"
    
    atoms, box = read_lammps_structure(input_file)
    bond_map = find_oh_bonds(atoms, box, cutoff=1.15)
    write_bond_topology(bond_map, output_file)
    
    print(f"Bond topology written to {output_file}")

if __name__ == "__main__":
    main()