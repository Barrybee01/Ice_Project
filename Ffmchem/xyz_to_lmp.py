def parse_symbol_map(symbol_map=None):
    if symbol_map is None:
        raise ValueError("A symbol_map is required, for example: {'O': 1, 'H': 2}")

    parsed_map = {}

    for key, value in symbol_map.items():
        parsed_map[str(key)] = int(value)
    return parsed_map

def symbol_to_atom_type(symbol, symbol_map=None):
    symbol_map = parse_symbol_map(symbol_map)

    if symbol not in symbol_map:
        raise ValueError(f"No atom type defined for symbol '{symbol}'.")
    return symbol_map[symbol]


def xyz_to_lmp(xyz_file, lmp_file, symbol_map=None, mass_map=None):
    symbol_map = parse_symbol_map(symbol_map)

    with open(xyz_file, "r") as infile:
        lines = infile.readlines()

    if len(lines) < 2:
        raise ValueError("XYZ file must contain at least two lines.")

    n_atoms = int(lines[0].strip())

    cell_parts = lines[1].split()

    if len(cell_parts) < 6:
        raise ValueError("XYZ comment line must contain: a b c xy xz yz")

    a = float(cell_parts[0])
    b = float(cell_parts[1])
    c = float(cell_parts[2])
    xy = float(cell_parts[3])
    xz = float(cell_parts[4])
    yz = float(cell_parts[5])

    atom_lines = lines[2:]

    if len(atom_lines) < n_atoms:
        raise ValueError(
            f"XYZ file says there are {n_atoms} atoms, but only {len(atom_lines)} atom lines were found.")

    atoms = []

    for atom_id, line in enumerate(atom_lines[:n_atoms], start=1):
        parts = line.split()

        if len(parts) < 4:
            raise ValueError(f"Invalid XYZ atom line: {line}")

        symbol = parts[0]
        x = float(parts[1])
        y = float(parts[2])
        z = float(parts[3])

        atom_type = symbol_to_atom_type(symbol, symbol_map)

        atoms.append((atom_id, atom_type, x, y, z))

    atom_types = sorted(set(atom[1] for atom in atoms))
    if mass_map is None:
        mass_map = {atom_type: 1.0 for atom_type in atom_types}

    type_map = {atom_type: symbol for symbol, atom_type in symbol_map.items()}

    with open(lmp_file, "w") as outfile:
        outfile.write("# Generated from XYZ\n\n")
        outfile.write(f"{len(atoms)} atoms\n")
        outfile.write(f"{len(atom_types)} atom types\n\n")

        outfile.write(f"0.0 {a:.10f} xlo xhi\n")
        outfile.write(f"0.0 {b:.10f} ylo yhi\n")
        outfile.write(f"0.0 {c:.10f} zlo zhi\n")
        outfile.write(f"{xy:.10f} {xz:.10f} {yz:.10f} xy xz yz\n\n")

        outfile.write("Masses\n\n")
        for atom_type in atom_types:
            mass = mass_map.get(atom_type, 1.0)
            symbol = type_map.get(atom_type, "")
            outfile.write(f"{atom_type:12d} {mass:15.8f} # {symbol}\n")

        outfile.write("\n")
        outfile.write("Atoms # atomic\n\n")

        for atom_id, atom_type, x, y, z in atoms:
            outfile.write(
                f"{atom_id:8d} {atom_type:4d} "
                f"{x:15.9f} {y:15.9f} {z:15.9f}\n")


if __name__ == "__main__": #to test
    xyz_to_lmp("iceIV_ortho_x464.xyz","iceIV_ortho_x464_from_xyz.lmp",symbol_map={"O": 1, "H": 2},mass_map={1: 15.999, 2: 1.008})
