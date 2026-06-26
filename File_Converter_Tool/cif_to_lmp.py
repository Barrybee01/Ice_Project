def parse_symbol_map(symbol_map=None):
    if symbol_map is None:
        raise ValueError( "A symbol_map is required, for example: {'O': 1, 'H': 2}")

    parsed_map = {}
    for key, value in symbol_map.items():
        parsed_map[str(key)] = int(value)
    return parsed_map

def symbol_to_atom_type(symbol, symbol_map=None):
    symbol_map = parse_symbol_map(symbol_map)

    if symbol not in symbol_map:
        raise ValueError(f"No atom type defined for symbol '{symbol}'.")
    return symbol_map[symbol]

def unscale_coordinates(xs, ys, zs, cell_lengths):
    x = xs * cell_lengths[0]
    y = ys * cell_lengths[1]
    z = zs * cell_lengths[2]
    return x, y, z

def cif_to_lmp(cif_file, output_lmp, symbol_map=None):
    symbol_map = parse_symbol_map(symbol_map)

    with open(cif_file, "r") as infile:
        lines = infile.readlines()

    a = b = c = None
    atoms = []
    read_atoms = False

    for line in lines:
        stripped = line.strip()
        parts = stripped.split()

        if "_cell_length_a" in line:
            a = float(parts[1])

        elif "_cell_length_b" in line:
            b = float(parts[1])

        elif "_cell_length_c" in line:
            c = float(parts[1])

        elif "_atom_site_fract_z" in line:
            read_atoms = True
            continue

        elif read_atoms:
            if not stripped:
                continue

            if stripped.startswith("#"):
                continue

            parts = stripped.split()

            if len(parts) < 5:
                continue

            atom_symbol = parts[1]
            xs = float(parts[2])
            ys = float(parts[3])
            zs = float(parts[4])

            atom_type = symbol_to_atom_type(atom_symbol, symbol_map)
            x, y, z = unscale_coordinates(xs, ys, zs, (a, b, c))

            atoms.append((atom_type, x, y, z))

    if a is None:
        raise ValueError("Could not find _cell_length_a in CIF file.")

    if b is None:
        raise ValueError("Could not find _cell_length_b in CIF file.")

    if c is None:
        raise ValueError("Could not find _cell_length_c in CIF file.")

    atom_types = sorted(set(atom[0] for atom in atoms))

    with open(output_lmp, "w") as outfile:
        outfile.write("# Generated from CIF\n\n")
        outfile.write(f"{len(atoms)} atoms\n")
        outfile.write(f"{len(atom_types)} atom types\n\n")

        outfile.write(f"0.0 {a:.10f} xlo xhi\n")
        outfile.write(f"0.0 {b:.10f} ylo yhi\n")
        outfile.write(f"0.0 {c:.10f} zlo zhi\n\n")

        outfile.write("Atoms # atomic\n\n")

        for atom_id, (atom_type, x, y, z) in enumerate(atoms, start=1):
            outfile.write(
                f"{atom_id:8d} {atom_type:4d} "
                f"{x:15.9f} {y:15.9f} {z:15.9f}\n")
