def parse_type_map(type_map=None):
    if type_map is None:
        return {}

    parsed_map = {}

    for key, value in type_map.items():
        parsed_map[int(key)] = str(value)
    return parsed_map

def atom_type_to_symbol(atom_type, type_map=None):
    type_map = parse_type_map(type_map)
    return type_map.get(atom_type, f"X{atom_type}")

def lmp_to_cif(lmp_file,cif_file,type_map=None,lattice_type="orthorhombic",space_group="P 1",):
    type_map = parse_type_map(type_map)

    with open(lmp_file, "r") as infile:
        lines = infile.readlines()

    xlo = xhi = None
    ylo = yhi = None
    zlo = zhi = None
    xy = 0.0
    xz = 0.0
    yz = 0.0

    atoms = []
    read_atoms = False

    for line in lines:
        stripped = line.strip()
        parts = stripped.split()

        if len(parts) >= 4 and parts[-2:] == ["xlo", "xhi"]:
            xlo = float(parts[0])
            xhi = float(parts[1])

        elif len(parts) >= 4 and parts[-2:] == ["ylo", "yhi"]:
            ylo = float(parts[0])
            yhi = float(parts[1])

        elif len(parts) >= 4 and parts[-2:] == ["zlo", "zhi"]:
            zlo = float(parts[0])
            zhi = float(parts[1])

        elif len(parts) >= 6 and parts[-3:] == ["xy", "xz", "yz"]:
            xy = float(parts[0])
            xz = float(parts[1])
            yz = float(parts[2])

        elif stripped.startswith("Atoms"):
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

            atom_id = int(parts[0])
            atom_type = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            z = float(parts[4])

            atoms.append((atom_id, atom_type, x, y, z))

    if xlo is None or xhi is None:
        raise ValueError("Could not find xlo/xhi bounds in LMP file.")

    if ylo is None or yhi is None:
        raise ValueError("Could not find ylo/yhi bounds in LMP file.")

    if zlo is None or zhi is None:
        raise ValueError("Could not find zlo/zhi bounds in LMP file.")

    a = xhi - xlo
    b = yhi - ylo
    c = zhi - zlo

    atoms.sort(key=lambda atom: atom[0])

    with open(cif_file, "w") as outfile:
        outfile.write("data_generated_from_lmp\n")
        outfile.write("#\n")
        outfile.write("_audit_creation_method        'Generated from LAMMPS data file'\n")
        outfile.write(f"_cell_length_a                {a:.12f}\n")
        outfile.write(f"_cell_length_b                {b:.12f}\n")
        outfile.write(f"_cell_length_c                {c:.12f}\n")
        outfile.write("_cell_angle_alpha             90.0\n")
        outfile.write("_cell_angle_beta              90.0\n")
        outfile.write("_cell_angle_gamma             90.0\n")
        outfile.write(f"_symmetry_cell_setting        '{lattice_type}'\n")
        outfile.write(f"_symmetry_space_group_name_H-M   '{space_group}'\n\n")
        outfile.write("loop_\n")
        outfile.write("  _symmetry_equiv_pos_as_xyz\n")
        outfile.write("  'x, y, z'\n\n")
        outfile.write("loop_\n")
        outfile.write("_atom_site_label\n")
        outfile.write("_atom_site_type_symbol\n")
        outfile.write("_atom_site_fract_x\n")
        outfile.write("_atom_site_fract_y\n")
        outfile.write("_atom_site_fract_z\n")

        for atom_id, atom_type, x, y, z in atoms:
            symbol = atom_type_to_symbol(atom_type, type_map)

            frac_x = (x - xlo) / a
            frac_y = (y - ylo) / b
            frac_z = (z - zlo) / c

            label = f"{symbol}{atom_id}"

            outfile.write(
                f"{label:>8s} {symbol:>4s} "
                f"{frac_x:12.8f} {frac_y:12.8f} {frac_z:12.8f}\n")
