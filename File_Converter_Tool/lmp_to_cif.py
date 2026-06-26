def atom_type_to_symbol(atom_type):
    if atom_type == 1:
        return "O"
    elif atom_type == 2:
        return "H"
    else:
        return f"X{atom_type}"


def lmp_to_cif(lmp_file, cif_file, lattice_type="orthorhombic", space_group="P 1"):
    with open(lmp_file, "r") as infile:
        lines = infile.readlines()

    xlo = xhi = None
    ylo = yhi = None
    zlo = zhi = None
    atoms = []
    read_atoms = False

    for line in lines:
        parts = line.split()

        if len(parts) >= 4 and parts[-2:] == ["xlo", "xhi"]:
            xlo = float(parts[0])
            xhi = float(parts[1])

        elif len(parts) >= 4 and parts[-2:] == ["ylo", "yhi"]:
            ylo = float(parts[0])
            yhi = float(parts[1])

        elif len(parts) >= 4 and parts[-2:] == ["zlo", "zhi"]:
            zlo = float(parts[0])
            zhi = float(parts[1])

        elif line.strip().startswith("Atoms"):
            read_atoms = True
            continue

        elif read_atoms:
            if not line.strip():
                continue

            parts = line.split()

            if len(parts) < 5:
                continue

            atom_id = int(parts[0])
            atom_type = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            z = float(parts[4])

            atoms.append((atom_id, atom_type, x, y, z))

    a = xhi - xlo
    b = yhi - ylo
    c = zhi - zlo

    with open(cif_file, "w") as outfile:
        outfile.write("data_generated_from_lmp\n")
        outfile.write("#\n")
        outfile.write("# Generated from LAMMPS data file\n")

        outfile.write(f"_cell_length_a                {a:.12f}\n")
        outfile.write(f"_cell_length_b                {b:.12f}\n")
        outfile.write(f"_cell_length_c                {c:.12f}\n")
        outfile.write("_cell_angle_alpha             90.0\n")
        outfile.write("_cell_angle_beta              90.0\n")
        outfile.write("_cell_angle_gamma             90.0\n\n")

        outfile.write(f"_symmetry_cell_setting        '{lattice_type}'\n")
        outfile.write(f"_symmetry_space_group_name_H-M   '{space_group} '\n\n")

        outfile.write("loop_\n")
        outfile.write("  _symmetry_equiv_pos_as_xyz\n")
        outfile.write("X,Y,Z\n\n")

        outfile.write("loop_\n")
        outfile.write("_atom_site_label\n")
        outfile.write("_atom_site_type_symbol\n")
        outfile.write("_atom_site_fract_x\n")
        outfile.write("_atom_site_fract_y\n")
        outfile.write("_atom_site_fract_z\n")

        for atom_id, atom_type, x, y, z in atoms:
            symbol = atom_type_to_symbol(atom_type)

            frac_x = (x - xlo) / a
            frac_y = (y - ylo) / b
            frac_z = (z - zlo) / c

            label = f"{symbol}{atom_id - 1}"

            outfile.write(
                f"{label:>6s} {symbol:>5s} "
                f"{frac_x:10.6f} {frac_y:10.6f} {frac_z:10.6f}\n")
