import math

def vector_length(vector):
    return math.sqrt(sum(component * component for component in vector))

def angle_between_vectors(vector_1, vector_2):
    dot_product = sum(a * b for a, b in zip(vector_1, vector_2))
    length_1 = vector_length(vector_1)
    length_2 = vector_length(vector_2)

    cos_angle = dot_product / (length_1 * length_2)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))

def lammps_box_to_cell(a, b, c, xy, xz, yz):
    a_vector = (a, 0.0, 0.0)
    b_vector = (xy, b, 0.0)
    c_vector = (xz, yz, c)

    cell_a = vector_length(a_vector)
    cell_b = vector_length(b_vector)
    cell_c = vector_length(c_vector)

    alpha = angle_between_vectors(b_vector, c_vector)
    beta = angle_between_vectors(a_vector, c_vector)
    gamma = angle_between_vectors(a_vector, b_vector)
    return cell_a, cell_b, cell_c, alpha, beta, gamma

def cartesian_to_fractional(x, y, z, a, b, c, xy, xz, yz):
    fz = z / c
    fy = (y - yz * fz) / b
    fx = (x - xy * fy - xz * fz) / a
    return fx, fy, fz


def xyz_to_cif(xyz_file, cif_file,lattice_type="triclinic",space_group="P 1"):
    with open(xyz_file, "r") as infile:
        lines = infile.readlines()

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

    cell_a, cell_b, cell_c, alpha, beta, gamma = lammps_box_to_cell(a, b, c, xy, xz, yz)
    atom_lines = lines[2:2 + n_atoms]

    with open(cif_file, "w") as outfile:
        outfile.write("data_generated_from_xyz\n")
        outfile.write("#\n")
        outfile.write("_audit_creation_method        'Generated from XYZ file'\n")
        outfile.write(f"_cell_length_a                {cell_a:.12f}\n")
        outfile.write(f"_cell_length_b                {cell_b:.12f}\n")
        outfile.write(f"_cell_length_c                {cell_c:.12f}\n")
        outfile.write(f"_cell_angle_alpha             {alpha:.8f}\n")
        outfile.write(f"_cell_angle_beta              {beta:.8f}\n")
        outfile.write(f"_cell_angle_gamma             {gamma:.8f}\n")
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

        for atom_id, line in enumerate(atom_lines, start=1):
            parts = line.split()

            if len(parts) < 4:
                raise ValueError(f"Invalid XYZ atom line: {line}")

            symbol = parts[0]
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])

            fx, fy, fz = cartesian_to_fractional(x, y, z, a, b, c, xy, xz, yz)
            label = f"{symbol}{atom_id}"
            outfile.write(
                f"{label:>8s} {symbol:>4s} "
                f"{fx:12.8f} {fy:12.8f} {fz:12.8f}\n")

if __name__ == "__main__":
    xyz_to_cif("structure.xyz", "structure.cif",lattice_type="triclinic",space_group="P 1")
