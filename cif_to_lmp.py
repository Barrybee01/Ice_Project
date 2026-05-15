import numpy as np


def unscale_coordinates(xs, ys, zs, cell_lengths):
    x = xs * cell_lengths[0]
    y = ys * cell_lengths[1]
    z = zs * cell_lengths[2]
    return x, y, z


def convert_cif_to_lmp(cif_file, output_lmp):
    with open(cif_file, 'r') as infile:
        lines = infile.readlines()

    a = b = c = None
    atoms = []
    read_atoms = False
    for line in lines:
        if "_cell_length_a" in line:
            a = float(line.split()[1])

        elif "_cell_length_b" in line:
            b = float(line.split()[1])

        elif "_cell_length_c" in line:
            c = float(line.split()[1])

        elif "_atom_site_fract_z" in line:
            read_atoms = True
            continue

        elif read_atoms:
            if not line.strip():
                continue
            parts = line.split()

            if len(parts) < 5:
                continue

            label = parts[0]
            atom_symbol = parts[1]

            xs = float(parts[2])
            ys = float(parts[3])
            zs = float(parts[4])
            x, y, z = unscale_coordinates(xs, ys, zs, (a, b, c))

            if atom_symbol == 'O':
                atom_type = 1

            elif atom_symbol == 'H':
                atom_type = 2

            atoms.append((atom_type, x, y, z))

    with open(output_lmp, 'w') as outfile:
        outfile.write("#Generated from CIF\n\n")
        outfile.write(f"{len(atoms)} atoms\n")
        outfile.write("2 atom types\n\n")
        outfile.write(f"0.0 {a:.10f} xlo xhi\n")
        outfile.write(f"0.0 {b:.10f} ylo yhi\n")
        outfile.write(f"0.0 {c:.10f} zlo zhi\n\n")
        outfile.write("Masses\n\n")
        outfile.write("1 15.999 # O\n")
        outfile.write("2 1.008  # H\n\n")
        outfile.write("Atoms # atomic\n\n")

        for i, (atom_type, x, y, z) in enumerate(atoms, start=1):

            outfile.write(
                f"{i:8d} {atom_type:4d} "
                f"{x:15.9f} {y:15.9f} {z:15.9f}\n"
            )


if __name__ == "__main__":

    convert_cif_to_lmp(
        "Ice_1h_Supercell.cif",
        "test_structure.lmp"
    )
