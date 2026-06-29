def parse_gro_atom_line(line):
    atom_name = line[10:15].strip()
    x = float(line[20:28])
    y = float(line[28:36])
    z = float(line[36:44])
    return atom_name, x, y, z

def parse_gro_box(line):
    parts = [float(value) for value in line.split()]

    if len(parts) == 3:
        a = parts[0]
        b = parts[1]
        c = parts[2]
        xy = 0.0
        xz = 0.0
        yz = 0.0

    elif len(parts) == 9:
        a = parts[0]
        b = parts[1]
        c = parts[2]
        xy = parts[5]
        xz = parts[7]
        yz = parts[8]

    else:
        raise ValueError("GRO box line must contain either 3 values or 9 values.")

    return a, b, c, xy, xz, yz

def gro_to_xyz(gro_file, xyz_file):
    with open(gro_file, "r") as infile:
        lines = infile.readlines()

    if len(lines) < 3:
        raise ValueError("GRO file is too short.")

    n_atoms = int(lines[1].strip())

    atom_lines = lines[2:2 + n_atoms]
    box_line = lines[2 + n_atoms].strip()

    if len(atom_lines) < n_atoms:
        raise ValueError(
            f"GRO file says there are {n_atoms} atoms, "
            f"but only {len(atom_lines)} atom lines were found.")

    scale = 10.0
    atoms = []
    for line in atom_lines:
        atom_name, x, y, z = parse_gro_atom_line(line)

        atoms.append((atom_name, x * scale, y * scale, z * scale))

    a, b, c, xy, xz, yz = parse_gro_box(box_line)

    a *= scale
    b *= scale
    c *= scale
    xy *= scale
    xz *= scale
    yz *= scale

    with open(xyz_file, "w") as outfile:
        outfile.write(f"{n_atoms}\n")
        outfile.write(
            f"{a:.10f} {b:.10f} {c:.10f} "
            f"{xy:.10f} {xz:.10f} {yz:.10f}\n")

        for atom_name, x, y, z in atoms:
            outfile.write(f"{atom_name} {x:.10f} {y:.10f} {z:.10f}\n")

if __name__ == "__main__":
    gro_to_xyz("structure.gro", "structure.xyz")
