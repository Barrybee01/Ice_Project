def xyz_to_gro(xyz_file,gro_file,residue_name="MOL",title="Generated from XYZ",):
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

    atom_lines = lines[2:2 + n_atoms]

    if len(atom_lines) < n_atoms:
        raise ValueError(
            f"XYZ file says there are {n_atoms} atoms, "
            f"but only {len(atom_lines)} atom lines were found.")

    scale = 0.1 
    with open(gro_file, "w") as outfile:
        outfile.write(f"{title}\n")
        outfile.write(f"{n_atoms:5d}\n")

        residue_id = 1

        for atom_id, line in enumerate(atom_lines, start=1):
            parts = line.split()

            if len(parts) < 4:
                raise ValueError(f"Invalid XYZ atom line: {line}")

            atom_name = parts[0]
            x = float(parts[1]) * scale
            y = float(parts[2]) * scale
            z = float(parts[3]) * scale

            outfile.write(
                f"{residue_id % 100000:5d}"
                f"{residue_name[:5]:<5s}"
                f"{atom_name[:5]:>5s}"
                f"{atom_id % 100000:5d}"
                f"{x:8.3f}"
                f"{y:8.3f}"
                f"{z:8.3f}\n")

        a *= scale
        b *= scale
        c *= scale
        xy *= scale
        xz *= scale
        yz *= scale

        if xy == 0.0 and xz == 0.0 and yz == 0.0:
            outfile.write(f"{a:10.5f} {b:10.5f} {c:10.5f}\n")
        else:
            outfile.write(
                f"{a:10.5f} {b:10.5f} {c:10.5f} "
                f"{0.0:10.5f} {0.0:10.5f} "
                f"{xy:10.5f} {0.0:10.5f} "
                f"{xz:10.5f} {yz:10.5f}\n")

if __name__ == "__main__":
    xyz_to_gro("structure.xyz","structure.gro",residue_name="MOL",title="Generated from XYZ")
