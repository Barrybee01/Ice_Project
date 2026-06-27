def unscale_coordinates(xs, ys, zs, box_bounds, xy=0.0, xz=0.0, yz=0.0):
    lx = box_bounds[0][1] - box_bounds[0][0]
    ly = box_bounds[1][1] - box_bounds[1][0]
    lz = box_bounds[2][1] - box_bounds[2][0]

    x = xs * lx + ys * xy + zs * xz
    y = ys * ly + zs * yz
    z = zs * lz
    return x, y, z

def read_box_bounds(lines, start_index):
    box_bounds = []
    xy = 0.0
    xz = 0.0
    yz = 0.0

    box_header = lines[start_index].split()
    triclinic = ("xy" in box_header and "xz" in box_header and "yz" in box_header)

    for i in range(3):
        parts = lines[start_index + 1 + i].split()
        box_bounds.append((float(parts[0]), float(parts[1])))

        if triclinic:
            if i == 0:
                xy = float(parts[2])
            elif i == 1:
                xz = float(parts[2])
            elif i == 2:
                yz = float(parts[2])

    a = box_bounds[0][1] - box_bounds[0][0]
    b = box_bounds[1][1] - box_bounds[1][0]
    c = box_bounds[2][1] - box_bounds[2][0]

    return box_bounds, a, b, c, xy, xz, yz


def get_coordinate_indices(atom_columns, coordinate_mode):
    if coordinate_mode not in ["auto", "scaled", "unscaled"]:
        raise ValueError("coordinate_mode must be 'auto', 'scaled', or 'unscaled'.")

    if coordinate_mode == "scaled":
        return (atom_columns.index("xs"), atom_columns.index("ys"), atom_columns.index("zs"), "scaled")

    if coordinate_mode == "unscaled":
        if "x" in atom_columns and "y" in atom_columns and "z" in atom_columns:
            return (atom_columns.index("x"), atom_columns.index("y"), atom_columns.index("z"), "unscaled")

        if "xu" in atom_columns and "yu" in atom_columns and "zu" in atom_columns:
            return (atom_columns.index("xu"),atom_columns.index("yu"),atom_columns.index("zu"),"unscaled")

        raise ValueError("Could not find x/y/z or xu/yu/zu columns.")

    if "xs" in atom_columns and "ys" in atom_columns and "zs" in atom_columns:
        return (atom_columns.index("xs"),atom_columns.index("ys"),atom_columns.index("zs"),"scaled")

    if "x" in atom_columns and "y" in atom_columns and "z" in atom_columns:
        return (atom_columns.index("x"),atom_columns.index("y"), atom_columns.index("z"),"unscaled")

    if "xu" in atom_columns and "yu" in atom_columns and "zu" in atom_columns:
        return (atom_columns.index("xu"),atom_columns.index("yu"),atom_columns.index("zu"),"unscaled")

    raise ValueError("Could not find xs/ys/zs, x/y/z, or xu/yu/zu columns.")

def lammpstrj_to_modified_xyz(lammpstrj_file,xyz_file,coordinate_mode="auto"):
    with open(lammpstrj_file, "r") as infile:
        lines = infile.readlines()

    with open(xyz_file, "w") as outfile:
        i = 0

        while i < len(lines):
            if not lines[i].startswith("ITEM: TIMESTEP"):
                i += 1
                continue

            timestep = int(lines[i + 1].strip())

            if not lines[i + 2].startswith("ITEM: NUMBER OF ATOMS"):
                raise ValueError("Expected ITEM: NUMBER OF ATOMS after timestep.")

            n_atoms = int(lines[i + 3].strip())

            if not lines[i + 4].startswith("ITEM: BOX BOUNDS"):
                raise ValueError("Expected ITEM: BOX BOUNDS after number of atoms.")

            box_bounds, a, b, c, xy, xz, yz = read_box_bounds(lines, i + 4)

            atom_header_index = i + 8

            if not lines[atom_header_index].startswith("ITEM: ATOMS"):
                raise ValueError("Expected ITEM: ATOMS after box bounds.")

            atom_columns = lines[atom_header_index].split()[2:]

            id_index = atom_columns.index("id")
            type_index = atom_columns.index("type")

            x_index, y_index, z_index, active_coordinate_mode = get_coordinate_indices(atom_columns,coordinate_mode)

            atoms = []
            atom_start = atom_header_index + 1
            atom_end = atom_start + n_atoms

            for line in lines[atom_start:atom_end]:
                parts = line.split()

                atom_id = int(parts[id_index])
                atom_type = int(parts[type_index])

                x_raw = float(parts[x_index])
                y_raw = float(parts[y_index])
                z_raw = float(parts[z_index])

                if active_coordinate_mode == "scaled":
                    x, y, z = unscale_coordinates(x_raw,y_raw,z_raw,box_bounds,xy,xz,yz,)
                else:
                    x, y, z = x_raw, y_raw, z_raw

                atoms.append((atom_id, atom_type, x, y, z))

            atoms.sort(key=lambda atom: atom[0])

            outfile.write(f"{n_atoms}\n")
            outfile.write(
                f"{a:.10f} {b:.10f} {c:.10f} "
                f"{xy:.10f} {xz:.10f} {yz:.10f} "
                f"timestep {timestep}\n")

            for atom_id, atom_type, x, y, z in atoms:
                outfile.write(f"{atom_type} {x:.10f} {y:.10f} {z:.10f}\n")

            i = atom_end

if __name__ == "__main__":
    lammpstrj_to_modified_xyz("dump.lammpstrj", "dump_modified.xyz",coordinate_mode="auto")
