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


def unscale_coordinates(xs, ys, zs, box_bounds):
    x = xs * (box_bounds[0][1] - box_bounds[0][0]) + box_bounds[0][0]
    y = ys * (box_bounds[1][1] - box_bounds[1][0]) + box_bounds[1][0]
    z = zs * (box_bounds[2][1] - box_bounds[2][0]) + box_bounds[2][0]
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

        if triclinic:
            box_bounds.append((float(parts[0]), float(parts[1])))

            if i == 0:
                xy = float(parts[2])
            elif i == 1:
                xz = float(parts[2])
            elif i == 2:
                yz = float(parts[2])

        else:
            box_bounds.append((float(parts[0]), float(parts[1])))

    a = box_bounds[0][1] - box_bounds[0][0]
    b = box_bounds[1][1] - box_bounds[1][0]
    c = box_bounds[2][1] - box_bounds[2][0]

    return box_bounds, a, b, c, xy, xz, yz


def get_coordinate_indices(atom_columns, coordinate_mode):
    if coordinate_mode not in ["auto", "scaled", "unscaled"]:
        raise ValueError("coordinate_mode must be 'auto', 'scaled', or 'unscaled'.")

    if coordinate_mode == "scaled":
        if "xs" not in atom_columns or "ys" not in atom_columns or "zs" not in atom_columns:
            raise ValueError("coordinate_mode='scaled' requires xs, ys, and zs columns.")

        x_index = atom_columns.index("xs")
        y_index = atom_columns.index("ys")
        z_index = atom_columns.index("zs")

        return x_index, y_index, z_index, "scaled"

    if coordinate_mode == "unscaled":
        if "x" in atom_columns and "y" in atom_columns and "z" in atom_columns:
            x_index = atom_columns.index("x")
            y_index = atom_columns.index("y")
            z_index = atom_columns.index("z")

        elif "xu" in atom_columns and "yu" in atom_columns and "zu" in atom_columns:
            x_index = atom_columns.index("xu")
            y_index = atom_columns.index("yu")
            z_index = atom_columns.index("zu")

        else:
            raise ValueError("coordinate_mode='unscaled' requires x/y/z or xu/yu/zu columns.")

        return x_index, y_index, z_index, "unscaled"

    if "xs" in atom_columns and "ys" in atom_columns and "zs" in atom_columns:
        x_index = atom_columns.index("xs")
        y_index = atom_columns.index("ys")
        z_index = atom_columns.index("zs")

        return x_index, y_index, z_index, "scaled"

    if "x" in atom_columns and "y" in atom_columns and "z" in atom_columns:
        x_index = atom_columns.index("x")
        y_index = atom_columns.index("y")
        z_index = atom_columns.index("z")

        return x_index, y_index, z_index, "unscaled"

    if "xu" in atom_columns and "yu" in atom_columns and "zu" in atom_columns:
        x_index = atom_columns.index("xu")
        y_index = atom_columns.index("yu")
        z_index = atom_columns.index("zu")

        return x_index, y_index, z_index, "unscaled"

    raise ValueError("Could not find xs/ys/zs, x/y/z, or xu/yu/zu columns.")


def lammpstrj_to_xyz(lammpstrj_file, xyz_file, type_map=None, coordinate_mode="auto"):
    type_map = parse_type_map(type_map)

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

            if "id" not in atom_columns:
                raise ValueError("Could not find id column in ITEM: ATOMS header.")

            if "type" not in atom_columns:
                raise ValueError("Could not find type column in ITEM: ATOMS header.")

            id_index = atom_columns.index("id")
            type_index = atom_columns.index("type")

            x_index, y_index, z_index, active_coordinate_mode = get_coordinate_indices(atom_columns, coordinate_mode)

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
                    x, y, z = unscale_coordinates(x_raw, y_raw, z_raw, box_bounds)
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
                symbol = atom_type_to_symbol(atom_type, type_map)
                outfile.write(f"{symbol} {x:.10f} {y:.10f} {z:.10f}\n")

            i = atom_end

if __name__ == "__main__":
    lammpstrj_to_xyz("dump.lammpstrj", "dump_trj.xyz", type_map={1: "O", 2: "H"}, coordinate_mode="auto")
