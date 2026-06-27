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

def cell_to_lammps_box(a, b, c, alpha, beta, gamma):
    alpha = math.radians(alpha)
    beta = math.radians(beta)
    gamma = math.radians(gamma)

    lx = a
    xy = b * math.cos(gamma)
    xz = c * math.cos(beta)

    ly_squared = b**2 - xy**2
    ly = math.sqrt(max(ly_squared, 0.0))

    yz = (b * c * math.cos(alpha) - xy * xz) / ly

    lz_squared = c**2 - xz**2 - yz**2
    lz = math.sqrt(max(lz_squared, 0.0))
    return lx, ly, lz, xy, xz, yz

def fractional_to_cartesian(fx, fy, fz, lx, ly, lz, xy, xz, yz):
    x = fx * lx + fy * xy + fz * xz
    y = fy * ly + fz * yz
    z = fz * lz
    return x, y, z

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
    alpha = beta = gamma = 90.0

    atom_headers = []
    atoms = []
    read_atom_headers = False
    read_atoms = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        parts = stripped.split()

        if not stripped:
            continue

        if stripped.startswith("_cell_length_a"):
            a = float(parts[1])

        elif stripped.startswith("_cell_length_b"):
            b = float(parts[1])

        elif stripped.startswith("_cell_length_c"):
            c = float(parts[1])

        elif stripped.startswith("_cell_angle_alpha"):
            alpha = float(parts[1])

        elif stripped.startswith("_cell_angle_beta"):
            beta = float(parts[1])

        elif stripped.startswith("_cell_angle_gamma"):
            gamma = float(parts[1])

    if a is None:
        raise ValueError("Could not find _cell_length_a in CIF file.")

    if b is None:
        raise ValueError("Could not find _cell_length_b in CIF file.")

    if c is None:
        raise ValueError("Could not find _cell_length_c in CIF file.")

    lx, ly, lz, xy, xz, yz = cell_to_lammps_box(a,b,c,alpha,beta,gamma,)

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()

        if stripped == "loop_":
            headers = []
            j = i + 1

            while j < len(lines):
                candidate = lines[j].strip()

                if candidate.startswith("_"):
                    headers.append(candidate)
                    j += 1
                else:
                    break

            if ("_atom_site_type_symbol" in headers and "_atom_site_fract_x" in headers and "_atom_site_fract_y" in headers and "_atom_site_fract_z" in headers):
                type_index = headers.index("_atom_site_type_symbol")
                fx_index = headers.index("_atom_site_fract_x")
                fy_index = headers.index("_atom_site_fract_y")
                fz_index = headers.index("_atom_site_fract_z")

                k = j
                while k < len(lines):
                    atom_line = lines[k].strip()

                    if not atom_line:
                        break

                    if atom_line.startswith("#"):
                        break

                    if atom_line.startswith("loop_"):
                        break

                    if atom_line.startswith("_"):
                        break

                    parts = atom_line.split()

                    if len(parts) < len(headers):
                        k += 1
                        continue

                    symbol = parts[type_index]
                    fx = float(parts[fx_index])
                    fy = float(parts[fy_index])
                    fz = float(parts[fz_index])
                    fx = fx % 1 #wrapping the coords
                    fy = fy % 1
                    fz = fz % 1

                    atom_type = symbol_to_atom_type(symbol, symbol_map)
                    x, y, z = fractional_to_cartesian(fx,fy,fz,lx,ly,lz,xy,xz,yz)
                    atoms.append((atom_type, x, y, z))

                    k += 1
                break
            i = j
        else:
            i += 1

    if not atoms:
        raise ValueError("Could not find atom loop with _atom_site_type_symbol and fractional coordinates.")

    atom_types = sorted(set(atom[0] for atom in atoms))

    with open(output_lmp, "w") as outfile:
        outfile.write("# Generated from CIF\n\n")
        outfile.write(f"{len(atoms)} atoms\n")
        outfile.write(f"{len(atom_types)} atom types\n\n")
        outfile.write(f"0.0 {lx:.10f} xlo xhi\n")
        outfile.write(f"0.0 {ly:.10f} ylo yhi\n")
        outfile.write(f"0.0 {lz:.10f} zlo zhi\n")
        outfile.write(f"{xy:.10f} {xz:.10f} {yz:.10f} xy xz yz\n\n")
        outfile.write("Atoms # atomic\n\n")

        for atom_id, (atom_type, x, y, z) in enumerate(atoms, start=1):
            outfile.write(
                f"{atom_id:8d} {atom_type:4d} "
                f"{x:15.9f} {y:15.9f} {z:15.9f}\n")

if __name__ == "__main__":
    cif_to_lmp("iceIV_ortho_x464.cif", "iceIV_ortho_x464_from_cif.lmp", symbol_map={"O": 1, "H": 2})
                f"{atom_id:8d} {atom_type:4d} "
                f"{x:15.9f} {y:15.9f} {z:15.9f}\n")
