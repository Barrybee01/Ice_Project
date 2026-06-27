import math

def cell_to_lammps_box(a, b, c, alpha, beta, gamma):
    alpha = math.radians(alpha)
    beta = math.radians(beta)
    gamma = math.radians(gamma)

    lx = a
    xy = b * math.cos(gamma)
    xz = c * math.cos(beta)

    ly_squared = b * b - xy * xy
    ly = math.sqrt(max(ly_squared, 0.0))
    yz = (b * c * math.cos(alpha) - xy * xz) / ly
    lz_squared = c * c - xz * xz - yz * yz
    lz = math.sqrt(max(lz_squared, 0.0))
    return lx, ly, lz, xy, xz, yz


def fractional_to_cartesian(fx, fy, fz, lx, ly, lz, xy, xz, yz):
    x = fx * lx + fy * xy + fz * xz
    y = fy * ly + fz * yz
    z = fz * lz
    return x, y, z

def cif_to_xyz(cif_file, xyz_file):
    with open(cif_file, "r") as infile:
        lines = infile.readlines()

    a = b = c = None
    alpha = beta = gamma = 90.0
    atoms = []

    for line in lines:
        stripped = line.strip()
        parts = stripped.split()

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

    if a is None or b is None or c is None:
        raise ValueError("Could not find complete cell lengths in CIF file.")

    lx, ly, lz, xy, xz, yz = cell_to_lammps_box(a, b, c, alpha, beta, gamma)

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
                    fx = fx % 1
                    fy = fy % 1
                    fz = fz % 1

                    x, y, z = fractional_to_cartesian(fx, fy, fz, lx, ly, lz, xy, xz, yz)
                    atoms.append((symbol, x, y, z))

                    k += 1
                break
            i = j
        else:
            i += 1

    if not atoms:
        raise ValueError("Could not find atom loop with fractional coordinates.")

    with open(xyz_file, "w") as outfile:
        outfile.write(f"{len(atoms)}\n")
        outfile.write(
            f"{lx:.10f} {ly:.10f} {lz:.10f} "
            f"{xy:.10f} {xz:.10f} {yz:.10f}\n")

        for symbol, x, y, z in atoms:
            outfile.write(f"{symbol} {x:.10f} {y:.10f} {z:.10f}\n")

if __name__ == "__main__":
    cif_to_xyz("structure.cif","structure.xyz",)
