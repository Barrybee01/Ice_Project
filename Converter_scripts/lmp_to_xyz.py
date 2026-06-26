import numpy as np

def unscale_coordinates(xs, ys, zs, box_bounds):
    x = xs * (box_bounds[0][1] - box_bounds[0][0]) + box_bounds[0][0]
    y = ys * (box_bounds[1][1] - box_bounds[1][0]) + box_bounds[1][0]
    z = zs * (box_bounds[2][1] - box_bounds[2][0]) + box_bounds[2][0]
    return x, y, z


def convert_single_frame(lmp_file, xyz_file):
    with open(lmp_file, 'r') as infile, open(xyz_file, 'w') as outfile:

        infile.readline()  # ITEM: TIMESTEP
        timestep = infile.readline().strip()

        infile.readline()  # ITEM: NUMBER OF ATOMS
        num_atoms = int(infile.readline().strip())

        infile.readline()  # ITEM: BOX BOUNDS
        box_bounds = []
        for _ in range(3):
            bounds = list(map(float, infile.readline().strip().split()))
            box_bounds.append(bounds)

        infile.readline()  # ITEM: ATOMS ...

        outfile.write(f"{num_atoms}\n")
        outfile.write(f"Timestep: {timestep}\n")

        for _ in range(num_atoms):
            data = infile.readline().strip().split()

            atom_type = data[1]
            xs = float(data[2])
            ys = float(data[3])
            zs = float(data[4])

            x, y, z = unscale_coordinates(xs, ys, zs, box_bounds)

            # Map atom types
            if atom_type == '1':
                atom_type = 'O'
            elif atom_type == '2':
                atom_type = 'H'

            outfile.write(f"{atom_type} {x:.6f} {y:.6f} {z:.6f}\n")


if __name__ == "__main__":
    convert_single_frame("final_state.lmp", "final_state.xyz")
