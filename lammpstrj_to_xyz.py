# Author: Rielly Castle

import numpy as np

def unscale_coordinates(xs, ys, zs, box_bounds):
    """Unscale the coordinates based on box bounds."""
    x = xs * (box_bounds[0][1] - box_bounds[0][0]) + box_bounds[0][0]
    y = ys * (box_bounds[1][1] - box_bounds[1][0]) + box_bounds[1][0]
    z = zs * (box_bounds[2][1] - box_bounds[2][0]) + box_bounds[2][0]
    return x, y, z

def compute_lattice_params(box_bounds):
    """Compute lattice parameters (a, b, c, alpha, beta, gamma) from box bounds."""
    xlo, xhi, xy = box_bounds[0]
    ylo, yhi, xz = box_bounds[1]
    zlo, zhi, yz = box_bounds[2]
    
    # Basis vectors
    a_vec = np.array([xhi - xlo, 0, 0])
    b_vec = np.array([xy, yhi - ylo, 0])
    c_vec = np.array([xz, yz, zhi - zlo])
    
    # Lattice lengths
    a = np.linalg.norm(a_vec)
    b = np.linalg.norm(b_vec)
    c = np.linalg.norm(c_vec)
    
    # Lattice angles (degrees)
    alpha = np.degrees(np.arccos(np.dot(b_vec, c_vec) / (b * c)))
    beta = np.degrees(np.arccos(np.dot(a_vec, c_vec) / (a * c)))
    gamma = np.degrees(np.arccos(np.dot(a_vec, b_vec) / (a * b)))
    
    return a, b, c, alpha, beta, gamma

def convert_to_xyz(lammps_dump_file, xyz_file):
    with open(lammps_dump_file, 'r') as infile, open(xyz_file, 'w') as outfile:
        while True:
            timestep = infile.readline().strip()
            if not timestep:
                break  # End of file

            infile.readline()  # Skip timestep value

            infile.readline()  # Skip "ITEM: NUMBER OF ATOMS"
            num_atoms = int(infile.readline().strip())

            infile.readline()  # Skip "ITEM: BOX BOUNDS"
            box_bounds = [list(map(float, infile.readline().strip().split())) for _ in range(3)]

            a, b, c, alpha, beta, gamma = compute_lattice_params(box_bounds)

            infile.readline()  # Skip "ITEM: ATOMS id type xs ys zs"

            outfile.write(f"{num_atoms}\n")
            outfile.write(f"{a:.6f} {b:.6f} {c:.6f} {alpha:.6f} {beta:.6f} {gamma:.6f}\n")

            for _ in range(num_atoms):
                data = infile.readline().strip().split()
                atom_id = data[0]
                atom_type = data[1]
                xs, ys, zs = map(float, data[2:5])
                x, y, z = unscale_coordinates(xs, ys, zs, box_bounds)

                atom_type = 'O' if atom_type == '1' else 'H' if atom_type == '2' else atom_type
                outfile.write(f"{atom_id} {atom_type} {x:.6f} {y:.6f} {z:.6f}\n")

if __name__ == "__main__":
    lammps_dump_file = "dump.lammpstrj"
    xyz_file = "xyz_trj.xyz"
    
    convert_to_xyz(lammps_dump_file, xyz_file)
    print(f"Conversion complete. XYZ file saved as {xyz_file}.")
    
    convert_to_xyz(lammps_dump_file, xyz_file)
    print(f"Conversion complete. XYZ file saved as {xyz_file}.")
