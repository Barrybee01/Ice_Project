import os
import csv
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.neighborlist import neighbor_list

def read_crystal(filename, cell_parameters): #grab atom positions
    atoms = read(filename, format='xyz', index=0)
    values = np.asarray(cell_parameters, dtype=float)
    if values.shape == (3,): #orthorhombic
        lx, ly, lz = values
        vectors = [[lx, 0, 0], [0, ly, 0], [0, 0, lz]]
    elif values.shape == (6,): #triclinic
        lx, ly, lz, xy, xz, yz = values
        vectors = [[lx, 0, 0], [xy, ly, 0], [xz, yz, lz]]
    atoms.set_cell(vectors)
    atoms.pbc = True #idk why I need to have this here, this should be default
    return atoms

def write_xyz(path, first, second): #make xyz file to view motifs
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'{1 + len(first) + len(second)}\n')
        f.write(f'Central O; shell 1: {len(first)} O; shell 2: {len(second)} O; nonperiodic excerpt\n')
        f.write('O 0.0000000000 0.0000000000 0.0000000000\n')
        for array in (first, second):
            for xyz in array:
                f.write(f'O {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}\n')


def save_reference(path, vectors):
    centered = vectors - vectors.mean(axis=0) if len(vectors) else vectors
    np.savetxt(path, centered, fmt='%.10f', header=f'Neighbor coordinates; N={len(vectors)}; centroid at origin; Angstrom')

def analyze(input_file, output_dir, r1, r2, cell=None, export_all=False, rdf_bin=0.025):
    atoms = read_crystal(input_file, cell)
    oxygen = np.flatnonzero(np.asarray(atoms.get_chemical_symbols()) == 'O')
    is_oxygen = np.zeros(len(atoms), dtype=bool) #just in case my dumbass uses the wrong input structure or I missed removing ghost atoms from tip4p structures
    is_oxygen[oxygen] = True

    # D is an unwrapped vector from central site i to periodic neighbor image j.
    i, j, D, shifts = neighbor_list('ijDS', atoms, r2)
    valid = is_oxygen[i] & is_oxygen[j]
    i, j, D, shifts = i[valid], j[valid], D[valid], shifts[valid]
    distances = np.linalg.norm(D, axis=1)
    valid = (distances > 1e-8) & (distances <= r2 + 1e-8)
    i, j, D, shifts, distances = i[valid], j[valid], D[valid], shifts[valid], distances[valid]
    order = np.lexsort((distances, i))
    i, j, D, shifts, distances = i[order], j[order], D[order], shifts[order], distances[order]

    out = os.path.abspath(output_dir)
    os.makedirs(out, exist_ok=True)
    examples = os.path.join(out, 'coordination_groups')
    os.makedirs(examples, exist_ok=True)
    if export_all:
        os.makedirs(os.path.join(out, 'all_sites'), exist_ok=True)

    site_rows, neighbor_rows = [], []
    groups = Counter()
    representatives = {}
    for center in oxygen:
        lo, hi = np.searchsorted(i, center, side='left'), np.searchsorted(i, center, side='right')
        inside = np.arange(lo, hi)
        first_ids = inside[distances[inside] <= r1 + 1e-8] #change tolerance as needed
        second_ids = inside[distances[inside] > r1 + 1e-8]
        cn1, cn2 = len(first_ids), len(second_ids)
        key = (cn1, cn2)
        groups[key] += 1
        site_rows.append([int(center + 1), cn1, cn2, cn1 + cn2])
        first, second = D[first_ids], D[second_ids]
        if key not in representatives:
            representatives[key] = (int(center + 1), first.copy(), second.copy())
        for shell, selected in ((1, first_ids), (2, second_ids)):
            for rank, k in enumerate(selected, 1):
                neighbor_rows.append([int(center + 1), shell, rank, int(j[k] + 1),
                                      *map(int, shifts[k]), float(distances[k]), *map(float, D[k])])
        if export_all:
            write_xyz(os.path.join(out, 'all_sites', f'oxygen_{center + 1:06d}.xyz'), first, second)

    def write_csv(name, columns, rows):
        with open(os.path.join(out, name), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(rows)

    write_csv('site_coordination.csv', ['oxygen_atom_id', 'CN1', 'CN2', 'CN_total'], site_rows)
    write_csv('neighbor_coordinates.csv', ['oxygen_atom_id', 'shell', 'neighbor_rank',
              'neighbor_atom_id', 'image_x', 'image_y', 'image_z', 'distance_A', 'dx_A', 'dy_A', 'dz_A'], neighbor_rows)
    population_rows = []
    for (cn1, cn2), count in sorted(groups.items(), key=lambda x: (-x[1], x[0])):
        name = f'CN1_{cn1}_CN2_{cn2}'
        center_id, first, second = representatives[(cn1, cn2)]
        population_rows.append([cn1, cn2, cn1 + cn2, count, count / len(oxygen), center_id])
        write_xyz(os.path.join(examples, f'{name}.xyz'), first, second)
        save_reference(os.path.join(examples, f'{name}_first_reference.txt'), first)
        save_reference(os.path.join(examples, f'{name}_second_reference.txt'), second)
        save_reference(os.path.join(examples, f'{name}_combined_reference.txt'), np.vstack((first, second)))
    write_csv('coordination_populations.csv', ['CN1', 'CN2', 'CN_total', 'count', 'fraction',
              'representative_oxygen_atom_id'], population_rows)

    # OO radial distribution up to r2, computed from central O neighbour pairs.
    bins = np.arange(0, r2 + rdf_bin, rdf_bin)
    if bins[-1] < r2:
        bins = np.append(bins, r2)
    counts, edges = np.histogram(distances, bins=bins)
    shell_volumes = (4 * np.pi / 3) * (edges[1:] ** 3 - edges[:-1] ** 3)
    rho = len(oxygen) / atoms.get_volume()
    rdf = counts / (len(oxygen) * rho * shell_volumes)
    r_mid = (edges[:-1] + edges[1:]) / 2
    write_csv('oxygen_rdf.csv', ['r_A', 'g_OO_r', 'pair_count'], zip(r_mid, rdf, counts))

    fig, ax = plt.subplots(figsize=(7, 4)) #plot the rdf for sanity reasons. Verify RDF with vesta or ovito or some shit
    ax.plot(r_mid, rdf, lw=1.5)
    ax.axvline(r1, linestyle='--', label=f'First-shell boundary: {r1:g} Å') #index specific peaks
    ax.axvline(r2, linestyle=':', label=f'Second-shell boundary: {r2:g} Å')
    ax.set(xlabel=r'$r_{OO}$ (Å)', ylabel='g$_{OO}$(r)', xlim=(0, r2))
    ax.set_xlim(2.5, 4.5) #check rdf elsewhere to compare with this, and set the range appropriately
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'oxygen_rdf.png'), dpi=250)
    plt.close(fig)

    print(f'Oxygen sites: {len(oxygen)}')
    print(f'Shell boundaries: r1 = {r1:g} Å, r2 = {r2:g} Å')
    print('CN1  CN2  Total  Sites  Fraction')
    for cn1, cn2, total, count, fraction, _ in population_rows:
        print(f'{cn1:3d}  {cn2:3d}  {total:5d}  {count:5d}  {fraction:.4f}')
    print(f'Output: {out}')


def main():
    input_dir = r"E:\PhD project\Amorphous ice calculations\1h to HDA to LDA full and final ring stats\Local order metric\Ice 1h and ice iv reference motifs\ice iv"
    input_filename = 'Ocentric_iceIV_ortho_x464.xyz'
    input_file = os.path.join(input_dir, input_filename)
    output_dir = os.path.join(input_dir, "Coordination_Results")

    r1 = 3  #first OO coordination shell (A)
    r2 = 4.3  #second OO coordination shell (A)
    rdf_bin = 0.01 #change this as needed, initial guess was 0.025

    # Orthorhombic: cell = [lx, ly, lz]
    # Triclinic (grab from lammps trj): cell = [lx, ly, lz, xy, xz, yz]. Should not need this case because triclinic sucks balls
    cell = [67.6798533721, 51.9277324739, 59.9609806330] #change as needed from lammps structure file. Try to stick with orthorhombic systems

    # True exports an O-centered xyz file for every central O site
    #For ideal xtal systems, I should only see a small number of motifs show up. If not, turn this to true to see if there is any difference in total number of motifs
    export_all = False
    analyze(input_file, output_dir, r1, r2, cell=cell, export_all=export_all, rdf_bin=rdf_bin)

if __name__ == '__main__':
    main()
