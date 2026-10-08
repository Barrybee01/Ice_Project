import os
import csv
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.neighborlist import neighbor_list

def read_crystal(filename, cell_parameters):
    atoms = read(filename, format='xyz', index=0)
    values = np.asarray(cell_parameters, dtype=float)

    if values.shape == (3,):
        lx, ly, lz = values
        vectors = [[lx, 0, 0], [0, ly, 0], [0, 0, lz]]
    elif values.shape == (6,):
        lx, ly, lz, xy, xz, yz = values
        vectors = [[lx, 0, 0], [xy, ly, 0], [xz, yz, lz]]

    atoms.set_cell(vectors)
    atoms.pbc = True
    return atoms

def bonded_hydrogens(atoms, oxygen, hydrogen, oh_bond_cutoff): #Attach neighbouring H atom to O atoms within motif
    is_o = np.zeros(len(atoms), dtype=bool)
    is_h = np.zeros(len(atoms), dtype=bool)
    is_o[oxygen], is_h[hydrogen] = True, True

    i, j, D = neighbor_list('ijD', atoms, oh_bond_cutoff)
    valid = is_o[i] & is_h[j]
    i, j, D = i[valid], j[valid], D[valid]

    distances = np.linalg.norm(D, axis=1)
    best = {}

    for o_idx, h_idx, vector, distance in zip(i, j, D, distances):
        o_idx, h_idx = int(o_idx), int(h_idx)

        if h_idx not in best or distance < best[h_idx][2]:
            best[h_idx] = (o_idx, np.array(vector, copy=True), float(distance))

    if len(best) != len(hydrogen):
        missing = len(hydrogen) - len(best)
        raise ValueError(f'{missing} H atoms have no O within {oh_bond_cutoff:g} Å. Check the cell and OH bond cutoff.')

    attached = {int(o): [] for o in oxygen}

    for h_idx, (o_idx, vector, _) in sorted(best.items()):
        attached[o_idx].append((h_idx, vector))

    counts = Counter(len(attached[int(o)]) for o in oxygen)
    return attached

def write_xyz(path, first, second, center_idx, first_indices, second_indices, attached): #make xyz files with the O atoms and their attached H atoms
    oxygens = [(int(center_idx), np.zeros(3))]
    oxygens.extend((int(idx), xyz) for idx, xyz in zip(first_indices, first))
    oxygens.extend((int(idx), xyz) for idx, xyz in zip(second_indices, second))

    lines = []

    for o_idx, position in oxygens:
        lines.append(('O', position))

        for _, bond_vector in attached[o_idx]:
            lines.append(('H', position + bond_vector))

    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'{len(lines)}\n')
        f.write(f'O-centered environment: CN1={len(first)} CN2={len(second)}; hydrogens bonded to displayed oxygen images\n')

        for species, xyz in lines:
            f.write(f'{species} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}\n')

def save_reference(path, vectors): #Considers just the OO pairs to find reference motifs. H added later for visualization
    centered = vectors - vectors.mean(axis=0) if len(vectors) else vectors
    np.savetxt(path, centered, fmt='%.10f', header=f'Neighbor coordinates; N={len(vectors)}; centroid at origin; Angstrom')


def write_csv(path, columns, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)


def save_rdf(atoms, oxygen, hydrogen, out, rdf_bin, rdf_max, r1, r2): #Make rdf figs for sanity check. Verify with other software
    n_atoms = len(atoms)
    is_o = np.zeros(n_atoms, dtype=bool)
    is_h = np.zeros(n_atoms, dtype=bool)
    is_o[oxygen], is_h[hydrogen] = True, True

    i, j, distances = neighbor_list('ijd', atoms, rdf_max)

    valid = (distances > 1e-8) & (distances <= rdf_max + 1e-8)
    i, j, distances = i[valid], j[valid], distances[valid]

    edges = np.linspace(0, rdf_max, int(np.ceil(rdf_max / rdf_bin)) + 1)
    r_mid = (edges[:-1] + edges[1:]) / 2
    shell_volumes = (4 * np.pi / 3) * (edges[1:]**3 - edges[:-1]**3)
    volume = atoms.get_volume()

    pair_definitions = [
        ('OO', is_o[i] & is_o[j], len(oxygen), len(oxygen)),
        ('OH', is_o[i] & is_h[j], len(oxygen), len(hydrogen)),
        ('HH', is_h[i] & is_h[j], len(hydrogen), len(hydrogen))]

    for pair, selected, n_ref, n_target in pair_definitions:
        counts, _ = np.histogram(distances[selected], bins=edges)
        rdf = counts / (n_ref * (n_target / volume) * shell_volumes)

        if pair == 'OO':
            filename = 'oxygen_rdf'
            ylabel = 'g$_{OO}$(r)'
        elif pair == 'OH':
            filename = 'hydrogen_oxygen_rdf'
            ylabel = 'g$_{OH}$(r)'
        else:
            filename = 'hydrogen_rdf'
            ylabel = 'g$_{HH}$(r)'

        write_csv(os.path.join(out, f'{filename}.csv'),
                  ['r_A', f'g_{pair}_r', 'pair_count'], zip(r_mid, rdf, counts))

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(r_mid, rdf, lw=1.5)

        if pair == 'OO':
            ax.axvline(r1, linestyle='--', label=f'First-shell boundary: {r1:g} Å')
            ax.axvline(r2, linestyle=':', label=f'Second-shell boundary: {r2:g} Å')
            ax.legend()

        ax.set(xlabel='Separation (Å)', ylabel=ylabel, xlim=(0, rdf_max))
        fig.tight_layout()
        fig.savefig(os.path.join(out, f'{filename}.png'), dpi=250)
        plt.close(fig)

def analyze(input_file, output_dir, r1, r2, cell=None, export_all=False, rdf_bin=0.01, rdf_max=None, oh_bond_cutoff=1.25):
    atoms = read_crystal(input_file, cell)
    symbols = np.asarray(atoms.get_chemical_symbols())

    oxygen = np.flatnonzero(symbols == 'O')
    hydrogen = np.flatnonzero(symbols == 'H')

    attached = bonded_hydrogens(atoms, oxygen, hydrogen, oh_bond_cutoff)
    is_oxygen = symbols == 'O'
    i, j, D, shifts = neighbor_list('ijDS', atoms, r2)

    valid = is_oxygen[i] & is_oxygen[j]
    i, j, D, shifts = i[valid], j[valid], D[valid], shifts[valid]

    distances = np.linalg.norm(D, axis=1)
    valid = (distances > 1e-8) & (distances <= r2 + 1e-8) #scan system with some tolerance value

    i = i[valid]
    j = j[valid]
    D = D[valid]
    shifts = shifts[valid]
    distances = distances[valid]
    order = np.lexsort((distances, i))

    i = i[order]
    j = j[order]
    D = D[order]
    shifts = shifts[order]
    distances = distances[order]

    out = os.path.abspath(output_dir)
    examples = os.path.join(out, 'coordination_groups')

    os.makedirs(examples, exist_ok=True)

    if export_all:
        os.makedirs(os.path.join(out, 'all_sites'), exist_ok=True)

    site_rows = []
    neighbor_rows = []

    groups = Counter()
    representatives = {}

    for center in oxygen:
        lo = np.searchsorted(i, center, side='left')
        hi = np.searchsorted(i, center, side='right')
        inside = np.arange(lo, hi)

        first_ids = inside[distances[inside] <= r1 + 1e-8]
        second_ids = inside[distances[inside] > r1 + 1e-8]

        cn1 = len(first_ids)
        cn2 = len(second_ids)
        key = (cn1, cn2)

        groups[key] += 1
        site_rows.append([int(center + 1), cn1, cn2, cn1 + cn2])

        first = D[first_ids]
        second = D[second_ids]

        first_o = j[first_ids]
        second_o = j[second_ids]

        if key not in representatives:
            representatives[key] = (int(center), first.copy(), second.copy(), first_o.copy(), second_o.copy())

        for shell, selected in ((1, first_ids), (2, second_ids)):
            for rank, k in enumerate(selected, 1):
                neighbor_rows.append([
                    int(center + 1), shell, rank, int(j[k] + 1),
                    *map(int, shifts[k]), float(distances[k]),
                    *map(float, D[k])])

        if export_all:
            write_xyz(os.path.join(out, 'all_sites', f'oxygen_{center + 1:06d}.xyz'), first, second, center, first_o, second_o, attached)

    write_csv(os.path.join(out, 'site_coordination.csv'), ['oxygen_atom_id', 'CN1', 'CN2', 'CN_total'], site_rows)
    write_csv(os.path.join(out, 'neighbor_coordinates.csv'),
              ['oxygen_atom_id', 'shell', 'neighbor_rank', 'neighbor_atom_id',
               'image_x', 'image_y', 'image_z', 'distance_A',
               'dx_A', 'dy_A', 'dz_A'], neighbor_rows)

    population_rows = []

    for (cn1, cn2), count in sorted(groups.items(), key=lambda x: (-x[1], x[0])):
        name = f'CN1_{cn1}_CN2_{cn2}'
        center, first, second, first_o, second_o = representatives[(cn1, cn2)]

        population_rows.append([cn1, cn2, cn1 + cn2, count, count / len(oxygen), center + 1])

        write_xyz(os.path.join(examples, f'{name}.xyz'),
                  first, second, center, first_o, second_o, attached)

        save_reference(os.path.join(examples, f'{name}_first_reference.txt'), first)
        save_reference(os.path.join(examples, f'{name}_second_reference.txt'), second)
        save_reference(os.path.join(examples, f'{name}_combined_reference.txt'), np.vstack((first, second)))

    write_csv(os.path.join(out, 'coordination_populations.csv'),['CN1', 'CN2', 'CN_total', 'count', 'fraction',
               'representative_oxygen_atom_id'], population_rows)

    save_rdf(atoms, oxygen, hydrogen, out, rdf_bin, rdf_max, r1, r2)

    print(f'Oxygen sites: {len(oxygen)}; hydrogen sites: {len(hydrogen)}')
    print(f'Shell boundaries: r1 = {r1:g} Å, r2 = {r2:g} Å')
    print('CN1  CN2  Total  Sites  Fraction')

    for cn1, cn2, total, count, fraction, _ in population_rows:
        print(f'{cn1:3d}  {cn2:3d}  {total:5d}  {count:5d}  {fraction:.4f}')

    print(f'Output: {out}')

def main():
    input_dir = r'E:\PhD project\Amorphous ice calculations\1h to HDA to LDA full and final ring stats\Local order metric\Ice 1h and ice iv reference motifs\ice iv'
    input_filename = 'iceIV_ortho_x464.xyz'
    input_file = os.path.join(input_dir, input_filename)
    output_dir = os.path.join(input_dir, 'Coordination_Results_with_H')

    r1 = 3 #OO cutoff distances. Change as needed. Distance unit in A
    r2 = 4.3

    rdf_bin = 0.01 #This should be good enough
    rdf_max = 6.0 #Change as needed. Should be large enough to see first 2 shells

    oh_bond_cutoff = 1.0 #Only a fool would change this

    # Orthorhombic: [lx, ly, lz]
    # Triclinic LAMMPS: [lx, ly, lz, xy, xz, yz]
    cell = [67.6798533721, 51.9277324739, 59.9609806330] #Try and only use orthorhombic

    export_all = False # Export XYZ for every central oxygen if True. Only use for debugging if more motifs than expected come out

    analyze(input_file, output_dir, r1, r2, cell=cell, export_all=export_all,
            rdf_bin=rdf_bin, rdf_max=rdf_max, oh_bond_cutoff=oh_bond_cutoff)

if __name__ == '__main__':
    main()
