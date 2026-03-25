import os
import tarfile
import numpy as np
import matplotlib.pyplot as plt
import ase.io
import homcloud.interface as hc
import h5py
from mpi4py import MPI


def extract_birth_death_lifetime(pd1):
    births = np.asarray(pd1.births, dtype=np.float64)
    deaths = np.asarray(pd1.deaths, dtype=np.float64)
    lifetimes = deaths - births
    return births, deaths, lifetimes


def make_pd_for_xyz(xyz_file, split_folder, pdgm_folder, raw_pd_folder):
    xyz_input_path = os.path.join(split_folder, xyz_file)
    base_name = os.path.splitext(xyz_file)[0]

    pdgm_path = os.path.join(pdgm_folder, f"{base_name}.pdgm")
    raw_pd_path = os.path.join(raw_pd_folder, f"{base_name}.png")

    amorph_ice = ase.io.read(xyz_input_path)

    weights = np.array([
        0.175**2 if atom == "O" else 0.775**2
        for atom in amorph_ice.get_chemical_symbols()
    ])

    pd1 = hc.PDList.from_alpha_filtration(
        amorph_ice.get_positions(),
        vertex_symbols=amorph_ice.get_chemical_symbols(),
        weight=weights,
        save_boundary_map=True,
        save_phtrees=True,
        save_to=pdgm_path
    ).dth_diagram(1)

    births, deaths, lifetimes = extract_birth_death_lifetime(pd1)

    fig, ax = plt.subplots(figsize=(6, 6))
    pd1.histogram((0, 8), 250).plot(
        colorbar={"type": "log", "colormap": "plasma"},
        font_size=18,
        ax=ax
    )
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)

    fig.savefig(raw_pd_path, dpi=300)
    plt.close(fig)

    timestep = int(base_name.split("_")[1])

    return timestep, births, deaths, lifetimes


def write_rank_hdf5(rank, temp_h5_dir, local_results):
    rank_path = os.path.join(temp_h5_dir, f"rank_{rank}.h5")
    vlen_dtype = h5py.vlen_dtype(np.float64)

    local_results = sorted(local_results, key=lambda x: x[0])
    timesteps = np.array([item[0] for item in local_results], dtype=np.int64)

    with h5py.File(rank_path, "w") as f:
        f.create_dataset("timesteps", data=timesteps)

        births_ds = f.create_dataset("births", shape=(len(local_results),), dtype=vlen_dtype)
        deaths_ds = f.create_dataset("deaths", shape=(len(local_results),), dtype=vlen_dtype)
        lifetimes_ds = f.create_dataset("lifetimes", shape=(len(local_results),), dtype=vlen_dtype)

        for i, (_, births, deaths, lifetimes) in enumerate(local_results):
            births_ds[i] = births
            deaths_ds[i] = deaths
            lifetimes_ds[i] = lifetimes


def merge_rank_hdf5(size, temp_h5_dir, base_path):
    all_timesteps = []
    all_births = []
    all_deaths = []
    all_lifetimes = []

    for rank in range(size):
        rank_path = os.path.join(temp_h5_dir, f"rank_{rank}.h5")

        with h5py.File(rank_path, "r") as f:
            timesteps = f["timesteps"][...]
            births = f["births"]
            deaths = f["deaths"]
            lifetimes = f["lifetimes"]

            for i in range(len(timesteps)):
                all_timesteps.append(int(timesteps[i]))
                all_births.append(np.asarray(births[i], dtype=np.float64))
                all_deaths.append(np.asarray(deaths[i], dtype=np.float64))
                all_lifetimes.append(np.asarray(lifetimes[i], dtype=np.float64))

    merged = sorted(
        zip(all_timesteps, all_births, all_deaths, all_lifetimes),
        key=lambda x: x[0]
    )

    sorted_timesteps = np.array([x[0] for x in merged], dtype=np.int64)
    sorted_births = [x[1] for x in merged]
    sorted_deaths = [x[2] for x in merged]
    sorted_lifetimes = [x[3] for x in merged]

    vlen_dtype = h5py.vlen_dtype(np.float64)

    birth_h5_path = os.path.join(base_path, "births.h5")
    death_h5_path = os.path.join(base_path, "deaths.h5")
    lifetime_h5_path = os.path.join(base_path, "lifetimes.h5")

    with h5py.File(birth_h5_path, "w") as f:
        f.create_dataset("timesteps", data=sorted_timesteps)
        ds = f.create_dataset("births", shape=(len(sorted_births),), dtype=vlen_dtype)
        for i, arr in enumerate(sorted_births):
            ds[i] = arr

    with h5py.File(death_h5_path, "w") as f:
        f.create_dataset("timesteps", data=sorted_timesteps)
        ds = f.create_dataset("deaths", shape=(len(sorted_deaths),), dtype=vlen_dtype)
        for i, arr in enumerate(sorted_deaths):
            ds[i] = arr

    with h5py.File(lifetime_h5_path, "w") as f:
        f.create_dataset("timesteps", data=sorted_timesteps)
        ds = f.create_dataset("lifetimes", shape=(len(sorted_lifetimes),), dtype=vlen_dtype)
        for i, arr in enumerate(sorted_lifetimes):
            ds[i] = arr

    return birth_h5_path, death_h5_path, lifetime_h5_path


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    base_path = "/home/rielly/scratch/HDA_VHDA_PDs/1h_to_HDA_to_VHDA_fullringstats/22d5kbar/P3/Full_PD/Comprehensive_ringstats"
    split_folder = os.path.join(base_path, "Split_xyz_trj")
    pdgm_folder = os.path.join(base_path, "pdgm_files")
    raw_pd_folder = os.path.join(base_path, "Raw_PDs")
    temp_h5_dir = os.path.join(base_path, "temp_rank_hdf5")

    if rank == 0:
        os.makedirs(pdgm_folder, exist_ok=True)
        os.makedirs(raw_pd_folder, exist_ok=True)
        os.makedirs(temp_h5_dir, exist_ok=True)

        xyz_files = sorted(
            [
                f for f in os.listdir(split_folder)
                if os.path.isfile(os.path.join(split_folder, f)) and f.lower().endswith(".xyz")
            ],
            key=lambda x: int(os.path.splitext(x)[0].split("_")[1])
        )
    else:
        xyz_files = None

    xyz_files = comm.bcast(xyz_files, root=0)
    comm.Barrier()

    my_files = xyz_files[rank::size]
    print(f"Rank {rank} processing {len(my_files)} files")

    local_results = []

    for xyz_file in my_files:
        try:
            timestep, births, deaths, lifetimes = make_pd_for_xyz(
                xyz_file,
                split_folder,
                pdgm_folder,
                raw_pd_folder
            )
            local_results.append((timestep, births, deaths, lifetimes))
            print(f"Rank {rank} finished {xyz_file}")
        except Exception as e:
            print(f"Rank {rank} failed on {xyz_file}: {e}")

    write_rank_hdf5(rank, temp_h5_dir, local_results)

    comm.Barrier()

    if rank == 0:
        birth_h5, death_h5, lifetime_h5 = merge_rank_hdf5(size, temp_h5_dir, base_path)

        tar_path = os.path.join(base_path, "pdgm_files.tar.gz")
        print("Compressing pdgm files...")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(pdgm_folder, arcname="pdgm_files")
        print(f"Compressed archive saved to: {tar_path}")

        print("All persistence diagram jobs completed.")
        print(f"PDGM files saved in: {pdgm_folder}")
        print(f"Raw persistence diagrams saved in: {raw_pd_folder}")
        print(f"Birth values saved in: {birth_h5}")
        print(f"Death values saved in: {death_h5}")
        print(f"Lifetime values saved in: {lifetime_h5}")