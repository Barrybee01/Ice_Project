import os
import numpy as np
import h5py
import matplotlib.pyplot as plt


def load_ordered_hdf5_arrays(h5_path, data_key):
    with h5py.File(h5_path, "r") as f:
        timesteps = f["timesteps"][...]
        values = [np.asarray(arr, dtype=np.float64) for arr in f[data_key]]
    return timesteps, values


def build_binned_matrix(values_per_timestep, bin_edges):
    n_timesteps = len(values_per_timestep)
    n_bins = len(bin_edges) - 1
    matrix = np.zeros((n_timesteps, n_bins), dtype=np.float64)

    for i, values in enumerate(values_per_timestep):
        counts, _ = np.histogram(values, bins=bin_edges)
        matrix[i, :] = counts

    return matrix


def make_contour_plot(
    timesteps,
    matrix,
    bin_edges,
    title,
    output_path,
    xlabel="Timestep",
    ylabel="Scale",
    use_log=False,
):
    x = timesteps
    y = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    X, Y = np.meshgrid(x, y, indexing="xy")

    Z = matrix.T

    if use_log:
        Z = np.log10(Z + 1.0)

    fig, ax = plt.subplots(figsize=(8, 6))

    contour = ax.contourf(X, Y, Z, levels=100, cmap="plasma")
    cbar = fig.colorbar(contour, ax=ax)

    if use_log:
        cbar.set_label("log10(count + 1)")
    else:
        cbar.set_label("Count")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    base_path = "/home/rielly/scratch/HDA_VHDA_PDs/1h_to_HDA_to_VHDA_fullringstats/22d5kbar/P3/Full_PD/Comprehensive_ringstats"

    births_h5 = os.path.join(base_path, "births.h5")
    deaths_h5 = os.path.join(base_path, "deaths.h5")
    lifetimes_h5 = os.path.join(base_path, "lifetimes.h5")

    # Adjust these if needed for your system
    birth_bins = np.linspace(0, 8, 251)
    death_bins = np.linspace(0, 8, 251)
    lifetime_bins = np.linspace(0, 8, 251)

    # -------- Birth contour --------
    timesteps_birth, births = load_ordered_hdf5_arrays(births_h5, "births")
    birth_matrix = build_binned_matrix(births, birth_bins)

    make_contour_plot(
        timesteps=timesteps_birth,
        matrix=birth_matrix,
        bin_edges=birth_bins,
        title="Birth Scale Contour Map",
        output_path=os.path.join(base_path, "births_contour.png"),
        ylabel="Birth Scale",
        use_log=True,
    )

    # -------- Death contour --------
    timesteps_death, deaths = load_ordered_hdf5_arrays(deaths_h5, "deaths")
    death_matrix = build_binned_matrix(deaths, death_bins)

    make_contour_plot(
        timesteps=timesteps_death,
        matrix=death_matrix,
        bin_edges=death_bins,
        title="Death Scale Contour Map",
        output_path=os.path.join(base_path, "deaths_contour.png"),
        ylabel="Death Scale",
        use_log=True,
    )

    # -------- Lifetime contour --------
    timesteps_lifetime, lifetimes = load_ordered_hdf5_arrays(lifetimes_h5, "lifetimes")
    lifetime_matrix = build_binned_matrix(lifetimes, lifetime_bins)

    make_contour_plot(
        timesteps=timesteps_lifetime,
        matrix=lifetime_matrix,
        bin_edges=lifetime_bins,
        title="Lifetime Contour Map",
        output_path=os.path.join(base_path, "lifetimes_contour.png"),
        ylabel="Lifetime",
        use_log=True,
    )

    print(f"Contour maps saved in: {base_path}")