
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_density(file_path, start_time=None, end_time=None):
    data = np.loadtxt(file_path, skiprows=2)

    if data.ndim != 2 or data.shape[1] != 2: #dat files from lammps output has two columns
        raise ValueError("Expected two columns: time and density.")

    time = data[:, 0]
    density = data[:, 1]
    valid = np.isfinite(time) & np.isfinite(density)
    time = time[valid]
    density = density[valid]
    mask = np.ones(len(time), dtype=bool)

    if start_time is not None: #Need time info for ACF
        mask &= time >= start_time

    if end_time is not None:
        mask &= time <= end_time

    time = time[mask]
    density = density[mask]

    if len(time) < 30:
        raise ValueError("Insufficient density measurements.") #trajectory should be sufficiently long. Quick runs will not work here

    return time, density


#ACF is used to make sure the measurements are decorrelated, and provides some info on reliability of measurement
def autocorrelation(density, max_lag=None): #Get ACF of density
    density = np.asarray(density, dtype=float)
    n = len(density)
    centered = density - np.mean(density)

    if np.var(centered) == 0:
        return np.ones(1)

    fft_size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=fft_size)
    correlation = np.fft.irfft(spectrum * np.conjugate(spectrum),n=fft_size)[:n]

    correlation /= np.arange(n, 0, -1)
    correlation /= correlation[0]

    if max_lag is not None:
        correlation = correlation[:max_lag + 1]

    return correlation

def integrated_autocorrelation_time(acf, window_factor=5.0): #For ACF derived error (approximation)
    if len(acf) < 2:
        return np.nan, 0, False

    negative = np.where(acf[1:] <= 0)[0]

    if len(negative) > 0:
        upper = int(negative[0] + 1)
    else:
        upper = len(acf) - 1

    if upper < 1:
        return np.nan, 0, False

    cumulative = 0.5 + np.cumsum(acf[1:upper + 1])
    lags = np.arange(1, upper + 1)

    candidates = np.where(lags >= window_factor * cumulative)[0]

    if len(candidates) > 0:
        index = int(candidates[0])
        window = int(lags[index])
        tau = float(cumulative[index])
        converged = True
    else:
        window = upper
        tau = float(cumulative[-1])
        converged = False

    if window < 5: #need reasonable window
        converged = False

    return max(tau, 0.5), window, converged


def block_analysis(density, block_sizes, min_blocks=10): #Following this https://espressomd.github.io/tutorials4.2.2/error_analysis/error_analysis_part1.html
    density = np.asarray(density, dtype=float)
    n = len(density)

    results = []
    for block_size in block_sizes:
        block_size = int(block_size)
        n_blocks = n // block_size

        trimmed = density[:n_blocks * block_size]
        blocks = trimmed.reshape(n_blocks, block_size)

        block_means = np.mean(blocks, axis=1)
        mean_density = np.mean(block_means)

        block_std = np.std(block_means, ddof=1.5)
        sem = block_std / np.sqrt(n_blocks)

        results.append({
            "Block_Size": block_size,
            "Number_of_Blocks": n_blocks,
            "Mean_Density": mean_density,
            "Block_STD": block_std,
            "SEM": sem})

    return pd.DataFrame(results)


def find_sem_plateau(results, tolerance=0.10, points=5, min_blocks=20, min_block_size=1): #Error with increasing number of blocks will converge, this tells you about that
    eligible = results[(results["Number_of_Blocks"] >= min_blocks) &(results["Block_Size"] >= min_block_size)].reset_index(drop=True)

    for i in range(len(eligible) - points, -1, -1):
        segment = eligible.iloc[i:i + points]
        values = segment["SEM"].to_numpy()
        median = np.median(values)

        relative_deviation = (np.max(np.abs(values - median)) / median)
        if relative_deviation <= tolerance:
            return int(segment.iloc[points // 2]["Block_Size"])
    return None


def running_average(density): #cumulative mean avg
    return np.cumsum(density) / np.arange(1, len(density) + 1)


def save_figure(fig, output_dir, filename, dpi=300):
    fig.savefig(os.path.join(output_dir, filename),dpi=dpi,bbox_inches="tight")
    plt.close(fig)


def analyze_density(input_file, output_dir, start_time=None,end_time=None, time_unit="ps",density_unit="g/cm³", max_lag=None,min_blocks=10, max_block_fraction=0.1,
                    plateau_tolerance=0.10,manual_block_size=None):

    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="ticks", context="paper", font_scale=1.2)

    time, density = load_density(input_file, start_time, end_time)

    n = len(density)
    dt_values = np.diff(time)
    dt = np.median(dt_values)

    regular_sampling = np.allclose(dt_values, dt, rtol=1e-3, atol=1e-10)

    mean_density = np.mean(density)
    std_density = np.std(density, ddof=1)
    conventional_sem = std_density / np.sqrt(n)

    print("\n" + "=" * 55)
    print("DENSITY ERROR ANALYSIS")
    print("=" * 55)

    print(f"Input file: {input_file}")
    print(f"Measurements: {n}")
    print(f"Mean density: {mean_density:.8f}")
    print(f"Standard deviation: {std_density:.8f}")
    print(f"Conventional SEM: {conventional_sem:.8f}")

    if not regular_sampling:
        print("WARNING: Sampling intervals are not uniform.") #diagnostics


    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time, density, lw=0.7, alpha=0.6,label="Raw Density")
    ax.plot(time, running_average(density),lw=1.5, label="Running Average")
    ax.axhline(mean_density, color="black",ls="--", lw=1, label="Mean Density")
    ax.set_xlabel(f"Time ({time_unit})")
    ax.set_ylabel(f"Density ({density_unit})")
    ax.legend(frameon=False)

    sns.despine()
    fig.tight_layout()
    save_figure(fig, output_dir, "density_vs_time.png")

    if max_lag is None:
        max_lag = min(n // 4, 200000)

    max_lag = min(int(max_lag), n - 1)
    acf = autocorrelation(density, max_lag=max_lag)
    tau_int, window, acf_converged = (integrated_autocorrelation_time(acf))

    lag_indices = np.arange(len(acf))
    lag_values = (lag_indices * dt if regular_sampling else lag_indices)

    acf_df = pd.DataFrame({
        "Lag": lag_values,
        "ACF": acf})

    acf_df.to_csv(os.path.join(output_dir, "density_autocorrelation.csv"),index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lag_values, acf, lw=1.2)
    ax.axhline(0, color="black", ls="--", lw=0.8)

    if window > 0 and acf_converged:
        ax.axvline(lag_values[window],color="red",ls=":",label="Integration Window")
        ax.legend(frameon=False)

    if regular_sampling:
        ax.set_xlabel(f"Lag ({time_unit})")
    else:
        ax.set_xlabel("Lag (Samples)")

    ax.set_ylabel("Normalized Autocorrelation")

    sns.despine()
    fig.tight_layout()
    save_figure(fig, output_dir, "density_autocorrelation.png")

    if acf_converged:
        effective_samples = n / (2 * tau_int)
        acf_sem = std_density / np.sqrt(effective_samples)

        print(f"\nIntegrated ACF time: {tau_int:.4f} samples")
        print(f"Effective sample size: {effective_samples:.2f}")
        print(f"ACF-based SEM: {acf_sem:.8f}")
    else:
        effective_samples = np.nan
        acf_sem = np.nan

        print("\nACF estimate is inconclusive.")
        print("Block averaging will continue independently.")

    maximum_block = min(int(n * max_block_fraction),n // min_blocks)
    maximum_block = max(1, maximum_block)
    block_sizes = np.unique(np.geomspace(1, maximum_block, 150).astype(int))

    if manual_block_size is not None:
        manual_block_size = int(manual_block_size)
        block_sizes = np.unique(np.append(block_sizes, manual_block_size))

    results = block_analysis(density,block_sizes,min_blocks=min_blocks)
    results["Block_Duration"] = (results["Block_Size"] * dt if regular_sampling else np.nan)
    results.to_csv(os.path.join(output_dir, "density_block_analysis.csv"),index=False)

    selection_method = "None"
    if manual_block_size is not None:
        matching = results[results["Block_Size"] == manual_block_size]

        plateau_size = manual_block_size
        selection_method = "Manual"

    else:
        plateau_size = find_sem_plateau(results,tolerance=plateau_tolerance,points=5,min_blocks=20,min_block_size=1)

        if plateau_size is not None:
            selection_method = "Automatic candidate"

    selected_sem = np.nan
    selected_blocks = np.nan
    selected_mean = np.nan

    if plateau_size is not None:
        selected = results[results["Block_Size"] == plateau_size].iloc[0]
        selected_sem = float(selected["SEM"])
        selected_blocks = int(selected["Number_of_Blocks"])
        selected_mean = float(selected["Mean_Density"])

        print("\nBLOCK AVERAGING")
        print(f"Selection method: {selection_method}")
        print(f"Selected block size: {plateau_size}")
        print(f"Number of blocks: {selected_blocks}")
        print(f"Block-averaged mean: {selected_mean:.8f}")
        print(f"Block-averaged SEM: {selected_sem:.8f}")

    else:
        print("\nNo candidate SEM plateau was identified.")
        print("Inspect density_sem_convergence.png.")
        print("You may specify manual_block_size.")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results["Block_Size"],results["SEM"],marker="o",markersize=3,lw=1.2,label="Block-Averaged SEM")
    ax.axhline(conventional_sem,color="tab:orange",ls=":",label="Conventional SEM")

    if np.isfinite(acf_sem):
        ax.axhline(acf_sem,color="tab:green",ls="--",label="ACF-Based SEM")

    if plateau_size is not None:
        ax.axvline(plateau_size,color="tab:red",ls="--",label="Selected Block Size")
        ax.scatter([plateau_size],[selected_sem],color="tab:red",s=45,zorder=5)

    ax.set_xscale("log")
    ax.set_xlabel("Block Size (Samples)")
    ax.set_ylabel(f"Density SEM ({density_unit})")
    ax.legend(frameon=False)
    sns.despine()
    fig.tight_layout()
    save_figure(fig, output_dir, "density_sem_convergence.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results["Block_Size"],results["Number_of_Blocks"],lw=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Block Size (Samples)")
    ax.set_ylabel("Number of Blocks")
    sns.despine()
    fig.tight_layout()
    save_figure(fig, output_dir, "number_of_blocks.png")

    midpoint = n // 2
    first_mean = np.mean(density[:midpoint])
    second_mean = np.mean(density[midpoint:])
    half_difference = second_mean - first_mean

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time, density, lw=0.7, alpha=0.5)
    ax.axhline(first_mean,color="tab:blue",ls="--",label="First-Half Mean")
    ax.axhline(second_mean,color="tab:red",ls="--", label="Second-Half Mean")
    ax.axvline(time[midpoint], color="black", ls=":")
    ax.set_xlabel(f"Time ({time_unit})")
    ax.set_ylabel(f"Density ({density_unit})")
    ax.legend(frameon=False)
    sns.despine()
    fig.tight_layout()
    save_figure(fig, output_dir, "stationarity_check.png")


    summary = {
        "Input_File": os.path.abspath(input_file),
        "Number_of_Samples": n,
        "Start_Time": time[0],
        "End_Time": time[-1],
        "Sampling_Interval": dt,
        "Regular_Sampling": regular_sampling,
        "Mean_Density": mean_density,
        "Density_STD": std_density,
        "Conventional_SEM": conventional_sem,
        "ACF_SEM": acf_sem,
        "Integrated_ACF_Time_Samples": tau_int,
        "ACF_Converged": acf_converged,
        "Effective_Sample_Size": effective_samples,
        "Selection_Method": selection_method,
        "Selected_Block_Size": plateau_size,
        "Selected_Block_Count": selected_blocks,
        "Block_Averaged_Mean": selected_mean,
        "Block_Averaged_SEM": selected_sem,
        "First_Half_Mean": first_mean,
        "Second_Half_Mean": second_mean,
        "Half_Mean_Difference": half_difference}

    pd.DataFrame([summary]).to_csv(os.path.join(output_dir, "density_summary.csv"),index=False)
    report_path = os.path.join(output_dir, "density_summary.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("DENSITY ERROR ANALYSIS\n")
        f.write("=" * 55 + "\n\n")

        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

        f.write("\nNOTES\n")
        f.write("-" * 55 + "\n")
        f.write("Block averaging is independent of ACF convergence.\n")

        f.write(
            "Automatic plateau selection is heuristic and "
            "requires visual verification.\n")

        f.write(
            "If SEM does not stabilize, a reliable uncertainty "
            "may not be available from this trajectory.\n")

        f.write(
            "Stationarity is required for interpreting the "
            "result as uncertainty in a constant mean.\n")

        f.write(
            "The first-half versus second-half comparison "
            "is a diagnostic, not a formal stationarity test.\n")


    print("\n" + "=" * 55)
    print("FINAL DENSITY STATISTICS")
    print("=" * 55)

    print(f"Mean Density:       {mean_density:.8f} {density_unit}")
    print(f"Standard Deviation: {std_density:.8f} {density_unit}")
    print(f"Conventional SEM:   {conventional_sem:.8f} {density_unit}")

    if np.isfinite(acf_sem):
        print(f"ACF-Based SEM:      {acf_sem:.8f} {density_unit}")
    else:
        print("ACF-Based SEM:      Inconclusive")

    if np.isfinite(selected_sem):
        print(f"Block-Averaged SEM: {selected_sem:.8f} {density_unit}")
    else:
        print("Block-Averaged SEM: No candidate plateau")

    print(f"\nResults saved to:\n{output_dir}")


if __name__ == "__main__":
    input_file = r"PATH\TO\DAT-FILE"
    input_directory = os.path.dirname(os.path.abspath(input_file))
    output_directory = os.path.join(input_directory, "Density_Error_Analysis")

    start_time = None
    end_time = None

    time_unit = "ps"
    density_unit = "g/cm³"
    max_lag = None
    min_blocks = 10
    max_block_fraction = 0.1
    plateau_tolerance = 0.10
    manual_block_size = None

    analyze_density(input_file=input_file,output_dir=output_directory,start_time=start_time,end_time=end_time,time_unit=time_unit,density_unit=density_unit,
        max_lag=max_lag,min_blocks=min_blocks,max_block_fraction=max_block_fraction,plateau_tolerance=plateau_tolerance,manual_block_size=manual_block_size)
