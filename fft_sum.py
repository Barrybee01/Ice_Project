import h5py
import numpy as np

input_file = 'OH1_fft_results.h5'
output_file = 'OH1_sum_fft.txt'

with h5py.File(input_file, 'r') as f:
    # Load the wavenumber axis (X values)
    wn = f["wavenumber_cm^-1"][:]

    # Get all dataset names except the wavenumber
    dataset_names = [name for name in f.keys() if name != "wavenumber_cm^-1"]
    dataset_names.sort()

    if not dataset_names:
        raise ValueError("No FFT datasets found.")

    # Use the first dataset to initialize the sum
    sum_fft = f[dataset_names[0]][:]  # this becomes our running sum

    # Loop over the remaining datasets and add in-place
    for name in dataset_names[1:]:
        sum_fft += f[name][:]  # in-place accumulation

# Combine wavenumber and sum
output_array = np.column_stack((wn, sum_fft))

# Save to text file
np.savetxt(output_file, output_array, fmt='%.6f', header="wavenumber_cm^-1\tsum_fft")

print(f"Saved summed FFT to: {output_file}")
