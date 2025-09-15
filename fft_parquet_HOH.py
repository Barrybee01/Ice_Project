import numpy as np
import pyarrow.parquet as pq
import h5py

# Constants
c_cm_per_s = 2.99792458e10  # Speed of light in cm/s
fs_to_s = 1e-15  # Femtosecond to second

# FFT function
def compute_fft_wavenumbers(t_fs, signal):
    dt_s = 1 * fs_to_s
    N = len(signal)
    yf = np.fft.fft(signal - np.mean(signal))
    xf_hz = np.fft.fftfreq(N, dt_s)
    wavenumbers = xf_hz / c_cm_per_s
    return wavenumbers[:N // 2], 2.0 / N * np.abs(yf[:N // 2])

# Preprocessing
def preprocess_signal(signal):
    signal = signal.copy()
    for i in range(2, len(signal) - 1):
        if signal[i] < 95 or signal[i] > 115:
            signal[i] = 0.5 * (signal[i - 1] + signal[i - 2])
    return signal

# Input/output paths
parquet_path = "HOH.parquet"
output_h5_path = "HOHtest_fft_results.h5"

# Open Parquet file
parquet_file = pq.ParquetFile(parquet_path)
columns = parquet_file.schema.names
time_column = columns[0]
signal_columns = columns[1:]

# Read time column once
t_hoh = parquet_file.read(columns=[time_column])[time_column].to_numpy()

# Start output HDF5 file
with h5py.File(output_h5_path, 'w') as h5f:
    wavenumber_written = False

    for i, col in enumerate(signal_columns):
        print(f"Processing {col}...")

        # Read just this signal column
        v_hoh = parquet_file.read(columns=[col])[col].to_numpy()

        # Preprocess
        v_hoh = preprocess_signal(v_hoh)

        # FFT
        wn, fft_vals = compute_fft_wavenumbers(t_hoh, v_hoh)

        # Write wavenumbers once
        if not wavenumber_written:
            h5f.create_dataset("wavenumber_cm^-1", data=wn, dtype='float64')
            wavenumber_written = True

        # Write FFT column
        h5f.create_dataset(col, data=fft_vals, dtype='float64')

print(f"\nHDF5 FFT results written to: {output_h5_path}")