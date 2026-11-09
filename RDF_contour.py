import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

def read_rdf_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    timestep = None
    n_data_points = None
    initial_timestep = None

    for line in lines:
        if line.startswith('#'):
            continue
        if len(line.split()) == 2:
            # Extract timestep and number of data points
            if timestep is not None:
                # Store previous timestep data if any
                df = pd.DataFrame(data, columns=['ID', 'r', 'RDF1', 'RDF2', 'RDF3', 'RDF4', 'RDF5', 'RDF6'])
                df['Timestep'] = (timestep - initial_timestep) / 1e6  # Convert to ns
                all_data.append(df)
                data = []

            timestep, n_data_points = map(int, line.split())
            if initial_timestep is None:
                initial_timestep = timestep
        else:
            data.append(list(map(float, line.split())))

    # Add the last collected data
    if data:
        df = pd.DataFrame(data, columns=['ID', 'r', 'RDF1', 'RDF2', 'RDF3', 'RDF4', 'RDF5', 'RDF6'])
        df['Timestep'] = (timestep - initial_timestep) / 1e6  # Convert to ns
        all_data.append(df)

    return pd.concat(all_data, ignore_index=True)


def create_contour_plot(data, rdf_column):
    """
    Creates a contour plot of the RDF data.
    """
    # Pivot the DataFrame to get r, Timestep, and the selected RDF column
    pivot_table = data.pivot_table(index='Timestep', columns='r', values=rdf_column)

    X, Y = np.meshgrid(pivot_table.columns, pivot_table.index)
    Z = pivot_table.values

    line_list = [0.095, 0.24, 0.385, 0.92, 1.315, 1.515]

    plt.figure(figsize=(12, 8))
    cp = plt.contourf(X, Y, Z, cmap='magma', levels=90)
    plt.colorbar(cp, label=r'$g_{OO}(r)$')
    plt.xlabel(r'Radial Distance ($\AA$)', fontsize=12)
    plt.ylabel('Timestep (ns)', fontsize=12)
    plt.show()

def create_contour_plot_seaborn(data, rdf_column):
    pivot_table = data.pivot_table(index='Timestep', columns='r', values=rdf_column)
    X, Y = np.meshgrid(pivot_table.columns, pivot_table.index)
    Z = pivot_table.values

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))
    cp = plt.contourf(X, Y, Z, cmap='icefire', levels=90, norm=colors.PowerNorm(0.8))
    cbar = plt.colorbar(cp)
    # Set colorbar label with font properties
    cbar.set_label(r'$g_{OH}(r)$', fontsize=17, fontweight='bold', family='serif')
    plt.xlabel(r'Radial Distance ($\AA$)', fontsize=20)
    plt.ylabel('Timestep (ns)', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.show()

file_path = 'LDA_formation.rdf'
all_data = []
rdf_data = read_rdf_file(file_path)

create_contour_plot(rdf_data, 'RDF1') #For water, there are 3 pair types, so I have RDF1, RDF2, and RDF3. Chage according to species types
create_contour_plot_seaborn(rdf_data, 'RDF3')
