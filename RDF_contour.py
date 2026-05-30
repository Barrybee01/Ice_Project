import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

def finalize_block(data, timestep, initial_timestep, timeunit="ns"):
    df = pd.DataFrame(data,columns=["ID","r","RDF1","RDF2","RDF3","RDF4","RDF5","RDF6"])

    if timeunit == "ns":
        df["Timestep"] = (timestep - initial_timestep) / 1e6
    elif timeunit == "ps":
        df["Timestep"] = (timestep - initial_timestep) / 1e3

    return df


def read_rdf_file(file_path, timeunit="ns"):
    all_data = []
    with open(file_path, "r") as f:
        lines = f.readlines()

    data = []
    timestep = None
    initial_timestep = None

    for line in lines:
        if line.startswith("#"):
            continue

        fields = line.split()
        if len(fields) == 2:
            if timestep is not None:
                all_data.append(finalize_block(data,timestep,initial_timestep,timeunit))

            timestep, nrows = map(int, fields)
            if initial_timestep is None:
                initial_timestep = timestep
            data = []
        else:
            data.append(list(map(float, fields)))

    if data:
        all_data.append(finalize_block(data,timestep,initial_timestep,timeunit))

    return pd.concat(all_data, ignore_index=True)


def convert_rdf_to_ir(data, rdf_column):
    data = data.copy()
    g = np.maximum(data[rdf_column], 1e-10)
    data["I(r)"] = (g * np.log(g) - g + 1) * data["r"]**2
    return data

def create_contour_plot(data,value_column,cbar_label,outfile):
    pivot_table = data.pivot_table(index="Timestep",columns="r",values=value_column,)
    X, Y = np.meshgrid(pivot_table.columns, pivot_table.index)
    Z = pivot_table.values
    sns.set_theme(style="white")
    plt.figure(figsize=(12, 8))
    cp = plt.contourf(X,Y,Z,levels=100,cmap="icefire",norm=colors.PowerNorm(0.8))
    cbar = plt.colorbar(cp)
    cbar.set_label(cbar_label,fontsize=16)
    plt.xlabel(r"Radial Distance ($\AA$)",fontsize=18)
    plt.ylabel("Time (ns)",fontsize=18)
    plt.tight_layout()
    plt.savefig(outfile,dpi=300,bbox_inches="tight")
    plt.close()

def create_waterfall_plot(data,value_column,ylabel,outfile,N=3,offset=0.5):
    sns.set_theme(style="white")
    timesteps = np.sort(data["Timestep"].unique())
    selected_timesteps = timesteps[::N]
    cmap = sns.color_palette("flare", as_cmap=True)
    plt.figure(figsize=(12, 8))

    for i, timestep in enumerate(selected_timesteps):
        subset = data[
            data["Timestep"] == timestep]

        plt.plot(subset["r"],subset[value_column] + i * offset,color=cmap(i / max(len(selected_timesteps) - 1, 1)),lw=1.5,alpha=0.9,label=f"{timestep:.3f} ns")

    plt.xlabel(r"Radial Distance ($\AA$)",fontsize=18)
    plt.ylabel(ylabel,fontsize=18)
    plt.xlim(0.5, 6)
    plt.tick_params(axis="both",which="major",labelsize=14)
    plt.legend(title="Time (ns)",loc="center left",bbox_to_anchor=(1.02, 0.5),frameon=False,reverse=True)
    plt.tight_layout()
    plt.savefig(outfile,dpi=300,bbox_inches="tight")
    plt.close()

def analyze_file(rdf_file):
    data = read_rdf_file(rdf_file)
    rdf_pairs = {"RDF1": "OO","RDF3": "OH","RDF5": "HH"}
    basename = os.path.splitext(os.path.basename(rdf_file))[0]
    for rdf_column, pair in rdf_pairs.items():
        create_contour_plot(data=data,value_column=rdf_column,cbar_label=fr"$g_{{{pair}}}(r)$",outfile=f"{basename}_{pair}_RDF_contour.png")
        create_waterfall_plot(data=data,value_column=rdf_column,ylabel=fr"$g_{{{pair}}}(r)$",outfile=f"{basename}_{pair}_RDF_waterfall.png")

        ir_data = convert_rdf_to_ir(data,rdf_column)
        create_contour_plot(data=ir_data,value_column="I(r)",cbar_label=fr"$I_{{{pair}}}(r)$",outfile=f"{basename}_{pair}_IR_contour.png")
        create_waterfall_plot(data=ir_data,value_column="I(r)",ylabel=fr"$I_{{{pair}}}(r)$",outfile=f"{basename}_{pair}_IR_waterfall.png")

if __name__ == "__main__":
    rdf_files = sorted([f for f in os.listdir(".") if f.endswith(".rdf")])
    for rdf_file in rdf_files:
        analyze_file(rdf_file)
