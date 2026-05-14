import os
import ase
import ase.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import homcloud.interface as hc
import sklearn.linear_model as lm  # Machine learning
from sklearn.decomposition import PCA  # for PCA
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import scipy

###READ INPUT FOLDERS AND DEFINE OUTPUT FOLDER###

dataset_1 = r"E:\PhD project\Amorphous ice calculations\1h to HDA to LDA full and final ring stats\Large scale PCA and regression model analysis\Training_data\0kbar_init"
dataset_2 = r"E:\PhD project\Amorphous ice calculations\1h to HDA to LDA full and final ring stats\Large scale PCA and regression model analysis\Training_data\6kbar_compression"

training_set1 = sorted([f for f in os.listdir(dataset_1) if f.endswith(".xyz")])
training_set2 = sorted([f for f in os.listdir(dataset_2) if f.endswith(".xyz")])

output_dir = r"E:\PhD project\Amorphous ice calculations\1h to HDA to LDA full and final ring stats\Large scale PCA and regression model analysis\Compress_0-6kbar"

last_file_set1 = os.path.join(dataset_1, training_set1[-1])
last_file_set2 = os.path.join(dataset_2, training_set2[-1])

###RANDOM ASS FUNCTIONS###

def make_PD(xyz_path, dataset_label, output_dir):
    amorph_ice = ase.io.read(xyz_path)
    weights = np.array([0.175**2 if atom == 'O' else 0.775**2 for atom in amorph_ice.get_chemical_symbols()])#0.175 for O, 0.775 for H

    base_name = os.path.splitext(os.path.basename(xyz_path))[0]
    output_name = f"{dataset_label}_{base_name}"
    pdgm_path = os.path.join(output_dir, f"{output_name}.pdgm")

    pd1 = hc.PDList.from_alpha_filtration(
        amorph_ice.get_positions(),
        vertex_symbols=amorph_ice.get_chemical_symbols(),
        weight=weights,
        save_boundary_map=True,
        save_phtrees=True,
        save_to=pdgm_path
    ).dth_diagram(1)

    print(f"Saved PDGM: {pdgm_path}")
    return pd1, output_name

def compute_birth_lifetime(pd1):
    births = pd1.births
    deaths = pd1.deaths
    lifetimes = deaths - births
    df = pd.DataFrame({"Birth": births, "Lifetime": lifetimes})
    return df

def save_figure(fig, output_dir, filename):
    output_path = os.path.join(output_dir, filename)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved image: {output_path}")

def make_PD_plot(pd1, output_name, output_dir):
    pd1.histogram((0, 8), 250).plot(colorbar={"type": "log", "colormap": "plasma"}, font_size=18)
    fig = plt.gcf()
    save_figure(fig, output_dir,f"{output_name}_PD.png")
    return fig

def make_lifetime_birth_plot(birth_lifetime_df, output_name, output_dir, point_size=8, alpha=0.7,
        figsize=(8, 6), log_x=False, log_y=False):
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(data=birth_lifetime_df,x="Birth",y="Lifetime",s=point_size,alpha=alpha,ax=ax)

    ax.set_xlabel(r"Birth ($\AA^2$)", fontsize=16)
    ax.set_ylabel(r"Lifetime ($\AA^2$)", fontsize=16)

    if log_x:
        ax.set_xscale("log")

    if log_y:
        ax.set_yscale("log")

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    save_figure(fig, output_dir,f"{output_name}_lifetime_birth.png")
    return fig

###I STOLE THIS FROM A MATPLOTLIB EXAMPLE, MAYBE THIS WILL HELP###
#https://matplotlib.org/stable/gallery/images_contours_and_fields/colormap_normalizations.html
class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
###################################################################

def make_persistence_surface_and_image(birth_lifetime_df,output_name,output_dir,weight_function="w2",sigma=0.002,resolution=128,
    birth_range=(0, 8),lifetime_range=(0, 8),cmap="plasma",levels=100, vmin=0, vmax=0.9):

    def linear_weight(lifetime, c):
        return c * lifetime

    def atan_weight(lifetime, c, p):
        return np.arctan(c * (lifetime ** p))

    weight_options = {
        "w1": ("linear", 0.5),
        "w2": ("atan", 0.01, 3),
        "w3": ("atan", 0.5, 1)
    }

    selected_weight = weight_options[weight_function]

    births = birth_lifetime_df["Birth"].values
    lifetimes = birth_lifetime_df["Lifetime"].values

    x = np.linspace(birth_range[0], birth_range[1], resolution)
    y = np.linspace(lifetime_range[0], lifetime_range[1], resolution)
    X, Y = np.meshgrid(x, y)

    surface = np.zeros_like(X)

    for b, l in zip(births, lifetimes):
        if selected_weight[0] == "linear":
            weight = linear_weight(l, selected_weight[1])
        elif selected_weight[0] == "atan":
            weight = atan_weight(l, selected_weight[1], selected_weight[2])

        gaussian = weight * np.exp(-((X - b) ** 2 + (Y - l) ** 2) / (2 * sigma ** 2))
        surface += gaussian

    sns.set_style("white")
    #vmin = np.percentile(surface, 15)
    #vmax = np.percentile(surface, 100)

    fig1, ax1 = plt.subplots(figsize=(8, 7))
    contour = ax1.contourf(X, Y, np.log10(1+surface),levels=levels,cmap=cmap,vmin=vmin,vmax=vmax,
                           norm=None)
    fig1.colorbar(contour)
    ax1.set_xlabel("Birth")
    ax1.set_ylabel("Lifetime")
    plt.tight_layout()
    save_figure(fig1, output_dir, f"{output_name}_persistence_surface_weight-{selected_weight}_bins-{resolution}.png")

    fig2, ax2 = plt.subplots(figsize=(8, 7))
    im = ax2.imshow(np.log10(1+surface),cmap=cmap,origin="lower", extent=[birth_range[0], birth_range[1], lifetime_range[0], lifetime_range[1]],
        vmin=vmin,vmax=vmax,aspect="auto")

    fig2.colorbar(im, ax=ax2)
    ax2.set_xlabel("Birth")
    ax2.set_ylabel("Lifetime")
    plt.tight_layout()
    save_figure(fig2, output_dir, f"{output_name}_persistence_image_weight-{selected_weight}_bins-{resolution}.png")
    return surface

def vectorize_persistence_diagrams(pds, weight_function="w2", birth_range=(0, 8), resolution=128, sigma=0.006): #Need for large scale pca
    weights = {
        "w1": ("linear", 0.5),
        "w2": ("atan", 0.01, 3),
        "w3": ("atan", 0.5, 1)}

    selected_weight = weights[weight_function]
    vectorize_spec = hc.PIVectorizeSpec(birth_range, resolution, sigma=sigma, weight=selected_weight)

    pdvects = np.vstack([vectorize_spec.vectorize(pd) for pd in pds])
    pdvects = pdvects / pdvects.max()
    return pdvects

def prepare_PD_data(dataset_1, dataset_2, training_set1, training_set2, output_dir): #fixing names for PDs to put into one big list for PCA analysis
    all_pds = []
    all_labels = []
    all_names = []

    for filename in training_set1:
        xyz_path = os.path.join(dataset_1, filename)
        pd1, output_name = make_PD(xyz_path,"0kbar",output_dir)
        all_pds.append(pd1)
        all_labels.append(0)
        all_names.append(output_name)

    for filename in training_set2:
        xyz_path = os.path.join(dataset_2,filename)
        pd1, output_name = make_PD(xyz_path,"6kbar",output_dir)
        all_pds.append(pd1)
        all_labels.append(1)
        all_names.append(output_name)

    all_labels = np.array(all_labels)
    return all_pds, all_labels, all_names

def save_parameter_distributions(birth_lifetime_df, dataset_label, output_dir, num_points=1000):
    births = birth_lifetime_df["Birth"].values
    lifetimes = birth_lifetime_df["Lifetime"].values
    deaths = births + lifetimes

    distributions = {
        "birth": births,
        "death": deaths,
        "lifetime": lifetimes
    }

    for parameter_name, values in distributions.items():
        values = np.asarray(values)
        kde = scipy.stats.gaussian_kde(values, bw_method=None)
        xmin = values.min()
        xmax = values.max()
        x = np.linspace(xmin,xmax,num_points)
        y = kde(x)

        output_file = os.path.join(output_dir,f"{dataset_label}_{parameter_name}_distribution.txt")
        data = np.column_stack((x, y))
        np.savetxt(output_file,data,header="x density")
        print(f"Saved distribution: {output_file}")

###EXECUTE MAIN FUNCTIONS###

pd1_set1, output_name1 = make_PD(last_file_set1,"0kbar", output_dir)
pd1_set2, output_name2 = make_PD(last_file_set2,"6kbar", output_dir)

bl_df1 = compute_birth_lifetime(pd1_set1)
bl_df2 = compute_birth_lifetime(pd1_set2)

save_parameter_distributions(bl_df1,"0kbar",output_dir)
save_parameter_distributions(bl_df2,"6kbar",output_dir)

###EXECUTE PCA RELATED FUNCTIONS


###MAKE PLOTTY###

make_PD_plot(pd1_set1, output_name1, output_dir)
make_PD_plot(pd1_set2, output_name2, output_dir)

make_lifetime_birth_plot(bl_df1,output_name1,output_dir,log_x=False,log_y=False, point_size=4)
make_lifetime_birth_plot(bl_df2,output_name2,output_dir,log_x=False,log_y=False, point_size=4)

make_persistence_surface_and_image(bl_df1,output_name1,output_dir,weight_function="w3",sigma=0.008,resolution=128,cmap="magma",
levels=100)

make_persistence_surface_and_image(bl_df2,output_name2,output_dir,weight_function="w3",sigma=0.008,resolution=128,cmap="magma",
levels=100)