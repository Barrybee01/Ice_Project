import os
import re
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

dataset_1 = r"E:\Path\To\First Training Set"
dataset_2 = r"E:\Path\To\Second Training Set"

training_set1 = sorted([f for f in os.listdir(dataset_1) if f.endswith(".xyz")])
training_set2 = sorted([f for f in os.listdir(dataset_2) if f.endswith(".xyz")])

output_dir = r"E:\Path\To\Output Directory"

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
        save_to=None #Not saving pdgm for now, keeping path defined just in case they need to be saved
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

def extract_step_number(filename):
    match = re.search(r'step_(\d+)', filename)
    return int(match.group(1))

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
        "w3": ("atan", 0.5, 1)}

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

    fig1, ax1 = plt.subplots(figsize=(8, 7))
    contour = ax1.contourf(X, Y, np.log10(1+surface),levels=levels,cmap=cmap,vmin=vmin,vmax=vmax)
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

def vectorize_persistence_diagrams(pds, weight_function="w2", birth_range=(0, 8), resolution=128, sigma=0.06): #Need for large scale pca
    weights = {
        "w1": ("linear", 0.5),
        "w2": ("atan", 0.01, 3),
        "w3": ("atan", 0.5, 1)}

    selected_weight = weights[weight_function]
    vectorize_spec = hc.PIVectorizeSpec(birth_range, resolution, sigma=sigma, weight=selected_weight)

    pdvects = np.vstack([vectorize_spec.vectorize(pd) for pd in pds])
    pdvects = pdvects / pdvects.max()
    return pdvects, vectorize_spec

def prepare_PD_data_with_steps(dataset_path, file_list, dataset_label, output_dir): #fixing names for PDs to put into one big list for PCA analysis
    all_pds = []
    all_step_numbers = []
    all_names = []

    for filename in file_list:
        xyz_path = os.path.join(dataset_path, filename)
        step_number = extract_step_number(filename)
        pd1, output_name = make_PD(xyz_path, dataset_label,output_dir)
        all_pds.append(pd1)
        all_step_numbers.append(step_number)
        all_names.append(output_name)

    all_step_numbers = np.array(all_step_numbers)
    return all_pds, all_step_numbers, all_names

def save_parameter_distributions(birth_lifetime_df, dataset_label, output_dir, bins=200, density=False): #basic binned histograms
    births = birth_lifetime_df["Birth"].values
    lifetimes = birth_lifetime_df["Lifetime"].values
    deaths = births + lifetimes

    distributions = {
        "birth": births,
        "death": deaths,
        "lifetime": lifetimes}

    for parameter_name, values in distributions.items():
        values = np.asarray(values)

        hist, bin_edges = np.histogram(values,bins=bins,density=density)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        output_file = os.path.join(output_dir,f"{dataset_label}_{parameter_name}_histogram.txt")
        data = np.column_stack((bin_centers, hist))

        np.savetxt(output_file,data,header="bin_center density")

        print(f"Saved histogram: {output_file}")

def perform_pca_regression(pd_vectors, target_values, n_components=10, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(pd_vectors, target_values, test_size=test_size, random_state=random_state)

    pca = PCA(n_components=n_components) #actual pca step
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    regressor = lm.LinearRegression() #see if other regression library is better
    regressor.fit(X_train_pca, y_train)

    y_pred_train = regressor.predict(X_train_pca)
    y_pred_test = regressor.predict(X_test_pca) #prediction

    train_score = regressor.score(X_train_pca, y_train)
    test_score = regressor.score(X_test_pca, y_test)
    return {'pca': pca, 'regressor': regressor, 'X_train_pca': X_train_pca, 'X_test_pca': X_test_pca, 'y_train': y_train,
        'y_test': y_test, 'y_pred_train': y_pred_train, 'y_pred_test': y_pred_test, 'train_score': train_score, 'test_score': test_score}

def plot_pca_results(pca_results, output_dir, title_prefix="PCA_Regression"): #I thought this was fucking hilarious
    sns.set_style("whitegrid")

    fig1, ax1 = plt.subplots(figsize=(12, 7)) #explained variance ratio plot
    explained_variance_ratio = pca_results['pca'].explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    components = range(1, len(explained_variance_ratio) + 1)

    df_var = pd.DataFrame({'Component': components, 'Individual Variance': explained_variance_ratio, 'Cumulative Variance': cumulative_variance})
    sns.barplot(data=df_var, x='Component', y='Individual Variance', alpha=0.7, color='steelblue', ax=ax1)
    sns.lineplot(data=df_var, x='Component', y='Cumulative Variance', marker='o', color='red', linewidth=2, markersize=8, ax=ax1)
    ax1.axhline(y=0.95, color='green', linestyle='--', alpha=0.7, linewidth=2, label='95% variance threshold')
    ax1.set_xlabel('Principal Component', fontsize=14)
    ax1.set_ylabel('Explained Variance Ratio', fontsize=14)
    ax1.set_title(f'{title_prefix} - Explained Variance', fontsize=16)
    ax1.legend(loc='upper left', fontsize=12)

    n_components_95 = np.where(cumulative_variance >= 0.95)[0]

    if len(n_components_95) > 0:
        ax1.text(0.02, 0.98, f'{n_components_95[0] + 1} components explain 95% variance',
                 transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    save_figure(fig1, output_dir, f"{title_prefix}_explained_variance.png")

    fig2, ax2 = plt.subplots(figsize=(8, 8)) #predictions vs actual (test set)

    df_pred = pd.DataFrame({
        'Actual Step Number': pca_results['y_test'],
        'Predicted Step Number': pca_results['y_pred_test']
    })

    sns.regplot(data=df_pred, x='Actual Step Number', y='Predicted Step Number',
                scatter=False,
                line_kws={'color': 'red', 'linewidth': 2, 'linestyle': '--'},
                ax=ax2)

    ax2.scatter(df_pred['Actual Step Number'], df_pred['Predicted Step Number'],alpha=0.6, s=60, edgecolors='black', linewidths=0.5)

    min_val = min(df_pred['Actual Step Number'].min(), df_pred['Predicted Step Number'].min())
    max_val = max(df_pred['Actual Step Number'].max(), df_pred['Predicted Step Number'].max())

    ax2.plot([min_val, max_val], [min_val, max_val], 'g--', linewidth=2, label='Perfect prediction', alpha=0.7)
    ax2.set_xlabel('Actual Step Number', fontsize=14)
    ax2.set_ylabel('Predicted Step Number', fontsize=14)
    ax2.set_title(f'Test Set Predictions\nR² = {pca_results["test_score"]:.4f}', fontsize=14)
    ax2.legend(fontsize=12)
    ax2.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    save_figure(fig2, output_dir, f"{title_prefix}_predictions_vs_actual.png")

    fig3, ax3 = plt.subplots(figsize=(10, 6)) #residual plot
    residuals = pca_results['y_test'] - pca_results['y_pred_test']

    df_residual = pd.DataFrame({
        'Predicted Step Number': pca_results['y_pred_test'],
        'Residuals': residuals})

    sns.scatterplot(data=df_residual, x='Predicted Step Number', y='Residuals', alpha=0.6, s=50, color='coral', edgecolor='black', ax=ax3)
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=2)
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)

    ax3.axhline(y=mean_residual, color='blue', linestyle=':', linewidth=1.5, label=f'Mean residual = {mean_residual:.3f}')
    ax3.axhline(y=mean_residual + 2 * std_residual, color='gray', linestyle=':', alpha=0.7, linewidth=1)
    ax3.axhline(y=mean_residual - 2 * std_residual, color='gray',linestyle=':', alpha=0.7, linewidth=1,label=f'±2σ = {2 * std_residual:.3f}')
    ax3.set_xlabel('Predicted Step Number', fontsize=14)
    ax3.set_ylabel('Residuals (Actual - Predicted)', fontsize=14)
    ax3.set_title('Residual Plot', fontsize=14)
    ax3.legend(fontsize=10)
    plt.tight_layout()
    save_figure(fig3, output_dir, f"{title_prefix}_residuals.png")

    fig4, ax4 = plt.subplots(figsize=(10, 6)) #residual distribution
    sns.histplot(residuals, bins=20, kde=True, stat='density',color='coral', edgecolor='black', alpha=0.7, ax=ax4)

    mu, std = np.mean(residuals), np.std(residuals)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / std) ** 2)

    ax4.plot(x, y, 'b-', linewidth=2, label=f'Normal fit (μ={mu:.3f}, σ={std:.3f})')
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero residual')
    ax4.set_xlabel('Residuals', fontsize=14)
    ax4.set_ylabel('Density', fontsize=14)
    ax4.set_title('Residual Distribution', fontsize=14)
    ax4.legend(fontsize=12)
    plt.tight_layout()
    save_figure(fig4, output_dir, f"{title_prefix}_residual_distribution.png")

    fig5, ax5 = plt.subplots(figsize=(8, 6)) #training vs test performance

    df_performance = pd.DataFrame({
        'Dataset': ['Train', 'Test'],
        'R² Score': [pca_results['train_score'], pca_results['test_score']]})

    sns.barplot(data=df_performance, x='Dataset', y='R² Score',hue='Dataset', palette=['green', 'steelblue'],
                legend=False, alpha=0.7, edgecolor='black', ax=ax5)

    for i, (bar, score) in enumerate(zip(ax5.patches, df_performance['R² Score'])):
        ax5.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01,
                 f'{score:.4f}',
                 ha='center', va='bottom', fontsize=12)

    ax5.set_ylim(0, 1.05)
    ax5.set_ylabel('R² Score', fontsize=14)
    ax5.set_title('Model Performance Comparison', fontsize=14)
    plt.tight_layout()
    save_figure(fig5, output_dir, f"{title_prefix}_performance_comparison.png")

    fig6, ax6 = plt.subplots(figsize=(14, 10)) #PCA components heatmap

    n_components_show = min(10, pca_results['pca'].n_components_)

    components_df = pd.DataFrame(
        pca_results['pca'].components_[:n_components_show, :],
        index=[f'PC{i + 1}' for i in range(n_components_show)],
        columns=[f'Feature_{i}' for i in range(pca_results['pca'].components_.shape[1])])

    sns.heatmap(components_df, cmap='RdBu_r', center=0,cbar_kws={'label': 'Component Weight', 'shrink': 0.8},ax=ax6)
    ax6.set_xlabel('Original Feature Index', fontsize=14)
    ax6.set_ylabel('Principal Component', fontsize=14)
    ax6.set_title(f'First {n_components_show} PCA Components', fontsize=14)
    plt.tight_layout()
    save_figure(fig6, output_dir, f"{title_prefix}_components_heatmap.png")
    fig7, ax7 = plt.subplots(figsize=(10, 8)) #2D PCA projection

    all_pca = pca_results['X_train_pca']

    df_projection = pd.DataFrame({
        'PC1': all_pca[:, 0],
        'PC2': all_pca[:, 1],
        'Step Number': pca_results['y_train']})

    sns.scatterplot(data=df_projection, x='PC1', y='PC2',hue='Step Number', palette='viridis',
                    alpha=0.7, s=60, edgecolor='black', ax=ax7)

    ax7.set_xlabel(f'First Principal Component ({explained_variance_ratio[0] * 100:.1f}% variance)', fontsize=14)
    ax7.set_ylabel(f'Second Principal Component ({explained_variance_ratio[1] * 100:.1f}% variance)', fontsize=14)
    ax7.set_title('2D PCA Projection (Training Data)', fontsize=14)
    plt.legend(title='Step Number', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    save_figure(fig7, output_dir, f"{title_prefix}_2d_projection.png")

    if pca_results['pca'].n_components_ >= 3: #pairplot of first 3 principal components
        df_pcs = pd.DataFrame({
            'PC1': all_pca[:, 0],
            'PC2': all_pca[:, 1],
            'PC3': all_pca[:, 2],
            'Step Number': pca_results['y_train']})

        pairplot = sns.pairplot(df_pcs, hue='Step Number', palette='viridis',
                                diag_kind='kde',
                                plot_kws={'alpha': 0.6, 's': 30},
                                diag_kws={'alpha': 0.7})

        pairplot.fig.suptitle('Pairwise Relationships of First 3 Principal Components', fontsize=16, y=1.02)
        plt.tight_layout()
        save_figure(pairplot.fig, output_dir, f"{title_prefix}_3d_pairplot.png")

    fig9, ax9 = plt.subplots(figsize=(10, 8)) #loadings plot for first 2 principal components
    loadings = pca_results['pca'].components_[:2, :]
    df_loadings = pd.DataFrame({
        'PC1': loadings[0, :],
        'PC2': loadings[1, :]})

    sns.scatterplot(data=df_loadings, x='PC1', y='PC2',alpha=0.6, s=40, ax=ax9)

    for i in range(min(50, loadings.shape[1])):
        ax9.arrow(0, 0, loadings[0, i], loadings[1, i],
                  head_width=0.01, head_length=0.01,
                  alpha=0.3, color='gray')

    ax9.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax9.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax9.set_xlabel(f'PC1 ({explained_variance_ratio[0] * 100:.1f}% variance)', fontsize=14)
    ax9.set_ylabel(f'PC2 ({explained_variance_ratio[1] * 100:.1f}% variance)', fontsize=14)
    ax9.set_title('PCA Loadings Plot (Feature Contributions)', fontsize=14)
    ax9.grid(True, alpha=0.3)
    plt.tight_layout()
    save_figure(fig9, output_dir, f"{title_prefix}_loadings_plot.png")
    return

def save_pca_results_to_csv(pca_results, output_dir, title_prefix="PCA_Regression"):
    explained_variance_df = pd.DataFrame({
        'component': range(1, len(pca_results['pca'].explained_variance_ratio_) + 1),
        'explained_variance_ratio': pca_results['pca'].explained_variance_ratio_,
        'cumulative_variance_ratio': np.cumsum(pca_results['pca'].explained_variance_ratio_)})

    explained_variance_df.to_csv(os.path.join(output_dir, f"{title_prefix}_explained_variance.csv"), index=False)
    print(f"Saved: {title_prefix}_explained_variance.csv")

    train_predictions_df = pd.DataFrame({
        'actual_step': pca_results['y_train'],
        'predicted_step': pca_results['y_pred_train'],
        'residual': pca_results['y_train'] - pca_results['y_pred_train']})

    train_predictions_df.to_csv(os.path.join(output_dir, f"{title_prefix}_train_predictions.csv"),index=False)
    print(f"Saved: {title_prefix}_train_predictions.csv")

    test_predictions_df = pd.DataFrame({
        'actual_step': pca_results['y_test'],
        'predicted_step': pca_results['y_pred_test'],
        'residual': pca_results['y_test'] - pca_results['y_pred_test']})

    test_predictions_df.to_csv(os.path.join(output_dir, f"{title_prefix}_test_predictions.csv"), index=False)
    print(f"Saved: {title_prefix}_test_predictions.csv")

    components_df = pd.DataFrame(pca_results['pca'].components_, columns=[f'feature_{i}' for i in range(pca_results['pca'].components_.shape[1])])
    components_df.to_csv(os.path.join(output_dir, f"{title_prefix}_components.csv"),index=False)
    print(f"Saved: {title_prefix}_components.csv")

    train_scores_df = pd.DataFrame(pca_results['X_train_pca'], columns=[f'PC{i + 1}' for i in range(pca_results['X_train_pca'].shape[1])])
    train_scores_df['step_number'] = pca_results['y_train']
    train_scores_df.to_csv(os.path.join(output_dir, f"{title_prefix}_train_scores.csv"),index=False)
    print(f"Saved: {title_prefix}_train_scores.csv")

    test_scores_df = pd.DataFrame(pca_results['X_test_pca'],columns=[f'PC{i + 1}' for i in range(pca_results['X_test_pca'].shape[1])])
    test_scores_df['step_number'] = pca_results['y_test']
    test_scores_df.to_csv(os.path.join(output_dir, f"{title_prefix}_test_scores.csv"),index=False)
    print(f"Saved: {title_prefix}_test_scores.csv")

    coefficients_df = pd.DataFrame({'pc_component': [f'PC{i + 1}' for i in range(len(pca_results['regressor'].coef_))],'coefficient': pca_results['regressor'].coef_})
    coefficients_df.to_csv(os.path.join(output_dir, f"{title_prefix}_regression_coefficients.csv"),index=False)
    print(f"Saved: {title_prefix}_regression_coefficients.csv")

    loadings_df = pd.DataFrame({
        'feature_index': range(pca_results['pca'].components_.shape[1]),
        'PC1_loading': pca_results['pca'].components_[0, :],
        'PC2_loading': pca_results['pca'].components_[1, :] if pca_results['pca'].n_components_ > 1 else np.nan,
        'PC3_loading': pca_results['pca'].components_[2, :] if pca_results['pca'].n_components_ > 2 else np.nan})
    loadings_df.to_csv(os.path.join(output_dir, f"{title_prefix}_feature_loadings.csv"),index=False)
    print(f"Saved: {title_prefix}_feature_loadings.csv")

    with open(os.path.join(output_dir, f"{title_prefix}_summary.txt"), 'w') as f:
        f.write("PCA REGRESSION SUMMARY\n")
        f.write("=" * 30 + "\n\n")
        f.write("MODEL PERFORMANCE\n")
        f.write("-" * 30 + "\n")
        f.write(f"Training R² Score: {pca_results['train_score']:.6f}\n")
        f.write(f"Test R² Score: {pca_results['test_score']:.6f}\n\n")
        f.write("PCA INFORMATION\n")
        f.write("-" * 30 + "\n")
        f.write(f"Number of PCA components: {pca_results['pca'].n_components_}\n")
        f.write(f"Total explained variance: {np.sum(pca_results['pca'].explained_variance_ratio_):.6f}\n\n")
        f.write("EXPLAINED VARIANCE PER COMPONENT\n")
        f.write("-" * 30 + "\n")
        for i, var in enumerate(pca_results['pca'].explained_variance_ratio_):
            f.write(f"  PC{i + 1}: {var:.6f} ({var * 100:.2f}%)\n")
        f.write("\nCUMULATIVE EXPLAINED VARIANCE\n")
        f.write("-" * 30 + "\n")
        cumsum = 0
        for i, var in enumerate(pca_results['pca'].explained_variance_ratio_):
            cumsum += var
            f.write(f"  PC{i + 1}: {cumsum:.6f} ({cumsum * 100:.2f}%)\n")

        f.write("\nREGRESSION INFORMATION\n")
        f.write("-" * 30 + "\n")
        f.write(f"Regression intercept: {pca_results['regressor'].intercept_:.6f}\n")
        f.write("\nRegression coefficients:\n")
        for i, coef in enumerate(pca_results['regressor'].coef_):
            f.write(f"  PC{i + 1}: {coef:.6f}\n")

        f.write("\nDATA INFORMATION\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total samples: {len(pca_results['y_train']) + len(pca_results['y_test'])}\n")
        f.write(f"Training samples: {len(pca_results['y_train'])}\n")
        f.write(f"Test samples: {len(pca_results['y_test'])}\n")
        f.write(f"Feature dimension before PCA: {pca_results['X_train'].shape[1]}\n")
        f.write(f"Feature dimension after PCA: {pca_results['X_train_pca'].shape[1]}\n")

        f.write("\n" + "=" * 30 + "\n")

    print(f"Saved: {title_prefix}_summary.txt")
    print(f"\nAll CSV files saved to: {output_dir}")

def plot_pca_persistence_components(pca,vectorize_spec,output_dir,title_prefix="PCA",cmap="RdBu_r",midpoint=0):
    fig_mean = plt.figure(figsize=(7, 6))
    vectorize_spec.histogram_from_vector(pca.mean_).plot(colorbar={"type": "log", "max":1E1})
    plt.title("Mean Persistence Image")
    mean_path = os.path.join(output_dir,f"{title_prefix}_mean_persistence_image.png")
    plt.savefig(mean_path, dpi=300, bbox_inches="tight")
    plt.close(fig_mean)
    print(f"Saved: {mean_path}")

    fig_pc1 = plt.figure(figsize=(7, 6))
    vectorize_spec.histogram_from_vector(pca.components_[0, :]).plot(colorbar={"type": "linear-midpoint","midpoint": midpoint})
    plt.title("PC1 Persistence Image")
    pc1_path = os.path.join(output_dir,f"{title_prefix}_PC1_persistence_image.png")
    plt.savefig(pc1_path, dpi=300, bbox_inches="tight")
    plt.close(fig_pc1)
    print(f"Saved: {pc1_path}")

    fig_pc2 = plt.figure(figsize=(7, 6))
    vectorize_spec.histogram_from_vector(pca.components_[1, :]).plot(colorbar={"type": "linear-midpoint","midpoint": midpoint})
    plt.title("PC2 Persistence Image")
    pc2_path = os.path.join(output_dir,f"{title_prefix}_PC2_persistence_image.png")
    plt.savefig(pc2_path, dpi=300, bbox_inches="tight")
    plt.close(fig_pc2)
    print(f"Saved: {pc2_path}")

### LOGISTIC REGRESSION ###

def perform_logistic_regression(pd_vectors, target_labels, test_size=0.2, random_state=42, C=0.01, solver="lbfgs"):
    X_train, X_test, y_train, y_test = train_test_split(pd_vectors, target_labels, test_size=test_size, random_state=random_state, stratify=target_labels) #split up data
    model = lm.LogisticRegression(C=C, solver=solver, max_iter=1000, random_state=random_state) #train
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_pred_proba_test = model.predict_proba(X_test)

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
    return {'model': model,'X_train': X_train,'X_test': X_test,'y_train': y_train,'y_test': y_test,'y_pred_train': y_pred_train,'y_pred_test': y_pred_test,
        'y_pred_proba_test': y_pred_proba_test,'train_score': train_score,'test_score': test_score,'classification_report': classification_report(y_test, y_pred_test),
        'confusion_matrix': confusion_matrix(y_test, y_pred_test),'roc_auc': roc_auc_score(y_test, y_pred_proba_test[:, 1]),'roc_curve': roc_curve(y_test, y_pred_proba_test[:, 1])}


def plot_logistic_regression_results(lr_results, output_dir, title_prefix="Logistic_Regression"):
    sns.set_style("whitegrid")

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    cm = lr_results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklabels=['0kbar', '6kbar'],yticklabels=['0kbar', '6kbar'],cbar_kws={'label': 'Count'}, ax=ax1)
    ax1.set_xlabel('Predicted', fontsize=14)
    ax1.set_ylabel('Actual', fontsize=14)
    ax1.set_title(f'Confusion Matrix\nAccuracy: {lr_results["test_score"]:.4f}', fontsize=14)
    plt.tight_layout()
    save_figure(fig1, output_dir, f"{title_prefix}_confusion_matrix.png")

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    fpr, tpr, thresholds = lr_results['roc_curve']
    ax2.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {lr_results["roc_auc"]:.4f})')
    ax2.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier (AUC = 0.5)')
    ax2.set_xlabel('False Positive Rate', fontsize=14)
    ax2.set_ylabel('True Positive Rate', fontsize=14)
    ax2.set_title('ROC Curve', fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    save_figure(fig2, output_dir, f"{title_prefix}_roc_curve.png")

    fig3, ax3 = plt.subplots(figsize=(8, 6))
    df_performance = pd.DataFrame({'Dataset': ['Train', 'Test'],'Accuracy': [lr_results['train_score'], lr_results['test_score']]})
    sns.barplot(data=df_performance, x='Dataset', y='Accuracy',palette=['green', 'steelblue'], alpha=0.7, edgecolor='black', ax=ax3)
    for i, (bar, score) in enumerate(zip(ax3.patches, df_performance['Accuracy'])):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,f'{score:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 1.05)
    ax3.set_ylabel('Accuracy', fontsize=14)
    ax3.set_title('Logistic Regression Performance', fontsize=14)
    plt.tight_layout()
    save_figure(fig3, output_dir, f"{title_prefix}_performance.png")

    fig4, ax4 = plt.subplots(figsize=(10, 6))
    prob_class0 = lr_results['y_pred_proba_test'][lr_results['y_test'] == 0][:, 1]
    prob_class1 = lr_results['y_pred_proba_test'][lr_results['y_test'] == 1][:, 1]
    ax4.hist(prob_class0, bins=20, alpha=0.6, label='True 0kbar', color='blue', edgecolor='black')
    ax4.hist(prob_class1, bins=20, alpha=0.6, label='True 6kbar', color='orange', edgecolor='black')
    ax4.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Decision Boundary')
    ax4.set_xlabel('Predicted Probability of 6kbar', fontsize=14)
    ax4.set_ylabel('Frequency', fontsize=14)
    ax4.set_title('Prediction Probability Distribution', fontsize=14)
    ax4.legend(fontsize=12)
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    save_figure(fig4, output_dir, f"{title_prefix}_probability_distribution.png")

    fig5, ax5 = plt.subplots(figsize=(10, 6))
    report = lr_results['classification_report']
    report_lines = report.split('\n')
    metrics_data = []
    for line in report_lines[2:5]:  # Get the class rows
        if line.strip():
            parts = line.split()
            if len(parts) >= 4:
                class_name = parts[0]
                precision = float(parts[1])
                recall = float(parts[2])
                f1_score = float(parts[3])
                support = int(parts[4]) if len(parts) > 4 else 0
                metrics_data.append([class_name, precision, recall, f1_score, support])

    df_report = pd.DataFrame(metrics_data, columns=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
    df_report_melted = df_report.melt(id_vars=['Class', 'Support'],value_vars=['Precision', 'Recall', 'F1-Score'],var_name='Metric', value_name='Score')
    pivot_report = df_report_melted.pivot_table(index='Class', columns='Metric', values='Score')
    sns.heatmap(pivot_report, annot=True, fmt='.3f', cmap='YlGnBu',cbar_kws={'label': 'Score'}, ax=ax5)
    ax5.set_title('Classification Report Metrics', fontsize=14)
    plt.tight_layout()
    save_figure(fig5, output_dir, f"{title_prefix}_classification_report.png")
    return


def plot_logistic_regression_coefficients(model, vectorize_spec, output_dir, title_prefix="Logistic_Regression",cmap="RdBu_r", midpoint=0.0):
    fig_coef = plt.figure(figsize=(10, 8))
    coef_vector = model.coef_.ravel()
    img = vectorize_spec.histogram_from_vector(coef_vector)
    img.plot(colorbar={"type": "linear-midpoint", "midpoint": midpoint})
    plt.title("Logistic Regression Coefficients (Feature Importance)", fontsize=16)
    plt.tight_layout()
    coef_path = os.path.join(output_dir, f"{title_prefix}_coefficients_persistence.png")
    plt.savefig(coef_path, dpi=300, bbox_inches="tight")
    plt.close(fig_coef)
    print(f"Saved: {coef_path}")

    fig_bar, ax_bar = plt.subplots(figsize=(12, 6))
    n_top = 20
    top_indices = np.argsort(np.abs(coef_vector))[-n_top:]
    top_coefs = coef_vector[top_indices]
    top_indices_sorted = top_indices[np.argsort(top_coefs)]
    top_coefs_sorted = coef_vector[top_indices_sorted]
    colors = ['red' if x < 0 else 'green' for x in top_coefs_sorted]
    ax_bar.barh(range(len(top_coefs_sorted)), top_coefs_sorted, color=colors, alpha=0.7)
    ax_bar.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax_bar.set_xlabel('Coefficient Value', fontsize=14)
    ax_bar.set_ylabel('Feature Index', fontsize=14)
    ax_bar.set_title(f'Top {n_top} Most Important Features (Absolute Coefficient)', fontsize=14)
    ax_bar.grid(True, alpha=0.3)
    plt.tight_layout()
    top_features_path = os.path.join(output_dir, f"{title_prefix}_top_features.png")
    plt.savefig(top_features_path, dpi=300, bbox_inches="tight")
    plt.close(fig_bar)
    print(f"Saved: {top_features_path}")
    return


def save_logistic_regression_results(lr_results, output_dir, title_prefix="Logistic_Regression"):
    predictions_df = pd.DataFrame({
        'actual_label': lr_results['y_test'],
        'predicted_label': lr_results['y_pred_test'],
        'probability_0kbar': lr_results['y_pred_proba_test'][:, 0],
        'probability_6kbar': lr_results['y_pred_proba_test'][:, 1]})
    predictions_df.to_csv(os.path.join(output_dir, f"{title_prefix}_test_predictions.csv"), index=False)
    print(f"Saved: {title_prefix}_test_predictions.csv")

    coefficients_df = pd.DataFrame({'feature_index': range(len(lr_results['model'].coef_.ravel())),'coefficient': lr_results['model'].coef_.ravel()})
    coefficients_df.to_csv(os.path.join(output_dir, f"{title_prefix}_coefficients.csv"), index=False)
    print(f"Saved: {title_prefix}_coefficients.csv")

    with open(os.path.join(output_dir, f"{title_prefix}_summary.txt"), 'w') as f:
        f.write("=" * 30 + "\n")
        f.write("LOGISTIC REGRESSION SUMMARY\n")
        f.write("=" * 30 + "\n\n")

        f.write("MODEL PERFORMANCE\n")
        f.write("-" * 30 + "\n")
        f.write(f"Training Accuracy: {lr_results['train_score']:.6f}\n")
        f.write(f"Test Accuracy: {lr_results['test_score']:.6f}\n")
        f.write(f"ROC AUC Score: {lr_results['roc_auc']:.6f}\n\n")

        f.write("CLASSIFICATION REPORT\n")
        f.write("-" * 30 + "\n")
        f.write(lr_results['classification_report'])
        f.write("\n")

        f.write("CONFUSION MATRIX\n")
        f.write("-" * 30 + "\n")
        # Fixed spacing for confusion matrix
        f.write("                 Predicted\n")
        f.write("                0kbar  6kbar\n")
        f.write(
            f"Actual 0kbar:    {lr_results['confusion_matrix'][0, 0]:4d}   {lr_results['confusion_matrix'][0, 1]:4d}\n")
        f.write(
            f"       6kbar:    {lr_results['confusion_matrix'][1, 0]:4d}   {lr_results['confusion_matrix'][1, 1]:4d}\n\n")

        f.write("MODEL INFORMATION\n")
        f.write("-" * 30 + "\n")
        f.write(f"Regularization strength (C): {lr_results['model'].C}\n")
        f.write(f"Solver: {lr_results['model'].solver}\n")
        if hasattr(lr_results['model'], 'n_iter_'):
            f.write(f"Number of iterations: {lr_results['model'].n_iter_[0]}\n")
        else:
            f.write("Number of iterations: N/A\n")
        f.write("\n")

        f.write("DATA INFORMATION\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total samples: {len(lr_results['y_train']) + len(lr_results['y_test'])}\n")
        f.write(f"Training samples: {len(lr_results['y_train'])}\n")
        f.write(f"Test samples: {len(lr_results['y_test'])}\n")
        f.write(f"Feature dimension: {lr_results['X_train'].shape[1]}\n")
        f.write(
            f"Class distribution (0kbar/6kbar): {np.sum(lr_results['y_train'] == 0)} / {np.sum(lr_results['y_train'] == 1)}\n")

        f.write("\n" + "=" * 60 + "\n")

    print(f"Saved: {title_prefix}_summary.txt")
    print(f"\nAll logistic regression files saved to: {output_dir}")

### EXECUTE MAIN FUNCTIONS ###

dataset1 = {}
dataset2 = {}

dataset1["pd"], dataset1["name"] = make_PD(last_file_set1, "0kbar", output_dir)
dataset2["pd"], dataset2["name"] = make_PD(last_file_set2,"6kbar",output_dir)

dataset1["birth_lifetime"] = compute_birth_lifetime(dataset1["pd"])
dataset2["birth_lifetime"] = compute_birth_lifetime(dataset2["pd"])

save_parameter_distributions(dataset1["birth_lifetime"],"0kbar",output_dir)
save_parameter_distributions(dataset2["birth_lifetime"],"6kbar",output_dir)

### MAKE PLOTTY ###

make_PD_plot(dataset1["pd"],dataset1["name"],output_dir)
make_PD_plot(dataset2["pd"],dataset2["name"],output_dir)
make_lifetime_birth_plot(dataset1["birth_lifetime"],dataset1["name"],output_dir,log_x=False,log_y=False,point_size=4)
make_lifetime_birth_plot(dataset2["birth_lifetime"],dataset2["name"],output_dir,log_x=False,log_y=False,point_size=4)
make_persistence_surface_and_image(dataset1["birth_lifetime"],dataset1["name"],output_dir,weight_function="w3",
    sigma=0.008,resolution=128,cmap="magma",levels=100)
make_persistence_surface_and_image(dataset2["birth_lifetime"],dataset2["name"],output_dir,weight_function="w3",
    sigma=0.008,resolution=128,cmap="magma",levels=100)

### PCA RELATED STUFF ###

print("\nPreparing all PD data for PCA")
pds_1, steps_1, names_1 = prepare_PD_data_with_steps(dataset_1, training_set1, "0kbar", output_dir)
pds_2, steps_2, names_2 = prepare_PD_data_with_steps(dataset_2, training_set2, "6kbar", output_dir)

all_pds = pds_1 + pds_2
all_step_numbers = np.concatenate([steps_1, steps_2])
all_names = names_1 + names_2

print(f"Total samples: {len(all_pds)}")
print(f"Step numbers range: {all_step_numbers.min()} to {all_step_numbers.max()}")
print(f"Unique step numbers: {np.unique(all_step_numbers)}")
print(f"Dataset 1 steps: {steps_1}")
print(f"Dataset 2 steps: {steps_2}")

print("\nVectorizing persistence diagrams")
pd_vectors, vectorize_spec = vectorize_persistence_diagrams(all_pds, weight_function="w3", birth_range=(0, 8), resolution=128, sigma=0.06)
print(f"Vectorized data shape: {pd_vectors.shape}")

print("\nPerforming PCA regression")
pca_results = perform_pca_regression(pd_vectors, all_step_numbers, n_components=20, test_size=0.2, random_state=42)

print("PCA REGRESSION RESULTS")
print("=" * 30)
print(f"Training R² score: {pca_results['train_score']:.4f}")
print(f"Test R² score: {pca_results['test_score']:.4f}")
print(f"Total explained variance by {pca_results['pca'].n_components_} components: {np.sum(pca_results['pca'].explained_variance_ratio_):.4f}")

plot_pca_results(pca_results, output_dir, "Step_Number_Regression")
save_pca_results_to_csv(pca_results, output_dir, "Step_Number_Regression")
plot_pca_persistence_components(pca=pca_results['pca'],vectorize_spec=vectorize_spec,output_dir=output_dir,title_prefix="Step_Number_Regression")

### EXECUTE LOGISTIC REGRESSION ANALYSIS ###

# Prepare binary labels (0 for dataset_1/0kbar, 1 for dataset_2/6kbar)
all_labels = np.array([0] * len(pds_1) + [1] * len(pds_2))

print("\nPerforming logistic regression")
lr_results = perform_logistic_regression(pd_vectors, all_labels, test_size=0.2, random_state=69, C=0.01,solver="lbfgs")

print("\nLOGISTIC REGRESSION RESULTS")
print("=" * 30)
print(f"Training Accuracy: {lr_results['train_score']:.4f}")
print(f"Test Accuracy: {lr_results['test_score']:.4f}")
print(f"ROC AUC Score: {lr_results['roc_auc']:.4f}")
print("\nClassification Report:")
print(lr_results['classification_report'])

plot_logistic_regression_results(lr_results, output_dir, "Step_Number_Logistic")

print("\nGenerating coefficient persistence images")

plot_logistic_regression_coefficients(lr_results['model'], vectorize_spec, output_dir, title_prefix="Step_Number_Logistic",cmap="RdBu_r",midpoint=0.0)

print("\nSaving logistic regression results")
save_logistic_regression_results(lr_results, output_dir, "Step_Number_Logistic")

print("Fin")
