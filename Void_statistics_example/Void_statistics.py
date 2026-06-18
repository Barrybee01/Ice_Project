import os
import ase.io
import homcloud.interface as hc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def make_PD(xyz_file, sample_name, output_dir):
    amorph_ice = ase.io.read(xyz_file)
    weights = np.array([0.175**2 if atom == "O" else 0.775**2 for atom in amorph_ice.get_chemical_symbols()])

    pdgm_path = os.path.join(output_dir, f"{sample_name}.pdgm")

    pd1 = hc.PDList.from_alpha_filtration(
        amorph_ice.get_positions(),
        vertex_symbols=amorph_ice.get_chemical_symbols(),
        weight=weights,
        save_boundary_map=True,
        save_phtrees=True,
        save_to=pdgm_path
    ).dth_diagram(2)

    print(f"Saved PDGM: {pdgm_path}")
    return pd1


def compute_birth_lifetime(pd1):
    births = pd1.births
    deaths = pd1.deaths
    lifetimes = deaths - births

    df = pd.DataFrame({"Birth": births, "Death": deaths, "Lifetime": lifetimes})
    return df


def save_figure(fig, output_dir, filename):
    output_path = os.path.join(output_dir, filename)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved image: {output_path}")


def make_PD_plot(pd1, sample_name, output_dir):
    pd1.histogram((0, 8), 250).plot(colorbar={"type": "log", "colormap": "plasma"}, font_size=18)
    fig = plt.gcf()
    save_figure(fig=fig, output_dir=output_dir, filename=f"{sample_name}_PD.png")

def calculate_fan_angle(pd1, n_wedges):
    births = np.asarray(pd1.births)
    deaths = np.asarray(pd1.deaths)

    distances_from_zero = np.sqrt(births**2 + deaths**2)
    origin_index = np.argmin(distances_from_zero) #find point closest to lower left corner

    origin_birth = births[origin_index]
    origin_death = deaths[origin_index]

    dx = births - origin_birth
    dy = deaths - origin_death

    nonzero = (dx != 0) | (dy != 0)

    dx_nonzero = dx[nonzero]
    dy_nonzero = dy[nonzero]

    point_angles = np.arctan2(dy_nonzero, dx_nonzero)
    radii = np.sqrt(dx_nonzero**2 + dy_nonzero**2)

    diagonal_angle = np.pi / 4
    min_radius_for_angle = 1.0
    angle_mask = radii >= min_radius_for_angle
    upper_arm_angle = np.max(point_angles[angle_mask])

    fan_angle_rad = upper_arm_angle - diagonal_angle
    fan_angle_deg = np.degrees(fan_angle_rad)

    plot_max = max(np.max(births), np.max(deaths))
    arm_length = 1.1 * np.sqrt((plot_max - origin_birth)**2 + (plot_max - origin_death)**2)

    wedge_angles = np.linspace(diagonal_angle, upper_arm_angle, n_wedges + 1)

    wedge_lines = []
    for wedge_index, wedge_angle in enumerate(wedge_angles):
        wedge_line = {"wedge_boundary": wedge_index, "angle_degrees": np.degrees(wedge_angle), "x": [origin_birth, origin_birth + arm_length * np.cos(wedge_angle)], "y": [origin_death, origin_death + arm_length * np.sin(wedge_angle)]}
        wedge_lines.append(wedge_line)

    diagonal_line = {"x": [origin_birth, origin_birth + arm_length * np.cos(diagonal_angle)], "y": [origin_death, origin_death + arm_length * np.sin(diagonal_angle)]}

    upper_arm_line = {"x": [origin_birth, origin_birth + arm_length * np.cos(upper_arm_angle)], "y": [origin_death, origin_death + arm_length * np.sin(upper_arm_angle)]}

    return {
    	"births": births,
    	"deaths": deaths,
    	"origin_birth": origin_birth,
    	"origin_death": origin_death,
    	"min_radius_for_angle": min_radius_for_angle,
    	"n_wedges": n_wedges,
    	"diagonal_angle_degrees": np.degrees(diagonal_angle),
    	"upper_arm_angle_degrees": np.degrees(upper_arm_angle),
    	"fan_angle_degrees": fan_angle_deg,
    	"fan_angle_radians": fan_angle_rad,
    	"wedge_angles_radians": wedge_angles,
    	"wedge_angles_degrees": np.degrees(wedge_angles),
    	"plot_max": plot_max,
    	"diagonal_line": diagonal_line,
    	"upper_arm_line": upper_arm_line,
    	"wedge_lines": wedge_lines}

def make_fan_angle_plot(sample_name, output_dir, fan_results):
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(fan_results["births"], fan_results["deaths"], s=2, alpha=0.5)

    for wedge_line in fan_results["wedge_lines"]:
        if wedge_line["wedge_boundary"] == 0:
            ax.plot(wedge_line["x"], wedge_line["y"], linewidth=2, label="Birth-death arm")
        elif wedge_line["wedge_boundary"] == fan_results["n_wedges"]:
            ax.plot(wedge_line["x"], wedge_line["y"], linewidth=2, label="Upper fan arm")
        else:
            ax.plot(wedge_line["x"], wedge_line["y"], linewidth=1, linestyle="--", alpha=0.7)

    ax.scatter(fan_results["origin_birth"], fan_results["origin_death"], s=60, marker="x", label="Fan origin")

    ax.set_xlabel("Birth", fontsize=16)
    ax.set_ylabel("Death", fontsize=16)

    ax.set_xlim(0, fan_results["plot_max"])
    ax.set_ylim(0, fan_results["plot_max"])
    ax.set_aspect("equal", adjustable="box")
    ax.legend()

    save_figure(fig=fig, output_dir=output_dir, filename=f"{sample_name}_fan_angle.png")

def assign_pairs_to_wedges(pd1, fan_results):
    pairs = pd1.pairs()

    origin_birth = fan_results["origin_birth"]
    origin_death = fan_results["origin_death"]
    wedge_angles = fan_results["wedge_angles_radians"]
    n_wedges = fan_results["n_wedges"]

    pair_records = []

    for pair_index, pair in enumerate(pairs):
        birth = pair.birth
        death = pair.death
        lifetime = death - birth

        dx = birth - origin_birth
        dy = death - origin_death

        angle = np.arctan2(dy, dx)

        wedge_index = np.searchsorted(wedge_angles, angle, side="right") - 1

        if wedge_index == n_wedges:
            wedge_index = n_wedges - 1

        if 0 <= wedge_index < n_wedges:
            pair_records.append({"pair_index": pair_index, "birth": birth, "death": death, "lifetime": lifetime, "angle_degrees": np.degrees(angle), "wedge": wedge_index})

    wedge_pair_df = pd.DataFrame(pair_records)

    return wedge_pair_df


def analyze_xyz_file(xyz_file, sample_name, output_dir, n_wedges):
    pd1 = make_PD(xyz_file=xyz_file, sample_name=sample_name, output_dir=output_dir)

    birth_lifetime_df = compute_birth_lifetime(pd1)
    birth_lifetime_csv_path = os.path.join(output_dir, f"{sample_name}_birth_lifetime.csv")
    birth_lifetime_df.to_csv(birth_lifetime_csv_path, index=False)
    print(f"Saved CSV: {birth_lifetime_csv_path}")

    fan_results = calculate_fan_angle(pd1, n_wedges=n_wedges)

    fan_summary_df = pd.DataFrame([{
        "origin_birth": fan_results["origin_birth"],
        "origin_death": fan_results["origin_death"],
        "min_radius_for_angle": fan_results["min_radius_for_angle"],
        "n_wedges": fan_results["n_wedges"],
        "diagonal_angle_degrees": fan_results["diagonal_angle_degrees"],
        "upper_arm_angle_degrees": fan_results["upper_arm_angle_degrees"],
        "fan_angle_degrees": fan_results["fan_angle_degrees"],
        "fan_angle_radians": fan_results["fan_angle_radians"]}])

    fan_csv_path = os.path.join(output_dir, f"{sample_name}_fan_angle.csv")
    fan_summary_df.to_csv(fan_csv_path, index=False)
    print(f"Saved fan angle CSV: {fan_csv_path}")

    wedge_pair_df = assign_pairs_to_wedges(pd1, fan_results)
    wedge_pair_csv_path = os.path.join(output_dir, f"{sample_name}_wedge_pairs.csv")
    wedge_pair_df.to_csv(wedge_pair_csv_path, index=False)
    print(f"Saved wedge pair CSV: {wedge_pair_csv_path}")

    make_PD_plot(pd1=pd1, sample_name=sample_name, output_dir=output_dir)

    make_fan_angle_plot(sample_name=sample_name, output_dir=output_dir, fan_results=fan_results)




