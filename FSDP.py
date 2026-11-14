import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

df = pd.read_csv('LDA_ring_sizes.csv') #change file name as needed

def S_star(Q, deltas, sigma=0.05):
    total = np.zeros_like(Q, dtype=float)
    for delta in deltas:
        mu = 2 * np.pi / delta
        total += np.exp(-((Q - mu)**2) / (2 * sigma**2))
    return total / len(deltas)

def peak_position_error(Q, columns, sigma=0.05):
    peak_positions = {}
    all_deltas = []
    for name, deltas in columns.items():
        deltas = np.asarray(deltas)
        deltas = deltas[deltas > 0]
        all_deltas.extend(deltas)
        y = S_star(Q, deltas, sigma)
        peak_positions[name] = Q[np.argmax(y)]

    peaks = np.array(list(peak_positions.values()))

    avg_y = S_star(Q, np.array(all_deltas), sigma)
    avg_peak = Q[np.argmax(avg_y)]

    std_peak = np.std(peaks, ddof=1)
    stderr_peak = std_peak / np.sqrt(len(peaks))

    return avg_peak, std_peak, stderr_peak, peak_positions

sigma = 0.05
Q_rng = np.arange(0,8,1000) #probably won't need to go out as far as 8

columns = {col: df[col].dropna().values for col in df.columns}

all_deltas = np.concatenate(list(columns.values()))
mus = 2 * np.pi / all_deltas

Q_min = mus.min() - 3 * sigma
Q_max = mus.max() + 3 * sigma
Q = np.linspace(Q_min, Q_max, 500)

rows = []
for name, deltas in columns.items():
    y = S_star(Q, deltas, sigma)

    for q, val in zip(Q, y):
        rows.append((q, val, name))

avg_y = S_star(Q, all_deltas, sigma)
for q, val in zip(Q, avg_y):
    rows.append((q, val, "average"))

plot_df = pd.DataFrame(rows, columns=["Q", "S_star", "group"])

sns.set_style("white")
plt.figure(figsize=(10, 6))

base_df = plot_df[plot_df["group"] != "average"]
avg_df = plot_df[plot_df["group"] == "average"]

palette = sns.color_palette("rainbow", n_colors=base_df["group"].nunique())
ax = sns.lineplot(data=base_df, x="Q", y="S_star", hue="group",palette=palette, linewidth=2)
sns.lineplot(data=avg_df, x="Q", y="S_star", color="black", linestyle="--", linewidth=2.5)

leg_list = ['4', '5', '6', '7', '8', '9', '10', '10<i<15', '15<i<20', '20<i<30'] #These are just legend handles to make the subscripts in the legend

formatted_labels = [rf"$L_{{{s}}}$" for s in leg_list]
formatted_labels.append(r"$L_{\mathrm{avg}}$")
handles, _ = ax.get_legend_handles_labels()
handles.append(Line2D([], [], color='black', linestyle='--', linewidth=2.5))
ax.legend(handles=handles, labels=formatted_labels,bbox_to_anchor=(1.05, 1),loc="upper left", fontsize=12)
plt.xlabel(r"Q ($\AA$)", fontsize=15)
plt.ylabel("S*(Q)", fontsize=15)
plt.ylim(0, 0.6)
plt.margins(x=0)
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.tight_layout()
plt.show()


avg_peak, std_peak, stderr_peak, peak_positions = \
    peak_position_error(Q, columns, sigma)

print(f"Q_peak = {avg_peak:.4f}")
print(f"SD     = {std_peak:.4f}")
print(f"SE     = {stderr_peak:.4f}")

print("\nPeak positions by loop type")
print("-" * 60)
print(f"{'Loop type':<15}{'Q_peak':>12}{'Difference':>15}{'Rel. diff (%)':>18}")
print("-" * 60)

relative_differences = {}

for name, peak in peak_positions.items():
    difference = peak - avg_peak
    relative_difference = abs(difference) / avg_peak * 100
    relative_differences[name] = relative_difference

    print(
        f"{name:<15}"
        f"{peak:>12.4f}"
        f"{difference:>15.4f}"
        f"{relative_difference:>18.3f}")

best_match = min(relative_differences, key=relative_differences.get)

print("-" * 60)
print(f"\nBest match to averaged curve: {best_match}")
print(f"Peak position              : {peak_positions[best_match]:.4f}")
print(f"Average peak position      : {avg_peak:.4f}")
print(f"Relative difference        : {relative_differences[best_match]:.3f}%")
