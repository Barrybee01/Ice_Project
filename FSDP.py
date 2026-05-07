import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

df = pd.read_csv('LDA_ring_sizes.csv') #change file name as needed

sigma = 0.05
Q_rng = np.arange(0,8,1000) #probably won't need to go out as far as 8

columns = {col: df[col].dropna().values for col in df.columns}

all_deltas = np.concatenate(list(columns.values()))

mus = 2 * np.pi / all_deltas

Q_min = mus.min() - 3 * sigma
Q_max = mus.max() + 3 * sigma
Q = np.linspace(Q_min, Q_max, 500)

def S_star(Q, deltas, sigma=0.05):
    total = np.zeros_like(Q, dtype=float)
    for delta in deltas:
        mu = 2 * np.pi / delta
        total += np.exp(-((Q - mu)**2) / (2 * sigma**2))
    return total / len(deltas)

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

ax = sns.lineplot(data=base_df, x="Q", y="S_star", hue="group",
    palette=palette, linewidth=2)

sns.lineplot(data=avg_df, x="Q", y="S_star", color="black", 
             linestyle="--", linewidth=2.5)

leg_list = ['4', '5', '6', '7', '8', '9', '10', '10<i<15', '15<i<20', '20<i<30'] #These are just legend handles to make the subscripts in the legend

formatted_labels = [rf"$L_{{{s}}}$" for s in leg_list]
formatted_labels.append(r"$L_{\mathrm{avg}}$")
handles, _ = ax.get_legend_handles_labels()
handles.append(Line2D([], [], color='black', linestyle='--', linewidth=2.5))

ax.legend(handles=handles, labels=formatted_labels,bbox_to_anchor=(1.05, 1),
          loc="upper left", fontsize=12)

#plt.axvline(x=2.3, c='k', linestyle='--') 
plt.xlabel(r"Q ($\AA$)", fontsize=15)
plt.ylabel("S*(Q)", fontsize=15)
plt.ylim(0, 0.6)
plt.margins(x=0)

plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13

plt.tight_layout()
plt.show()
