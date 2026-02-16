sigma = 0.05
qmin, qmax = 0.0, 5.0
threshold = 1e-30

in_file  = "0kbar_finalstate_ringsizes.csv"
out_file = "fsdp.csv"

def ring_stat(Q, deltas, sigma, threshold):
    deltas = np.asarray(deltas, float)
    deltas = deltas[np.isfinite(deltas) & (deltas != 0)]
    if deltas.size == 0:
        return np.zeros_like(Q)

    centers = 2 * np.pi / deltas
    S = np.exp(-((Q - centers[:, None])**2) / (2 * sigma**2)).mean(axis=0)
    S[S < threshold] = 0.0
    return S
	
def interp_to_common(Q_common, Q, S):
    Q = np.asarray(Q, float)
    S = np.asarray(S, float)
    m = np.isfinite(Q) & np.isfinite(S)
    Q, S = Q[m], S[m]
    idx = np.argsort(Q)
    Q, S = Q[idx], S[idx]
    return np.interp(Q_common, Q_unique, S_unique, left=0.0, right=0.0)

with open(in_file, "r") as f:
    header = f.readline().strip()

names = [h.strip() for h in header.split(",")]
df = pd.read_csv(in_file, skiprows=1, header=None)

Qs = []
Ss = []
lengths = []

for col in range(3):
    deltas = df[col].dropna().to_numpy()
    n = len(deltas)
    lengths.append(n)
    if n < 2:
        Q = np.array([qmin])
        S = np.zeros_like(Q)
    else:
        Q = np.linspace(qmin, qmax, n)
        S = ring_stat(Q, deltas, sigma, threshold)

    Qs.append(Q)
    Ss.append(S)

max_len = max(lengths)
out_dict = {}

for i in range(3):
    qcol = np.zeros(max_len)
    scol = np.zeros(max_len)
    qcol[:len(Qs[i])] = Qs[i]
    scol[:len(Ss[i])] = Ss[i]
    out_dict[f"Q_{names[i]}"] = qcol
    out_dict[f"S_{names[i]}"] = scol

out_df = pd.DataFrame(out_dict)

out_df.to_csv(out_file, index=False, float_format="%.10g")

fsdp_dat = pd.read_csv("fsdp.csv")

Q1 = np.array(fsdp_dat["Q_H1"])
S1 = np.array(fsdp_dat["S_H1"])
Q2 = np.array(fsdp_dat["Q_H2"])
S2 = np.array(fsdp_dat["S_H2"])
Q3 = np.array(fsdp_dat["Q_H3"])
S3 = np.array(fsdp_dat["S_H3"])

Q_common = np.linspace(0, 5, 500)

S1_i = interp_to_common(Q_common, Q1, S1)
S2_i = interp_to_common(Q_common, Q2, S2)
S3_i = interp_to_common(Q_common, Q3, S3)

S_avg = (S1_i + S2_i + S3_i) / 3

out_file = "average_SQ.txt"
out = np.column_stack((Q_common, S_avg))

np.savetxt(
    out_file,
    out,
    header="Q\tS_avg",
    fmt="%.10g",
    comments="" 
)