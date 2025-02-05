import os
import h5py
import numpy as np
import collections
import pandas as pd
from tqdm import tqdm

root = os.path.abspath(os.path.dirname(__file__))

MIN_PERCENTILE = 99
MIN_PROPORTION = 0.3
TOP_N = 100
MIN_ABS_PROBA = 0.2

print("Reading data...")
h5_path = os.path.join(root, "..", "processed", "10_predictions.h5")
output_csv = os.path.join(root, "..", "processed", "11_predictions.csv")

print("Getting columns")
with h5py.File(h5_path, "r") as f:
    columns = sorted([x for x in f.keys() if x not in ["smiles", "identifier"]])
print(columns)
print(len(columns))

cutoffs = {}
for col in tqdm(columns, desc="Calculating percentiles"):
    with h5py.File(h5_path, "r") as f:
        values = f[col][:]
        cutoffs[col] = int(np.percentile(values, MIN_PERCENTILE))

print(cutoffs)

idx_counts = collections.defaultdict(int)
for col in tqdm(columns, desc="Counting hits per identifier"):
    with h5py.File(h5_path, "r") as f:
        values = f[col][:]
    cutoff = cutoffs[col]
    sel_idxs = np.where(values >= cutoff)[0]
    for sel_idx in sel_idxs:
        idx_counts[int(sel_idx)] += 1

print(sorted(idx_counts.items(), key = lambda x: -x[1])[:100])

print("Keeping only identifiers with at least {0} counts".format(int(MIN_PROPORTION*len(columns))))
min_counts = int(MIN_PROPORTION*len(columns))
print(min_counts, "is the minumum number of hit columns")
keep_idxs = []
for k,v in idx_counts.items():
    if v >= min_counts:
        keep_idxs += [k]
keep_idxs = set(keep_idxs)
print(len(keep_idxs), "kept so far")

for col in tqdm(columns, desc="Getting top hits per column"):
    with h5py.File(h5_path, "r") as f:
        values = f[col][:]
    sel_idxs = np.argsort(values)[::-1][:TOP_N]
    keep_idxs.update([int(sel_idx) for sel_idx in sel_idxs])

print(len(keep_idxs), "kept at the end")

keep_idxs = sorted(keep_idxs)
print("Creating a dataframe")
with h5py.File(h5_path, "r") as f:
    print("Reading identifiers")
    identifiers = list(f["identifier"][keep_idxs].astype(str))
    print("Reading SMILES")
    smiles = list(f["smiles"][keep_idxs].astype(str))
    df = pd.DataFrame({"identifier": identifiers, "smiles": smiles})
    for col in tqdm(columns, desc="Iterating over columns"):
        values = f[col][keep_idxs]/10000
        df[col] = list(values)

print("Filtering by minimum absolute probability")
df = df[df[columns].gt(MIN_ABS_PROBA).any(axis=1)]

df.to_csv(output_csv, index=False)
print("Done. Output shape", df.shape)