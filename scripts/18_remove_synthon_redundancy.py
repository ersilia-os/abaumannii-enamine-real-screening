import pandas as pd
import os
import numpy as np
import collections

MAX_INSTANCES_PER_SERIES = 3

root = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(root, "..", "processed", "17_predictions.csv"))

synthon_counts = collections.defaultdict(int)

identifiers = df["identifiers"].tolist()

columns = list(df.columns)
R = []
for r in df.values:
    keep = True
    r = list(r)
    synthons = list(set(r[0].split("____")[1:]))
    print(synthons)
    for synthon in synthons:
        if synthon_counts[synthon] >= MAX_INSTANCES_PER_SERIES:
            keep = False
        synthon_counts[synthon] += 1
    if keep:
        R += [r]

print(sorted(synthon_counts.items(), key=lambda x: -x[1]))
df = pd.DataFrame(R, columns=columns)
df.to_csv(os.path.join(root, "..", "processed", "18_predictions.csv"), index=False)