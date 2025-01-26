import pandas as pd
import os
import csv
import sys
from tqdm import tqdm
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_curve, auc
import joblib

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, ".."))

from src.featurizer import Featurizer

df = pd.read_csv(os.path.join(root, "..", "processed", "07_predictions.csv"))
h5_path = os.path.join(root, "..", "processed", "04_fingerprints.h5")

columns = df.columns.tolist()[1:]

R = []

all_positive_idxs = set()
all_exclude_idxs = set()
for col in columns:
    print("Getting top 1000 indices for column", col)
    vals = np.array(df[col].tolist())
    idxs = np.array([i for i in range(len(vals))], dtype=int)
    print("Checking if there are enough values above 0.5")
    potential_positive_idxs = np.array([i for i in idxs if vals[i] > 0.5], dtype=int)
    if len(potential_positive_idxs) >= 1000:
        print("Best scenario")
        print("Sorting potential positive indices")
        potential_positive_idxs = sorted(potential_positive_idxs, key=lambda x: -vals[x])
        positive_idxs = potential_positive_idxs[:1000]
        exclude_idxs = potential_positive_idxs[1000:100000]
        assert np.all(vals[positive_idxs] > 0.5), "Error"
    else:
        print("Suboptimal scenario")
        print("Not enough values above 0.5, getting top 1000 from the ones above 0.1")
        potential_positive_idxs = np.array([i for i in idxs if vals[i] > 0.1], dtype=int)
        potential_positive_idxs = sorted(potential_positive_idxs, key=lambda x: -vals[x])
        positive_idxs = potential_positive_idxs[:1000]
        print(vals[positive_idxs])
        exclude_idxs = []
        assert np.all(vals[positive_idxs] > 0.1), "Error"
    all_positive_idxs.update([int(x) for x in positive_idxs])
    all_exclude_idxs.update([int(x) for x in exclude_idxs])

y = np.zeros((len(df),), dtype=int)

for i in range(len(y)):
    if i in all_positive_idxs:
        y[i] = 1
    elif i in all_exclude_idxs:
        y[i] = -1
    else:
        continue

print(len(all_positive_idxs), len(all_exclude_idxs))

y = np.array(y, dtype=int)

featurizer = Featurizer()
clf = MultinomialNB()

start_idx = 0
for i, data in tqdm(enumerate(featurizer.h5_chunk_iterator(h5_path)), "Training on chunks"):
    X = data["X"]
    n = X.shape[0]
    end_idx = start_idx + n
    y_chunk = y[start_idx:end_idx]
    mask = y_chunk != -1
    X = X[mask]
    y_chunk = y_chunk[mask]
    start_idx = end_idx
    if i == 9:
        print("Evaluating on the last chunk")
        y_hat = clf.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y_chunk, y_hat)
        roc_auc = auc(fpr, tpr)
        print("AUROC", roc_auc)
        with open(os.path.join(root, "..", "processed", "08_performance.csv"), "w") as f:
            writer = csv.writer(f)
            writer.writerow(["auroc"])
            writer.writerow([roc_auc])
    clf.partial_fit(X, y_chunk, classes=[0, 1])

joblib.dump(clf, os.path.join(root, "..", "processed", "08_model.pkl"))