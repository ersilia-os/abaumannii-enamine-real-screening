import os
import sys
import pandas as pd
import h5py
import numpy as np
from tqdm import tqdm

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, ".."))

from src.featurizer import Featurizer
from src.model import FlamlClassificationModel as ClassificationModel

h5_fingerprints = os.path.join(root, "..", "processed", "09_fingerprints.h5")
h5_output = os.path.join(root, "..", "processed", "10_predictions.h5")

ds = pd.read_csv(os.path.join(root, "..", "processed", "03_flaml_models_summary.csv"))
print(ds)

featurizer = Featurizer()
string_dtype = h5py.special_dtype(vlen=str)

for i, row in tqdm(ds.iterrows(), desc="Predicting over different models"):
    source = row["source"]
    task = row["task"]
    col = "_".join([source, task])
    if os.path.exists(h5_output):
        with h5py.File(h5_output, "r") as f:
            datasets = list(f.keys())
            if col in datasets:
                print("Skipping", col)
                continue
    model_dir = os.path.join(root, "..", "processed", "flaml_models", source, task)
    model = ClassificationModel(model_dir=model_dir)
    print("Loading model from", model_dir)
    identifiers = []
    smiles = []
    y_hat = []
    R = []
    for j, chunk in tqdm(enumerate(featurizer.h5_chunk_iterator(h5_fingerprints)), desc="Predicting over chunks"):
        print("Working on chunk", j)
        X = chunk["X"]
        if i == 0:
            identifiers_ = chunk["identifier"]
            identifiers += identifiers_.tolist()
            smiles_ = chunk["smiles"]
            smiles += smiles_.tolist()
        y_hat_ = model.predict_proba(X)
        y_hat += y_hat_.tolist()
    y_hat = np.array(np.array(y_hat)*10000, dtype="int16")
    if not os.path.exists(h5_output):
        with h5py.File(h5_output, "w") as f:
            f.create_dataset(col, data=y_hat)
            f.create_dataset("identifier", data=identifiers, dtype=string_dtype)
            f.create_dataset("smiles", data=smiles, dtype=string_dtype)
    else:
        with h5py.File(h5_output, "a") as f:
            f.create_dataset(col, data=y_hat)