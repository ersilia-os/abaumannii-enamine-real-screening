import os
import joblib
import os
import sys
import numpy as np
from tqdm import tqdm
import h5py

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, ".."))
from src.featurizer import Featurizer

clf = joblib.load(os.path.join(root, "..", "processed", "08_model.pkl"))

N_BITS = 2048

fp_file_prefix = "06"

h5_paths = []
for fn in os.listdir(os.path.join(root, "..", "processed")):
    if fn.startswith(fp_file_prefix) and fn.endswith(".h5"):
        h5_path = os.path.join(root, "..", "processed", fn)
        h5_paths.append(h5_path)

h5_paths = sorted(h5_paths)
print(h5_paths)

featurizer = Featurizer()

cutoff = 0.8

h5_output = os.path.join(root, "..", "processed", "09_fingerprints.h5")
if os.path.exists(h5_output):
    os.remove(h5_output)

string_dtype = h5py.special_dtype(vlen=str)

def _initialize_h5_datasets(h5_path, nbits, string_dtype):
    with h5py.File(h5_path, "w") as h5_file:
        h5_file.create_dataset(
            "X",
            shape=(0, nbits),
            maxshape=(None, nbits),
            dtype="int8",
            chunks=True,
            compression="gzip",
        )
        h5_file.create_dataset(
            "identifier",
            shape=(0,),
            maxshape=(None,),
            dtype=string_dtype,
            compression="gzip",
        )
        h5_file.create_dataset(
            "smiles",
            shape=(0,),
            maxshape=(None,),
            dtype=string_dtype,
            compression="gzip",
        )


def _append_to_h5(h5_path, dataset_name, data):
    with h5py.File(h5_path, "a") as h5_file:
        dataset = h5_file[dataset_name]
        dataset.resize((dataset.shape[0] + len(data)), axis=0)
        dataset[-len(data) :] = data

_initialize_h5_datasets(h5_output, N_BITS, string_dtype)

for h5 in h5_paths:
    print("Predicting on", h5)
    for chunk in tqdm(featurizer.h5_chunk_iterator(h5), desc="Predicting on chunk"):
        X = chunk["X"]
        identifier = chunk["identifier"]
        smiles = chunk["smiles"]
        y_proba = clf.predict_proba(X)[:, 1]
        mask = y_proba > cutoff
        X = X[mask]
        identifier = identifier[mask]
        smiles = smiles[mask]
        assert len(X) == len(identifier) == len(smiles), "Error: variable lengths"
        _append_to_h5(h5_output, "X", X)
        _append_to_h5(h5_output, "identifier", identifier)
        _append_to_h5(h5_output, "smiles", smiles)
        print(np.sum(mask), "predicted as active", len(mask), "total")
