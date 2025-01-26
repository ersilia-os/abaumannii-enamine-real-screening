import pandas as pd
import os
import csv
import sys
from tqdm import tqdm
import numpy as np

root = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(root, "..", "src"))
from similarity import TrainingSetScaffoldSimilarity, TrainingSetHighestSimilarity, ChemblHighestSimilarity

if not os.path.exists(os.path.join(root, "..", "processed", "07_smiles.csv")):
    df = pd.read_csv(os.path.join(root, "..", "data", "prediction", "2024.07_Enamine_REAL_DB_9.6M.cxsmiles.bz2"), sep="\t").head(1000)
    smiles_list = df["smiles"].tolist()
    with open(os.path.join(root, "..", "processed", "07_smiles.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles"])
        for smiles in tqdm(smiles_list, desc="Writing SMILES"):
            writer.writerow([smiles])
else:
    df = pd.read_csv(os.path.join(root, "..", "processed", "07_smiles.csv"))
    smiles_list = df["smiles"].tolist()

ss = TrainingSetScaffoldSimilarity()
keep_scaffold = ss.filter_by_scaffold(smiles_list)

print(np.sum(keep_scaffold), len(keep_scaffold), "Kept by scaffold")

st = TrainingSetHighestSimilarity()
keep_similarity = st.filter_by_highest_similarity(smiles_list)

print(np.sum(keep_similarity), len(keep_similarity), "Kept by similarity")


sc = ChemblHighestSimilarity()
keep_chembl = sc.filter_by_highest_similarity(smiles_list)

print(np.sum(keep_chembl), len(keep_chembl), "Kept by chembl similarity")