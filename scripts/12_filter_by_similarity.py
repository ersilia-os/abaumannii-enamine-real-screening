import pandas as pd
from rdkit import Chem
import os
import csv
import sys
from tqdm import tqdm
import numpy as np
from standardiser import standardise

root = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(root, "..", "src"))
from similarity import TrainingSetScaffoldSimilarity, TrainingSetHighestSimilarity, ChemblHighestSimilarity

input_csv = os.path.join(root, "..", "processed", "11_predictions.csv")
output_csv = os.path.join(root, "..", "processed", "12_predictions.csv")

df = pd.read_csv(input_csv)
smiles_list = df["smiles"].tolist()

def has_at_least_one_ring(smiles_list):
    ring_molecules = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        # Check if the molecule was parsed successfully
        if mol:
            ring_info = mol.GetRingInfo()
            if ring_info.NumRings() > 0:
                ring_molecules.append(s)
    return ring_molecules

print("Keeping molecules that have at least one ring")
print(len(smiles_list))
smiles_list = has_at_least_one_ring(smiles_list)
print(len(smiles_list))

ss = TrainingSetScaffoldSimilarity()
keep_scaffold = ss.filter_by_scaffold(smiles_list)
print(keep_scaffold[:10])
smiles_list = [smiles for i, smiles in enumerate(smiles_list) if keep_scaffold[i] is True]

print(np.sum(keep_scaffold), len(keep_scaffold), "Kept by scaffold")
print(len(smiles_list))

st = TrainingSetHighestSimilarity()
keep_similarity = st.filter_by_highest_similarity(smiles_list)
print(keep_similarity[:10])
smiles_list = [smiles for i, smiles in enumerate(smiles_list) if keep_similarity[i] is True]

print(np.sum(keep_similarity), len(keep_similarity), "Kept by similarity")
print(len(smiles_list))

sc = ChemblHighestSimilarity()
keep_chembl = sc.filter_by_highest_similarity(smiles_list)
print(keep_chembl[:10])
print(np.sum(keep_chembl), len(keep_chembl), "Kept by chembl similarity")
smiles_list = [smiles for i, smiles in enumerate(smiles_list) if keep_chembl[i] is True]
print(len(smiles_list))

print("Saving to {0}".format(output_csv))
df = df[df["smiles"].isin(smiles_list)].reset_index(drop=True)

keep_idxs = []
std_smiles = []
for i, smiles in tqdm(enumerate(df["smiles"].tolist())):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        continue
    try:
        mol = standardise.run(mol)
    except:
        continue
    if mol is None:
        continue
    std_smiles += [Chem.MolToSmiles(mol)]
    keep_idxs += [i]

df = df.iloc[keep_idxs].reset_index(drop=True)
df.to_csv(output_csv, index=False)

print("Saving Ersilia input too, using standardised smiles")
with open(os.path.join(root, "..", "processed", "12_smiles.csv"), "w") as f:
    writer = csv.writer(f)
    writer.writerow(["smiles"])
    for smiles in std_smiles:
        writer.writerow([smiles])