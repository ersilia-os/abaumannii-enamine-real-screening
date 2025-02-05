import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from rdkit.Chem import rdFingerprintGenerator
from rdkit import Chem
from rdkit.Chem import DataStructs

SIMILARITY_CUTOFF = 0.8

root = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(root, "..", "processed", "16_predictions.csv"))

agg_cols = [x for x in list(df.columns)[3:] if not x.startswith("eos42ez") and not x in ["eos74km_acid-fast", "eos74km_fungi", "eos74km_gram-positive", "eos74km_inactive"]]

X = np.array(df[agg_cols])
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
agg = np.sum(X, axis=1)

df["aggregate_score"] = agg

df = df.sort_values("aggregate_score", ascending=False)
print(df)

def remove_redundant_smiles(smiles_list, similarity_cutoff):
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    unique_smiles = []
    fingerprints = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue 
        fp = mfpgen.GetFingerprint(mol)
        is_unique = True
        for existing_fp in fingerprints:
            similarity = DataStructs.TanimotoSimilarity(fp, existing_fp)
            if similarity >= similarity_cutoff:
                is_unique = False
                break
        if is_unique:
            unique_smiles.append(smi)
            fingerprints.append(fp)
    return unique_smiles

smiles_list = remove_redundant_smiles(df["std_smiles"].tolist(), similarity_cutoff=SIMILARITY_CUTOFF)
df = df[df["std_smiles"].isin(smiles_list)]

df.to_csv(os.path.join(root, "..", "processed", "17_predictions.csv"), index=False)