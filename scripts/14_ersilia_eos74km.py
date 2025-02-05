import pandas as pd
import os
from ersilia import ErsiliaModel
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import rdFingerprintGenerator

root = os.path.dirname(os.path.abspath(__file__))

if not os.path.exists(os.path.join(root, "..", "processed", "14_predictions.csv")):
    with ErsiliaModel("eos74km") as mdl:
        mdl.run(input=os.path.join(root, "..", "processed", "12_smiles.csv"), output=os.path.join(root, "..", "processed", "14_predictions.csv"))