import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from FPSim2 import FPSim2Engine
from tqdm import tqdm

root = os.path.dirname(os.path.abspath(__file__))


class TrainingSetScaffoldSimilarity(object):

    def __init__(self):
        self.reference_inchikeys = set(pd.read_csv(os.path.join(root, "..", "data", "other", "fpsim2_database_scaffolds.csv"))["inchikey"].tolist())
        print(len(self.reference_inchikeys))
        print(self.reference_inchikeys)

    def filter_by_scaffold(self, smiles_list):
        keep = []
        for smiles in tqdm(smiles_list, desc="Scaffold similarity"):
            mol = Chem.MolFromSmiles(smiles)
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            if scaffold is None:
                keep += [False]
                continue
            inchikey = Chem.inchi.MolToInchiKey(scaffold)
            if inchikey in self.reference_inchikeys:
                print(inchikey)
                keep += [False]
            else:
                keep += [True]
        return keep


class TrainingSetHighestSimilarity(object):

    def __init__(self, similarity_threshold=0.7):
        self.similarity_threshold = similarity_threshold
        self.fp_database = os.path.join(root, "..", "data", "other", "fpsim2_database_training_data.h5")
        self.fpe = FPSim2Engine(self.fp_database, in_memory_fps=False)

    def filter_by_highest_similarity(self, smiles_list):
        keep = []
        for smiles in tqdm(smiles_list, desc="Training set similarity"):
            results = self.fpe.on_disk_similarity(smiles, metric="tanimoto", threshold=self.similarity_threshold, n_workers=1)
            if len(results) > 0:
                keep += [False]
            else:
                keep += [True]
        return keep


class ChemblHighestSimilarity(object):

    def __init__(self, similarity_threshold=0.7):
        self.similarity_threshold = similarity_threshold
        self.fp_database = os.path.join(root, "..", "data", "other", "chembl_35_v0.6.0.h5")
        self.fpe = FPSim2Engine(self.fp_database, in_memory_fps=False)

    def filter_by_highest_similarity(self, smiles_list):
        keep = []
        for smiles in tqdm(smiles_list, desc="Chembl similarity"):
            results = self.fpe.on_disk_similarity(smiles, threshold=self.similarity_threshold, metric="tanimoto", n_workers=1)
            if len(results) > 0:
                keep += [False]
            else:
                keep += [True]
        return keep