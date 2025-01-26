
import os
import pandas as pd
from ftplib import FTP
import csv
from tqdm import tqdm
from FPSim2.io import create_db_file
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

root = os.path.dirname(os.path.abspath(__file__))


local_file = os.path.join(root, "..", "data", "other", "chembl_35_v0.6.0.h5")

if not os.path.exists(local_file):
    print("Downloading ChEMBL database file")
    ftp_server = "ftp.ebi.ac.uk"
    ftp_path = "/pub/databases/chembl/ChEMBLdb/releases/chembl_35/chembl_35_v0.6.0.h5"
    ftp = FTP(ftp_server)
    ftp.login()
    with open(local_file, "wb") as file:
        ftp.retrbinary(f"RETR {ftp_path}", file.write)

    ftp.quit()
    print(f"Downloaded {local_file} successfully!")
else:
    print(f"{local_file} already exists!")

smiles_list = []
for subdir in os.listdir(os.path.join(root, "..", "data", "training")):
    if subdir.startswith("."):
        continue
    for task in os.listdir(os.path.join(root, "..", "data", "training", subdir, "tasks")):
        df = pd.read_csv(os.path.join(root, "..", "data", "training", subdir, "tasks", task))
        smiles_list += df["smiles"].tolist()

smiles_list = sorted(set(smiles_list))

mols = [[smiles, i] for i, smiles in enumerate(smiles_list)]

print("Creating a database file with Morgan fingerprints")

create_db_file(
    mols_source=mols,
    filename=os.path.join(root, "..", "data", "other", "fpsim2_database_training_data.h5"),
    mol_format='smiles',
    fp_type='Morgan',
    fp_params={'radius': 2, 'fpSize': 1024}
)

with open(os.path.join(root, "..", "data", "other", "fpsim2_database_smiles.csv"), "w") as f:
    writer = csv.writer(f)
    writer.writerow(["smiles", "index"])
    for smiles, i in mols:
        writer.writerow([smiles, i])

print("Done creating the database file!")

print("Getting Murcko scaffolds")

scaffolds = []
for smiles in tqdm(smiles_list):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    scaffolds.append(Chem.MolToSmiles(scaffold))

print("Converting Murcko scaffolds to InChIKeys and SMILES")

scaffolds = sorted(set(scaffolds))
scaffolds_inchikeys = []
scaffolds_smiles = []
for scaffold in tqdm(scaffolds):
    mol = Chem.MolFromSmiles(scaffold)
    if mol is None:
        continue
    inchikey = Chem.inchi.MolToInchiKey(mol)
    if scaffold is None or scaffold == "":
        continue
    if inchikey is None or inchikey == "":
        continue
    scaffolds_inchikeys.append(inchikey)
    scaffolds_smiles.append(scaffold)

with open(os.path.join(root, "..", "data", "other", "fpsim2_database_scaffolds.csv"), "w") as f:
    writer = csv.writer(f)
    writer.writerow(["inchikey", "smiles"])
    for smi, ik in zip(scaffolds_smiles, scaffolds_inchikeys):
        writer.writerow([ik, smi])