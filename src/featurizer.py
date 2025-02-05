import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit.Chem import rdFingerprintGenerator
from rdkit import Chem
import os

from .default import BATCH_SIZE, N_BITS, RADIUS


class Featurizer(object):

    def __init__(self, batch_size=BATCH_SIZE):
        self.batch_size = batch_size

    @staticmethod
    def _clip_sparse(vect, nbits):
        l = [0]*nbits
        for i,v in vect.GetNonzeroElements().items():
            l[i] = v if v < 255 else 255
        return l

    def calculate(self, smiles_list):
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
        X = np.zeros((len(smiles_list), N_BITS), dtype="int8")
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            v = mfpgen.GetCountFingerprint(mol)
            X[i,:] = self._clip_sparse(v, N_BITS)
        return X
    
    def calculate_csv(self, csv_path):
        smiles_list = pd.read_csv(csv_path)['smiles'].tolist()
        return self.calculate(smiles_list)

    def calculate_cxsmiles2h5(self, cxsmiles_path, h5_path, chunksize=None, max_rows_ceil=None):
        if os.path.exists(h5_path):
            os.remove(h5_path)

        if chunksize is None:
            chunksize = self.batch_size
        # Define special dtype for variable-length strings
        string_dtype = h5py.special_dtype(vlen=str)
        
        with h5py.File(h5_path, "a") as f:
            identifier_col = 'id'
            smiles_col = 'smiles'
            if cxsmiles_path.endswith('.bz2'):
                df_chunker = pd.read_csv(cxsmiles_path, chunksize=chunksize, sep="\t", compression='bz2', usecols=[identifier_col, smiles_col])
            else:
                df_chunker = pd.read_csv(cxsmiles_path, chunksize=chunksize, sep="\t", usecols=[identifier_col, smiles_col])
            for i, chunk in tqdm(enumerate(df_chunker)):
                # Convert columns to lists of Python strings
                identifiers_list = chunk[identifier_col].astype(str).to_list()
                smiles_list = chunk[smiles_col].astype(str).to_list()
                X = self.calculate(smiles_list).astype('int8')  # Integer matrix
                if i == 0:
                    # Create datasets with variable-length string dtype
                    f.create_dataset(
                        "X",
                        data=X,
                        shape=X.shape,
                        maxshape=(None, X.shape[1]),
                        dtype='int8',  # Integer type for X
                        chunks=(chunksize, X.shape[1]),
                        compression="gzip",
                    )
                    f.create_dataset(
                        "identifier",
                        data=identifiers_list,
                        shape=(len(identifiers_list),),
                        maxshape=(None,),
                        dtype=string_dtype,  # Variable-length string
                        compression="gzip",
                    )
                    f.create_dataset(
                        "smiles",
                        data=smiles_list,
                        shape=(len(smiles_list),),
                        maxshape=(None,),
                        dtype=string_dtype,  # Variable-length string
                        compression="gzip",
                    )
                else:
                    # Append to datasets
                    f["X"].resize((f["X"].shape[0] + X.shape[0]), axis=0)
                    f["X"][-X.shape[0]:] = X
                    f["identifier"].resize((f["identifier"].shape[0] + len(identifiers_list)), axis=0)
                    f["identifier"][-len(identifiers_list):] = identifiers_list
                    f["smiles"].resize((f["smiles"].shape[0] + len(smiles_list)), axis=0)
                    f["smiles"][-len(smiles_list):] = smiles_list
                if max_rows_ceil is not None and f["X"].shape[0] >= max_rows_ceil:
                    break

    def h5_chunk_iterator(self, h5_path, batch_size=None):
        datasets = ["X", "identifier", "smiles"]
        if batch_size is None:
            batch_size = self.batch_size

        with h5py.File(h5_path, "r") as f:
            num_rows = f[datasets[0]].shape[0]  # Assume all datasets have the same length
            for i in tqdm(range(0, num_rows, batch_size)):
                chunk = {}
                for dataset in datasets:
                    data = f[dataset][i:i+batch_size]
                    # Decode only if dataset uses vlen=str dtype
                    if h5py.check_dtype(vlen=f[dataset].dtype) == str:
                        chunk[dataset] = data.astype(str)  # Convert to Python strings
                    else:
                        chunk[dataset] = data  # Use as is for non-string datasets
                yield chunk
