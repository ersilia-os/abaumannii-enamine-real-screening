import os
import h5py
import time
import pandas as pd
import numpy as np
from rdkit.Chem import rdFingerprintGenerator
from rdkit import Chem
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os

N_BITS = 2048
RADIUS = 3
BATCH_SIZE = 800000
DTYPE = np.uint8


class Featurizer:
    def __init__(self, batch_size=BATCH_SIZE):
        self.batch_size = batch_size

    @staticmethod
    def _clip_sparse(vect, nbits):
        result = np.zeros(nbits, dtype=np.int8)
        for idx, val in vect.GetNonzeroElements().items():
            result[idx] = min(val, 255)
        return result

    @staticmethod
    def _process_smiles(smiles_list):
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
        results = np.zeros((len(smiles_list), N_BITS), dtype=np.int8)
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol:
                vect = mfpgen.GetCountFingerprint(mol)
                results[i] = Featurizer._clip_sparse(vect, N_BITS)
        return results

    def calculate(self, smiles_list):
        num_cores = cpu_count()
        chunk_size = max(1, len(smiles_list) // num_cores)
        chunks = [
            smiles_list[i : i + chunk_size]
            for i in range(0, len(smiles_list), chunk_size)
        ]
        with Pool(num_cores) as pool:
            results = list(
                tqdm(
                    pool.imap(self._process_smiles, chunks),
                    total=len(chunks),
                    desc="Processing SMILES",
                )
            )
        return np.vstack(results)

    @staticmethod
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

    @staticmethod
    def _append_to_h5(h5_path, dataset_name, data):
        with h5py.File(h5_path, "a") as h5_file:
            dataset = h5_file[dataset_name]
            dataset.resize((dataset.shape[0] + len(data)), axis=0)
            dataset[-len(data) :] = data

    def calculate_cxsmiles2h5(self, cxsmiles_path, h5_path, max_rows_ceil=None):
        if os.path.exists(h5_path):
            os.remove(h5_path)

        string_dtype = h5py.special_dtype(vlen=str)
        chunker = pd.read_csv(
            cxsmiles_path,
            chunksize=self.batch_size,
            sep="\t",
            compression="bz2" if cxsmiles_path.endswith(".bz2") else None,
            usecols=["id", "smiles"],
        )
        create = True
        n_rows = 0
        h5_name = h5_path.split(".h5")[0]
        h5_path = h5_name + "_00.h5"
        for chunk in tqdm(chunker, desc="Processing Chunks"):
            identifiers = chunk["id"].astype(str).tolist()
            smiles_list = chunk["smiles"].astype(str).tolist()
            X = self.calculate(smiles_list)
            if create:
                self._initialize_h5_datasets(h5_path, N_BITS, string_dtype)
                create = False
            self._append_to_h5(h5_path, "X", X)
            self._append_to_h5(h5_path, "identifier", identifiers)
            self._append_to_h5(h5_path, "smiles", smiles_list)
            with h5py.File(h5_path, "r") as h5_file:
                shape = h5_file["X"].shape
            if max_rows_ceil and shape[0] >= max_rows_ceil:
                break
            n_rows += len(chunk)
            if n_rows >= BATCH_SIZE*100:
                idx = int(h5_path.split(".h5")[0].split("_")[-1]) + 1
                h5_path = h5_name + "_{0}.h5".format(str(idx).zfill(2))
                n_rows = 0
                create = True

    def h5_chunk_iterator(self, h5_path, batch_size=None):
        datasets = ["X", "identifier", "smiles"]
        batch_size = batch_size or self.batch_size
        with h5py.File(h5_path, "r") as h5_file:
            num_rows = h5_file["X"].shape[0]
            for i in range(0, num_rows, batch_size):
                yield {
                    dataset: h5_file[dataset][i : i + batch_size]
                    for dataset in datasets
                }


if __name__ == "__main__":
    
    st = time.time()
    root = os.path.dirname(os.path.abspath(__file__))

    def main():
        csv_path = os.path.join(root, "..", "data", "prediction", "Enamine_REAL_350-3_lead-like_cxsmiles.cxsmiles.bz2")
        h5_path = os.path.join(root, "..", "processed", "06_fingerprints.h5")
        featurizer = Featurizer()
        featurizer.calculate_cxsmiles2h5(cxsmiles_path=csv_path, h5_path=h5_path)

    main()
    et = time.time()
    print(f"Time taken: {et - st:.2f} seconds")
