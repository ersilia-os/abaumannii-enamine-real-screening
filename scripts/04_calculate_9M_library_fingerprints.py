import os
import sys

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, ".."))

from src.featurizer import Featurizer

cxsmiles_file = os.path.join(root, "..", "data", "prediction", "2024.07_Enamine_REAL_DB_9.6M.cxsmiles.bz2")

featurizer = Featurizer()
featurizer.calculate_cxsmiles2h5(cxsmiles_file, os.path.join(root, "..", "processed", "04_fingerprints.h5"))

print("Calculating features done!")