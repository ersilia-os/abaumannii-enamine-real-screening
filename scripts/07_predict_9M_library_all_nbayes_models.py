import os
import sys
import pandas as pd
from tqdm import tqdm

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, ".."))

from src.featurizer import Featurizer
from src.model import NaiveBayesClassificationModel as ClassificationModel

h5_fingerprints = os.path.join(root, "..", "processed", "04_fingerprints.h5")
csv_output = os.path.join(root, "..", "processed", "07_predictions.csv")

ds = pd.read_csv(os.path.join(root, "..", "processed", "01_nbayes_models_summary.csv"))
print(ds)

featurizer = Featurizer()

for i, row in tqdm(ds.iterrows()):
    source = row["source"]
    task = row["task"]
    model_dir = os.path.join(root, "..", "processed", "nbayes_models", source, task)
    model = ClassificationModel(model_dir=model_dir)
    identifiers = []
    y_hat = []
    R = []
    for j, chunk in enumerate(featurizer.h5_chunk_iterator(h5_fingerprints)):
        X = chunk["X"]
        if i == 0:
            identifiers_ = chunk["identifier"]
            identifiers += identifiers_.tolist()
        y_hat_ = model.predict_proba(X)
        y_hat += y_hat_.tolist()
    col = source + "_" + task
    if i == 0:
        df = pd.DataFrame({"identifier": identifiers, col: y_hat})
    else:
        df[col] = y_hat

df.to_csv(csv_output, index=False)