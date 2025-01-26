import os
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
import shutil
import collections

root = os.path.dirname(os.path.abspath(__file__))
print(root)

import sys
sys.path.append(os.path.join(root, ".."))
from src.featurizer import Featurizer
from src.model import RandomForestClassificationModel as ClassificationModel

training_data_dir = os.path.abspath(os.path.join(root, "..", "data", "training"))

print("Getting data paths from ChEMBL")

csv_paths = []
for fn in os.listdir(os.path.join(training_data_dir, "chembl", "tasks")):
    if fn.endswith(".csv"):
        csv_paths.append(os.path.join(training_data_dir, "chembl", "tasks", fn))


print("Getting data paths from Co-ADD")
for fn in os.listdir(os.path.join(training_data_dir, "coadd", "tasks")):
    if fn.endswith(".csv"):
        csv_paths.append(os.path.join(training_data_dir, "coadd", "tasks", fn))

print("Getting data paths from Spark")
for fn in os.listdir(os.path.join(training_data_dir, "spark", "tasks")):
    if fn.endswith(".csv"):
        csv_paths.append(os.path.join(training_data_dir, "spark", "tasks", fn))

print("Training models and storing them in a similar folder structure in /processed")

featurizer = Featurizer()

for csv_path in tqdm(csv_paths):
    print(csv_path)
    output_dir = os.path.join(root, "..", "processed", "baseline_models")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    prefix = csv_path.split("/data/training/")[1].split("/")[0]
    X = featurizer.calculate_csv(csv_path)
    df = pd.read_csv(csv_path)
    columns = list(df.columns)
    outcome_columns = [c for c in columns if c != "inchikey" and c != "smiles" and not c.startswith("Unnamed")]
    print(outcome_columns)
    if len(outcome_columns) == 1:
        add_suffix = False
    else:
        add_suffix = True
    for column in outcome_columns:
        model_dir = os.path.join(output_dir, prefix, os.path.basename(csv_path).replace(".csv", ""))
        if os.path.exists(os.path.join(model_dir, "model.pkl")):
            print("Model already trained, skipping")
            continue
        if add_suffix:
            model_dir = model_dir + "_" + column
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        y = np.array(df[column].tolist())
        model = ClassificationModel(model_dir=model_dir)
        model.fit_cv(X, y)
        if model.results["auroc"] >= 0.75:
            model.save()
        else:
            print("Model did not pass threshold, not saving")

print("Done training models!")

print("Assembling results for summarization")

R = []
for subdir in ["chembl", "coadd", "spark"]:
    if not os.path.exists(os.path.join(root, "..", "processed", "baseline_models", subdir)):
        continue
    for model_dir in os.listdir(os.path.join(root, "..", "processed", "baseline_models", subdir)):
        if not os.path.exists(os.path.join(root, "..", "processed", "baseline_models", subdir, model_dir, "results.json")):
            shutil.rmtree(os.path.join(root, "..", "processed", "baseline_models", subdir, model_dir))
            continue
        with open(os.path.join(root, "..", "processed", "baseline_models", subdir, model_dir, "results.json"), "r") as f:
            results = json.load(f)
        data = collections.OrderedDict()
        data["source"] = subdir
        data["task"] = model_dir
        data["auroc"] = results["auroc"]
        data["num_pos"] = results["num_pos"]
        data["num_tot"] = results["num_tot"]
        data["opt_cut"] = results["opt_cut"]
        R += [data]

pd.DataFrame(R).to_csv(os.path.join(root, "..", "processed", "02_baseline_models_summary.csv"), index=False)

print("Done assembling results!")