import pandas as pd
import os
import collections

root = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(root, "..", "processed", "12_predictions.csv"))

data = collections.OrderedDict()

data["identifiers"] = df["identifier"].tolist()
data["smiles"] = df["smiles"].tolist()
data["std_smiles"] = pd.read_csv(os.path.join(root, "..", "processed", "12_smiles.csv"))["smiles"].tolist()

df = pd.DataFrame(data)
print(df[["smiles"]])

def df_to_dict(csv_file):
    df = pd.read_csv(csv_file)
    columns = list(df.columns)[2:]
    smiles_list = df["input"].tolist()
    data = {}
    for i, v in enumerate(df[columns].values):
        data[smiles_list[i]] = list(v)
    return data, columns

data_1, cols_1 = df_to_dict(os.path.join(root, "..", "processed", "13_predictions.csv"))
data_2, cols_2 = df_to_dict(os.path.join(root, "..", "processed", "14_predictions.csv"))
data_3, cols_3 = df_to_dict(os.path.join(root, "..", "processed", "15_predictions.csv"))

R = []
for smiles in df["std_smiles"].tolist():
    R += [data_1[smiles]]
df = pd.concat([df, pd.DataFrame(R, columns=["eos3804_"+x for x in cols_1])], axis=1)

R = []
for smiles in df["std_smiles"].tolist():
    R += [data_2[smiles]]
df = pd.concat([df, pd.DataFrame(R, columns=["eos74km_"+x for x in cols_2])], axis=1)

R = []
for smiles in df["std_smiles"].tolist():
    R += [data_3[smiles]]
df = pd.concat([df, pd.DataFrame(R, columns=["eos42ez_"+x for x in cols_3])], axis=1)

df_ = pd.read_csv(os.path.join(root, "..", "processed", "12_predictions.csv"))
columns = [x for x in list(df_.columns) if x not in ["identifier", "smiles"]]

df = pd.concat([df, df_[columns]], axis=1)

df.to_csv(os.path.join(root, "..", "processed", "16_predictions.csv"), index=False)