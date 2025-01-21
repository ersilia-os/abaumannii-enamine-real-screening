import os


class ClassificationModel(object):

    def __init__(self, model_path):
        self.model_path = os.path.abspath(model_path)

    def fit(self, file_path, column_name=None):
        pass

    def predict(self, smiles_list):
        pass

    def save(self):
        pass