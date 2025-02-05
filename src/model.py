import os
import sys
import json
import numpy as np
import joblib
import shutil
import collections
from flaml import AutoML
from flaml.default import RandomForestClassifier as ZeroShotRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_curve, auc, roc_auc_score

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)

from tools import ghostml

N_FOLDS = 5
SPLITTING_ROUNDS = 3

FLAML_ESTIMATOR_LIST = ["rf"]

FLAML_COLD_MINIMUM_TIME_BUDGET_SECONDS = 250
FLAML_COLD_MAXIMUM_TIME_BUDGET_SECONDS = 500
FLAML_WARM_MINIMUM_TIME_BUDGET_SECONDS = 50
FLAML_WARM_MAXIMUM_TIME_BUDGET_SECONDS = 250

FLAML_COLD_MINIMUM_ITERATIONS = 50
FLAML_COLD_MAXIMUM_ITERATIONS = 250
FLAML_WARM_MINIMUM_ITERATIONS = 25
FLAML_WARM_MAXIMUM_ITERATIONS = 100


class NaiveBayesClassificationModel(object):

    def __init__(self, model_dir):
        print("Initializing model in {0}".format(model_dir))
        self.model_dir = model_dir
        if os.path.exists(os.path.join(model_dir, "model.pkl")):
            self.model = joblib.load(os.path.join(model_dir, "model.pkl"))
        else:
            self.model = None
        if os.path.exists(os.path.join(model_dir, "results.json")):
            with open(os.path.join(model_dir, "results.json"), "r") as f:
                self.results = json.load(f)
        else:
            self.results = None

    def _fit(self, X, y):
        model = MultinomialNB()
        model.fit(X, y)
        return model

    def fit_cv(self, X, y):
        y = np.array(y, dtype=int)
        print("Fitting model with CV")
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True)
        aurocs = []
        optimal_thresholds = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model = self._fit(X_train, y_train)
            model.predict_proba(X_test)
            fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
            auroc = auc(fpr, tpr)
            aurocs.append(auroc)
            youden_index = tpr - fpr
            optimal_idx = np.argmax(youden_index)
            optimal_threshold = thresholds[optimal_idx]
            optimal_thresholds.append(optimal_threshold)
        self.results = {"auroc": float(np.mean(aurocs)), "num_pos": int(sum(y)), "num_tot": int(len(y)), "opt_cut": float(np.mean(optimal_thresholds))}
        print(self.results)
        self.model = self._fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def save(self):
        joblib.dump(self.model, os.path.join(self.model_dir, "model.pkl"))
        with open(self.model_dir + "/results.json", "w") as f:
            json.dump(self.results, f, indent=4)


class RandomForestClassificationModel(object):

    def __init__(self, model_dir):
        print("Initializing model in {0}".format(model_dir))
        self.model_dir = model_dir
        if os.path.exists(os.path.join(model_dir, "model.pkl")):
            self.model = joblib.load(os.path.join(model_dir, "model.pkl"))
        else:
            self.model = None
        if os.path.exists(os.path.join(model_dir, "results.json")):
            with open(os.path.join(model_dir, "results.json"), "r") as f:
                self.results = json.load(f)
        else:
            self.results = None

    def _fit(self, X, y):
        zero_shot = ZeroShotRandomForestClassifier()
        hyperparams = zero_shot.suggest_hyperparams(X, y)[0]
        print(hyperparams)
        model = RandomForestClassifier(**hyperparams)
        model.fit(X, y)
        return model

    def fit_cv(self, X, y):
        y = np.array(y, dtype=int)
        print("Fitting model with CV")
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True)
        aurocs = []
        optimal_thresholds = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model = self._fit(X_train, y_train)
            model.predict_proba(X_test)
            fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
            auroc = auc(fpr, tpr)
            aurocs.append(auroc)
            youden_index = tpr - fpr
            optimal_idx = np.argmax(youden_index)
            optimal_threshold = thresholds[optimal_idx]
            optimal_thresholds.append(optimal_threshold)
        self.results = {"auroc": float(np.mean(aurocs)), "num_pos": int(sum(y)), "num_tot": int(len(y)), "opt_cut": float(np.mean(optimal_thresholds))}
        print(self.results)
        self.model = self._fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def save(self):
        joblib.dump(self.model, os.path.join(self.model_dir, "model.pkl"))
        with open(self.model_dir + "/results.json", "w") as f:
            json.dump(self.results, f, indent=4)


# FLAML AutoML modeling

class Splitter(object):
    def __init__(self, X, y):
        self.splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True)
        self.X = X
        self.y = y

    def split(self):
        for _ in range(SPLITTING_ROUNDS):
            for tr_idxs, te_idxs in self.splitter.split(self.X, self.y):
                yield tr_idxs, te_idxs


class GhostLight(object):
    def __init__(self):
        pass

    def _get_class_balance(self, y):
        return np.sum(y) / len(y)

    def get_threshold(self, y, y_hat):
        max_prop = np.max([self._get_class_balance(y), 0.6])
        thresholds = np.round(
            np.arange(0.05, max_prop, 0.05), 2
        )
        threshold = ghostml.optimize_threshold_from_predictions(
            y, y_hat, thresholds, ThOpt_metrics="Kappa"
        )
        return threshold


class FlamlClassificationModel(object):

    def __init__(self, model_dir):
        print("Initializing model in {0}".format(model_dir))
        self.model_dir = model_dir
        if os.path.exists(os.path.join(model_dir, "model.pkl")):
            self.model = joblib.load(os.path.join(model_dir, "model.pkl"))
        else:
            self.model = None
        if os.path.exists(os.path.join(model_dir, "results.json")):
            with open(os.path.join(model_dir, "results.json"), "r") as f:
                self.results = json.load(f)
        else:
            self.results = None

    def _clean_log(self, automl_settings):
        cwd = os.getcwd()
        log_file = os.path.join(cwd, automl_settings["log_file_name"])
        if os.path.exists(log_file):
            os.remove(log_file)
        catboost_info = os.path.join(cwd, "catboost_info")
        if os.path.exists(catboost_info):
            shutil.rmtree(catboost_info)
        catboost_folders = []
        for f in os.listdir(cwd):
            if f.startswith("catboost"):
                catboost_folders += [f]
        for f in catboost_folders:
            shutil.rmtree(os.path.join(cwd, f))

    @staticmethod
    def _get_starting_point(model):
        best_models = model.best_config_per_estimator
        best_estimator = model.best_estimator
        starting_point = {best_estimator: best_models[best_estimator]}
        return starting_point

    def get_automl_settings(self, time_budget, num_samples):
        automl_settings = {
            "time_budget": int(
                np.clip(
                    time_budget,
                    FLAML_COLD_MINIMUM_TIME_BUDGET_SECONDS,
                    FLAML_COLD_MAXIMUM_TIME_BUDGET_SECONDS,
                )
            ),
            "metric": "roc_auc",
            "task": "classification",
            "log_file_name": "flaml.log",
            "log_training_metric": True,
            "verbose": 3,
            "early_stop": True,
            "max_iter": int(
                np.clip(
                    num_samples / 3,
                    FLAML_COLD_MINIMUM_ITERATIONS,
                    FLAML_COLD_MAXIMUM_ITERATIONS,
                )
            ),
            "estimator_list": FLAML_ESTIMATOR_LIST
        }
        return automl_settings
    
    @staticmethod
    def _get_starting_point(model):
        best_models = model.best_config_per_estimator
        best_estimator = model.best_estimator
        starting_point = {best_estimator: best_models[best_estimator]}
        return starting_point

    def _fit_cold(self, X, y):
        automl_settings = self.get_automl_settings(
            time_budget=(FLAML_COLD_MINIMUM_TIME_BUDGET_SECONDS + FLAML_COLD_MAXIMUM_TIME_BUDGET_SECONDS) / 2,
            num_samples=len(y)
        )
        model = AutoML()
        _automl_settings = dict((k, v) for k, v in automl_settings.items())
        _automl_settings["eval_method"] = "auto"
        _automl_settings["split_type"] = None
        _automl_settings["groups"] = None
        model.fit(X_train=X, y_train=y, **_automl_settings)
        automl_settings = automl_settings
        model = self._fit_warm(X, y, model, automl_settings)
        return model, automl_settings

    def _fit_warm(self, X, y, cold_model, automl_settings):
        automl_settings = automl_settings.copy()
        automl_settings["time_budget"] = int(
            np.clip(
                automl_settings["time_budget"] * 0.5,
                FLAML_WARM_MINIMUM_TIME_BUDGET_SECONDS,
                FLAML_WARM_MAXIMUM_TIME_BUDGET_SECONDS,
            )
        )
        y = np.array(y)
        best_estimator = cold_model.best_estimator
        starting_point = self._get_starting_point(cold_model)
        tag = "warm"
        automl_settings["log_file_name"] = "{0}_flaml.log".format(tag)
        automl_settings["eval_method"] = "auto"
        automl_settings["split_type"] = None
        automl_settings["groups"] = None
        automl_settings["estimator_list"] = [best_estimator]
        automl_settings["max_iter"] = int(
            np.clip(
                int(automl_settings["max_iter"] * 0.5) + 1,
                FLAML_WARM_MINIMUM_ITERATIONS,
                FLAML_WARM_MAXIMUM_ITERATIONS,
            )
        )
        model = AutoML()
        model.fit(
            X_train=X, y_train=y, starting_points=starting_point, **automl_settings
        )
        self._clean_log(automl_settings)
        return model
    
    def _fit_predict_cv(self, X, y, cold_model, automl_settings):
        automl_settings = automl_settings.copy()
        automl_settings["time_budget"] = int(
            np.clip(
                automl_settings["time_budget"] * 0.5,
                FLAML_WARM_MINIMUM_TIME_BUDGET_SECONDS,
                FLAML_WARM_MAXIMUM_TIME_BUDGET_SECONDS,
            )
        )
        y = np.array(y)
        best_estimator = cold_model.best_estimator
        starting_point = self._get_starting_point(cold_model)
        splitter = Splitter(X, y)
        k = 0
        results = collections.defaultdict(list)
        for tr_idxs, te_idxs in splitter.split():
            tag = "oos_{0}".format(k)
            automl_settings["log_file_name"] = "{0}_automl.log".format(tag)
            X_tr = X[tr_idxs]
            X_te = X[te_idxs]
            y_tr = y[tr_idxs]
            automl_settings["eval_method"] = "auto"
            automl_settings["split_type"] = None
            automl_settings["groups"] = None
            automl_settings["estimator_list"] = [best_estimator]
            automl_settings["max_iter"] = int(
                np.clip(
                    int(automl_settings["max_iter"] * 0.25) + 1,
                    FLAML_WARM_MINIMUM_ITERATIONS,
                    FLAML_WARM_MAXIMUM_ITERATIONS,
                )
            )
            model = AutoML()
            model.fit(
                X_train=X_tr,
                y_train=y_tr,
                starting_points=starting_point,
                **automl_settings
            )
            model = model.model.estimator
            print("Fitting a calibrated model")
            try:
                model = CalibratedClassifierCV(estimator=model, n_jobs=-1)
            except:
                model = model.model.estimator
                print("Could not calibrate model. Continuing...")
                print("This is the model", model)
            model.fit(X_tr, y_tr)
            y_te_hat = model.predict_proba(X_te)[:, 1]
            for i, idx in enumerate(te_idxs):
                results[idx] += [y_te_hat[i]]
            self._clean_log(automl_settings)
            k += 1
        y_hat = []
        for i in range(len(y)):
            y_hat += [np.mean(results[i])]
        return {"y_hat": np.array(y_hat), "y": y}
    
    def fit_cv(
        self,
        X,
        y
    ):
        model, automl_settings = self._fit_cold(X, y)
        results = self._fit_predict_cv(X, y, model, automl_settings)
        threshold = GhostLight().get_threshold(
            results["y"], results["y_hat"]
        )
        try:
            cal_model = CalibratedClassifierCV(model.model.estimator)
            cal_model.fit(X, y)
        except:
            print("Could not calibrate model. Continuing...")
            cal_model = model.model.estimator
        auroc = roc_auc_score(results["y"], results["y_hat"])
        self.results = {"auroc": float(auroc), "num_pos": int(sum(y)), "num_tot": int(len(y)), "opt_cut": threshold}
        print(self.results)
        self.model = cal_model

    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def save(self):
        joblib.dump(self.model, os.path.join(self.model_dir, "model.pkl"))
        with open(self.model_dir + "/results.json", "w") as f:
            json.dump(self.results, f, indent=4)