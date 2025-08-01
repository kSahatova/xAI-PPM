import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple

from abc import ABC, abstractmethod
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

import hyperopt
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll.base import scope

from functools import partial

import xgboost as xgb

from xai_ppm.dataset_manager_opt import DatasetManager
from preprocessing.encoding import get_encoder
from preprocessing.bucketing import get_bucketer


class AbstractExperimentRunner(ABC):
    def __init__(
        self,
        dataset_name: str,
        dataset_manager: DatasetManager,
        bucket_method: str = 'single',
        encoding_methods: List[str] = ['static', 'agg'],
        encoding_args: Dict[str, Any] = {},
        cls_method: str = 'xgboost',
        cls_args: Dict = {},
        random_state: int = 22,
    ):
        self.dataset_name = dataset_name
        self.dm = dataset_manager

        self.bucket_method = bucket_method
        self.encoding_methods = encoding_methods
        self.encoding_args = encoding_args

        self.cls_method = cls_method
        self.cls_args = cls_args
        self.random_state = random_state

        # Initialize components
        self.bucketer = get_bucketer(self.bucket_method, case_id_col=self.dm.case_id)

    def preprocess_event_log(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        min_prefix_length: int = 1,
        max_prefix_length: int = 20,
        gap: int = 1,
    ):
        """Preprocess a single dataset with the specified bucketing and encoding"""

        df_train_prefixes = self.dm.generate_prefix_data(
            train, min_prefix_length, max_prefix_length, gap=gap
        )
        df_test_prefixes = self.dm.generate_prefix_data(
            test, min_prefix_length, max_prefix_length, gap=gap
        )
        print("Length of the train prefixes: ", df_train_prefixes.shape[0])
        print("Length of the test prefixes: ", df_test_prefixes.shape[0])

        # Create buckets
        print(f'\nCreating buckets with the "{self.bucket_method}" bucket method')
        train_bucket_labels = self.bucketer.fit_predict(df_train_prefixes) # type: ignore
        test_bucket_labels = self.bucketer.predict(df_test_prefixes) # type: ignore

        # Process each bucket (merge all possible buckets)
        unique_buckets = set(train_bucket_labels) | set(test_bucket_labels)
        dataset_results = {"train": {}, "test": {}}

        for bucket_id in unique_buckets:
            print(f"    Processing bucket: {bucket_id}")

            # Get bucket indices
            train_bucket_mask = train_bucket_labels == bucket_id
            test_bucket_mask = test_bucket_labels == bucket_id

            if not np.any(train_bucket_mask) or not np.any(test_bucket_mask):
                print(f"    Skipping bucket {bucket_id} - insufficient data")
                continue

            # Extract bucket data
            train_bucket_indices = self.dm.get_indexes(df_train_prefixes)[
                train_bucket_mask
            ]
            test_bucket_indices = self.dm.get_indexes(df_test_prefixes)[
                test_bucket_mask
            ]

            df_train_bucket = self.dm.get_data_by_indexes(
                df_train_prefixes, train_bucket_indices
            )
            df_test_bucket = self.dm.get_data_by_indexes(
                df_test_prefixes, test_bucket_indices
            )

            # Get labels
            _, train_y = self.dm.get_labels(df_train_bucket)
            _, test_y = self.dm.get_labels(df_test_bucket)

            print(
                "       Shape of the train bucket and its labels after labels extraction: ",
                df_train_bucket.shape,
                train_y.shape,
            )
            print(
                "       Shape of the test bucket and its labels after labels extraction: ",
                df_test_bucket.shape,
                test_y.shape,
            )

            self.encoders = [
                (method, get_encoder(method, **self.encoding_args))
                for method in self.encoding_methods
            ]
            self.feature_combiner = FeatureUnion(self.encoders)

            # Fit on training data and transform both sets
            encoded_train_bucket = self.feature_combiner.fit_transform(df_train_bucket)
            encoded_test_bucket = self.feature_combiner.transform(df_test_bucket)

            print(
                "\n       Shape of the train bucket after encoding: ",
                encoded_train_bucket.shape,
            )
            print(
                "       Shape of the test bucket after encoding: ",
                encoded_test_bucket.shape,
            )

            # Get feature names
            feature_names = []
            for name, transformer in self.feature_combiner.transformer_list:
                for fname in transformer.get_feature_names():
                    feature_names.append(f"{name}_{fname}")

            # Store results
            bucket_key = f"bucket_{bucket_id}"

            dataset_results["train"][bucket_key] = {
                "features": pd.DataFrame(encoded_train_bucket, columns=feature_names),
                "labels": train_y,
            }

            dataset_results["test"][bucket_key] = {
                "features": pd.DataFrame(encoded_test_bucket, columns=feature_names),
                "labels": test_y,
            }

            print(f"    Finished processing bucket: {bucket_id}")

        return dataset_results

    @abstractmethod
    def run_experiment(self, *args, **kwargs):
        raise NotImplementedError(
            "Method self.run_experiment() has not been implemented for this class"
        )


class MLExperimentRunner(AbstractExperimentRunner):
    """Main experiment orchestrator combining all components"""

    def __init__(
        self,
        dataset_name: str,
        dataset_manager: DatasetManager,
        bucket_method: str = 'single',
        encoding_methods: List[str] = ['static', 'agg'],
        encoding_args: Dict[str, Any] = {},
        cls_method: str = 'xgboost',
        cls_args: Dict = {},
        random_state: int = 22,
    ):
        super().__init__(dataset_name, dataset_manager, bucket_method, 
                         encoding_methods, encoding_args, cls_method,
                         cls_args, random_state)

    def create_classifier(self):
        """Factory method for creating classifiers"""
        args = self.cls_args

        if self.cls_method == "rf":
            return RandomForestClassifier(
                n_estimators=args["n_estimators"],
                max_features=args["max_features"],
                random_state=self.random_state,
                verbose=True
            )
        elif self.cls_method == "xgboost":
            return xgb.XGBClassifier(
                objective="binary:logistic",
                n_estimators=args["n_estimators"],
                learning_rate=args["learning_rate"],
                subsample=args["subsample"],
                max_depth=int(args["max_depth"]),
                colsample_bytree=args["colsample_bytree"],
                min_child_weight=int(args["min_child_weight"]),
                seed=self.random_state,
                verbose=True
            )
        elif self.cls_method == "logit":
            return LogisticRegression(C=2 ** args["C"], random_state=self.random_state, verbose=1)
        elif self.cls_method == "svm":
            return SVC(
                C=2 ** args["C"],
                gamma=2 ** args["gamma"],
                random_state=self.random_state,
                verbose=True
            )
        else:
            raise ValueError(f"Unknown classifier method: {self.cls_method}")

    def run_experiment(self, train_data, test_data):

        # trained_classifiers = {}
        result = {}

        # TODO: Optimize!
        for train_bucket_id, test_bucket_id in zip(train_data, test_data):
            train_data_X = train_data[train_bucket_id]["features"]
            train_y = train_data[train_bucket_id]["labels"]

            test_data_X = test_data[test_bucket_id]["features"]
            test_y = test_data[test_bucket_id]["labels"]

            classifier = self.create_classifier()
            print(f"***Fitting the created {classifier.__class__.__name__} classifier***")
            classifier.fit(train_data_X, train_y)

            print("\n***Estimating the fitted classifier***")
            # predictions
            preds_pos_label_idx = np.where(classifier.classes_ == 1)[0][0]
            preds = classifier.predict_proba(test_data_X)[:, preds_pos_label_idx]

            # TODO: Change ROC-AUC calculation for the bucketing method with multiple buckets
            # auc total
            bucket_auc = roc_auc_score(test_y, preds)
            print(f"ROC-AUC of the classifier on the {test_bucket_id}: ", bucket_auc)
            result[test_bucket_id] = dict(auc=bucket_auc, model=classifier)
            # result[test_bucket_id]["auc"] = bucket_auc
            # result[test_bucket_id]["model"] = classifier

        return result
    

class CrossValidationExperimentRunner(AbstractExperimentRunner):
    """
    Execution logic of this class includes (i) the extraction of hold out folds from the training data;
    (ii) instantiation of a new model, (iii) training of the model and estimation on the k-th fold.
    """
    def __init__(self, dataset_name: str,
                dataset_manager: DatasetManager,
                bucket_method: str,
                encoding_methods: List[str],
                encoding_args: Dict[str, Any],
                cls_method: str,
                cls_args: Dict,
                random_state: int = 22,
                k_folds: int = 3):
        super().__init__(dataset_name, dataset_manager, bucket_method, 
                        encoding_methods, encoding_args, cls_method,
                        cls_args, random_state)
        self.k_folds = k_folds

    def get_cv_folds(self, data: pd.DataFrame,
                        min_prefix_length: int = 1,
                        max_prefix_length: int = 20,
                        gap: int = 1,
                        ):
        df_prefixes = []
        class_ratios = []
        for train_chunk, test_chunk in self.dm.get_stratified_split_generator(data, n_splits=self.k_folds):
            class_ratios.append(self.dm.get_class_ratio(train_chunk))            
            df_prefixes.append(
                self.dm.generate_prefix_data(test_chunk, min_prefix_length, max_prefix_length, gap=gap))
        del data

        for cv_iter in range(self.k_folds):
            test_fold = df_prefixes[cv_iter]
            train_fold = pd.concat([df_prefixes[i] for i in range(self.k_folds) if i != cv_iter], axis=0)
            yield train_fold, test_fold

    def preprocess_fold(self, train_prefixes, test_prefixes):
        # TODO: Check the shape of features and labels
        # Get labels
        _, y_train = self.dm.get_labels(train_prefixes)
        _, y_test = self.dm.get_labels(test_prefixes)

        print(
            "       Shape of the train prefixes and labels after labels extraction: ",
            train_prefixes.shape, y_train.shape,
        )
        print(
            "       Shape of the test prefixes and labels after labels extraction: ",
            test_prefixes.shape, y_test.shape,)

        self.feature_combiner = FeatureUnion([
            (method, get_encoder(method, **self.encoding_args))
            for method in self.encoding_methods
        ])

        # Fit on training data and transform both sets
        encoded_train_X = self.feature_combiner.fit_transform(train_prefixes)
        encoded_test_X = self.feature_combiner.transform(test_prefixes)

        print(
            "\n       Shape of the train bucket after encoding: ", encoded_train_X.shape)
        print(
            "       Shape of the test bucket after encoding: ", encoded_test_X.shape)

        # Get feature names
        feature_names = []
        for name, transformer in self.feature_combiner.transformer_list:
            if hasattr(transformer, "get_feature_names"):
                for fname in transformer.get_feature_names():
                    feature_names.append(f"{name}_{fname}")
        
        X_train = pd.DataFrame(encoded_train_X, columns=feature_names) 
        X_test = pd.DataFrame(encoded_test_X, columns=feature_names) 
        return X_train, y_train, X_test, y_test
        
    def create_classifier(self, args):
        """Factory method for creating classifiers"""

        if self.cls_method == "rf":
            return RandomForestClassifier(
                n_estimators=500,
                max_features=args["max_features"],
                random_state=self.random_state,
                verbose=True
            )
        elif self.cls_method == "xgboost":
            return xgb.XGBClassifier(
                objective="binary:logistic",
                n_estimators=args["n_estimators"],
                learning_rate=args["learning_rate"],
                subsample=args["subsample"],
                max_depth=int(args["max_depth"]),
                colsample_bytree=args["colsample_bytree"],
                min_child_weight=int(args["min_child_weight"]),
                seed=self.random_state,
                verbose=True
            )
        elif self.cls_method == "logit":
            return LogisticRegression(C=2 ** args["C"], random_state=self.random_state, verbose=1)
        elif self.cls_method == "svm":
            return SVC(
                C=2 ** args["C"],
                gamma=2 ** args["gamma"],
                random_state=self.random_state,
                verbose=True
            )
        else:
            raise ValueError(f"Unknown classifier method: {self.cls_method}")
        
    def create_objective(self, data, 
                    min_prefix_length,
                    max_prefix_length, 
                    gap, args):
        score = 0
        prefixes_generator = self.get_cv_folds(data, min_prefix_length, max_prefix_length, gap)
        for i, (train_prefixes, test_prefixes) in enumerate(prefixes_generator):
            print(f"Started processing the fold {i}")
            X_train, y_train, X_test, y_test = self.preprocess_fold(train_prefixes, test_prefixes)
            classifier = self.create_classifier(args)
            classifier.fit(X_train, y_train)
            preds_pos_label_idx = np.where(classifier.classes_ == 1)[0][0]
            preds = classifier.predict_proba(X_test)[:,preds_pos_label_idx]

            score += roc_auc_score(y_test, preds)
            
        return {'loss': -score / self.k_folds, 'status': STATUS_OK}
    

    def run_experiment(self, train_data: pd.DataFrame, min_prefix_length, max_prefix_length, gap):
        objective_fn = partial(self.create_objective, train_data, 
                               min_prefix_length, max_prefix_length, gap)

        space = {}
        if self.cls_method == "xgboost":
            space = {'learning_rate': hp.uniform("learning_rate", 0, 1),
                'subsample': hp.uniform("subsample", 0.5, 1),
                'max_depth': scope.int(hp.quniform('max_depth', 4, 30, 1)),
                'colsample_bytree': hp.uniform("colsample_bytree", 0.5, 1),
                'min_child_weight': scope.int(hp.quniform('min_child_weight', 1, 6, 1))}
      
        elif self.cls_method == "rf":
            space = {'max_features': hp.uniform('max_features', 0, 1)}

        elif self.cls_method == "lr":
            space = {'C': hp.uniform('C', -15, 15)}

        trials = Trials()
        best = fmin(objective_fn, space, algo=tpe.suggest, max_evals=4, trials=trials)
        print("The best parameters observed: \n", hyperopt.space_eval(space, best))
        return best, trials
