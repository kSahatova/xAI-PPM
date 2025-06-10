import numpy as np
import pandas as pd
from typing import List, Dict

from abc import ABC, abstractmethod
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

import xgboost as xgb

from dataset_manager_optimized import DatasetManager, CVFoldsManager
from preprocessing.encoding import get_encoder
from preprocessing.bucketing import get_bucketer


class AbstarctExperimentRunner(ABC):
    def __init__(
        self,
        dataset_name: str,
        dataset_manager: DatasetManager,
        bucket_method: str,
        encoding_methods: List[str],
        encoding_args: Dict[str, any],
        cls_method: str,
        cls_args: Dict,
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
        data: pd.DataFrame,
        train_ratio: int = 0.8,
        min_prefix_length: int = 1,
        max_prefix_length: int = 20,
        gap: int = 1,
    ):
        """Preprocess a single dataset with proper train/test split and bucketing"""

        dataset_results = {"train": {}, "test": {}}

        # Splitting the data into train and test set
        train, test = self.dm.split_data_strict(
            data, train_ratio=train_ratio, split="temporal"
        )
        print(
            "Shape of the train set: ",
            train.shape,
            "\nShape of the test set: ",
            test.shape,
        )

        # Generate train and test prefixes
        # Calculate max prefix length
        max_prefix_length = min(
            max_prefix_length, self.dm.get_pos_case_length_quantile(data, 0.90)
        )
        print(
            f"\nGenerating train and test prefixes with the max length {max_prefix_length}"
        )

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
        train_bucket_labels = self.bucketer.fit_predict(df_train_prefixes)
        test_bucket_labels = self.bucketer.predict(df_test_prefixes)

        # Process each bucket (merge all possible buckets)
        unique_buckets = set(train_bucket_labels) | set(test_bucket_labels)

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
                # 'raw_data': df_train_bucket
            }

            dataset_results["test"][bucket_key] = {
                "features": pd.DataFrame(encoded_test_bucket, columns=feature_names),
                "labels": test_y,
                # 'raw_data': df_test_bucket
            }

            print(f"    Finished processing bucket: {bucket_id}")

        return dataset_results

    @abstractmethod
    def run_experiment(self):
        raise NotImplementedError(
            "Method self.run_experiment() has not been implemented for this class"
        )


class MLExperimentRunner(AbstarctExperimentRunner):
    """Main experiment orchestrator combining all components"""

    def __init__(
        self,
        dataset_name: str,
        dataset_manager: DatasetManager,
        bucket_method: str,
        encoding_methods: List[str],
        encoding_args: Dict[str, any],
        cls_method: str,
        cls_args: Dict,
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

    def create_classifier(self):
        """Factory method for creating classifiers"""
        args = self.cls_args

        if self.cls_method == "rf":
            return RandomForestClassifier(
                n_estimators=args["n_estimators"],
                max_features=args["max_features"],
                random_state=self.random_state,
                verbose=1
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
                verbose=1
            )
        elif self.cls_method == "logit":
            return LogisticRegression(C=2 ** args["C"], random_state=self.random_state, verbose=1)
        elif self.cls_method == "svm":
            return SVC(
                C=2 ** args["C"],
                gamma=2 ** args["gamma"],
                random_state=self.random_state,
                verbose=1
            )

    def run_experiment(
        self, encoded_train: Dict, encoded_test: Dict, phase: str = "offline"
    ):
        # TODO: add online prediction

        trained_classifiers = {}
        result = {}

        # TODO: Optimize!
        for train_bucket_id, test_bucket_id in zip(encoded_train, encoded_test):
            encoded_train_X = encoded_train[train_bucket_id]["features"]
            train_y = encoded_train[train_bucket_id]["labels"]

            encoded_test_X = encoded_test[test_bucket_id]["features"]
            test_y = encoded_test[test_bucket_id]["labels"]

            classifier = self.create_classifier()
            print(f"***Fitting the created {classifier.__class__.__name__} classifier***")
            classifier.fit(encoded_train_X, train_y)

            print("\n***Estimating the fitted classifier***")
            # predictions
            preds_pos_label_idx = np.where(classifier.classes_ == 1)[0][0]
            preds = classifier.predict_proba(encoded_test_X)[:, preds_pos_label_idx]

            # TODO: Change ROC-AUC calculation for the bucketing method with multiple buckets
            # auc total
            bucket_auc = roc_auc_score(test_y, preds)
            print(f"ROC-AUC of the classifier on the {test_bucket_id}: ", bucket_auc)
            result[test_bucket_id] = dict(auc=bucket_auc, model=classifier)
            # result[test_bucket_id]["auc"] = bucket_auc
            # result[test_bucket_id]["model"] = classifier

        return result


"""def run_cross_validation_experiment(self, args: Dict, n_splits: int = 3) -> Dict:
        Run complete CV experiment with timing
        # Load and prepare data
        data = self.dataset_manager.read_dataset()
        train, _ = self.dataset_manager.split_data_strict(data, 0.8, split="temporal")
        
        # Generate prefix data for CV
        prefix_generator = PrefixGenerator(self.dataset_manager, random_state=self.random_state)
        
        # Create CV splits
        dt_prefixes_raw, class_ratios = self.cv_manager.create_cv_splits(self.dataset_manager, train)
        
        # Generate prefixes for each split
        dt_prefixes = []
        for split_data in dt_prefixes_raw:
            prefixes = prefix_generator.generate_prefixes(split_data, self.dataset_name)
            dt_prefixes.append(prefixes)
        
        total_score = 0
        scores_per_bucket = defaultdict(int) if "prefix" in f"{self.bucket_method}_{self.cls_encoding}" else None
        
        # Run CV iterations
        for cv_iter in range(n_splits):
            dt_train_prefixes, dt_test_prefixes = self.cv_manager.get_train_test_split(dt_prefixes, cv_iter)
            
            # Create and fit bucketer
            bucketer_kwargs = {}
            if self.bucket_method == "cluster":
                bucketer_kwargs["n_clusters"] = args["n_clusters"]
            
            bucketer = self.bucketing_manager.create_bucketer(**bucketer_kwargs)
            bucket_assignments_train = bucketer.fit_predict(dt_train_prefixes)
            bucket_assignments_test = bucketer.predict(dt_test_prefixes)
            
            # Process each bucket
            preds_all, test_y_all = self._process_buckets(
                dt_train_prefixes, dt_test_prefixes,
                bucket_assignments_train, bucket_assignments_test,
                class_ratios[cv_iter], args, scores_per_bucket
            )
            
            # Calculate score for this CV iteration
            from sklearn.metrics import roc_auc_score
            total_score += roc_auc_score(test_y_all, preds_all)
        
        return {
            'score': total_score / n_splits,
            'bucket_scores': scores_per_bucket
        }
"""
