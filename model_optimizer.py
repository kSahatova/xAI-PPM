import os 
import os.path as osp
import hyperopt
import pandas as pd

# import EncoderFactory
# from DatasetManager import DatasetManager
# import BucketFactory

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import time
import pickle
from collections import defaultdict
from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
import hyperopt
from hyperopt.pyll.base import scope
from hyperopt.pyll.stochastic import sample

from preprocessing.encoding import get_encoder


def prepare_data_splits(dataset_manager, data, train_ratio=0.8, n_splits=3):
    """Prepare data splits for cross-validation"""
    # Determine prefix lengths
    min_prefix_length = dataset_manager.config.min_prefix_length
    max_prefix_length = dataset_manager.config.max_prefix_length
    
    # Split into training and test
    train, _ = dataset_manager.split_data_strict(data, train_ratio, split="temporal")
    
    # Prepare chunks for CV
    test_cv_prefixes = []
    train_cv_class_ratios = []
    
    for train_chunk, test_chunk in dataset_manager.get_stratified_split_generator(train, n_splits=n_splits):
        train_cv_class_ratios.append(dataset_manager.get_class_ratio(train_chunk))
        test_cv_prefixes.append(
            dataset_manager.generate_prefix_data(test_chunk, min_prefix_length, max_prefix_length)
        )
    
    return test_cv_prefixes, train_cv_class_ratios


def setup_hyperparameter_space(cls_method: str = 'rf', bucket_method: str = 'single'):
    """Setup hyperparameter search space based on classifier and bucketing method"""
    space = {}
    
    if cls_method == "rf":
        space = {'max_features': hp.uniform('max_features', 0, 1)}
    elif cls_method == "xgboost":
        space = {
            'learning_rate': hp.uniform("learning_rate", 0, 1),
            'subsample': hp.uniform("subsample", 0.5, 1),
            'max_depth': scope.int(hp.quniform('max_depth', 4, 30, 1)),
            'colsample_bytree': hp.uniform("colsample_bytree", 0.5, 1),
            'min_child_weight': scope.int(hp.quniform('min_child_weight', 1, 6, 1))
        }
    elif cls_method == "logit":
        space = {'C': hp.uniform('C', -15, 15)}
    elif cls_method == "svm":
        space = {
            'C': hp.uniform('C', -15, 15),
            'gamma': hp.uniform('gamma', -15, 15)
        }
    
    if bucket_method == "cluster":
        space['n_clusters'] = scope.int(hp.quniform('n_clusters', 2, 50, 1))
    
    return space


def create_classifier(cls_method: str, args: dict, random_state: int):
    """Create classifier instance based on method and hyperparameters"""
    if cls_method == "rf":
        return RandomForestClassifier(
            n_estimators=500,
            max_features=args['max_features'],
            random_state=random_state
        )
    elif cls_method == "xgboost":
        return xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=500,
            learning_rate=args['learning_rate'],
            subsample=args['subsample'],
            max_depth=int(args['max_depth']),
            colsample_bytree=args['colsample_bytree'],
            min_child_weight=int(args['min_child_weight']),
            seed=random_state
        )
    elif cls_method == "logit":
        return LogisticRegression(
            C=2**args['C'],
            random_state=random_state
        )
    elif cls_method == "svm":
        return SVC(
            C=2**args['C'],
            gamma=2**args['gamma'],
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown classifier method: {cls_method}")


def create_pipeline(cls_method, feature_combiner, cls):
    """Create sklearn pipeline with appropriate preprocessing"""
    if cls_method in ["svm", "logit"]:
        return Pipeline([
            ('encoder', feature_combiner), 
            ('scaler', StandardScaler()), 
            ('cls', cls)
        ])
    else:
        return Pipeline([
            ('encoder', feature_combiner), 
            ('cls', cls)
        ])


def train_and_predict_bucket(dt_train_bucket, dt_test_bucket, train_y, test_y, 
                            encoding_methods, encoder_args, cls_method, cls_args, random_state):
    """Train model and make predictions for a single bucket"""
    
    if len(set(train_y)) < 2:
        # Not enough class diversity
        return [train_y[0]] * len(test_y)
    
    # Create feature encoder
    feature_combiner = FeatureUnion([
        (method, get_encoder(enc_method, **encoder_args)) 
        for enc_method in encoding_methods
    ])
    
    # Create classifier
    classifier = create_classifier(cls_method, cls_args, random_state)
    
    # Create pipeline
    pipeline = create_pipeline(cls_method, feature_combiner, classifier)
    
    # Train model
    pipeline.fit(dt_train_bucket, train_y)
    
    # Make predictions
    if cls_method == "svm":
        preds = pipeline.decision_function(dt_test_bucket)
    else:
        preds_pos_label_idx = np.where(classifier.classes_ == 1)[0][0]
        preds = pipeline.predict_proba(dt_test_bucket)[:, preds_pos_label_idx]
    
    return preds 


def evaluate_single_fold(cv_iter, dt_prefixes, dataset_manager, bucket_method, bucket_encoding,
                        class_ratios, methods, cls_encoder_args, cls_method, args, 
                        method_name, random_state, n_splits):
    """Evaluate model performance for a single CV fold"""
    
    dt_test_prefixes = dt_prefixes[cv_iter]
    dt_train_prefixes = pd.DataFrame()
    
    # Combine training folds
    for cv_train_iter in range(n_splits):
        if cv_train_iter != cv_iter:
            dt_train_prefixes = pd.concat([dt_train_prefixes, dt_prefixes[cv_train_iter]], axis=0)
    
    # Setup bucketing
    bucketer_args = {
        'encoding_method': bucket_encoding,
        'case_id_col': dataset_manager.case_id_col,
        'cat_cols': [dataset_manager.activity_col],
        'num_cols': [],
        'random_state': random_state
    }
    
    if bucket_method == "cluster":
        bucketer_args["n_clusters"] = args["n_clusters"]
    
    bucketer = BucketFactory.get_bucketer(bucket_method, **bucketer_args)
    bucket_assignments_train = bucketer.fit_predict(dt_train_prefixes)
    bucket_assignments_test = bucketer.predict(dt_test_prefixes)
    
    # Process each bucket
    preds_all = []
    test_y_all = []
    scores = defaultdict(int) if "prefix" in method_name else None
    
    for bucket in set(bucket_assignments_test):
        relevant_train_cases_bucket = dataset_manager.get_indexes(dt_train_prefixes)[bucket_assignments_train == bucket]
        relevant_test_cases_bucket = dataset_manager.get_indexes(dt_test_prefixes)[bucket_assignments_test == bucket]
        
        dt_test_bucket = dataset_manager.get_relevant_data_by_indexes(dt_test_prefixes, relevant_test_cases_bucket)
        test_y = dataset_manager.get_label_numeric(dt_test_bucket)
        
        if len(relevant_train_cases_bucket) == 0:
            # No training data for this bucket
            preds = [class_ratios[cv_iter]] * len(relevant_test_cases_bucket)
        else:
            dt_train_bucket = dataset_manager.get_relevant_data_by_indexes(dt_train_prefixes, relevant_train_cases_bucket)
            train_y = dataset_manager.get_label_numeric(dt_train_bucket)
            
            preds = train_and_predict_bucket(
                dt_train_bucket, dt_test_bucket, train_y, test_y,
                methods, cls_encoder_args, cls_method, args, random_state
            )
        
        # Track bucket-level scores if needed
        if "prefix" in method_name and scores is not None:
            auc = 0.5
            if len(set(test_y)) == 2:
                auc = roc_auc_score(test_y, preds)
            scores[bucket] += auc
        
        preds_all.extend(preds)
        test_y_all.extend(test_y)
    
    fold_score = roc_auc_score(test_y_all, preds_all)
    return fold_score, scores

def log_trial_results(trial_nr, dataset_name, cls_method, method_name, args, 
                     score, processing_time, scores_by_bucket, fout_all, n_splits):
    """Log trial results to file"""
    
    if "prefix" in method_name and scores_by_bucket:
        # Log bucket-level scores
        for k, v in args.items():
            for bucket, bucket_score in scores_by_bucket.items():
                fout_all.write(f"{trial_nr};{dataset_name};{cls_method};{method_name};{bucket};{k};{v};{bucket_score / n_splits}\n")
        fout_all.write(f"{trial_nr};{dataset_name};{cls_method};{method_name};0;processing_time;{processing_time};0\n")
    else:
        # Log overall scores
        for k, v in args.items():
            fout_all.write(f"{trial_nr};{dataset_name};{cls_method};{method_name};{k};{v};{score / n_splits}\n")
        fout_all.write(f"{trial_nr};{dataset_name};{cls_method};{method_name};processing_time;{processing_time};0\n")
    
    fout_all.flush()


def create_objective_function(dt_prefixes, class_ratios, dataset_manager, bucket_method, 
                            bucket_encoding, methods, cls_encoder_args, cls_method, 
                            method_name, random_state, n_splits, dataset_name, fout_all):
    """Create objective function for hyperparameter optimization"""
    
    trial_nr = {'count': 0}  # Use dict to make it mutable in closure
    
    def objective(args):
        trial_nr['count'] += 1
        start_time = time.time()
        
        total_score = 0
        all_bucket_scores = defaultdict(int)
        
        # Evaluate across all CV folds
        for cv_iter in range(n_splits):
            fold_score, bucket_scores = evaluate_single_fold(
                cv_iter, dt_prefixes, dataset_manager, bucket_method, bucket_encoding,
                class_ratios, methods, cls_encoder_args, cls_method, args,
                method_name, random_state, n_splits
            )
            
            total_score += fold_score
            
            if bucket_scores:
                for bucket, score in bucket_scores.items():
                    all_bucket_scores[bucket] += score
        
        processing_time = time.time() - start_time
        
        # Log results
        log_trial_results(
            trial_nr['count'], dataset_name, cls_method, method_name, args,
            total_score, processing_time, all_bucket_scores, fout_all, n_splits
        )
        
        return {'loss': -total_score / n_splits, 'status': STATUS_OK}
    
    return objective


def setup_dataset_configurations():
    """Setup dataset reference mappings and encoding dictionaries"""
    dataset_ref_to_datasets = {
        "bpic17": [],
        "insurance": ["insurance_activity", "insurance_followup"],
        "sepsis_cases": ["sepsis_cases_1", "sepsis_cases_2", "sepsis_cases_4"]
    }
    
    encoding_dict = {
        "laststate": ["static", "last"],
        "agg": ["static", "agg"],
        "index": ["static", "index"],
        "combined": ["static", "last", "agg"]
    }
    
    return dataset_ref_to_datasets, encoding_dict


def optimize_hyperparameters(dataset_name, bucket_method, cls_encoding, cls_method, 
                           params_dir, n_iter=100, n_splits=3, random_state=22):
    """Main optimization function"""
    
    # Setup configurations
    dataset_ref_to_datasets, encoding_dict = setup_dataset_configurations()
    datasets = [dataset_name] if dataset_name not in dataset_ref_to_datasets else dataset_ref_to_datasets[dataset_name]
    methods = encoding_dict[cls_encoding]
    method_name = f"{bucket_method}_{cls_encoding}"
    bucket_encoding = "last" if bucket_method == "state" else "agg"
    
    # Create results directory
    os.makedirs(params_dir, exist_ok=True)
    
    for dataset in datasets:
        print(f"Optimizing hyperparameters for dataset: {dataset}")
        
        # Initialize dataset manager and load data
        dataset_manager = DatasetManager(dataset)
        data = dataset_manager.read_dataset()
        
        # Setup encoder arguments
        cls_encoder_args = {
            'case_id_col': dataset_manager.case_id_col,
            'static_cat_cols': dataset_manager.static_cat_cols,
            'static_num_cols': dataset_manager.static_num_cols,
            'dynamic_cat_cols': dataset_manager.dynamic_cat_cols,
            'dynamic_num_cols': dataset_manager.dynamic_num_cols,
            'fillna': True
        }
        
        # Prepare data splits
        dt_prefixes, class_ratios = prepare_data_splits(dataset_manager, data, n_splits=n_splits)
        
        # Setup hyperparameter space
        space = setup_hyperparameter_space(cls_method, bucket_method)
        
        # Setup optimization
        trials = Trials()
        results_file = os.path.join(params_dir, f"param_optim_all_trials_{cls_method}_{dataset}_{method_name}.csv")
        
        with open(results_file, "w") as fout_all:
            # Write header
            if "prefix" in method_name:
                fout_all.write("iter;dataset;cls;method;nr_events;param;value;score\n")
            else:
                fout_all.write("iter;dataset;cls;method;param;value;score\n")
            
            # Create objective function
            objective = create_objective_function(
                dt_prefixes, class_ratios, dataset_manager, bucket_method,
                bucket_encoding, methods, cls_encoder_args, cls_method,
                method_name, random_state, n_splits, dataset, fout_all
            )
            
            # Run optimization
            best = fmin(objective, space, algo=tpe.suggest, max_evals=n_iter, trials=trials)
        
        # Save best parameters
        best_params = hyperopt.space_eval(space, best)
        outfile = os.path.join(params_dir, f"optimal_params_{cls_method}_{dataset}_{method_name}.pickle")
        
        with open(outfile, "wb") as fout:
            pickle.dump(best_params, fout)
        
        print(f"Best parameters for {dataset}: {best_params}")


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 7:
        print("Usage: python script.py <dataset_ref> <params_dir> <n_iter> <bucket_method> <cls_encoding> <cls_method>")
        sys.exit(1)
    
    dataset_ref = sys.argv[1]
    params_dir = sys.argv[2]
    n_iter = int(sys.argv[3])
    bucket_method = sys.argv[4]
    cls_encoding = sys.argv[5]
    cls_method = sys.argv[6]
    
    optimize_hyperparameters(
        dataset_name=dataset_ref,
        bucket_method=bucket_method,
        cls_encoding=cls_encoding,
        cls_method=cls_method,
        params_dir=params_dir,
        n_iter=n_iter
    )
