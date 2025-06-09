import numpy as np
import pandas as pd
import os.path as osp
from easydict import EasuDict as edict
from typing import Union, Optional, List, Dict

from sklearn.pipeline import FeatureUnion


class ExperimentRunner:
    """Main experiment orchestrator combining all components"""
    def __init__(self, dataset_name: str, bucket_method: str, cls_method: str, 
                encoding_method: str, random_state: int = 22):
        self.dataset_name = dataset_name
        self.bucket_method = bucket_method
        self.cls_method = cls_method
        self.encoding_method = encoding_method
        self.random_state = random_state
        
        # Initialize components
        self.dataset_manager = DatasetManager(dataset_name)
        self.cv_manager = CrossValidationManager(random_state=random_state)
        self.bucketing_manager = BucketingManager(bucket_method, self.dataset_manager, random_state)
        
        # Setup encoding methods
        encoding_dict = {
            "laststate": ["static", "last"],
            "agg": ["static", "agg"],
            "index": ["static", "index"],
            "combined": ["static", "last", "agg"]
        }
        self.methods = encoding_dict[encoding_method]
        
        # Setup encoder arguments
        self.cls_encoder_args = {
            'case_id_col': self.dataset_manager.case_id_col,
            'static_cat_cols': self.dataset_manager.static_cat_cols,
            'static_num_cols': self.dataset_manager.static_num_cols,
            'dynamic_cat_cols': self.dataset_manager.dynamic_cat_cols,
            'dynamic_num_cols': self.dataset_manager.dynamic_num_cols,
            'fillna': True
        }
        
        self.training_pipeline = ModelTrainingPipeline(
            cls_method, self.methods, self.cls_encoder_args, random_state
        )

    def run_cross_validation_experiment(self, args: Dict, n_splits: int = 3) -> Dict:
        """Run complete CV experiment with timing"""
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

