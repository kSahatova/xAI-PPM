import numpy as np
import pandas as pd
import os.path as osp
from easydict import EasuDict as edict
from typing import Union, Optional, List, Dict

from sklearn.pipeline import FeatureUnion
from dataset_manager_optimized import DatasetManager, CVFoldsManager
from preprocessing.encoding import get_encoder


class ExperimentRunner:
    """Main experiment orchestrator combining all components"""
    def __init__(self, dataset_name: str, dataset_manager: DatasetManager,  
                bucket_method: str, encoding_methods: List[str], preprocessing_args: edict,
                cls_method: str, cls_args: Dict, random_state: int = 22):
        self.dataset_name = dataset_name
        self.dataset_manager = dataset_manager
        self.bucket_method = bucket_method
        self.encoding_methods = encoding_methods
        self.preprocessing_args = preprocessing_args

        self.cls_method = cls_method
        self.cls_args = cls_args
        self.random_state = random_state
        
        # Initialize components
        cv_n_splits = preprocessing_args.cv_n_folds if preprocessing_args.cv_n_folds else 3
        self.cv_manager = CVFoldsManager(n_splits=cv_n_splits, 
                                        random_state=random_state)

        enc_args = self.preprocessing_args.encoding_args
        self.encoders =  [(enc_method, get_encoder(enc_method, **enc_args)) for enc_method in self.encoding_methods]              
        
        # self.training_pipeline = ModelTrainingPipeline(
        #     cls_method, self.methods, self.cls_encoder_args, random_state
        # )


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

