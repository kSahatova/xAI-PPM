
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
import time
import pickle
import os
from collections import defaultdict


class ModelTrainingPipeline:
    """Optimized model training pipeline"""
    
    def __init__(self, cls_method: str, cls_args: Dict, feature_combiner: FeatureUnion, random_state: int = 22):
        self.cls_method = cls_method
        self.classifier_args = cls_args
        self.feature_combiner = feature_combiner
        self.random_state = random_state
        self._pipeline_cache = {}
        
    def create_classifier(self):
        """Factory method for creating classifiers"""
        args = self.classifier_args

        if self.cls_method == "rf":
            return RandomForestClassifier(
                n_estimators=args['n_estimators'],
                max_features=args['max_features'],
                random_state=self.random_state
            )
        elif self.cls_method == "xgboost":
            return xgb.XGBClassifier(
                objective='binary:logistic',
                n_estimators=args['n_estimators'],
                learning_rate=args['learning_rate'],
                subsample=args['subsample'],
                max_depth=int(args['max_depth']),
                colsample_bytree=args['colsample_bytree'],
                min_child_weight=int(args['min_child_weight']),
                seed=self.random_state
            )
        elif self.cls_method == "logit":
            return LogisticRegression(
                C=2**args['C'],
                random_state=self.random_state
            )
        elif self.cls_method == "svm":
            return SVC(
                C=2**args['C'],
                gamma=2**args['gamma'],
                random_state=self.random_state
            )
        
    def create_pipeline(self) -> Pipeline:
        """Create sklearn pipeline with feature encoding"""
        
        cls = self.create_classifier(self.classifier_args)
        
        if self.cls_method in ["svm", "logit"]:
            return Pipeline([
                ('encoder', self.feature_combiner),
                ('scaler', StandardScaler()),
                ('cls', cls)
            ])
        else:
            return Pipeline([
                ('encoder', self.feature_combiner),
                ('cls', cls)
            ])
    
    def train_and_predict(self, dt_train_bucket, dt_test_bucket, train_y):
        """Train model and make predictions with timing"""
        start_time = time.time()
        
        if len(set(train_y)) < 2:
            # Handle edge case: only one class in training data
            preds = [train_y[0]] * len(dt_test_bucket)
            train_time = 0
        else:
            pipeline = self.create_pipeline()
            pipeline.fit(dt_train_bucket, train_y)
            train_time = time.time() - start_time
            
            # Make predictions
            if self.cls_method == "svm":
                preds = pipeline.decision_function(dt_test_bucket)
            else:
                preds_pos_label_idx = np.where(pipeline.named_steps['cls'].classes_ == 1)[0][0]
                preds = pipeline.predict_proba(dt_test_bucket)[:, preds_pos_label_idx]
        
        return preds, train_time


        