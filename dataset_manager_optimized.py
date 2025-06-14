import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Union, Tuple, List
from easydict import EasyDict as edict

from sklearn.model_selection import StratifiedKFold

from dataset_column_schema import DatasetColumnSchema


class DatasetManager:
    def __init__(self, ds_name: str, ds_column_schema: DatasetColumnSchema):
        self.dataset_name = ds_name

        self.case_id = ds_column_schema.case_id_col
        self.activity = ds_column_schema.activity_col
        self.timestamp = ds_column_schema.timestamp_col
        self.label = ds_column_schema.label_col
        self.pos_label = ds_column_schema.pos_label_col

        self.dynamic_cat_cols = ds_column_schema.dynamic_cat_cols
        self.static_cat_cols = ds_column_schema.static_cat_cols
        self.dynamic_num_cols = ds_column_schema.dynamic_num_cols
        self.static_num_cols = ds_column_schema.static_num_cols

        self.sorting_cols = [self.timestamp, self.activity]

    def read_dataset(self, file_path):
        """Reads the dataset from a given file path."""

        # set 'float' type for numeric columns and 'object' fir the rest
        dtypes = {col: "float" for col in self.dynamic_num_cols + self.static_num_cols} 
        for col in (self.dynamic_cat_cols + self.static_cat_cols
                    + [self.case_id, self.label, self.timestamp]):
            dtypes[col] = "object"

        data = pd.read_csv(file_path, sep=";", dtype=dtypes)
        data[self.timestamp] = pd.to_datetime(data[self.timestamp])
        return data
    
    def get_pos_case_length_quantile(self, data, quantile=0.90):
        return int(
            np.ceil(
                data[data[self.label] == self.pos_label]
                .groupby(self.case_id)
                .size()
                .quantile(quantile)
            )
        )

    def split_data(self, data, train_ratio, split="temporal", seed=22):
        # split into train and test using temporal split

        grouped = data.groupby(self.case_id)
        start_timestamps = grouped[self.timestamp].min().reset_index()
        if split == "temporal":
            start_timestamps = start_timestamps.sort_values(
                self.timestamp, ascending=True, kind="mergesort"
            )
        elif split == "random":
            np.random.seed(seed)
            start_timestamps = start_timestamps.reindex(
                np.random.permutation(start_timestamps.index)
            )
        train_ids = list(start_timestamps[self.case_id_col])[
            : int(train_ratio * len(start_timestamps))
        ]
        train = data[data[self.case_id].isin(train_ids)].sort_values(
            self.timestamp, ascending=True, kind="mergesort"
        )
        test = data[~data[self.case_id].isin(train_ids)].sort_values(
            self.timestamp, ascending=True, kind="mergesort"
        )

        return (train, test)

    def split_data_strict(self, data, train_ratio, split="temporal"):
        """Splits the data into train and test using temporal split and discard events that overlap the periods."""

        data = data.sort_values(self.sorting_cols, ascending=True, kind="mergesort")
        grouped = data.groupby(self.case_id)
        start_timestamps = grouped[self.timestamp].min().reset_index()
        start_timestamps = start_timestamps.sort_values(
            self.timestamp, ascending=True, kind="mergesort"
        )
        train_ids = list(start_timestamps[self.case_id])[
            : int(train_ratio * len(start_timestamps))
        ]
        train = data[data[self.case_id].isin(train_ids)].sort_values(
            self.sorting_cols, ascending=True, kind="mergesort"
        )
        test = data[~data[self.case_id].isin(train_ids)].sort_values(
            self.sorting_cols, ascending=True, kind="mergesort"
        )
        split_ts = test[self.timestamp].min()
        train = train[train[self.timestamp] < split_ts]
        
        return (train, test)

    def split_data_discard(self, data, train_ratio, split="temporal"):
        # split into train and test using temporal split and discard events that overlap the periods
        data = data.sort_values(self.sorting_cols, ascending=True, kind="mergesort")
        grouped = data.groupby(self.case_id_col)
        start_timestamps = grouped[self.timestamp_col].min().reset_index()
        start_timestamps = start_timestamps.sort_values(
            self.timestamp_col, ascending=True, kind="mergesort"
        )
        train_ids = list(start_timestamps[self.case_id_col])[
            : int(train_ratio * len(start_timestamps))
        ]
        train = data[data[self.case_id_col].isin(train_ids)].sort_values(
            self.sorting_cols, ascending=True, kind="mergesort"
        )
        test = data[~data[self.case_id_col].isin(train_ids)].sort_values(
            self.sorting_cols, ascending=True, kind="mergesort"
        )
        split_ts = test[self.timestamp_col].min()
        overlapping_cases = train[train[self.timestamp_col] >= split_ts][
            self.case_id_col
        ].unique()
        train = train[~train[self.case_id_col].isin(overlapping_cases)]
        return (train, test)

    def split_val(self, data, val_ratio, split="random", seed=22):
        # split into train and test using temporal split
        grouped = data.groupby(self.case_id_col)
        start_timestamps = grouped[self.timestamp_col].min().reset_index()
        if split == "temporal":
            start_timestamps = start_timestamps.sort_values(
                self.timestamp_col, ascending=True, kind="mergesort"
            )
        elif split == "random":
            np.random.seed(seed)
            start_timestamps = start_timestamps.reindex(
                np.random.permutation(start_timestamps.index)
            )
        val_ids = list(start_timestamps[self.case_id_col])[
            -int(val_ratio * len(start_timestamps)) :
        ]
        val = data[data[self.case_id_col].isin(val_ids)].sort_values(
            self.sorting_cols, ascending=True, kind="mergesort"
        )
        train = data[~data[self.case_id_col].isin(val_ids)].sort_values(
            self.sorting_cols, ascending=True, kind="mergesort"
        )
        return (train, val)

    def generate_prefix_data(self, data, min_length, max_length, gap=1):
        # generate prefix data (each possible prefix becomes a trace)
        data["case_length"] = data.groupby(self.case_id)[self.activity].transform('count')


        # Filter out cases that do not meet minimum length requirement
        valid_mask = data['case_length'] >= min_length

        prefix_lengths = list(range(min_length + gap, max_length + 1, gap))
        prefix_dataframes = []

        for prefix_length  in tqdm(prefix_lengths, desc="Generating prefixes"):
            # Use boolean indexing without copy for filtering
            eligible_mask = valid_mask & (data['case_length'] >= prefix_length)
            if not eligible_mask.any():
                continue

            prefix_data = (data[eligible_mask]
                      .groupby(self.case_id, group_keys=False)
                      .head(prefix_length)
                      .copy())  # Only copy the small result

            # Add metadata columns
            prefix_data["prefix_nr"] = prefix_length 
            prefix_data['orig_case_id'] = prefix_data[self.case_id]

            prefix_data[self.case_id] = (prefix_data[self.case_id].astype(str) + 
                                       '_' + str(prefix_length))

            prefix_dataframes.append(prefix_data)
        
        # Clean up the added column to restore original state
        data.drop('case_length', axis=1, inplace=True)
        
        if prefix_dataframes:
            return pd.concat(prefix_dataframes, axis=0, ignore_index=True)
        
        return pd.DataFrame()

    def get_indexes(self, data: pd.DataFrame) -> pd.Index:
        """
        Optimized version that avoids unnecessary groupby operations.
        Returns case IDs directly instead of full grouped data.
        """
        # Use drop_duplicates instead of groupby().first() for better performance
        unique_cases = data.drop_duplicates(subset=[self.case_id], keep='first')
        return unique_cases[self.case_id].index

    def get_data_by_indexes(self, data: pd.DataFrame, indices: Union[pd.Index, np.ndarray]) -> pd.DataFrame:
        """
        Optimized filtering using vectorized operations.
        Uses query() for better performance on large datasets.
        """

        return data.iloc[indices]

    def get_labels(self, data):
        # labels = data.groupby(self.case_id).first()[self.label]
        labels = data.drop_duplicates(subset=[self.case_id], keep='first')[self.label]
        # numeric_labels = np.asarray([1 if label == self.pos_label else 0 for label in labels])
        numeric_labels = (labels == self.pos_label).astype(int).values
        return labels, numeric_labels

    def get_class_ratio(self, data):
        class_freqs = data[self.label].value_counts()
        return class_freqs[self.pos_label] / class_freqs.sum()

    def get_stratified_split_generator(
        self, data, n_splits=5, shuffle=True, random_state=22
    ):
        grouped_firsts = data.groupby(self.case_id, as_index=False).first()
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )

        for train_index, test_index in skf.split(
            grouped_firsts, grouped_firsts[self.label]
        ):
            current_train_names = grouped_firsts[self.case_id][train_index]
            train_chunk = data[
                data[self.case_id].isin(current_train_names)
            ].sort_values(self.timestamp, ascending=True, kind="mergesort")
            test_chunk = data[
                ~data[self.case_id].isin(current_train_names)
            ].sort_values(self.timestamp, ascending=True, kind="mergesort")
            yield (train_chunk, test_chunk)

    def get_idx_split_generator(
        self, dt_for_splitting, n_splits=5, shuffle=True, random_state=22
    ):
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )

        for train_index, test_index in skf.split(
            dt_for_splitting, dt_for_splitting[self.label]
        ):
            current_train_names = dt_for_splitting[self.case_id][train_index]
            current_test_names = dt_for_splitting[self.case_id][test_index]
            yield (current_train_names, current_test_names)


class CVFoldsManager:
    """Manages cross-validation splits with consistent indexing"""
    
    def __init__(self, n_splits: int = 3, random_state: int = 22):
        self.n_splits = n_splits
        self.random_state = random_state
        
    def create_cv_splits(self, dataset_manager: DatasetManager, train_data: pd.DataFrame) -> Tuple[List[pd.DataFrame], List[float]]:
        """Create stratified CV splits and return prefixes with class ratios"""
        dt_prefixes = []
        class_ratios = []
        
        for train_chunk, test_chunk in dataset_manager.get_stratified_split_generator(
            train_data, n_splits=self.n_splits
        ):
            class_ratios.append(dataset_manager.get_class_ratio(train_chunk))
            dt_prefixes.append(test_chunk)
            
        return dt_prefixes, class_ratios
    
    def get_train_test_folds(self, dt_prefixes: List[pd.DataFrame], cv_iter: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get train/test split for specific CV iteration"""
        dt_test_prefixes = dt_prefixes[cv_iter]
        dt_train_prefixes = pd.DataFrame()
        
        for cv_train_iter in range(self.n_splits):
            if cv_train_iter != cv_iter:
                dt_train_prefixes = pd.concat([dt_train_prefixes, dt_prefixes[cv_train_iter]], axis=0)
                
        return dt_train_prefixes, dt_test_prefixes