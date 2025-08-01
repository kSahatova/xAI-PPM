import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Union, Tuple, List
from easydict import EasyDict as edict

from sklearn.model_selection import StratifiedKFold

from xai_ppm.dataset_column_schema import DatasetColumnSchema


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
        train_ids = list(start_timestamps[self.case_id])[
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
        overlapping_cases = train[train[self.timestamp] >= split_ts][
            self.case_id
        ].unique()
        train = train[~train[self.case_id].isin(overlapping_cases)]
        return (train, test)

    def split_val(self, data, val_ratio, split="random", seed=22):
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
        val_ids = list(start_timestamps[self.case_id])[
            -int(val_ratio * len(start_timestamps)) :
        ]
        val = data[data[self.case_id].isin(val_ids)].sort_values(
            self.sorting_cols, ascending=True, kind="mergesort"
        )
        train = data[~data[self.case_id].isin(val_ids)].sort_values(
            self.sorting_cols, ascending=True, kind="mergesort"
        )
        return (train, val)

    # def generate_prefix_data(self, data, min_length, max_length, gap=1):
    #     # generate prefix data (each possible prefix becomes a trace)
    #     data["case_length"] = data.groupby(self.case_id)[self.activity].transform('count')

    #     # Filter out cases that do not meet minimum length requirement
    #     valid_mask = data['case_length'] >= min_length

    #     prefix_lengths = list(range(min_length + gap, max_length + 1, gap))
    #     prefix_dataframes = []

    #     for prefix_length  in tqdm(prefix_lengths, desc="Generating prefixes"):
    #         # Use boolean indexing without copy for filtering
    #         eligible_mask = valid_mask & (data['case_length'] >= prefix_length)
    #         if not eligible_mask.any():
    #             continue

    #         prefix_data = (data[eligible_mask]
    #                   .groupby(self.case_id, group_keys=False)
    #                   .head(prefix_length)
    #                   .copy())  # Only copy the small result

    #         # Add metadata columns
    #         prefix_data["prefix_nr"] = prefix_length 
    #         prefix_data['orig_case_id'] = prefix_data[self.case_id]

    #         prefix_data[self.case_id] = (prefix_data[self.case_id].astype(str) + 
    #                                    '_' + str(prefix_length))

    #         prefix_dataframes.append(prefix_data)
        
    #     # Clean up the added column to restore original state
    #     data.drop('case_length', axis=1, inplace=True)
        
    #     if prefix_dataframes:
    #         return pd.concat(prefix_dataframes, axis=0, ignore_index=True)
        
    #     return pd.DataFrame()

    def generate_prefix_data(self, data, min_length, max_length, gap=1):
        # getting the length of each process instance
        data['case_length'] = data.groupby(self.case_id)[self.activity].transform(len)
        # getting instances which are longer than the minimum length and getting amount of data equivalent to the min length
        dt_prefixes = data[data['case_length'] >= min_length].groupby(self.case_id).head(min_length)
        # this is the first prefixed chunk
        dt_prefixes['prefix_nr'] = 1
        # keeping the original case id with each case
        dt_prefixes['original_case_id'] = dt_prefixes[self.case_id]
        # prefix-based bucketing requires certain nr_events
        # repeat the previous process while increasing the prefixed data bz the gap everytime
        for nr_events in range(min_length + gap, max_length + 1, gap):
            tmp = data[data['case_length'] >= nr_events].groupby(self.case_id).head(nr_events)
            tmp['original_case_id'] = tmp[self.case_id]
            tmp[self.case_id] = tmp[self.case_id].apply(lambda x: '%s_%s' % (x, nr_events))
            tmp['prefix_nr'] = nr_events
            dt_prefixes = pd.concat([dt_prefixes, tmp], axis=0)
        dt_prefixes['case_length'] = dt_prefixes['case_length'].apply(lambda x: min(max_length, x))
        return dt_prefixes

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
    
    def get_label_numeric(self, data):
        # get the label of the first row in a process instance, as they are grouped
        y = data.groupby(self.case_id).first()[self.label]
        return [1 if label == self.pos_label else 0 for label in y]

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


