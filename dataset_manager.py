import numpy as np
import pandas as pd
from tqdm import tqdm
from easydict import EasyDict as edict

from sklearn.model_selection import StratifiedKFold


class DatasetManager:
    def __init__(self, ds_name: str = "BPIC2017", ds_config: edict = None):
        self.dataset_name = ds_name
        self.config = ds_config

        self.case_id = ds_config.columns.case_id_col
        self.activity = ds_config.columns.activity_col
        self.timestamp = ds_config.columns.timestamp_col
        self.label = ds_config.columns.label_col
        self.pos_label = ds_config.columns.pos_label_col

        self.dynamic_cat_cols = ds_config.columns.dynamic_cat_cols
        self.static_cat_cols = ds_config.columns.static_cat_cols
        self.dynamic_num_cols = ds_config.columns.dynamic_num_cols
        self.static_num_cols = ds_config.columns.static_num_cols

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
        data["case_length"] = data.groupby(self.case_id)[self.activity].transform(len)

        dt_prefixes = (
            data[data["case_length"] >= min_length]
            .groupby(self.case_id)
            .head(min_length)
        )
        dt_prefixes["prefix_nr"] = 1
        dt_prefixes["orig_case_id"] = dt_prefixes[self.case_id]
        total_prefixes_num = range(min_length + gap, max_length + 1, gap)
        for nr_events in tqdm(total_prefixes_num, total=len(total_prefixes_num)):
            tmp = (
                data[data["case_length"] >= nr_events]
                .groupby(self.case_id)
                .head(nr_events)
            )
            tmp["orig_case_id"] = tmp[self.case_id]
            tmp[self.case_id] = tmp[self.case_id].apply(
                lambda x: "%s_%s" % (x, nr_events)
            )
            tmp["prefix_nr"] = nr_events
            dt_prefixes = pd.concat([dt_prefixes, tmp], axis=0)

        dt_prefixes["case_length"] = dt_prefixes["case_length"].apply(
            lambda x: min(max_length, x)
        )

        return dt_prefixes

    def get_indexes(self, data):
        return data.groupby(self.case_id).first().index

    def get_data_by_indexes(self, data, indexes):
        return data[data[self.case_id].isin(indexes)]

    def get_relevant_data_by_indexes(self, data, indexes):
        return data[data[self.case_id].isin(indexes)]

    def get_label(self, data):
        return data.groupby(self.case_id).first()[self.label]

    def get_prefix_lengths(self, data):
        return data.groupby(self.case_id).last()["prefix_nr"]

    def get_case_ids(self, data, nr_events=1):
        case_ids = pd.Series(data.groupby(self.case_id).first().index)
        if nr_events > 1:
            case_ids = case_ids.apply(lambda x: "_".join(x.split("_")[:-1]))
        return case_ids

    def get_label_numeric(self, data):
        y = self.get_label(data)  # one row per case
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
