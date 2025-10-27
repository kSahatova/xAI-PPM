import math
from re import sub
import string
import numpy as np

import torch
from itertools import compress
from jellyfish import damerau_levenshtein_distance
# from torch.utils.data import DataLoader
# from cohortshapley.similarity import similar_in_distance


def similar_in_distance(data, subject, vertex, ratio=0.1):
    # subject = subject.reshape(subject.shape[-1])
    xmax = np.amax(data, 0)
    xmin = np.amin(data, 0)
    xdist = (xmax - xmin) * ratio
    cmin = subject - xdist
    cmax = subject + xdist
    dataT = data.T
    ccond = np.ones(data.shape[0])
    for i in range(1, vertex.shape[-1] - 1):
        if vertex[i] == 0:
            continue
        cond = np.logical_and(
            np.greater_equal(dataT[i], cmin[i]), np.less_equal(dataT[i], cmax[i])
        )
        ccond = np.logical_and(ccond, cond)
    return ccond


class CohortExplainer:
    def __init__(
        self,
        model,
        dataset,
        y,
        threshold_similarity=0.8,
    ):
        self.model = model
        self.dataset = dataset
        self.y = y
        self.grand_mean = np.mean(self.y, axis=0)
        self.threshold_similarity = threshold_similarity
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _get_subject_data(self, subject_id):
        subject_fcat, subject_fnum, _, _ = self.dataset[subject_id]
        subject_cat = subject_fcat.squeeze().numpy()
        return subject_cat, subject_fnum

    def _extract_cohort_indices(
        self,
        subject_cat_features,
        subject_id,
        total_events_num=25,
        special_tokens_num=3,
    ):
        characters = string.ascii_uppercase
        start_ind = special_tokens_num
        end_ind = start_ind + total_events_num
        event_to_symbol_mapping = {
            i: characters[i - special_tokens_num] for i in range(start_ind, end_ind)
        }
        cohort_indices = []

        subject_activity_seq = "".join(
            [event_to_symbol_mapping[activity] for activity in subject_cat_features]
        )

        rest_traces_cat = [
            self.dataset[i][0].squeeze().numpy()
            for i in range(len(self.dataset))
            if i != subject_id
        ]
        max_trace_length = 0
        distances = []
        for trace_cat in rest_traces_cat:
            trace_length = len(trace_cat)
            if trace_length > max_trace_length:
                max_trace_length = trace_length

            # Transform the event log to a list of character sequences representing the traces
            activity_sequence = "".join(
                [event_to_symbol_mapping[activity] for activity in trace_cat]
            )

            # Calculate the DL distance between each pair of traces
            distance = damerau_levenshtein_distance(
                subject_activity_seq, activity_sequence
            )
            distances.append(distance)
        similarities = np.asarray([1 - (distance / max_trace_length) for distance in distances])
        cohort_indices = np.where(similarities > self.threshold_similarity)[0]
        return cohort_indices

    def cohort_value(
        self, cohort_indices, subject_data, feature_set, categorical_indices
    ):
        if (feature_set == categorical_indices).all():
            return np.asarray(list(compress(self.y, cohort_indices))).mean()
        else:
            cohort_data = [self.dataset[ind] for ind in cohort_indices]
            cohort_num_data = np.asarray(
                [trace[1].mean(axis=0) for trace in cohort_data]
            )
            subject_num_data = subject_data[1].mean(axis=0)

            updated_cohort_indices = similar_in_distance(
                cohort_num_data, subject_num_data, feature_set
            )

            avg_cohort_y = np.asarray(
                list(compress(self.y, updated_cohort_indices))
            ).mean()
            return avg_cohort_y

    def compute_cohort_shapley(self, subject_id: int, features_num: int):
        # Create a cohort with similar to the subject traces
        subject_data = self._get_subject_data(subject_id)
        subject_fcat, subject_fnum = subject_data

        categorical_indices = np.zeros(features_num)
        categorical_indices[0] = 1

        cohort_indices = self._extract_cohort_indices(subject_fcat, subject_id)

        shapley_values = np.zeros(features_num)
        shapley_values2 = np.zeros(features_num)
        phi_set = np.zeros(features_num)

        u_k = {}
        all_trace_indices = [i for i in range(len(self.dataset))]
        u_k[tuple(phi_set)] = self.cohort_value(
            all_trace_indices, subject_data, phi_set, categorical_indices
        )
        for k in range(features_num):
            coef = (
                math.factorial(k)
                * math.factorial(features_num - k - 1)
                / math.factorial(features_num - 1)
                / features_num
            )
            u_k_base = u_k
            u_k = {}
            for j in range(features_num):
                gain = 0
                gain2 = 0
                for fset in u_k_base.keys():
                    if fset[j] != 1:  # or u_k_base[tuple(fset)] != 1:
                        fset = np.asarray(fset)
                        fset_j = fset.copy()
                        fset_j[j] = 1
                        if tuple(fset_j) not in u_k.keys():
                            u_k[tuple(fset_j)] = self.cohort_value(
                                cohort_indices,
                                subject_data,
                                fset_j,
                                categorical_indices,
                            )

                        gain += u_k[tuple(fset_j)] - u_k_base[tuple(fset)]
                        gain2 += (u_k[tuple(fset_j)] - self.grand_mean) ** 2 - (
                            u_k_base[tuple(fset)] - self.grand_mean
                        ) ** 2
                shapley_values[j] += gain * coef
                shapley_values2[j] += gain2 * coef

        # TODO: Check if the division by number of traces in the cohort is needed
        return shapley_values, shapley_values2
