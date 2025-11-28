"""
This file is based on the original SHAP implementation:
https://github.com/slundberg/shap/blob/master/shap/explainers/_kernel.py
"""

import numpy as np
import pandas as pd
import scipy as sp
import logging
import copy
import itertools
from typing import List
import warnings 

import sklearn
from packaging import version
from sklearn.linear_model import Lasso, LassoLarsIC, lars_path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from shap.utils._legacy import convert_to_link, IdentityLink
from shap.utils._legacy import convert_to_instance, convert_to_model
from shap.utils._exceptions import DimensionError
from shap.explainers._kernel import KernelExplainer
from scipy.special import binom

from .utils import match_seq_model_to_data, convert_to_data, match_instance_to_data


log = logging.getLogger("shap")
    

class SeqShapKernel(KernelExplainer):
    """

    Parameters
    ----------
    model: function
        User supplied function that takes a 3D array (# samples x # sequence length x # features)
        and computes the output of the model for those samples. The output can be a vector
        (# samples) or a matrix (# samples x # model outputs).
        In order to use TimeSHAP in an optimized way, this model can also return the explained
        model's hidden state.

    background : numpy.array or pd.DataFrame
        The background event/sequence to use for integrating out features. To determine the impact
        of a feature, that feature is set to "missing" and the change in the model output
        is observed. Since most models aren't designed to handle arbitrary missing data at test
        time, we simulate "missing" by replacing the feature with the values it takes in the
        background dataset. So if the background dataset is a simple sample of all zeros, then
        we would approximate a feature being missing by setting it to zero.
        In TimeSHAP you can use an average event or average sequence.
        When using average events, consider using `timeshap.calc_avg_event` method to obtain it.
        When using average sequence, considering using `timeshap.calc_avg_sequence` method to obtain it.
        Note that when using the average sequence, all sequences of the dataset need to be the same length.

    rs: int
        Random seed for timeshap algorithm

    mode: str
        This method indicates what kind of explanations should be calculated.
        Possible values: ["pruning", "event", "feature", "cell"]
            - "pruning" - used for pruning algorithm
            - "event" - used for event explanations
            - "feature" - used for feature explanations
            - "cell" -used for cell explanations

    varying: Tuple
        index of varying indexes on cell level
        If mode == "cell": varying needs to be of len 2, the first the idx of
            events to preturb, and the second the idx of features

    link : "identity" or "logit"
        A generalized linear model link to connect the feature importance values to the model
        output. Since the feature importance values, phi, sum up to the model output, it often makes
        sense to connect them to the output with a link function where link(output) = sum(phi).
        If the model output is a probability then the LogitLink link function makes the feature
        importance values have log-odds units.
    """

    def __init__(
        self, model, background, rs, mode, varying=None, link=IdentityLink(), **kwargs
    ):
        self.background = background
        self.random_seed = rs
        self.mode = mode
        self.data = None
        self.varyingInds = None
        self.varying = varying
        self.returns_hs = None
        self.background_hs = None
        self.instance_hs = None
        # convert incoming inputs to standardized iml objects
        self.link = convert_to_link(link)  
        self.model = convert_to_model(model)
        self.keep_index = kwargs.get("keep_index", False)
        self.keep_index_ordered = kwargs.get("keep_index_ordered", False)
        if self.mode == "segment":
            self.segment_ids: List[List[int]] = kwargs["segment_ids"] 

    def set_variables_up(self, X: np.ndarray):
        """Sets variables up for explanations

        Parameters
        ----------
        X: Union[pd.DataFrame, np.ndarray]
            Instance being explained
        """
        if self.mode not in ["feature", "segment"]:
            raise ValueError("Provided mode is not supported currently. Only `feature` and `segment` are available.")
        
        bc_sequence = self.background  # Remove unnecessary deep copy for now
        ndim = bc_sequence.ndim  # Cache ndim to avoid repeated attribute access

        if ndim == 2:
            seq_len = bc_sequence.shape[0]

            if seq_len > 1 and seq_len != X.shape[1]:
                raise ValueError(
                    "When using background events, you can only pass one average event. "
                    "When using background sequence, your background must be the same sequence length of the explained sequence"
                )

            if seq_len == 1:
                # Average event - tile and expand in one step
                bc_sequence = np.tile(bc_sequence, (X.shape[1], 1))[np.newaxis, :]
            else:
                bc_sequence = bc_sequence[np.newaxis, :]

        elif ndim == 3:
            seq_len = bc_sequence.shape[1]

            if seq_len > 1 and seq_len != X.shape[1]:
                raise ValueError(
                    "When using background events, you can only pass one average event. "
                    "When using background sequence, your background must be the same sequence length of the explained sequence"
                )

            if seq_len == 1:
                bc_sequence = np.tile(bc_sequence, (1, X.shape[1], 1))
            # If seq_len == X.shape[1], no modification needed
        else:
            raise ValueError(
                f"Background must be 2D or 3D array, got {ndim}D. "
                "Please open a ticket on github"
            )

        kwargs = {}
        if self.mode == "segment":
            self.segment_boundaries = {i: (seg[0], seg[-1]) for i, seg in enumerate(self.segment_ids)}
            kwargs["segments_ind"] = list(self.segment_boundaries.keys()) 
        # Baseline sequence should be transformed to the DenseData class 
        try:
            self.data = convert_to_data(bc_sequence, self.mode, **kwargs)
        except Exception as e:
            print("Shap explainer only supports the SubseqDenseData input currently.", e)  
        
        # calculates model output over the given background dataset (model_null [self.data.shape[0] x n_outputs])
        model_null, returns_hs = match_seq_model_to_data(self.model, self.data)
        # TODO: revise how the hidden state can be used for SHAP values calculation
        self.returns_hs = returns_hs 

        if self.returns_hs:
            _, example_hs = self.model.f(X[:, -1:, :])
            if not isinstance(example_hs, tuple):
                example_hs = tuple(example_hs)
            self.instance_hs = tuple(
                        np.zeros_like(example_hs[i])
                        for i, x in enumerate(example_hs)
                    )
            self.background_hs = tuple(
                np.zeros_like(example_hs[i])
                for i, x in enumerate(example_hs)
            )    

        # warn users about large background data sets
        if len(self.data.weights) > 100:
            log.warning(
                "Using "
                + str(len(self.data.weights))
                + " background data samples could cause "
                + "slower run times. Consider using shap.sample(data, K) or shap.kmeans(data, K) to "
                + "summarize the background as K samples."
            )

        # initialize our parameters
        # N - number of instances, S - sequnce length, P - number of parameters
        self.N = self.data.data.shape[0] 
        self.S = self.data.data.shape[1]
        self.P = self.data.data.shape[2]
        # applies the function to each element of an array and returns an array of results
        self.linkfv = np.vectorize(self.link.f) 
        self.nsamplesAdded = 0
        self.nsamplesRun = 0

        # find the expected model output (9weighted average) over the background (null) dataset E_x[f(x)] 
        self.fnull = np.sum((model_null.T * self.data.weights).T, 0)
        self.expected_value = self.linkfv(self.fnull)

        # see if we have a vector output
        self.vector_out = True
        if len(self.fnull.shape) == 0:
            self.vector_out = False
            self.fnull = np.array([self.fnull])
            self.D = 1
            self.expected_value = float(self.expected_value)
        else:
            self.D = self.fnull.shape[0]

    def shap_values(self, X, **kwargs):
        """Estimate the SHAP values for a set of samples.

        Parameters
        ----------
        X : numpy.array
            A 3D matrix (#samples x #events x #features) on which to explain the model's output.

        nsamples : "auto" or int
            Number of times to re-evaluate the model when explaining each prediction. More samples
            lead to lower variance estimates of the SHAP values. The "auto" setting uses
            `nsamples = 2 * X.shape[1] + 2048`.

        l1_reg : "num_features(int)", "auto" (default for now, but deprecated), "aic", "bic", or float
            The l1 regularization to use for feature selection (the estimation procedure is based on
            a debiased lasso). The auto option currently uses "aic" when less that 20% of the possible sample
            space is enumerated, otherwise it uses no regularization. THE BEHAVIOR OF "auto" WILL CHANGE
            in a future version to be based on num_features instead of AIC.
            The "aic" and "bic" options use the AIC and BIC rules for regularization.
            Using "num_features(int)" selects a fix number of top features. Passing a float directly sets the
            "alpha" parameter of the sklearn.linear_model.Lasso model used for feature selection.

        Returns
        -------
        For models with a single output this returns a matrix of SHAP values
        (# samples x # features). Each row sums to the difference between the model output for that
        sample and the expected value of the model output (which is stored as expected_value
        attribute of the explainer). For models with vector outputs this returns a list
        of such matrices, one for each output.
        """
        assert isinstance(X, np.ndarray), "Instance must be 3D numpy array"
        
        self.set_variables_up(X)

        if sp.sparse.issparse(X) and not sp.sparse.isspmatrix_lil(X):
            X = X.tolil()

        # single instance
        if X.shape[0] == 1:
            explanation = self.explain(X, **kwargs)

            out = np.zeros(explanation.shape[0])
            if isinstance(explanation.shape, tuple) and len(explanation.shape) == 2:
                assert explanation.shape[1] == 1
                out[:] = explanation[:, 0]
            else:
                out[:] = explanation
            return out
        
        else:
            emsg = "Instance must have 1 or 2 dimensions!"
            raise DimensionError(emsg)

    def explain(self, incoming_instance, **kwargs):
        # convert input to be explained to a standardized iml object
        instance = convert_to_instance(incoming_instance)
        # safely assign group_display_values only if available on self.data
        groups = getattr(self.data, "groups", None) 
        instance.group_display_values = groups
        # not very useful check , better be optimized / removed
        match_instance_to_data(instance, self.data)

        # Find the feature groups we will test. If a feature does not change from its
        # current value then we know it doesn't impact the model
        
        if self.mode == "feature":
            self.varyingInds = self.varying_groups(instance.x)
        elif self.mode == "segment":
            self.varyingInds = self.varying_segments(instance.x) 
        

        if groups is None:
            self.varyingFeatureGroups = np.array([i for i in self.varyingInds])
            self.M = self.varyingFeatureGroups.shape[0]
        else:
            if self.mode in ["feature", "segment"]:
                self.varyingFeatureGroups = [
                    groups[i] for i in self.varyingInds
                ]
                self.M = len(self.varyingFeatureGroups)  
                
            # convert to numpy array as it is much faster if not jagged array (all groups of same length)
            if isinstance(self.varyingFeatureGroups, list) and all(
                len(groups[i]) == len(groups[0])
                for i in range(len(self.varyingFeatureGroups))
            ):
                self.varyingFeatureGroups = np.array(self.varyingFeatureGroups)
                # further performance optimization in case each group has a single value
                if self.varyingFeatureGroups.shape[1] == 1:
                    self.varyingFeatureGroups = self.varyingFeatureGroups.flatten()

        if self.returns_hs:
            # Removed the input variability to receive pd.series and DataFrame
            model_out, _ = self.model.f(instance.x)
        else:
            model_out = self.model.f(instance.x)

        self.fx = model_out[0]
        if not self.vector_out:
            self.fx = np.array([self.fx])

        # if no features vary then no feature has an effect
        if self.M == 0:
            phi = np.zeros((self.data.groups_size, self.D))

        # if only one feature varies then it has all the effect
        elif self.M == 1:
            phi = np.zeros((self.data.groups_size, self.D))
            diff = self.link.f(self.fx) - self.link.f(self.fnull)
            for d in range(self.D):
                phi[self.varyingInds[0], d] = diff[d]

        # if more than one feature varies then we have to do real work
        else:
            self.l1_reg = kwargs.get("l1_reg", "auto")

            # pick a reasonable number of samples if the user didn't specify how many they wanted
            self.nsamples = kwargs.get("nsamples", "auto")
            if self.nsamples == "auto":
                self.nsamples = 2 * self.M + 2**11

            # if we have enough samples to enumerate all subsets then ignore the unneeded samples
            self.max_samples = 2**30
            if self.M <= 30:
                self.max_samples = 2**self.M - 2
                if self.nsamples > self.max_samples:
                    self.nsamples = self.max_samples

            # reserve space for some of our computations
            self.allocate()

            # weight the different subset sizes (antithetic sampling)
            num_subset_sizes = int(np.ceil((self.M - 1) / 2.0))
            num_paired_subset_sizes = int(np.floor((self.M - 1) / 2.0))
            weight_vector = np.array(
                [
                    (self.M - 1.0) / (i * (self.M - i))
                    for i in range(1, num_subset_sizes + 1)
                ]
            )
            weight_vector[:num_paired_subset_sizes] *= 2
            weight_vector /= np.sum(weight_vector)
            log.debug("weight_vector = {0}".format(weight_vector))
            log.debug("num_subset_sizes = {0}".format(num_subset_sizes))
            log.debug("num_paired_subset_sizes = {0}".format(num_paired_subset_sizes))
            log.debug("M = {0}".format(self.M))

            # fill out all the subset sizes we can completely enumerate
            # given nsamples*remaining_weight_vector[subset_size]
            num_full_subsets = 0
            num_samples_left = self.nsamples
            group_inds = np.arange(self.M, dtype="int64")
            mask = np.zeros(self.M)
            remaining_weight_vector = copy.copy(weight_vector)
            for subset_size in range(1, num_subset_sizes + 1):
                # determine how many subsets (and their complements) are of the current size
                nsubsets = binom(self.M, subset_size)
                if subset_size <= num_paired_subset_sizes:
                    nsubsets *= 2
                log.debug("subset_size = {0}".format(subset_size))
                log.debug("nsubsets = {0}".format(nsubsets))
                log.debug(
                    "self.nsamples*weight_vector[subset_size-1] = {0}".format(
                        num_samples_left * remaining_weight_vector[subset_size - 1]
                    )
                )
                log.debug(
                    "self.nsamples*weight_vector[subset_size-1]/nsubsets = {0}".format(
                        num_samples_left
                        * remaining_weight_vector[subset_size - 1]
                        / nsubsets
                    )
                )

                # see if we have enough samples to enumerate all subsets of this size
                if (
                    num_samples_left
                    * remaining_weight_vector[subset_size - 1]
                    / nsubsets
                    >= 1.0 - 1e-8
                ):
                    num_full_subsets += 1
                    num_samples_left -= nsubsets

                    # rescale what's left of the remaining weight vector to sum to 1
                    if remaining_weight_vector[subset_size - 1] < 1.0:
                        remaining_weight_vector /= (
                            1 - remaining_weight_vector[subset_size - 1]
                        )

                    # add all the samples of the current subset size
                    w = weight_vector[subset_size - 1] / binom(self.M, subset_size)
                    if subset_size <= num_paired_subset_sizes:
                        w /= 2.0
                    for inds in itertools.combinations(group_inds, subset_size):
                        mask[:] = 0.0
                        mask[np.array(inds, dtype="int64")] = 1.0
                        self.add_sample(instance.x, mask, w)
                        if subset_size <= num_paired_subset_sizes:
                            mask[:] = np.abs(mask - 1)
                            self.add_sample(instance.x, mask, w)
                else:
                    break
            log.info("num_full_subsets = {0}".format(num_full_subsets))
            # add random samples from what is left of the subset space
            nfixed_samples = self.nsamplesAdded
            samples_left = self.nsamples - self.nsamplesAdded
            log.debug("samples_left = {0}".format(samples_left))
            np.random.seed(self.random_seed)
            if num_full_subsets != num_subset_sizes:
                remaining_weight_vector = copy.copy(weight_vector)
                remaining_weight_vector[:num_paired_subset_sizes] /= (
                    2  # because we draw two samples each below
                )
                remaining_weight_vector = remaining_weight_vector[num_full_subsets:]
                remaining_weight_vector /= np.sum(remaining_weight_vector)
                log.info(
                    "remaining_weight_vector = {0}".format(remaining_weight_vector)
                )
                log.info(
                    "num_paired_subset_sizes = {0}".format(num_paired_subset_sizes)
                )
                ind_set = np.random.choice(
                    len(remaining_weight_vector),
                    4 * samples_left,
                    p=remaining_weight_vector,
                )
                ind_set_pos = 0
                used_masks = {}
                while samples_left > 0 and ind_set_pos < len(ind_set):
                    mask.fill(0.0)
                    ind = ind_set[
                        ind_set_pos
                    ]  # we call np.random.choice once to save time and then just read it here
                    ind_set_pos += 1
                    subset_size = ind + num_full_subsets + 1
                    mask[np.random.permutation(self.M)[:subset_size]] = 1.0

                    # only add the sample if we have not seen it before, otherwise just
                    # increment a previous sample's weight
                    mask_tuple = tuple(mask)
                    new_sample = False
                    if mask_tuple not in used_masks:
                        new_sample = True
                        used_masks[mask_tuple] = self.nsamplesAdded
                        samples_left -= 1
                        self.add_sample(instance.x, mask, 1.0)
                    else:
                        self.kernelWeights[used_masks[mask_tuple]] += 1.0

                    # add the compliment sample
                    if samples_left > 0 and subset_size <= num_paired_subset_sizes:
                        mask[:] = np.abs(mask - 1)

                        # only add the sample if we have not seen it before, otherwise just
                        # increment a previous sample's weight
                        if new_sample:
                            samples_left -= 1
                            self.add_sample(instance.x, mask, 1.0)
                        else:
                            # we know the compliment sample is the next one after the original sample, so + 1
                            self.kernelWeights[used_masks[mask_tuple] + 1] += 1.0

                # normalize the kernel weights for the random samples to equal the weight left after
                # the fixed enumerated samples have been already counted
                weight_left = np.sum(weight_vector[num_full_subsets:])
                log.info("weight_left = {0}".format(weight_left))
                self.kernelWeights[nfixed_samples:] *= (
                    weight_left / self.kernelWeights[nfixed_samples:].sum()
                )

            # execute the model on the synthetic samples we have created
            self.run()

            # solve then expand the feature importance (Shapley value) vector to contain the non-varying features
            phi = np.zeros((self.data.groups_size, self.D))
            for d in range(self.D):
                vphi, _ = self.solve(self.nsamples / self.max_samples, d)
                phi[self.varyingInds, d] = vphi

        if not self.vector_out:
            phi = np.squeeze(phi, axis=1)

        return phi

    @staticmethod
    def not_equal(i, j):
        if isinstance(i, str) or isinstance(j, str):
            return 0 if i == j else 1
        return 0 if np.isclose(i, j, equal_nan=True) else 1


    def varying_segments(self, x):
        """
        Find which subsequences vary between instance x and background.
        
        Parameters
        ----------
        x : array-like, shape (1, sequence_length, n_features)
            Instance to explain
        subsequence_groups : list of tuples
            Each tuple is (start_idx, end_idx) defining a subsequence            
        
        Returns
        -------
        varying_subseq_indices : array
            Indices of subsequences that vary from background
        """
        num_subsequences = len(self.segment_boundaries)
        varying = np.zeros(num_subsequences, dtype=bool)
        
        for subseq_idx, (start, end) in self.segment_boundaries.items():
            # Extract subsequence from instance
            x_subseq = x[0, start:end, :]  # Shape: (subseq_length, n_features)
            
            # Extract corresponding subsequence from background
            bg_subseq = self.data.data[:, start:end, :]  # Shape: (n_bg_samples, subseq_length, n_features)
            
            # Check if this subsequence varies
            varies = self._check_subsequence_variation(x_subseq, bg_subseq)
            varying[subseq_idx] = varies
        
        return np.where(varying)[0]


    def _check_subsequence_variation(self, x_subseq, bg_subseq):
        """
        Check if a subsequence varies from background.
        
        Returns True if the subsequence differs meaningfully from ALL background samples.
        """
        n_bg_samples = bg_subseq.shape[0]
        
        # Strategy 1: Check if ANY timestep in subsequence varies
        for t in range(x_subseq.shape[0]):
            x_timestep = x_subseq[t, :]
            bg_timestep = bg_subseq[:, t, :]
            
            # Check if this timestep differs from all background samples
            for bg_sample_idx in range(n_bg_samples):
                diff = np.abs(x_timestep - bg_timestep[bg_sample_idx, :])
                if np.any(diff > 1e-7):
                    return True  # Found a difference
        
        return False  # No differences found in entire subsequence

    def varying_groups(self, x):
        """Find indices where values vary between background and instance x."""
        group_size = getattr(self.data, "groups_size")
        groups = getattr(self.data, "groups")
        bc_data = getattr(self.data, "data")
        if not sp.sparse.issparse(x):
            varying = np.zeros(group_size)
            for i in range(group_size):
                inds = groups[i]
                x_group = x[0, inds]
                if sp.sparse.issparse(x_group):
                    if all(j not in x.nonzero()[1] for j in inds):
                        varying[i] = False
                        continue
                    x_group = x_group.todense()
                varying[i] = self.not_equal(x_group, bc_data[:, inds])
            varying_indices = np.nonzero(varying)[0]
            return varying_indices
        else:
        
            # go over all nonzero columns in background and evaluation data
            # if both background and evaluation are zero, the column does not vary
            varying_indices = np.union1d(bc_data.nonzero()[1], x.nonzero()[1])

            remove_unvarying_indices = []
            for i in range(0, len(varying_indices)):
                varying_index = varying_indices[i]
                # now verify the nonzero values do vary
                data_rows = bc_data[:, [varying_index]]
                nonzero_rows = data_rows.nonzero()[0]

                if nonzero_rows.size > 0:
                    background_data_rows = data_rows[nonzero_rows]
                    if sp.sparse.issparse(background_data_rows):
                        background_data_rows = background_data_rows.toarray()
                    num_mismatches = np.sum(
                        np.abs(background_data_rows - x[0, varying_index]) > 1e-7
                    )
                    # Note: If feature column non-zero but some background zero, can't remove index
                    if num_mismatches == 0 and not (
                        np.abs(x[0, [varying_index]][0, 0]) > 1e-7
                        and len(nonzero_rows) < data_rows.shape[0]
                    ):
                        remove_unvarying_indices.append(i)
            mask = np.ones(len(varying_indices), dtype=bool)
            mask[remove_unvarying_indices] = False
            varying_indices = varying_indices[mask]
            return varying_indices

    def allocate(self):
        if sp.sparse.issparse(self.data.data):
            # We tile the sparse matrix in csr format but convert it to lil
            # for performance when adding samples
            shape = self.data.data.shape
            nnz = self.data.data.nnz
            data_rows, data_cols = shape
            rows = data_rows * self.nsamples
            shape = rows, data_cols
            if nnz == 0:
                self.synth_data = sp.sparse.csr_matrix(
                    shape, dtype=self.data.data.dtype
                ).tolil()
            else:
                data = self.data.data.data
                indices = self.data.data.indices
                indptr = self.data.data.indptr
                last_indptr_idx = indptr[len(indptr) - 1]
                indptr_wo_last = indptr[:-1]
                new_indptrs = []
                for i in range(0, self.nsamples - 1):
                    new_indptrs.append(indptr_wo_last + (i * last_indptr_idx))
                new_indptrs.append(indptr + ((self.nsamples - 1) * last_indptr_idx))
                new_indptr = np.concatenate(new_indptrs)
                new_data = np.tile(data, self.nsamples)
                new_indices = np.tile(indices, self.nsamples)
                self.synth_data = sp.sparse.csr_matrix(
                    (new_data, new_indices, new_indptr), shape=shape
                ).tolil()
        else:
            if self.returns_hs and self.background_hs is not None:
                if isinstance(self.background_hs, tuple):
                    self.synth_hidden_states = tuple(
                        np.tile(x, (1, self.nsamples, 1))
                        for x in self.background_hs
                    )
                else:
                    self.synth_hidden_states = np.tile(
                        self.background_hs, (1, self.nsamples, 1)
                    )

            self.synth_data = np.tile(self.data.data, (self.nsamples, 1, 1))

        self.maskMatrix = np.zeros((self.nsamples, self.M))
        self.kernelWeights = np.zeros(self.nsamples)
        self.y = np.zeros((self.nsamples * self.N, self.D))
        self.ey = np.zeros((self.nsamples, self.D))
        self.lastMask = np.zeros(self.nsamples)
        self.nsamplesAdded = 0
        self.nsamplesRun = 0
        if self.keep_index:
            self.synth_data_index = np.tile(self.data.index_value, self.nsamples)

    def add_sample(self, x, m, w):
        offset = self.nsamplesAdded * self.N
        mask = m == 1.0
        if self.mode == "feature":
            self.feat_add_sample(x, mask, offset)
        elif self.mode == "segment":
            self.segment_add_sample(x, mask, offset)

        self.maskMatrix[self.nsamplesAdded, :] = m
        self.kernelWeights[self.nsamplesAdded] = w
        self.nsamplesAdded += 1

    def activate_background(self, x, offset):
        # in case self.pruning_idx == sequence length, we dont prune anything.
        if self.returns_hs:
            # in case of using hidden state optimization, the background is the instance one
            if isinstance(self.synth_hidden_states, tuple):
                for i, i_layer_state in enumerate(self.synth_hidden_states):
                    i_layer_state[:, offset : offset + self.N, :] = self.instance_hs[i]
                
            else:
                self.synth_hidden_states[:, offset : offset + self.N, :] = (
                    self.instance_hs
                )
        else:
            # in case of not using hidden state optimization, we need to set the whole background to the original sequence
            evaluation_data = x[0:1, ...]
            self.synth_data[offset : offset + self.N,...] = evaluation_data

    def feat_add_sample(self, x, mask, offset):
        """
        Adds feature values to the constructed synthetic data
        """
        groups = self.varyingFeatureGroups[mask]

        evaluation_data = x[0:1, :, groups]
        if self.returns_hs:
            self.synth_data[offset : offset + self.N, :, groups] = evaluation_data
        else:
            self.synth_data[offset : offset + self.N, :, groups] = (
                evaluation_data
            )

    def segment_add_sample(self, x, mask, offset):
        """
        Adds segment values to the constructed synthetic data
        """
        # Fetch only boundaries of those segments that vary
        varying_segment_ids = self.varyingFeatureGroups[mask].tolist()
        varying_segments = [self.segment_boundaries[ind] for ind in varying_segment_ids]
        
        
        for (start, end) in varying_segments:
            evaluation_data = x[0:1, start:end, :]
            if self.returns_hs:
                self.synth_data[offset : offset + self.N, start:end, :] = evaluation_data
            else:
                self.synth_data[offset : offset + self.N, start:end, :] = (
                    evaluation_data
                )

    def run(self):
        num_to_run = self.nsamplesAdded * self.N - self.nsamplesRun * self.N

        data = self.synth_data[
            self.nsamplesRun * self.N : self.nsamplesAdded * self.N, :, :
        ]
        if self.returns_hs:
            modelOut, _ = self.model.f(data)
        else:
            modelOut = self.model.f(data)

        if isinstance(modelOut, (pd.DataFrame, pd.Series)):
            modelOut = modelOut.values
        
        self.y[self.nsamplesRun * self.N : self.nsamplesAdded * self.N, :] = np.reshape(
            modelOut, (num_to_run, self.D)
        )

        # find the expected value of each output
        for i in range(self.nsamplesRun, self.nsamplesAdded):
            eyVal = np.zeros(self.D)
            for j in range(0, self.N):
                eyVal += self.y[i * self.N + j, :] * self.data.weights[j]

            self.ey[i, :] = eyVal
            self.nsamplesRun += 1

    def solve(self, fraction_evaluated, dim):
        eyAdj = self.linkfv(self.ey[:, dim]) - self.link.f(self.fnull[dim])   # TODO: check the difference here
        s = np.sum(self.maskMatrix, 1)

        # do feature selection if we have not well enumerated the space
        nonzero_inds = np.arange(self.M)
        log.debug(f"{fraction_evaluated = }")
        
        if (self.l1_reg not in ["auto", False, 0]) or (fraction_evaluated < 0.2 and self.l1_reg == "auto"):
            w_aug = np.hstack((self.kernelWeights * (self.M - s), self.kernelWeights * s))
            log.info(f"{np.sum(w_aug) = }")
            log.info(f"{np.sum(self.kernelWeights) = }")
            
            w_sqrt_aug = np.sqrt(w_aug)
            eyAdj_aug = np.hstack((eyAdj, eyAdj - (self.link.f(self.fx[dim]) - self.link.f(self.fnull[dim]))))
            eyAdj_aug *= w_sqrt_aug
            mask_aug = np.transpose(w_sqrt_aug * np.transpose(np.vstack((self.maskMatrix, self.maskMatrix - 1))))

            # select a fixed number of top features
            if isinstance(self.l1_reg, str) and self.l1_reg.startswith("num_features("):
                r = int(self.l1_reg[len("num_features(") : -1])
                nonzero_inds = lars_path(mask_aug, eyAdj_aug, max_iter=r)[1]

            # use an adaptive regularization method
            elif self.l1_reg in ("auto", "bic", "aic"):
                c = "aic" if self.l1_reg == "auto" else self.l1_reg

                # "Normalize" parameter of LassoLarsIC was deprecated in sklearn version 1.2
                if version.parse(sklearn.__version__) < version.parse("1.2.0"):
                    kwg = dict(normalize=False)
                else:
                    kwg = {}
                model = make_pipeline(StandardScaler(with_mean=False), LassoLarsIC(criterion=c, **kwg))
                nonzero_inds = np.nonzero(model.fit(mask_aug, eyAdj_aug)[1].coef_)[0]

            # use a fixed regularization coefficient
            else:
                nonzero_inds = np.nonzero(Lasso(alpha=self.l1_reg).fit(mask_aug, eyAdj_aug).coef_)[0]

        if len(nonzero_inds) == 0:
            return np.zeros(self.M), np.ones(self.M)

        # eliminate one variable with the constraint that all features sum to the output
        eyAdj2 = eyAdj - self.maskMatrix[:, nonzero_inds[-1]] * (
            self.link.f(self.fx[dim]) - self.link.f(self.fnull[dim])
        )
        etmp = np.transpose(np.transpose(self.maskMatrix[:, nonzero_inds[:-1]]) - self.maskMatrix[:, nonzero_inds[-1]])
        log.debug(f"{etmp[:4, :] = }")

        # solve a weighted least squares equation to estimate phi
        # least squares:
        #     phi = min_w ||W^(1/2) (y - X w)||^2
        # the corresponding normal equation:
        #     (X' W X) phi = X' W y
        # with
        #     X = etmp
        #     W = np.diag(self.kernelWeights)
        #     y = eyAdj2
        #
        # We could just rely on sciki-learn
        #     from sklearn.linear_model import LinearRegression
        #     lm = LinearRegression(fit_intercept=False).fit(etmp, eyAdj2, sample_weight=self.kernelWeights)
        # Under the hood, as of scikit-learn version 1.3, LinearRegression still uses np.linalg.lstsq and
        # there are more performant options. See https://github.com/scikit-learn/scikit-learn/issues/22855.
        y = np.asarray(eyAdj2)
        X = etmp
        WX = self.kernelWeights[:, None] * X
        try:
            w = np.linalg.solve(X.T @ WX, WX.T @ y)
        except np.linalg.LinAlgError:
            warnings.warn(
                "Linear regression equation is singular, a least squares solutions is used instead.\n"
                "To avoid this situation and get a regular matrix do one of the following:\n"
                "1) turn up the number of samples,\n"
                "2) turn up the L1 regularization with num_features(N) where N is less than the number of samples,\n"
                "3) group features together to reduce the number of inputs that need to be explained."
            )
            # XWX = np.linalg.pinv(X.T @ WX)
            # w = np.dot(XWX, np.dot(np.transpose(WX), y))
            sqrt_W = np.sqrt(self.kernelWeights)
            w = np.linalg.lstsq(sqrt_W[:, None] * X, sqrt_W * y, rcond=None)[0]
        log.debug(f"{np.sum(w) = }")
        log.debug(
            f"self.link(self.fx) - self.link(self.fnull) = {self.link.f(self.fx[dim]) - self.link.f(self.fnull[dim])}"
        )
        log.debug(f"self.fx = {self.fx[dim]}")
        log.debug(f"self.link(self.fx) = {self.link.f(self.fx[dim])}")
        log.debug(f"self.fnull = {self.fnull[dim]}")
        log.debug(f"self.link(self.fnull) = {self.link.f(self.fnull[dim])}")
        phi = np.zeros(self.M)
        phi[nonzero_inds[:-1]] = w
        phi[nonzero_inds[-1]] = (self.link.f(self.fx[dim]) - self.link.f(self.fnull[dim])) - sum(w)
        log.info(f"{phi = }")

        # clean up any rounding errors
        for i in range(self.M):
            if np.abs(phi[i]) < 1e-10:
                phi[i] = 0

        return phi, np.ones(len(phi))
