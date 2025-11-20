import copy
import numpy as np
from typing import Any, List

import torch
from torch import nn

from timeshap.wrappers import TimeSHAPWrapper


class OutcomePredictorWrapper(TimeSHAPWrapper):
    def __init__(
        self,
        model: nn.Module,
        batch_budget: int,
        categorical_indices: List[int],
        target: str,
        device: torch.device,
    ):
        super(OutcomePredictorWrapper, self).__init__(model, batch_budget)
        self.categorical_indices = categorical_indices
        self.target = target
        self.device = device

    def prepare_input(self, input):
        sequence = copy.deepcopy(input)
        # check the dimensionality of the input
        if isinstance(sequence, np.ndarray):
            if len(sequence.shape) == 2:
                sequence = np.expand_dims(sequence, axis=0)
        else:
            raise ValueError(
                f"Input type not supported. Provided type and dimensions \
                             are {type(sequence), sequence.shape}, but requred np.ndarray of the shape BxLxD or LxD"
            )

        # separate categorical and numerical features
        total_features_num = sequence.shape[-1]
        numerical_indices = np.setdiff1d(
            np.arange(total_features_num), self.categorical_indices
        )
        categorical_features = sequence[:, :, self.categorical_indices]
        numerical_features = sequence[:, :, numerical_indices]

        x_cat, x_num = (
            torch.Tensor(categorical_features).long().to(self.device),
            torch.Tensor(numerical_features).to(self.device),
        )

        attention_mask = (x_cat[..., 0] != 0).long()

        # extract an attention mask

        return x_cat, x_num, attention_mask

    def __call__(self, sequences: np.ndarray, hidden_state: Any = None):
        x_cat, x_num, attention_mask = self.prepare_input(sequences)
        self.model = self.model.to(self.device)

        with torch.no_grad():
            self.model.eval()
            if hidden_state is not None:
                if isinstance(hidden_state, tuple):
                    if isinstance(hidden_state[0], tuple):
                        # for LSTM
                        hidden_state = tuple(
                            tuple(
                                torch.from_numpy(y).float().to(self.device) for y in x
                            )
                            for x in hidden_state
                        )
                    else:
                        hidden_state = tuple(
                            torch.from_numpy(x).float().to(self.device)
                            for x in hidden_state
                        )
                else:
                    hidden_state = (
                        torch.from_numpy(hidden_state).float().to(self.device)
                    )

            predictions = self.model(
                x_cat=x_cat, x_num=x_num, attention_mask=attention_mask, h=hidden_state
            )

            if isinstance(predictions, tuple) and len(predictions) == 2:
                predictions, hs = predictions
                predictions = predictions[:, -1, :]
                predictions = predictions.cpu().numpy().reshape(-1, 1)

                if isinstance(hs, tuple):
                    if isinstance(hs[0], tuple):
                        # for LSTM
                        hs = tuple(tuple(y.cpu().numpy() for y in x) for x in hs)
                    else:
                        hs = tuple(x.cpu().numpy() for x in hs)
                else:
                    hs = hs.cpu().numpy()
                return predictions, hs
            else:
                raise NotImplementedError(
                    "Only models that return predictions or predictions + hidden states are supported for now."
                )
