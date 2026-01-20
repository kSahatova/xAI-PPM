import math
from xml.sax import parseString
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch import nn
from sklearn.cluster import KMeans
from .autoencoder_torch import LSTMAutoencoder


class Explainer:
    """Model agnostic explainer class written in torch"""

    def __init__(
        self,
        input_dim: int,
        output_directory: Path,
        epochs: int,
        batch_size: int,
        n_concepts: int,
        latent_dim: int,
        build=True,
        sep_coef=1,
        **kwargs,
    ):
        """
        Initializes the explainer with specified settings
        """
        self.input_dim = input_dim
        self.output_directory = output_directory
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_concepts = n_concepts
        self.latent_dim = latent_dim
        self.sep_coef = sep_coef
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.explainer_kwargs = kwargs
        if build:
            self.explainer = self.build_explainer(**self.explainer_kwargs)

    def build_explainer(self, hidden_dim=64, num_layers=2, dropout=0.1, **extra_kwargs):
        """
        Builds explainer model
        **kwargs: Keyword arguments for tf.keras.Model.compile method
        :return: Tensorflow model for explanation of time series data
        """

        explainer = LSTMAutoencoder(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            latent_dim=self.latent_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        return explainer

    def fit_explainer(self, classifier: nn.Module, dataloader: DataLoader):
        """
        function fits model-agnostic explainer
        :param classifier: classifier to explain
        :param X: data
        """

        optimizer = torch.optim.Adam(self.explainer.parameters(), lr=1e-3)
        bce_loss = nn.BCELoss()
        mse_loss = nn.MSELoss()
        # loss_metric  =

        # classifier's weights should not be updated
        classifier.eval()

        for epoch in range(self.epochs):
            loss_info = {
                "mse_loss": 0.0,
                "bce_loss": 0.0,
                "sep_loss": 0.0,
                "total_loss": 0.0,
            }

            for batch in dataloader:
                x_cat, x_num, _, _ = batch
                x_cat, x_num = (x_cat.to(self.device), x_num.to(self.device))
                attention_mask = (x_cat[..., 0] != 0).long()

                out, _ = classifier(x_cat, x_num, attention_mask=attention_mask)
                y_pred = (out.squeeze(1) > 0.5).float()

                optimizer.zero_grad()
                input = torch.concat([x_cat.float(), x_num.float()], dim=-1)
                seq_len = input.shape[1]

                latent = self.explainer.encode(input)
                reconstructed = self.explainer.decode(latent, seq_len)

                x_cat_recon, x_num_recon = (
                    reconstructed[:, :, 0:1].to(torch.long),
                    reconstructed[:, :, 1:],
                )
                # attention_mask_recon = (x_cat_recon[..., 0] != 0).long()
                out_recon, _ = classifier(
                    x_cat=x_cat_recon,
                    x_num=x_num_recon,
                    attention_mask=attention_mask,
                )

                sep_loss_result = self.cluster_regularization(latent) * self.sep_coef
                mse_loss_result = mse_loss(reconstructed, input)
                bce_loss_result = bce_loss(y_pred, out_recon)
                loss = sep_loss_result + mse_loss_result + bce_loss_result
                loss.backward()
                optimizer.step()

                # Update loss results for all terms
                loss_info["mse_loss"] += mse_loss_result.item()
                loss_info["bce_loss"] += bce_loss_result.item()
                loss_info["sep_loss"] += sep_loss_result.item()
                loss_info["total_loss"] += loss.item()

            loss_info = {k: v / len(dataloader) for k, v in loss_info.items()}

            print(
                "Epoch {} | loss: {:.5f}, mse_loss: {:.5f}, bce_loss: {:.5f}".format(
                    epoch,
                    loss_info["total_loss"],
                    loss_info["mse_loss"],
                    loss_info["bce_loss"],
                )
            )

    def cluster_regularization(self, latent):
        """
        MAP separation regularization
        :param X: batch data
        :return: separation loss
        """
        latent = latent.detach().numpy()
        kmeans = KMeans(n_clusters=self.n_concepts, random_state=0).fit(latent)
        concepts = kmeans.cluster_centers_

        sep_loss = self.separation_regularization(concepts)

        return sep_loss

    def separation_regularization(self, concepts):
        """
        function from https://github.com/chihkuanyeh/concept_exp
        calculate Second regularization term, i.e. the similarity between concepts, to be minimized
        Note: it is important to MAKE SURE L2 GOES DOWN! that will let concepts separate from each other
        :param concepts: extracted concepts
        """

        all_concept_dot = concepts.T @ concepts
        mask = np.eye(len(concepts[0])) * -1 + 1  # the i==j positions are 0
        L_sparse_2_new = (
            torch.sum(torch.Tensor(all_concept_dot * mask))
            / (self.n_concepts * (self.n_concepts - 1))
            / self.batch_size
        )

        return L_sparse_2_new

    def get_concepts_kmeans(self, X):
        """
        :param X: data
        :return: reconstructed concepts and their lower dimensional prototypes
        """
        pass
