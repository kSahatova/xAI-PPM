import numpy as np
from typing import List
import torch
from torch import nn
from torch_geometric.data import Data, Batch
from tqdm import tqdm
import random
from utils.unsup_anomaly_detection.gama.gat_ae import GAT_AE
from utils.unsup_anomaly_detection.gama.graph_generator import graph_generator


class GAMA:
    def __init__(
        self,
        attribute_dims: List[int],
        n_epochs: int = 20,
        max_seq_len: int = 148,
        batch_size: int = 64,
        lr: float = 0.0005,
        b1: float = 0.5,
        b2: float = 0.999,
        hidden_dim: int = 64,
        GAT_heads: int = 4,
        decoder_num_layers: int = 2,
        TF_styles: str = "FAP",
        seed: int = 42,
    ):
        if TF_styles not in ["AN", "PAV", "FAP"]:
            raise Exception('"TF_styles" must be a value in ["AN","PAV", "FAP"]')

        self.attribute_dims = attribute_dims
        self.max_seq_len = max_seq_len
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.n_epochs = n_epochs
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.GAT_heads = GAT_heads
        self.decoder_num_layers = decoder_num_layers
        self.TF_styles = TF_styles
        self.name = "GAMA"
        if type(self.seed) is int:
            torch.manual_seed(self.seed)

    def __init_model(self):
        self.model = GAT_AE(
            self.attribute_dims,
            self.max_seq_len,
            self.hidden_dim,
            self.GAT_heads,
            self.decoder_num_layers,
            self.TF_styles,
        )
        self.model.to(self.device)

    def setup_model(self):
        """
        Initializes model, optimizer, and learning rate scheduler"""
        self.__init_model()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2)
        )
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, gamma=0.85
        )

    def setup_data(
        self,
        activities: np.ndarray,
        adjacency_matrix: np.ndarray,
        trace_lens: List[int],
        beta: float = 0.005,
    ):
        """
        Prepares data for training and evaluation
        """
        self.activities = activities
        self.trace_lens = trace_lens
        self.beta = beta

        self.mask = np.ones(activities.shape, dtype=bool)
        for m, j in zip(self.mask, self.trace_lens):
            m[:j] = False

        self.nodes, self.edge_indices = graph_generator(
            adjacency_matrix, activities, self.trace_lens, self.beta
        )

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.__init_model()
        self.model.load_state_dict(checkpoint)

        return self

    def fit(self):
        """
        Trains multigraph encoder and decoder on the provided dataset
        """
        if self.model is None or self.optimizer is None or self.scheduler is None:
            self.setup_model()

        criterion = nn.CrossEntropyLoss()
        Xs = torch.LongTensor(self.activities)

        print("*" * 10 + "training" + "*" * 10)
        for epoch in range(int(self.n_epochs)):
            train_loss = 0.0
            train_num = 0

            # Custom dataloader
            indexes = [i for i in range(len(self.nodes))]
            random.shuffle(indexes)

            for bathc_i in tqdm(
                range(self.batch_size, len(indexes) + 1, self.batch_size)
            ):
                current_batch_ind = indexes[bathc_i - self.batch_size : bathc_i]
                nodes_list = [self.nodes[i] for i in current_batch_ind]
                edge_indices_list = [self.edge_indices[i] for i in current_batch_ind]

                Xs_batch_list = [Xs[current_batch_ind].to(self.device)]
                graph_batch = Batch.from_data_list(
                    [
                        Data(x=nodes_list[b], edge_index=edge_indices_list[b])
                        for b in range(len(nodes_list))
                    ]
                )
                graph_batch_list = [graph_batch.to(self.device)]
                mask = torch.tensor(self.mask[current_batch_ind]).to(self.device)

                attr_reconstruction_outputs = self.model(
                    graph_batch_list, Xs_batch_list, mask, len(current_batch_ind)
                )
                self.optimizer.zero_grad()

                loss = 0.0
                mask[:, 0] = True
                for i in range(len(self.attribute_dims)):
                    pred = attr_reconstruction_outputs[i][~mask]
                    true = Xs_batch_list[i][~mask]
                    loss += criterion(pred, true)

                train_loss += loss.item() / len(self.attribute_dims)
                train_num += 1
                loss.backward()
                self.optimizer.step()

            ## Calculate the loss and accuracy of one epoch on the training set.
            train_loss_epoch = train_loss / train_num
            print(
                f"[Epoch {epoch + 1:{len(str(self.n_epochs))}}/{self.n_epochs}]"
                f"[loss: {train_loss_epoch:3f}]"
            )
            self.scheduler.step()

        return self

    def detect(self):
        """
        Detects anomalous traces in the provided dataset
        """
        self.model.eval()

        with torch.no_grad():
            final_res = []
            Xs = torch.LongTensor(self.activities)

            print("*" * 10 + "detecting" + "*" * 10)

            activities_num = self.activities.shape[0]
            Xs_list = [Xs.to(self.device)]
            graph_batch = Batch.from_data_list(
                [
                    Data(x=self.nodes[i], edge_index=self.edge_indices[i])
                    for i in range(len(self.nodes))
                ]
            )
            graph_batch_list = [graph_batch.to(self.device)]
            mask = torch.tensor(self.mask).to(self.device)

            attr_reconstruction_outputs = self.model(
                graph_batch_list, Xs_list, mask, activities_num
            )

            for attr_index in range(len(self.attribute_dims)):
                attr_reconstruction_outputs[attr_index] = torch.softmax(
                    attr_reconstruction_outputs[attr_index], dim=2
                )

            this_res = []
            for attr_index in range(len(self.attribute_dims)):
                # The sum of probabilities of taking the attribute value that is larger than the actual attribute value.
                temp = attr_reconstruction_outputs[attr_index]
                index = Xs_list[attr_index].unsqueeze(2)
                probs = temp.gather(2, index)
                temp[(temp <= probs)] = 0
                res = temp.sum(2)
                res = res * (~mask)
                this_res.append(res)

            final_res.append(torch.stack(this_res, 2))

            attr_level_abnormal_scores = np.array(
                torch.cat(final_res, 0).detach().cpu()
            )
            trace_level_abnormal_scores = attr_level_abnormal_scores.max((1, 2))
            event_level_abnormal_scores = attr_level_abnormal_scores.max((2))

            return (
                trace_level_abnormal_scores,
                event_level_abnormal_scores,
                attr_level_abnormal_scores,
            )
