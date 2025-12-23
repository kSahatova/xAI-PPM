import numpy as np
from typing import List
import torch
from torch_geometric.data import Data


def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def graph_generator(adjacency_matrix, padded_acivities, trace_lens: List[int], beta: float = 0.005):

    num_cases = len(trace_lens)
    adjacency_matrix =  np.array(adjacency_matrix >=  beta* num_cases, dtype='int32')
    onehot_encoded = to_categorical(padded_acivities)
    node_xs = []
    edge_indexs = []

    for case_index in range(num_cases):
        edge = []
        xs = torch.tensor(onehot_encoded[case_index, :, ])

        if  trace_lens[case_index] > 1:
            node = padded_acivities[case_index, : trace_lens[case_index]]
            for activity_index in range(0,  trace_lens[case_index]):
                out = np.argwhere(adjacency_matrix[padded_acivities[case_index, activity_index]] == 1).flatten()
                a = set(node.numpy())
                b = set(out)
                if activity_index + 1 < trace_lens[case_index]:
                    edge.append([activity_index, activity_index+1])
                for node_name in a.intersection(b):
                    for node_index in np.argwhere(node == node_name).flatten():
                        if  activity_index + 1 != node_index:
                            edge.append([activity_index, node_index.item()])

        edge_index = torch.tensor(edge, dtype=torch.long)
        node_xs.append(xs)
        edge_indexs.append(edge_index.T)
    
    