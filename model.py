import os, sys
import torch
import torch.nn as nn
from torch.nn import Module, Linear, Parameter
import torch.nn.functional as F
from torch_geometric.utils import scatter
from torch_geometric.nn.models.basic_gnn import GCN, GraphSAGE, GAT, BasicGNN
from torch_geometric.nn.conv import (
    MessagePassing,
    GCNConv,
    GraphConv,
    SAGEConv,
    GATConv,
)
import pennylane as qml
from typing import Final
from enum import Enum
from torch_scatter import scatter_add

from config import Embedder, ClassicalModel, Pooling


# EMBEDDINGS


class QFE_MeasureMethod(Enum):
    PROBS = "probs"
    EXP = "exp"


def build_qfe_circuit(dev, n_qubits, n_layers, qfe_method: QFE_MeasureMethod):
    @qml.qnode(dev, interface="torch")  # , diff_method="parameter-shift")
    def qfe_circuit(inputs, weights):
        # Embedding layer
        qml.templates.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")

        # Entangling layers
        for i in range(n_layers):
            for j in range(n_qubits):
                qml.RX(weights[i, j], wires=j)

            for j in range(n_qubits):
                qml.CNOT(wires=[j, (j + 1) % n_qubits])

        # Measurement
        if qfe_method == QFE_MeasureMethod.PROBS:
            return qml.probs(wires=range(n_qubits))
        elif qfe_method == QFE_MeasureMethod.EXP:
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        else:
            raise ValueError(f"Invalid QFE Measure Method: {qfe_method}")

    weight_shapes = {"weights": (n_layers, n_qubits)}
    return qml.qnn.TorchLayer(qfe_circuit, weight_shapes)


class QFE(Module):
    def __init__(self, features, layers, qfe_method: QFE_MeasureMethod):
        super().__init__()
        self.wires = features
        self.dev = qml.device("default.qubit", wires=self.wires)
        self.qfe = build_qfe_circuit(self.dev, self.wires, layers, qfe_method)

    def forward(self, x):
        prev_device = x.device
        x = x.cpu()
        return self.qfe(x).to(prev_device)


class MLPEmbedder(Module):
    def __init__(self, features, hidden_neurons, output_dim):
        super().__init__()
        self.fc1 = Linear(features, hidden_neurons)
        self.fc2 = Linear(hidden_neurons, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


def build_embedder(args, features):
    # Build Embedder
    if args.embedder.value.split("-")[0] == "QFE":
        qfe_method = QFE_MeasureMethod[args.embedder.value.split("-")[1].upper()]
        return QFE(features, args.qfe_layers, qfe_method)
    elif args.embedder.value.split("-")[0] == "MLP":
        mlp_type = args.embedder.value.split("-")[1]
        hidden_neurons = 2**features if mlp_type == "2^D" else features
        # Since the QML embedders can only output the same number of features
        # as the input, we choose the output dimension to be the same as the input
        return MLPEmbedder(features, hidden_neurons, features)
    elif args.embedder == Embedder.NONE:
        return None
    else:
        raise ValueError(f"Invalid Embedder Type: {args.embedder}")


# MODELS


# Modeled after GCN in torch_geometric.nn
class GraphConvModel(BasicGNN):
    supports_edge_weight: Final[bool] = True
    supports_edge_attr: Final[bool] = False
    supports_norm_batch: Final[bool]

    def init_conv(
        self, in_channels: int, out_channels: int, **kwargs
    ) -> MessagePassing:
        return GraphConv(in_channels, out_channels, **kwargs)


convolutions = {
    ClassicalModel.GCN: GCNConv,
    ClassicalModel.GraphConv: GraphConv,
    ClassicalModel.GraphSAGE: SAGEConv,
    ClassicalModel.GAT: GATConv,
}

models = {
    ClassicalModel.GCN: GCN,
    ClassicalModel.GraphConv: GraphConvModel,
    ClassicalModel.GraphSAGE: GraphSAGE,
    ClassicalModel.GAT: GAT,
}


class GNNModel(Module):
    def __init__(self, args, features, classes):
        Module.__init__(self)
        self.num_features = features
        self.num_classes = classes
        self.pooling = args.pooling

        self.embedder = build_embedder(args, features)

        in_features = 2**features if args.embedder == Embedder.QFE_PROBS else features

        self.model = models[args.model](
            in_channels=in_features,
            hidden_channels=args.hidden,
            num_layers=args.layers,
            dropout=args.dropout,
            out_channels=classes,
        )

    def forward(self, x, edge_index, batch=None):
        x = self.embedder(x) if self.embedder is not None else x
        x = self.model(x, edge_index, batch=batch)

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.int64, device=x.device)

        x = scatter(x, batch, dim=0, reduce=self.pooling.value)
        x = F.log_softmax(x, dim=1)
        return x


class SpecialGCN(MessagePassing):
    def __init__(self, in_channels, out_channels, cached=False, bias=True, **kwargs):
        super().__init__(aggr="add", **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cached = cached
        self.cached_result = None
        self.cached_num_edges = None

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight.data)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            nn.init.zeros_(self.bias.data)
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones(
                (edge_index.size(1),), dtype=dtype, device=edge_index.device
            )

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        x = torch.matmul(x, self.weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    "Cached {} number of edges, but found {}".format(
                        self.cached_num_edges, edge_index.size(1)
                    )
                )

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight, x.dtype)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__, self.in_channels, self.out_channels
        )


def build_model(args, features, classes):
    return GNNModel(args, features, classes)


def save_model(args, features, classes, model, path):
    torch.save(
        {
            "args": args,
            "features": features,
            "classes": classes,
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def load_model(path):
    data = torch.load(path)
    model = build_model(data["args"], data["features"], data["classes"])
    model.load_state_dict(data["model_state_dict"])
    return model
