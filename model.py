import os, sys
import torch
from torch.nn import Module, Linear
import torch.nn.functional as F
from torch_geometric.utils import scatter
from torch_geometric.nn.models.basic_gnn import GCN, GraphSAGE, GAT, BasicGNN
from torch_geometric.nn.conv import MessagePassing, GCNConv, GraphConv, SAGEConv, GATConv
import pennylane as qml
from typing import Final
from enum import Enum

# Import from HGP-SL
module_path = os.path.abspath("./HGP-SL")
if module_path not in sys.path:
    sys.path.insert(0, module_path)
from layers import HGPSLPool
from models import Model as HGPSLModel

from config import Embedding, ClassicalModel, Pooling

class QFE_GCN(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # self.qfe = QFE(in_features)
        # self.lin = torch.nn.Linear(in_features, in_features)

        self.gcn = GCN(
            in_channels=in_features,
            hidden_channels=64,
            num_layers=4,
            dropout = 0.1,
            out_channels=out_features
        )
    
    def forward(self, data):
        x = data.x
        # x = self.lin(x)
        # x = self.qfe(x.to(device="cpu")).to(device=x.device)

        x = self.gcn(x, data.edge_index, batch=data.batch)
        
        # x = global_mean_pool(x, data.batch)
        x = scatter(x, data.batch, dim=0, reduce="mean")
        x = F.log_softmax(x, dim=1)

        # out = torch.zeros(data.num_graphs, 2, device=x.device)

        # for i in range(data.num_graphs):
        #     out[i] = x[data.batch == i].mean(dim=0)
        
        # out = F.log_softmax(out, dim=1)
        # x = out
        
        return x


# EMBEDDINGS

class QFE_MeasureMethod(Enum):
    PROBS = "probs"
    EXP = "exp"

def build_qfe_circuit(dev, n_qubits, n_layers, qfe_method: QFE_MeasureMethod):
    @qml.qnode(dev, interface="torch") #, diff_method="parameter-shift")
    def qfe_circuit(inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
        # Ranges of 1 are used to closely match the figures in the QFE paper
        qml.templates.StronglyEntanglingLayers(weights, ranges=[1]*n_layers, wires=range(n_qubits))

        if qfe_method == QFE_MeasureMethod.PROBS:
            return qml.probs(wires=range(n_qubits))
        elif qfe_method == QFE_MeasureMethod.EXP:
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        else:
            raise ValueError(f"Invalid QFE Measure Method: {qfe_method}")
    
    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    return qml.qnn.TorchLayer(qfe_circuit, weight_shapes)

class QFE(Module):
    def __init__(self, features, layers, qfe_method: QFE_MeasureMethod):
        super().__init__()
        self.wires = features
        self.dev = qml.device("default.qubit", wires=self.wires)
        self.qfe = build_qfe_circuit(self.dev, self.wires, layers, qfe_method)
    
    def forward(self, x):
        return self.qfe(x)


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
    # Build Embedding
    if args.embedding.value.split("-")[0] == "QFE":
        qfe_method = QFE_MeasureMethod[args.embedding.value.split("-")[1]]
        return QFE(features, args.qml_embedding_layers, qfe_method)
    elif args.embedding.value.split("-")[0] == "MLP":
        mlp_type = args.embedding.value.split("-")[1]
        hidden_neurons = 2 ** features if mlp_type == "2^D" else features
        # Since the QML embedders can only output the same number of features
        # as the input, we choose the output dimension to be the same as the input
        return MLPEmbedder(features, hidden_neurons, features)
    elif args.embedding == Embedding.NONE:
        return None
    else:
        raise ValueError(f"Invalid Embedding Type: {args.embedding}")

# MODELS

# Modeled after GCN in torch_geometric.nn
class GraphConvModel(BasicGNN):
    supports_edge_weight: Final[bool] = True
    supports_edge_attr: Final[bool] = False
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        return GraphConv(in_channels, out_channels, **kwargs)

convolutions = {
    ClassicalModel.GCN: GCNConv,
    ClassicalModel.GraphConv: GraphConv,
    ClassicalModel.GraphSAGE: SAGEConv,
    ClassicalModel.GAT: GATConv
}

models = {
    ClassicalModel.GCN: GCN,
    ClassicalModel.GraphConv: GraphConvModel,
    ClassicalModel.GraphSAGE: GraphSAGE,
    ClassicalModel.GAT: GAT
}

class GNNModel(Module):
    def __init__(self, args, features, classes):
        Module.__init__(self)
        self.num_features = features
        self.num_classes = classes
        self.pooling = args.pooling

        self.embedder = build_embedder(args, features)

        self.model = models[args.model](
            in_channels=features,
            hidden_channels=args.hidden,
            num_layers=args.layers,
            dropout=args.dropout,
            out_channels=classes
        )

    def forward(self, data):
        x = data.x
        x = self.embedder(x) if self.embedder is not None else x
        x = self.model(x, data.edge_index, batch=data.batch)
        x = scatter(x, data.batch, dim=0, reduce=self.pooling.value)
        x = F.log_softmax(x, dim=1)
        return x

class HGPSLModelParametrized(HGPSLModel):
    def __init__(self, args, features, classes):
        Module.__init__(self)

        # Default parameters from HGP-SL are noted if not included in config.py
        self.num_features = features
        self.nhid = args.hidden # 128
        self.num_classes = classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout # 0.0
        self.sample = args.sample_neighbor
        self.sparse = args.sparse_attention
        self.sl = args.structure_learning
        self.lamb = args.lamb

        self.embedder = build_embedder(args, features)

        self.conv = convolutions[args.model]
        self.model = models[args.model]
        
        self.conv1 = self.conv(self.num_features, self.nhid)
        self.conv2 = self.model(self.nhid, self.nhid)
        self.conv3 = self.model(self.nhid, self.nhid)

        self.pool1 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)
        self.pool2 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)

        self.lin1 = Linear(self.nhid * 2, self.nhid)
        self.lin2 = Linear(self.nhid, self.nhid // 2)
        self.lin3 = Linear(self.nhid // 2, self.num_classes)
    
    def forward(self, data):
        data.x = self.embedder(data.x) if self.embedder is not None else data.x
        return super().forward(data)


def build_model(args, features, classes):
    if args.pooling == Pooling.HGPSL:
        return HGPSLModelParametrized(args, features, classes)
    else:
        return GNNModel(args, features, classes)