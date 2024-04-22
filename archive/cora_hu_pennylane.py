import pennylane as qml
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import numpy as np
import math
from tqdm import tqdm


dataset = Planetoid("/tmp/Cora", "Cora", split="random", num_train_per_class=int(140/7))

# Implement API similar to torchquantum
def op1_all_layers(op, wires):
    weight_index = 0
    for i in wires:
        op(i, weight_index)
        weight_index += 1
    
    return len(wires)

def op2_all_layers(op, wires):
    weight_index = 0
    for i in range(1,len(wires)):
        for j in range(len(wires)-i):
            op(wires[j], wires[j+i], weight_index)
            weight_index += 1
    
    return weight_index

def q_layer(n_wires, weights, weight_index=0):
    weight_index += op1_all_layers(lambda w, i: qml.RY(weights[i], wires=w), range(n_wires))
    weight_index += op2_all_layers(lambda w1, w2, i: qml.CRY(weight_index+i, wires=[w1, w2]), range(n_wires))
    weight_index += op1_all_layers(lambda w, i: qml.RY(weights[i+weight_index], wires=w), range(n_wires))

    weight_index += op1_all_layers(lambda w, i: qml.RX(weights[i+weight_index], wires=w), range(n_wires))
    weight_index += op2_all_layers(lambda w1, w2, i: qml.CRX(weight_index+i, wires=[w1, w2]), range(n_wires))
    weight_index += op1_all_layers(lambda w, i: qml.RX(weights[i+weight_index], wires=w), range(n_wires))

    weight_index += op1_all_layers(lambda w, i: qml.RZ(weights[i+weight_index], wires=w), range(n_wires))
    weight_index += op2_all_layers(lambda w1, w2, i: qml.CRZ(weight_index+i, wires=[w1, w2]), range(n_wires))
    weight_index += op1_all_layers(lambda w, i: qml.RZ(weights[i+weight_index], wires=w), range(n_wires))

    return weight_index

def build_circuit(dev, n_wires, layers, edge_index):
    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        qml.templates.AmplitudeEmbedding(inputs, wires=range(n_wires), pad_with=0, normalize=True)

        for i in range(layers):
            q_layer(n_wires, weights[i])

        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_wires)]

    weight_shape = {
        "weights": (layers, int(n_wires*6 + n_wires*(n_wires-1)*3/2),)
    }

    return qml.qnn.TorchLayer(circuit, weight_shape)


class QuGCN(torch.nn.Module):
    def __init__(self, n_nodes=140, edge_index=torch.tensor([]), in_features=1433, out_features=7, layers=8):
        super().__init__()
        self.wires = math.ceil(np.log2(n_nodes))
        self.qml_layer = build_circuit(qml.device("default.qubit", wires=self.wires), self.wires, layers, edge_index)
        self.fc = torch.nn.Linear(self.wires, out_features)
    
    def forward(self, x):
        # Take transpose so that the QC processes the data node-wise
        x = self.qml_layer(x.T).T
        x = self.fc(x)

        return x

n_nodes = 32

model = QuGCN(n_nodes=n_nodes, in_features=dataset.num_features, out_features=dataset.num_classes, layers=8)
data = dataset[0].subgraph(torch.tensor(range(n_nodes)))
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

losses = []

model.train()
for epoch in tqdm(range(200)):
    optimizer.zero_grad()
    # out = model(data.x[data.train_mask])
    out = model(data.x)
    # loss = F.nll_loss(out, data.y[data.train_mask])
    loss = F.nll_loss(out, data.y)
    loss.backward()
    losses.append(loss.detach())
    optimizer.step()

del loss, optimizer, out

data = dataset[0].subgraph(dataset.test_mask).subgraph(torch.tensor(range(n_nodes)))

model.eval()
pred = model(data.x).argmax(dim=1)
correct = (pred == data.y).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')