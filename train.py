import torch
import torch.nn.functional as F
from TextDataset import TextDataset
from gnn_model import gcn

data = TextDataset()
model = gcn(torch.eye(data.x_train[1]), 32, data.adj)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()
    F.mse_loss(model()[data.train_mask], data.y_train[data.train_mask]).backward()
    optimizer.step()
    return

@torch.no_grad()
def test():
    
    return 