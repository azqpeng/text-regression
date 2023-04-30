import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros

class gcn(nn.Module):
    def __init__(self, features, hidden_size, adj, bias=True): # X_size = num features
        super(gcn, self).__init__()

        # initialize matrices
        self.adj = torch.tensor(adj, requires_grad=False).float()
        self.weight = Parameter(torch.FloatTensor(features, hidden_size))
        self.weight2 = Parameter(torch.FloatTensor(hidden_size, 1))

        # initialize weights
        var = 2./(self.weight.size(1)+self.weight.size(0))
        self.weight.data.normal_(0,var)
        var2 = 2./(self.weight2.size(1)+self.weight2.size(0))
        self.weight2.data.normal_(0,var2)
        
    def forward(self, X): ### 2-layer GCN architecture
        if self.adj.is_sparse:
            X = torch.sparse.mm(X, self.weight)
            X = F.relu(torch.sparse.mm(self.adj, X))
            X = torch.sparse.mm(X, self.weight2)
            X = F.relu(torch.sparse.mm(self.adj, X))
        else:
            X = torch.mm(X, self.weight)
            X = F.relu(torch.mm(self.adj, X))
            X = torch.mm(X, self.weight2)
            X = F.relu(torch.mm(self.adj, X))
        return X
    
    