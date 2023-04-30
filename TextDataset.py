import torch
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data
from sklearn.model_selection import train_test_split
from build_graph import build_graph

class TextDataset(InMemoryDataset):
    def __init__(self, docList, labels, nodeCount, test_size, window = 5, transform = None):
        super(TextDataset, self).__init__('.', transform = transform)

        data = Data(x = docList, y = labels)
        vocabCount = nodeCount - len(docList)

        # do some sort of feature embedding here.
        self.X = torch.eye(nodeCount)

        # create adjacency matrix
        self.adj = build_graph(docList, "PMI", window = window)

        # split data into training, validation(?), test
        train_data, test_data, train_labels, test_labels = train_test_split(pd.Series(docList), pd.Series(labels), test_size = test_size)

        self.x_train = train_data
        self.x_test = test_data
        self.y_train = train_labels
        self.y_test = test_labels

        
        # create masks
        self.train_mask = torch.zeros(nodeCount, dtype = torch.bool)
        self.test_mask = torch.zeros(nodeCount, dtype = torch.bool)

        self.train_mask[train_data.index + vocabCount] = True
        self.test_mask[test_data.index + vocabCount] = True

        data['train_mask'] = self.train_mask
        data['test_mask'] = self.test_mask

        # whatever the fuck slices are
        self.data, self.slices = self.collate([data])

    def _download(self):
        return
    
    def _process(self):
        return
    
    def __repr__(self) -> str:
        return super().__repr__()
    





