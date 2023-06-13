from torch_geometric.nn import GATConv, Linear, to_hetero, SAGPooling
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

# construct a simple heterogeneous graph
graph = HeteroData()
graph["token"].x = torch.tensor([[1.]*8]*4, dtype=torch.float)
graph["sent"].x = torch.tensor([[1.]*8]*3, dtype=torch.float)
graph["doc"].x = torch.tensor([[1.]*8]*2, dtype=torch.float)
graph['token', 'similar_token', 'token'].edge_index = torch.tensor(
    [[0, 2, 3], [3, 1, 2]], dtype=torch.long)
graph['token', 'similar_token', 'token'].edge_attr = torch.reshape(torch.tensor([1., 1., 1.], dtype=torch.float), (-1, 1))

graph['sent', 'sent_contain_token', 'token'].edge_index = torch.tensor([[0, 1], [2, 0]], dtype=torch.long)
graph['sent', 'sent_contain_token', 'token'].edge_attr = torch.reshape(torch.tensor([0.5, 0.5], dtype=torch.float), (-1, 1))

graph['doc', 'doc_contain_sent', 'sent'].edge_index = torch.tensor([[0], [1]], dtype=torch.long)
graph['doc', 'doc_contain_sent', 'sent'].edge_attr = torch.reshape(torch.tensor([0.6], dtype=torch.float), (-1, 1))
graph = T.ToUndirected()(graph)

# ogb_mag = OGB_MAG(root='./test_data', preprocess='metapath2vec', transform=T.ToUndirected())
# graph = ogb_mag[0]

print(graph.metadata())


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False)
        self.lin2 = Linear(-1, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x)
        return x


model = GAT(hidden_channels=8, out_channels=1)
# model = to_hetero(model, graph.metadata(), aggr='sum')
model = to_hetero(model, (['token', 'sent', 'doc'], graph.metadata()[1]), aggr='sum')
# model = to_hetero(model, (['paper', 'author', 'institution', 'field_of_study'], graph.metadata()[1]), aggr='sum')
print(graph)
out = model(graph.x_dict, graph.edge_index_dict)
print("Output")
print(out)

# graph = graph.to_homogeneous()
print(graph)
sag_pooling = SAGPooling(in_channels=1, ratio=0.5)
graph["sent"].x = out["sent"]
out = sag_pooling(graph["sent"].x, graph['sent', 'sent_contain_token', 'token'].edge_index)
print(out)


