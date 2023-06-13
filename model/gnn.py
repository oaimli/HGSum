import torch.nn as nn
from torch_geometric.nn import GATConv, Linear, to_hetero
from typing import Callable, Optional, Union
import torch
from torch_geometric.nn import GraphConv
from torch_geometric.nn.pool.topk_pool import filter_adj, topk
from torch_geometric.utils import softmax


class SAGPooling(torch.nn.Module):
    def __init__(self, in_channels: int, ratio: Union[float, int] = 0.5,
                 GNN: Callable = GraphConv, min_score: Optional[float] = None,
                 multiplier: float = 1.0, nonlinearity: Callable = torch.tanh,
                 **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.gnn = GNN(in_channels, 1, **kwargs)
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity

        self.reset_parameters()

    def reset_parameters(self):
        self.gnn.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, batch=None, attn=None):
        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        attn = x if attn is None else attn
        attn = attn.unsqueeze(-1) if attn.dim() == 1 else attn
        score = self.gnn(attn, edge_index).view(-1)
        # print(score)

        if self.min_score is None:
            score = self.nonlinearity(score)
        else:
            score = softmax(score, batch)
        # print(score)

        perm = topk(score, self.ratio, batch, self.min_score)
        # print(score)
        x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        batch = batch[perm]
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm,
                                           num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm, score[perm]

    def __repr__(self) -> str:
        if self.min_score is None:
            ratio = f'ratio={self.ratio}'
        else:
            ratio = f'min_score={self.min_score}'

        return (f'{self.__class__.__name__}({self.gnn.__class__.__name__}, '
                f'{self.in_channels}, {ratio}, multiplier={self.multiplier})')


class GAT(nn.Module):
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


if __name__ == '__main__':
    # Using real case to test graph modeling and pooling
    import torch
    import random
    import jsonlines
    import torch_geometric.transforms as T
    import sys

    sys.path.append("../")
    from graph.building_graph import build_graph

    dataset_name = "multinews"
    token_type = "noun"
    samples = []
    with jsonlines.open("../data/%s_graph_%s_sentem.json" % (dataset_name, token_type)) as reader:
        for sample in reader:
            samples.append(sample)
            if len(samples) == 16:
                break

    random.seed(42)
    samples = random.sample(samples, 16)
    for sample in samples[4:]:
        print(sample.keys())
        print(sample["heterograph_source"].keys())
        heterograph_data_source = sample["heterograph_source"]
        heterograph_source = build_graph(sample["heterograph_source"], for_summary=False)
        print("graph data for source documents")
        print(heterograph_source)
        print(heterograph_source.metadata())
        tokens_positions_source = torch.tensor(heterograph_data_source["tokens_positions"])
        sents_positions_source = torch.tensor(heterograph_data_source["sents_positions"])
        docs_positions_source = torch.tensor(heterograph_data_source["docs_positions"])

        print(len(tokens_positions_source), len(sents_positions_source), len(docs_positions_source))
        heterograph_source["token"].x = torch.tensor([[0.0] * 8] * len(tokens_positions_source))
        heterograph_source["sent"].x = torch.tensor([[0.0] * 8] * len(sents_positions_source))
        heterograph_source["doc"].x = torch.tensor([[0.0] * 8] * len(docs_positions_source))
        model = GAT(hidden_channels=64, out_channels=8)
        model = to_hetero(model, heterograph_source.metadata(), aggr='sum')
        print(heterograph_source)
        print(torch.max(heterograph_source['token', 'followed_by', 'token'].edge_index))
        print(torch.max(heterograph_source['token', 'similar_token', 'token'].edge_index))
        print(torch.max(heterograph_source['sent', 'similar_sent_rouge', 'sent'].edge_index))
        print(torch.max(heterograph_source['sent', 'similar_sent_sbert', 'sent'].edge_index))
        print(torch.max(heterograph_source['sent', 'sent_contain_token', 'token'].edge_index[0]))
        print(torch.max(heterograph_source['sent', 'sent_contain_token', 'token'].edge_index[1]))
        print(torch.max(heterograph_source['doc', 'doc_contain_sent', 'sent'].edge_index[0]))
        print(torch.max(heterograph_source['doc', 'doc_contain_sent', 'sent'].edge_index[1]))
        print(torch.max(heterograph_source['doc', 'similar_doc_rouge', 'doc'].edge_index))
        heterograph_source = T.ToUndirected()(heterograph_source)
        print(heterograph_source.x_dict)
        print(heterograph_source.edge_index_dict[('doc', 'similar_doc_rouge', 'doc')])
        out = model(heterograph_source.x_dict, heterograph_source.edge_index_dict)
        print(out)
        sag_pooling = SAGPooling(in_channels=8, ratio=0.5)
        # heterograph_source["sent"].x = out["sent"]
        out = sag_pooling(heterograph_source["sent"].x, heterograph_source['sent', 'similar_sent_sbert', 'sent'].edge_index)
        print(out)

        print(sample["heterograph_tgt"].keys())
        heterograph_data_tgt = sample["heterograph_tgt"]
        heterograph_tgt = build_graph(sample["heterograph_tgt"], for_summary=True)
        print("graph data for target summary")
        print(heterograph_tgt)
        print(heterograph_tgt.metadata())
        heterograph_tgt = T.ToUndirected()(heterograph_tgt)
        tokens_positions_tgt = torch.tensor(heterograph_data_tgt["tokens_positions"])
        sents_positions_tgt = torch.tensor(heterograph_data_tgt["sents_positions"])