import torch
import torch.nn as nn
from torch.nn.functional import dropout
from torch_geometric.nn import GCNConv

class GCNBlock(nn.Module):
    # gcn_layers_dim: 列表，包含每层 GCN 层的输入和输出维度
    def __init__(self, gcn_layers_dim, dropout_rate=0.):
        super().__init__()
        self.conv_layers = torch.nn.ModuleList()
        for i in range(len(gcn_layers_dim) - 1):
            conv_layer = GCNConv(gcn_layers_dim[i], gcn_layers_dim[i + 1])
            self.conv_layers.append(conv_layer)
        # 与model.eval()联动，自动识别self.trainning标志，在训练时启用dropout，在评估时禁用dropout
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.relu = torch.nn.ReLU()

    def forward(self, x, edge_index, edge_weight, batch):
        output = x
        for conv_layer_index in range(len(self.conv_layers)):
            output = self.conv_layers[conv_layer_index](output, edge_index, edge_weight)
            output = self.relu(output)
            output = self.dropout(output)
        return output

class GCN(nn.Module):
    def __init__(self, gcn_layers_dim, dropout_rate=0.):
        super().__init__()
        self.num_layers = len(gcn_layers_dim) - 1
        self.graph_conv = GCNBlock(gcn_layers_dim, dropout_rate)

    def forward(self, graph_batchs):
        embedding_batchs = list(
            map(lambda graph: self.graph_conv(graph.x, graph.edge_index, graph.batch), graph_batchs))
        embeddings = []
        for i in range(self.num_layers):
            embeddings.append(torch.cat(list(map(lambda embedding_batch: embedding_batch[i], embedding_batchs)), 0))

        return embeddings
    


