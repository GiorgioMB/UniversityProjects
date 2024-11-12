from torch_geometric.nn import GATConv
import torch.nn as nn
from torch.nn import functional as F

class GATEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels, heads=1, dropout=0.2):
        super(GATEncoder, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv_mu = GATConv(hidden_channels * heads, latent_channels, heads=1, concat=False, dropout=dropout)
        self.conv_logvar = GATConv(hidden_channels * heads, latent_channels, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.elu(self.conv1(x, edge_index, edge_weight))
        mu = self.conv_mu(x, edge_index, edge_weight)
        logvar = self.conv_logvar(x, edge_index, edge_weight)
        return mu, logvar
