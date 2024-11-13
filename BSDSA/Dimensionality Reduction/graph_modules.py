from torch_geometric.nn import GATConv
import torch.nn as nn
from torch.nn import functional as F
import torch
class VGATEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels, heads=1, dropout=0.2):
        super(VGATEncoder, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout).to(self.device)
        self.conv_mu = GATConv(hidden_channels * heads, latent_channels, heads=1, concat=False, dropout=dropout).to(self.device)
        self.conv_logvar = GATConv(hidden_channels * heads, latent_channels, heads=1, concat=False, dropout=dropout).to(self.device)
        self.to(self.device)

    def forward(self, x, edge_index, edge_weight=None):
        x, edge_index = x.to(self.device), edge_index.to(self.device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(self.device)
        x = F.elu(self.conv1(x, edge_index, edge_weight))
        mu = self.conv_mu(x, edge_index, edge_weight)
        logvar = self.conv_logvar(x, edge_index, edge_weight)
        return mu, logvar
