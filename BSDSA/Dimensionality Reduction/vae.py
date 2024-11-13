import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, encoder, decoder, mean_module, logvar_module):
        super(VAE, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.mean_module = mean_module.to(self.device)
        self.logvar_module = logvar_module.to(self.device)
        self.to(self.device)
    
    def encode(self,x):
        x = x.to(self.device)
        mean = self.mean_module(self.encoder(x))
        logvar = self.logvar_module(self.encoder(x))
        return mean, logvar
    
    def decode(self,z):
        z = z.to(self.device)
        return self.decoder(z)
    
    def reparameterization(self, mean, var):
        mean = mean.to(self.device)
        var = var.to(self.device)
        epsilon = torch.randn_like(var).to(self.device)
        z = mean + var*epsilon
        return z
    
    def project(self, x):
        x = x.to(self.device)
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        return z

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        return self.decode(z), mean, logvar
    
    def recon_loss(self, recon_x, x):
        return nn.functional.mse_loss(recon_x, x, reduction='mean')
    
    def kl_loss(self, mean, logvar):
        return -0.5*torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels, num_layers):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.latent_channels = latent_channels
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(nn.Linear(in_channels, hidden_channels))
        self.convs.append(nn.LeakyReLU())
        for _ in range(num_layers - 2):
            self.convs.append(nn.Linear(hidden_channels, hidden_channels))
            self.convs.append(nn.LeakyReLU())
        self.convs.append(nn.Linear(hidden_channels, latent_channels))

    def forward(self, x):
        for layer in self.convs:
            x = layer(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(nn.Linear(in_channels, hidden_channels))
        self.convs.append(nn.LeakyReLU())
        for _ in range(num_layers - 2):
            self.convs.append(nn.Linear(hidden_channels, hidden_channels))
            self.convs.append(nn.LeakyReLU())
        self.convs.append(nn.Linear(hidden_channels, out_channels))
    
    def forward(self, z):
        for layer in self.convs:
            z = layer(z)
        return z
    
class MeanModule(nn.Module):
    def __init__(self, latent_channels, out_channels):
        super(MeanModule, self).__init__()
        self.latent_channels = latent_channels
        self.out_channels = out_channels
        self.fc = nn.Linear(latent_channels, out_channels)
    
    def forward(self, z):
        return self.fc(z)
    
class LogVarModule(nn.Module):
    def __init__(self, latent_channels, out_channels):
        super(LogVarModule, self).__init__()
        self.latent_channels = latent_channels
        self.out_channels = out_channels
        self.fc = nn.Linear(latent_channels, out_channels)
    
    def forward(self, z):
        return self.fc(z)
