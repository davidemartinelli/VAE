import torch
import torch.nn as nn

import numpy as np

from utils import get_gpu_memory_map

class MLP(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, n_layers):
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(n_input, n_hidden))

        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(n_hidden, n_hidden))

        self.layers.append(nn.Linear(n_hidden, n_output))

        self.activation = nn.ReLU()
    
    def forward(self, x):
        for i,layer in enumerate(self.layers):
            if i != 0:
                x = self.activation(x)

            x = layer(x)

        return x

class VAE(nn.Module):
    '''This class implements the variational autoencoder 
    as proposed in https://arxiv.org/pdf/1312.6114.pdf.
    In particular, this implementation was constructed
    to deal with binary data.'''

    def __init__(self, n_input, D, n_hidden, n_layers):
        super().__init__()

        self.D = D #latent space dimension

        self.encoder = MLP(n_input, D * 2, n_hidden, n_layers) # output is mean and std of z
        self.decoder = MLP(D, n_input, n_hidden, n_layers)

        self.sigmoid = nn.Sigmoid()

        self.device = torch.device('cuda:' + str(get_gpu_memory_map()) if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def encode(self, x):
        x = self.encoder(x)

        mu = x[:, :self.D] 
        std = torch.exp(x[:, self.D:]) + 1e-7

        z = mu + torch.randn_like(mu) * std

        return mu, std, z

    def decode(self, z):
        x = self.decoder(z)

        return self.sigmoid(x)

    def forward(self, x):
        mu, std, z = self.encode(x)
        out = self.decode(z) #(batch_size, 784)

        return mu, std, out

    def sample(self, n_samples=144):
        '''
        This function generates samples from the underlying generative model p(x,z).
        First z ~ p(z) (in this case, a N(0,1)), and then x ~ p(x|z).
        Note that in the end we are only interested in x.
        '''

        with torch.no_grad():
            z = torch.randn((n_samples, self.D), device=self.device)

            out = self.decode(z)
            samples = torch.bernoulli(out)
            
            return samples.view(n_samples, 1, 28, 28)
        
    def generate_manifold(self, width=20):
        assert self.D == 2, 'The latent space dimension must be 2 for the manifold to be visualized graphically!'
        
        with torch.no_grad():
            z = torch.zeros((width ** 2, self.D), device=self.device)

            x, y = np.meshgrid(np.linspace(-2, 2, width), np.linspace(2, -2, width))
            z[:, 0] = torch.from_numpy(x.flatten()).float().to(self.device)
            z[:, 1] = torch.from_numpy(y.flatten()).float().to(self.device)

            out = self.decode(z)
            
            return out.view(width ** 2, 1, 28, 28)