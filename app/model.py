"""
A Convolutional Variational Autoencoder
"""
from torch import nn
import torch
from collections import OrderedDict
from collections.abc import Sequence
from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers import Act
import torch.nn.functional as F
from generative.networks.nets.vqvae import Encoder, Decoder


class VAE(nn.Module):
    def __init__(self,
                 spatial_dims: int = 2,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 num_channels=(256, 256),
                 num_res_channels=(96, 96, 192),
                 num_res_layers: int = 2,
                 latent_size: int = 256,
                 downsample_parameters=((2, 4, 1, 1), (2, 4, 1, 1)),
                 upsample_parameters=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
                 num_embeddings: int = 256,
                 latent_dim=256,
                 embedding_dim: int = 32,
                 ):
        super().__init__()

        dropout_prob = 0.5
        self.encoder = Encoder(spatial_dims, in_channels, out_channels, num_channels, num_res_layers, num_res_channels,
                               downsample_parameters, dropout_prob, act='relu')

        self.fc_mu = nn.Linear(56 * 56, latent_dim)
        self.fc_logvar = nn.Linear(56 * 56, latent_dim)

        self.fc3 = nn.Linear(latent_dim, 56 * 56)

        self.decoder = Decoder(spatial_dims, in_channels=1, out_channels=1,
                               num_channels=num_channels, num_res_layers=num_res_layers,
                               num_res_channels=num_res_channels,
                               upsample_parameters=upsample_parameters, act='relu', dropout=dropout_prob,
                               output_act='relu')

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        z = self.encoder(x)
        z = z.view(z.size(0), -1)
        mu = self.fc_mu(z)
        logvar = self.fc_logvar(z)
        _z = self.reparameterize(mu, logvar)
        x = self.fc3(_z)
        z_out = x.view(x.size(0), 1, 56, 56)
        out = self.decoder(z_out)
        return out, mu, logvar


def main():
    vae = VAE()

    a = torch.randn(1, 1, 224, 224)
    model = VAE()
    #x, mu, logvar = model(a)
    print(model.forward(a))


if __name__ == '__main__':
    main()
