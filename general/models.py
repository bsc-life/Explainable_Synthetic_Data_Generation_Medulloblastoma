import torch
import torch.nn as nn

torch.manual_seed(0)


class VAE(nn.Module):
    def __init__(self, input_dim, mid_dim, features):
        super().__init__()
        self.input_dim = input_dim
        self.mid_dim = mid_dim
        self.features = features

        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=self.mid_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.mid_dim, out_features=self.features * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.features, out_features=self.mid_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.mid_dim, out_features=self.input_dim),
            nn.Tanh()
        )

    def reparametrize(self, mu, log_var):

        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            sample = mu + (eps * std)
        else:
            sample = mu
        return sample

    def forward(self, x, **kwargs):

        mu_logvar = self.encoder(x).view(-1, 2, self.features)
        mu = mu_logvar[:, 0, :]
        log_var = mu_logvar[:, 1, :]

        z = self.reparametrize(mu, log_var)
        reconstruction = self.decoder(z)

        return reconstruction, mu, log_var, z
