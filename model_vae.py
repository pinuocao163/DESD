import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEModel(nn.Module):
    def __init__(self):
        super(VAEModel, self).__init__()
        os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dropout = nn.Dropout(0.2)
        self.vae = None
        self.optimizer = None
        self.train_loader = None
        self.test_loader = None
        self.digit_size = 28

    def loss_function(self, recon_x, x, mu, log_var):
        x = x.detach()
        recon_x = self.dropout(torch.sigmoid(recon_x))
        bce = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return bce + kld

    class VAE(nn.Module):
        def __init__(self, input_size, hidden_size1, hidden_size2, latent_size):
            super(VAEModel.VAE, self).__init__()
            # Encoder layers
            self.fc1 = nn.Linear(input_size, hidden_size1)
            self.fc2 = nn.Linear(hidden_size1, hidden_size2)
            self.fc3_mean = nn.Linear(hidden_size2, latent_size)
            self.fc3_log_var = nn.Linear(hidden_size2, latent_size)

            # Decoder layers
            self.fc4 = nn.Linear(latent_size, hidden_size2)
            self.fc5 = nn.Linear(hidden_size2, hidden_size1)
            self.fc6 = nn.Linear(hidden_size1, input_size)

        def encode(self, x):
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            return self.fc3_mean(h), self.fc3_log_var(h)

        def reparameterize(self, mu, log_var):
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z):
            h = F.relu(self.fc4(z))
            h = F.relu(self.fc5(h))
            return torch.sigmoid(self.fc6(h))

        def forward(self, x):
            mu, log_var = self.encode(x.view(-1, 784))
            z = self.reparameterize(mu, log_var)
            return self.decode(z), mu, log_var, z

    def create_model(self):
        self.vae = self.VAE(784, 512, 256, 128).to(self.device)
        if torch.cuda.device_count() > 1:
            self.vae = nn.DataParallel(self.vae)
        self.optimizer = torch.optim.SGD(self.vae.parameters(), lr=1e-4, weight_decay=5e-4)

    def train_model(self, window_):
        loss_ = 10000000
        z_ = 0
        for epoch in range(100):
            self.vae.train()
            recon_batch, mu, log_var, z = self.vae(window_)
            loss = self.loss_function(recon_batch, window_, mu, log_var)
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()
            if loss_ > loss:
                loss_ =loss
                z_ = z
        return z_

    def forward(self, window_):
        vae_model = self.create_model()
        z = self.train_model(window_)
        return z

