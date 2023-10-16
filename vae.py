import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()
        # 编码器
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # 解码器
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
    # 编码器
    def encoder(self, x):
        h = nn.ReLU()(self.fc1(x))
        h = nn.ReLU()(self.fc2(h))
        return self.fc31(h), self.fc32(h)
    # 换算z的公式
    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    # 解码器
    def decoder(self, z):
        h = nn.ReLU()(self.fc4(z))
        h = nn.ReLU()(self.fc5(h))
        return nn.Sigmoid()(self.fc6(h))
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var, z
