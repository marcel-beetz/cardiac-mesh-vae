import torch
import torch.nn.functional as F
import torch.distributions
from model.layers import ChebConv_Coma, Pool


class MeshVAE(torch.nn.Module):

    def __init__(self, dataset, config, downsample_matrices, upsample_matrices, adjacency_matrices, num_nodes):
        super(MeshVAE, self).__init__()
        self.n_layers = config['n_layers']
        self.filters = config['num_conv_filters']
        self.filters.insert(0, dataset.num_features)  # To get initial features per node
        self.K = config['polygon_order']
        self.z = config['z']
        self.downsample_matrices = downsample_matrices
        self.upsample_matrices = upsample_matrices
        self.adjacency_matrices = adjacency_matrices
        self.A_edge_index, self.A_norm = zip(*[ChebConv_Coma.norm(self.adjacency_matrices[i]._indices(),
                                                                  num_nodes[i]) for i in range(len(num_nodes))])
        self.cheb = torch.nn.ModuleList([ChebConv_Coma(self.filters[i], self.filters[i+1], self.K[i])
                                         for i in range(len(self.filters)-2)])
        self.cheb_dec = torch.nn.ModuleList([ChebConv_Coma(self.filters[-i-1], self.filters[-i-2], self.K[i])
                                             for i in range(len(self.filters)-1)])
        self.cheb_dec[-1].bias = None  # No bias for last convolution layer
        self.pool = Pool()
        self.enc_lin_mu_1 = torch.nn.Linear(self.downsample_matrices[-1].shape[0]*self.filters[-1], self.z)
        self.enc_lin_sigma_1 = torch.nn.Linear(self.downsample_matrices[-1].shape[0]*self.filters[-1], self.z)
        self.dec_lin = torch.nn.Linear(self.z, self.filters[-1]*self.upsample_matrices[-1].shape[1])
        self.reset_parameters()


    def forward(self, data):

        # Reshape input data
        x, _ = data.x, data.edge_index
        batch_size = data.num_graphs
        x = x.reshape(batch_size, -1, self.filters[0])

        # Pass through encoder
        mu, log_var = self.encoder(x)

        # Apply reparameterization trick
        z = self.reparameterize(mu, log_var)

        # Pass through decoder
        x = self.decoder(z)

        # Reshape output data
        reconstruction = x.reshape(-1, self.filters[0])

        return reconstruction, mu, log_var, z


    def encoder(self, x):

        for i in range(self.n_layers):
            x = F.relu(self.cheb[i](x, self.A_edge_index[i], self.A_norm[i]))
            x = self.pool(x, self.downsample_matrices[i])

        z_mu = x.reshape(x.shape[0], self.enc_lin_mu_1.in_features)
        z_mu = F.relu(self.enc_lin_mu_1(z_mu))

        z_sigma = x.reshape(x.shape[0], self.enc_lin_sigma_1.in_features)
        z_sigma = F.relu(self.enc_lin_sigma_1(z_sigma))

        return z_mu, z_sigma


    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        sample = mu + (eps * std)

        return sample


    def decoder(self, x):

        x = F.relu(self.dec_lin(x))
        x = x.reshape(x.shape[0], -1, self.filters[-1])

        for i in range(self.n_layers):
            x = self.pool(x, self.upsample_matrices[-i-1])
            x = F.relu(self.cheb_dec[i](x, self.A_edge_index[self.n_layers-i-1], self.A_norm[self.n_layers-i-1]))
        x = self.cheb_dec[-1](x, self.A_edge_index[-1], self.A_norm[-1])

        return x


    def reset_parameters(self):
        torch.nn.init.normal_(self.enc_lin_mu_1.weight, 0, 0.1)
        torch.nn.init.normal_(self.enc_lin_sigma_1.weight, 0, 0.1)
        torch.nn.init.normal_(self.dec_lin.weight, 0, 0.1)

