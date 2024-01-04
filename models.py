import math
import torch
from utils import dft_matrix
import numpy as np
from torch import nn
from torch.nn import functional as F
from numpy import floor
from abc import abstractmethod
from typing import List, Any, TypeVar

Tensor = TypeVar('Tensor')
ndarray = TypeVar('ndarray')
dev = TypeVar('device')


def reparameterize(mu: Tensor, log_std: Tensor, **kwargs) -> Tensor:
    """
    Sample from std. gaussian and reparameterize with found parameters.
    :param mu: (Tensor) Mean of the latent Gaussian
    :param log_std: (Tensor) Standard deviation of the latent Gaussian
    :return:
    """
    B, M = mu.shape
    try:
        eps = torch.randn((B, M)).to(kwargs['device'])
    except KeyError:
        eps = torch.randn((B, M))
    std = torch.exp(log_std)
    return eps * std + mu, eps


def kl_div_diag_gauss(mu_p: Tensor, log_std_p: Tensor, mu_q: Tensor = None, log_std_q: Tensor = None):
    """Calculates the KL divergence between the two diagonal Gaussians p and q, i.e., KL(p||q)"""
    var_p = torch.exp(2 * log_std_p)
    if mu_q is not None:
        var_q = torch.exp(2 * log_std_q)
        kl_div = 0.5 * (torch.sum(2 * (log_std_q - log_std_p) + (mu_p - mu_q) ** 2 / (var_q + 1e-8)
                                  + var_p / var_q, dim=1) - mu_p.shape[1])
    else:
        # KL divergence for q=N(0,I)
        kl_div = 0.5 * (torch.sum(-2 * log_std_p + mu_p ** 2 + var_p, dim=1) - mu_p.shape[1])
    return kl_div


def get_activation(act_str):
    return {
        'relu': nn.ReLU(),
        'lrelu': nn.LeakyReLU(),
        'tanh': nn.Tanh()
    }[act_str]


class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, data: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, latent_code: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise RuntimeWarning()

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Tensor, **kwargs) -> Tensor:
        pass


class VAECircCov(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 stride: int,
                 kernel_szs: List,
                 latent_dim: int,
                 hidden_dims: List = None,
                 input_size: int = 16,
                 act: str = 'relu',
                 device: dev = 'cpu',
                 cond_as_input: int = 1) -> None:
        super(VAECircCov, self).__init__()

        self.latent_dim = latent_dim
        self.input_size = input_size
        self.in_channels = in_channels
        self.pad = 1
        self.stride = stride
        self.act = get_activation(act)
        self.device = device
        self.cond_as_input = cond_as_input
        self.pi = torch.tensor(math.pi).to(self.device)
        self.lambda_z = torch.tensor(1, device=self.device)
        self.N = torch.tensor(self.input_size, device=self.device)

        modules = []
        if hidden_dims is None:
            hidden_dims = [8, 16, 32]
        if kernel_szs is None:
            kernel_szs = [3 for _ in range(len(hidden_dims))]

        # calculate encoder output dims
        conv_out = []
        tmp_size = self.input_size
        for i in range(len(hidden_dims)):
            tmp_size = floor((tmp_size + 2 * self.pad - kernel_szs[i]) / self.stride + 1).astype(int)
            if tmp_size < 1:
                hidden_dims = hidden_dims[:i]
                kernel_szs = kernel_szs[:i]
                break
            self.pre_latent = tmp_size
            conv_out.append(tmp_size)
        self.hidden_dims = hidden_dims

        self.embed_data = nn.Conv1d(in_channels, hidden_dims[0], kernel_size=1)

        # build encoder
        in_channels = hidden_dims[0]
        for (i, h_dim) in enumerate(hidden_dims):
            modules.append(
                nn.Sequential(
                    nn.Conv1d(in_channels,
                              out_channels=h_dim,
                              kernel_size=kernel_szs[i],
                              stride=self.stride,
                              padding=self.pad),
                    # BatchNormMean1d(h_dim),
                    # nn.LayerNorm([h_dim, conv_out[i]]),
                    nn.BatchNorm1d(h_dim),
                    self.act)
            )
            in_channels = h_dim
        self.fc_mu_var = nn.Linear(self.pre_latent * hidden_dims[-1], 2 * self.latent_dim)

        self.encoder = nn.Sequential(*modules)

        # build decoder
        modules = []

        hidden_dims.reverse()
        kernel_szs.reverse()

        # calculate decoder output dims
        conv_out = []
        self.pre_out = self.pre_latent
        for i in range(len(hidden_dims)):
            self.pre_out = (self.pre_out - 1) * self.stride - 2 * self.pad + kernel_szs[i]
            conv_out.append(self.pre_out)

        self.decoder_input = nn.Linear(self.latent_dim, hidden_dims[0] * self.pre_latent)

        for i in range(len(hidden_dims)):
                modules.append(
                    nn.Sequential(
                        nn.ConvTranspose1d(hidden_dims[i],
                                           hidden_dims[i + 1] if i < len(hidden_dims) - 1 else 3,
                                           kernel_size=kernel_szs[i],
                                           stride=self.stride,
                                           padding=self.pad),
                        # BatchNormMean1d(hidden_dims[i + 1] if i < len(hidden_dims)-1 else 3),
                        nn.BatchNorm1d(hidden_dims[i + 1] if i < len(hidden_dims) - 1 else 3),
                        # nn.LayerNorm([hidden_dims[i+1] if i < len(hidden_dims) - 1 else 3, conv_out[i]]),
                        self.act)
                )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Linear(3 * self.pre_out, 3 * self.input_size)

        self.F = dft_matrix(self.input_size).to(self.device)

        self.apply(weights_init)

    def encode(self, data: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network and returns the latent codes.
        :param data: (Tensor) Input tensor to encoder [B x 2 x N]
        :return: (Tensor) List with mean and variance vector assuming a diagonal gaussian posterior
        """
        result = self.embed_data(data)
        result = self.encoder(result).flatten(start_dim=1)
        # Split the result into mu and var components of the latent Gaussian distribution
        [mu, log_std] = self.fc_mu_var(result).chunk(2, dim=1)
        return [mu, log_std]

    def decode(self, z: Tensor, **kwargs) -> Tensor:
        result = z
        # shape result to fit dimensions after all conv layers of encoder and decode
        result = self.decoder_input(result)
        result = result.view(len(result), -1, self.pre_latent)
        result = self.decoder(result)
        # flatten result and put into final layer to predict [mu_real, mu_imag, c_diag]
        result = torch.flatten(result, start_dim=1)
        result = self.final_layer(result)
        return result

    def forward(self, data: Tensor, **kwargs) -> List[Tensor]:
        if self.cond_as_input:
            encoder_input = kwargs['cond']
        else:
            encoder_input = data
        data = data[:, 0, :] + 1j * data[:, 1, :]

        mu_enc, log_std_enc = self.encode(encoder_input)

        if kwargs['train']:
            z, eps = reparameterize(mu_enc, log_std_enc, device=self.device)
        else:
            z, eps = mu_enc, torch.zeros_like(mu_enc).to(self.device)

        out = self.decode(z)
        mu_out_real, mu_out_imag, log_prec = out.chunk(3, dim=1)
        mu_out = mu_out_real + 1j * mu_out_imag

        if not kwargs['train']:
            c_diag = torch.diag_embed(torch.exp(-log_prec)).type(torch.complex64).to(self.device)
            C = self.F.conj().T @ c_diag @ self.F
            mu = mu_out @ self.F.conj()
        else:
            C, mu = None, None

        return [mu_out, data, log_prec, mu_enc, log_std_enc, z, mu, C]

    def loss_function(self, *args, **kwargs) -> dict:
        mu_out, data, log_prec, mu_enc, log_std_enc, z, mu, C = args

        rec_loss = torch.sum(log_prec - (log_prec.exp() * ((data - mu_out).abs() ** 2)), dim=1) \
            - self.N * torch.log(self.pi)

        kld_loss = kl_div_diag_gauss(mu_enc, log_std_enc)

        loss = rec_loss - kwargs['alpha'] * kld_loss
        loss_back = rec_loss.mean() - kwargs['alpha'] * torch.maximum(kld_loss.mean(), self.lambda_z)

        return {'loss': loss, 'loss_back': loss_back, 'rec_loss': rec_loss, 'KLD': kld_loss}


class VAECircCovReal(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 stride: int,
                 kernel_szs: List,
                 latent_dim: int,
                 hidden_dims: List = None,
                 input_size: int = 16,
                 act: str = 'relu',
                 device: dev = 'cpu') -> None:
        super(VAECircCovReal, self).__init__()

        self.latent_dim = latent_dim
        self.input_size = input_size
        self.in_channels = in_channels
        self.pad = 1
        self.stride = stride
        self.act = get_activation(act)
        self.device = device
        self.pi = torch.tensor(math.pi).to(self.device)
        self.lambda_z = torch.tensor(0.1, device=self.device)
        self.M = torch.tensor(self.input_size, device=self.device)

        self.embed_data = nn.Conv1d(in_channels, hidden_dims[0], kernel_size=1)

        modules = []
        if hidden_dims is None:
            hidden_dims = [8, 16, 32, 64]
        self.hidden_dims = hidden_dims
        if kernel_szs is None:
            kernel_szs = [3 for _ in range(len(hidden_dims))]

        # calculate encoder output dims
        tmp_size = self.input_size
        for i in range(len(hidden_dims)):
            tmp_size = floor((tmp_size + 2 * self.pad - kernel_szs[i]) / self.stride + 1).astype(int)
            if tmp_size < 1:
                hidden_dims = hidden_dims[:i]
                kernel_szs = kernel_szs[:i]
                break
            self.pre_latent = tmp_size

        # build encoder
        in_channels = hidden_dims[0]
        for (i, h_dim) in enumerate(hidden_dims):
            modules.append(
                nn.Sequential(
                    nn.Conv1d(in_channels,
                              out_channels=h_dim,
                              kernel_size=kernel_szs[i],
                              stride=self.stride,
                              padding=self.pad),
                    nn.BatchNorm1d(h_dim),
                    self.act)
            )
            in_channels = h_dim
        self.fc_mu_var = nn.Linear(self.pre_latent * hidden_dims[-1], 2 * self.latent_dim)

        self.encoder = nn.Sequential(*modules)
        in_channels = h_dim

        # build decoder
        modules = []

        hidden_dims.reverse()
        kernel_szs.reverse()

        self.decoder_input = nn.Linear(self.latent_dim, hidden_dims[0] * self.pre_latent)

        for i in range(len(hidden_dims)):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(hidden_dims[i],
                                       hidden_dims[i + 1] if i < len(hidden_dims) - 1 else 3,
                                       kernel_size=kernel_szs[i],
                                       stride=self.stride,
                                       padding=self.pad),
                    nn.BatchNorm1d(hidden_dims[i + 1] if i < len(hidden_dims) - 1 else 3),
                    self.act)
            )

        self.decoder = nn.Sequential(*modules)

        # calculate decoder output dims
        self.pre_out = self.pre_latent
        for i in range(len(hidden_dims)):
            self.pre_out = (self.pre_out - 1) * self.stride - 2 * self.pad + kernel_szs[i]

        self.final_layer = nn.Linear(3 * self.pre_out, 3 * self.input_size)

        self.F = dft_matrix(self.input_size).to(self.device)

        self.apply(weights_init)

    def encode(self, data: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network and returns the latent codes.
        :param data: (Tensor) Input tensor to encoder [B x 2 x N]
        :return: (Tensor) List with mean and variance vector assuming a diagonal gaussian posterior
        """
        result = self.embed_data(data)
        result = self.encoder(result).flatten(start_dim=1)
        # Split the result into mu and var components of the latent Gaussian distribution
        [mu, log_std] = self.fc_mu_var(result).chunk(2, dim=1)
        return [mu, log_std]

    def decode(self, z: Tensor, **kwargs) -> Tensor:
        result = z
        # shape result to fit dimensions after all conv layers of encoder and decode
        result = self.decoder_input(result)
        result = result.view(len(result), -1, self.pre_latent)
        result = self.decoder(result)
        # flatten result and put into final layer to predict [mu_real, mu_imag, c_diag]
        result = torch.flatten(result, start_dim=1)
        result = self.final_layer(result)
        return result

    def forward(self, data: Tensor, **kwargs) -> List[Tensor]:
        encoder_input = kwargs['cond']
        data = data[:, 0, :] + 1j * data[:, 1, :]
        cond = kwargs['cond'][:, 0, :] + 1j * kwargs['cond'][:, 1, :]

        mu_enc, log_std_enc = self.encode(encoder_input)

        if kwargs['train']:
            z, eps = reparameterize(mu_enc, log_std_enc, device=self.device)
        else:
            z, eps = mu_enc, torch.zeros_like(mu_enc).to(self.device)

        out = self.decode(z)
        mu_out_real, mu_out_imag, log_var = out.chunk(3, dim=1)
        mu_out = mu_out_real + 1j * mu_out_imag

        if not kwargs['train']:
            sigma = kwargs['sigma'].unsqueeze(-1)
            var_h = torch.exp(log_var)
            var_y = var_h + (sigma ** 2)
            c_h_diag = torch.diag_embed(var_h).type(torch.cfloat).to(self.device)
            C_h = self.F.conj().T @ c_h_diag @ self.F
            c_y_diag = torch.diag_embed(var_y).type(torch.cfloat).to(self.device)
            C_y = self.F.conj().T @ c_y_diag @ self.F
            C = (C_h, C_y)
            mu = mu_out @ self.F.conj()
        else:
            C, mu = None, None

        return [mu_out, data, cond, log_var, mu_enc, log_std_enc, z, mu, C]

    def loss_function(self, *args, **kwargs) -> dict:
        mu_out, data, cond, log_var, mu_enc, log_std_enc, z_0, z, jacobians, mu, C = args

        sigma = kwargs['sigma'].unsqueeze(-1)
        var = log_var.exp() + (sigma ** 2)

        rec_loss = (torch.sum(-torch.log(var) - (((cond - mu_out).abs() ** 2) / var), dim=1)
                    - self.M * torch.log(self.pi))

        kld_loss = kl_div_diag_gauss(mu_enc, log_std_enc)

        loss = rec_loss - kwargs['alpha'] * kld_loss
        loss_back = rec_loss.mean() - kwargs['alpha'] * torch.maximum(kld_loss.mean(), self.lambda_z)

        return {'loss': loss, 'loss_back': loss_back, 'rec_loss': rec_loss, 'KLD': kld_loss}


class VAENoisyHybrid(VAECircCov):

    def __init__(self,
                 in_channels: int,
                 stride: int,
                 kernel_szs: List,
                 latent_dim: int,
                 hidden_dims: List = None,
                 input_size: int = 16,
                 act: str = 'relu',
                 device: dev = 'cpu',
                 N: int = None,
                 rf: int = None) -> None:
        super(VAENoisyHybrid, self).__init__(in_channels, stride, kernel_szs, latent_dim, hidden_dims,
                                             input_size, act, device, 1)

        self.N = N
        self.rf = rf
        self.red = self.rf / self.N
        self.A_size = np.array([self.rf, self.N])
        self.sq_rf = torch.sqrt(torch.tensor(self.rf)).to(self.device)

        self.F = dft_matrix(self.N).to(self.device)

        self.A_pre = nn.Parameter(2*torch.pi*torch.rand((self.rf, self.N)), requires_grad=True)
        self.register_buffer('A', torch.exp(1j*2*torch.pi*torch.rand((self.rf, self.N), device=device)) / self.sq_rf)

        self.final_layer = nn.Linear(3 * self.pre_out, 3 * self.N)

    def update_Amat(self):
        self.A = torch.exp(1j*2*torch.pi*torch.rand((self.rf, self.N), device=self.device)) / self.sq_rf

    def encode(self, data: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network and returns the latent codes.
        :param data: (Tensor) Input tensor to encoder [B x 2 x N]
        :return: (Tensor) List with mean and variance vector assuming a diagonal gaussian posterior
        """
        result = self.embed_data(data)
        result = self.encoder(result).flatten(start_dim=1)
        # Split the result into mu and var components of the latent Gaussian distribution
        [mu, log_std] = self.fc_mu_var(result).chunk(2, dim=1)
        return [mu, log_std]

    def decode(self, z: Tensor, **kwargs) -> Tensor:
        # shape result to fit dimensions after all conv layers of encoder and decode
        result = self.decoder_input(z)
        result = result.view(len(result), -1, self.pre_latent)
        result = self.decoder(result)
        # flatten result and put into final layer to predict [mu_real, mu_imag, c_diag]
        result = torch.flatten(result, start_dim=1)
        result = self.final_layer(result)
        return result

    def forward(self, data: Tensor, **kwargs) -> List[Tensor]:
        data = data[:, 0, :] + 1j * data[:, 1, :]
        y_nf = data @ self.A.T

        if 'n' in kwargs:
            encoder_input = y_nf + kwargs['n']
        else:
            snr = -19 + 68 * torch.rand((len(data), 1), device=self.device)
            encoder_input = y_nf + torch.randn_like(y_nf, dtype=torch.cfloat) / torch.sqrt(10 ** (snr / 10))
        encoder_input = encoder_input @ self.A.conj()

        encoder_input = torch.unsqueeze(encoder_input @ self.F.T, 1)
        encoder_input = torch.cat([encoder_input.real, encoder_input.imag], 1)

        mu_enc, log_std_enc = self.encode(encoder_input)

        if kwargs['train']:
            z, eps = reparameterize(mu_enc, log_std_enc, device=self.device)
        else:
            z, eps = mu_enc, torch.zeros_like(mu_enc).to(self.device)

        out = self.decode(z)
        mu_out_real, mu_out_imag, log_prec = out.chunk(3, dim=1)
        mu_out = mu_out_real + 1j * mu_out_imag

        if not kwargs['train']:
            c_diag = torch.diag_embed(torch.exp(-log_prec)).type(torch.cfloat).to(self.device)
            C = self.F.H @ c_diag @ self.F
            mu = mu_out @ self.F.conj()
        else:
            C, mu = None, None

        return [mu_out, data, log_prec, mu_enc, log_std_enc, z, None, mu, C]

    def loss_function(self, *args, **kwargs) -> dict:
        mu_out, data, log_prec, mu_enc, log_std_enc, z, _, mu, C = args
        data = data @ self.F.T

        rec_loss = torch.sum(log_prec - (log_prec.exp() * ((data-mu_out).abs()**2)), dim=1) - self.N*torch.log(self.pi)

        kld_loss = kl_div_diag_gauss(mu_enc, log_std_enc)

        loss = rec_loss - kwargs['alpha'] * kld_loss
        loss_back = rec_loss.mean() - kwargs['alpha'] * torch.maximum(kld_loss.mean(), self.lambda_z)

        return {'loss': loss, 'loss_back': loss_back, 'rec_loss': rec_loss, 'KLD': kld_loss}


class VAENoisyWideband(VAECircCov):

    def __init__(self,
                 in_channels: int,
                 stride: int,
                 kernel_szs: List,
                 latent_dim: int,
                 hidden_dims: List = None,
                 input_size: int = 16,
                 act: str = 'relu',
                 device: dev = 'cpu',
                 Np: int = None,
                 Nt: int = None,
                 Nc: int = None) -> None:
        super(VAENoisyWideband, self).__init__(in_channels, stride, kernel_szs, latent_dim, hidden_dims,
                                               input_size, act, device, 1)

        self.Np = Np
        self.Nt = Nt
        self.Nc = Nc
        self.N = Nc * Nt

        Q_nt = dft_matrix(2*Nt)[:, :Nt]
        Q_nc = dft_matrix(2*Nc)[:, :Nc]
        self.Q = torch.kron(Q_nt, Q_nc).to(self.device)

        # create pilot pattern
        self.pilots_idx = None
        self.c_grid, self.t_grid = np.meshgrid(np.arange(self.Nc), np.arange(self.Nt), indexing='ij')

        self.register_buffer('A', torch.zeros((self.Np, self.N), device=self.device, dtype=torch.cfloat))
        self.update_Amat()

        self.final_layer = nn.Linear(3 * self.pre_out, 2 * self.N + 4*self.Nc*self.Nt)

    def create_pilot_pattern(self):
        pilots = np.zeros((self.Nc, self.Nt))
        for i in [0, 3, 7, 11]:
            for j in [0, 3, 6, 9, 13]:
                pilots[i, j] = 1
        pilots = np.reshape(pilots, (-1,), order='F')
        self.pilots_idx = np.nonzero(pilots)[0]

    def update_Amat(self):
        self.create_pilot_pattern()
        self.A = torch.zeros((self.Np, self.N), device=self.device, dtype=torch.cfloat)
        for i in range(self.Np):
            self.A[i, self.pilots_idx[i]] = 1

    def encode(self, data: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network and returns the latent codes.
        :param data: (Tensor) Input tensor to encoder [B x 2 x N]
        :return: (Tensor) List with mean and variance vector assuming a diagonal gaussian posterior
        """
        # result = self.embed_data(data + A_net_out)
        result = self.embed_data(data)
        result = self.encoder(result).flatten(start_dim=1)
        # Split the result into mu and var components of the latent Gaussian distribution
        [mu, log_std] = self.fc_mu_var(result).chunk(2, dim=1)
        return [mu, log_std]

    def decode(self, z: Tensor, **kwargs) -> Tensor:
        # shape result to fit dimensions after all conv layers of encoder and decode
        result = self.decoder_input(z)
        result = result.view(len(result), -1, self.pre_latent)
        result = self.decoder(result)
        # flatten result and put into final layer to predict [mu_real, mu_imag, c_diag]
        result = torch.flatten(result, start_dim=1)
        result = self.final_layer(result)
        return result

    def forward(self, data: Tensor, **kwargs) -> List[Tensor]:
        data = data[:, 0, :] + 1j * data[:, 1, :]
        y_nf = data @ self.A.T

        if 'n' in kwargs:
            y = y_nf + kwargs['n']
        else:
            snr = -19 + 68 * torch.rand((len(data), 1), device=self.device)
            sigma = 1 / torch.sqrt(10 ** (snr / 10))
            y = y_nf + torch.randn_like(y_nf, dtype=torch.cfloat) * sigma
        y_red = y @ self.A
        encoder_input = y_red
        encoder_input = torch.unsqueeze(encoder_input, 1)
        encoder_input = torch.cat([encoder_input.real, encoder_input.imag], 1)

        mu_enc, log_std_enc = self.encode(encoder_input)

        if kwargs['train']:
            z, eps = reparameterize(mu_enc, log_std_enc, device=self.device)
        else:
            z, eps = mu_enc, torch.zeros_like(mu_enc).to(self.device)

        out = self.decode(z)
        # mu_out_real, mu_out_imag, log_prec = out.chunk(3, dim=1)
        # mu_out = mu_out_real + 1j * mu_out_imag
        #
        # if not kwargs['train']:
        #     c_diag = torch.diag_embed(torch.exp(-log_prec)).type(torch.cfloat).to(self.device)
        #     C = self.Q.H @ c_diag @ self.Q
        #     mu = mu_out @ self.Q.conj()
        # else:
        #     C, mu = None, None

        mu_out_real, mu_out_imag = out[:, :2*self.N].chunk(2, dim=1)
        mu_out = mu_out_real + 1j * mu_out_imag

        log_c = out[:, 2*self.N:]
        c_diag = torch.diag_embed(torch.exp(log_c)).type(torch.cfloat).to(self.device)
        mu = mu_out
        C = self.Q.H @ c_diag @ self.Q

        return [mu_out, data, log_c, mu_enc, log_std_enc, z, None, mu, C]

    def loss_function(self, *args, **kwargs) -> dict:
        mu_out, data, log_c, mu_enc, log_std_enc, z, _, mu, C = args

        h_min_mu = torch.unsqueeze(data - mu, -1)
        C += 1e-3 * torch.eye(C.shape[-1], device=self.device).unsqueeze(0)  # add for numerical stability

        try:
            L = torch.linalg.cholesky(C)
            l_diag = torch.diagonal(L, dim1=1, dim2=2).real
            exp_arg = torch.linalg.norm(torch.linalg.solve_triangular(L, h_min_mu, upper=False), dim=1) ** 2
            rec_loss = -self.N * torch.log(self.pi) - 2 * torch.sum(torch.log(l_diag), -1) - torch.squeeze(exp_arg)
        except torch.linalg.LinAlgError:
            print('LinAlgError!')
            C_eig = F.relu(torch.linalg.eigvalsh(C)) + 1e-6
            exp_arg = torch.real(h_min_mu.mH @ torch.linalg.solve(C, h_min_mu))
            rec_loss = -self.N * torch.log(self.pi) - torch.sum(torch.log(C_eig), -1) - torch.squeeze(exp_arg)

        kld_loss = kl_div_diag_gauss(mu_enc, log_std_enc)

        loss = rec_loss - kwargs['alpha'] * kld_loss
        loss_back = rec_loss.mean() - kwargs['alpha'] * torch.maximum(kld_loss.mean(), self.lambda_z)

        return {'loss': loss, 'loss_back': loss_back, 'rec_loss': rec_loss, 'KLD': kld_loss}


class VAERealHybrid(VAECircCovReal):

    def __init__(self,
                 in_channels: int,
                 stride: int,
                 kernel_szs: List,
                 latent_dim: int,
                 hidden_dims: List = None,
                 input_size: int = 16,
                 act: str = 'relu',
                 device: dev = 'cpu',
                 N: int = 128,
                 rf: int = 32) -> None:
        super(VAERealHybrid, self).__init__(in_channels, stride, kernel_szs, latent_dim, hidden_dims, input_size,
                                            act, device)

        self.N = N
        self.rf = rf
        self.red = self.rf / self.N
        self.A_size = np.array([self.rf, self.N])
        self.sq_rf = torch.sqrt(torch.tensor(self.rf)).to(self.device)

        self.F = dft_matrix(self.N).to(self.device)

        self.register_buffer('A', torch.exp(1j*2*torch.pi*torch.rand((self.rf, self.N), device=device)) / self.sq_rf)

        self.final_layer = nn.Linear(3 * self.pre_out, 3 * self.N)

    def update_Amat(self):
        self.A = torch.exp(1j*2*torch.pi*torch.rand((self.rf, self.N), device=self.device)) / self.sq_rf

    def encode(self, data: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network and returns the latent codes.
        :param data: (Tensor) Input tensor to encoder [B x 2 x N]
        :return: (Tensor) List with mean and variance vector assuming a diagonal gaussian posterior
        """
        result = self.embed_data(data)
        result = self.encoder(result).flatten(start_dim=1)
        # Split the result into mu and var components of the latent Gaussian distribution
        [mu, log_std] = self.fc_mu_var(result).chunk(2, dim=1)
        return [mu, log_std]

    def decode(self, z: Tensor, **kwargs) -> Tensor:
        # shape result to fit dimensions after all conv layers of encoder and decoder
        result = self.decoder_input(z)
        result = result.view(len(result), -1, self.pre_latent)
        result = self.decoder(result)
        # flatten result and put into final layer to predict [mu_real, mu_imag, c_diag]
        result = torch.flatten(result, start_dim=1)
        result = self.final_layer(result)
        return result

    def forward(self, data: Tensor, **kwargs) -> List[Tensor]:
        data = data[:, 0, :] + 1j * data[:, 1, :]
        y_nf = data @ self.A.T

        if 'n' in kwargs:
            encoder_input = y_nf + kwargs['n']
            sigma = kwargs['sigma']
        else:
            snr = -19 + 68 * torch.rand((len(data), 1), device=self.device, dtype=torch.float)
            sigma = 1 / torch.sqrt(10 ** (snr / 10))
            encoder_input = y_nf + torch.randn_like(y_nf, dtype=torch.cfloat) * sigma
        y = encoder_input
        encoder_input = encoder_input @ self.A.conj()
        encoder_input = torch.unsqueeze(encoder_input @ self.F.T, 1)
        encoder_input = torch.cat([encoder_input.real, encoder_input.imag], 1)

        mu_enc, log_std_enc = self.encode(encoder_input)

        if kwargs['train']:
            z, eps = reparameterize(mu_enc, log_std_enc, device=self.device)
        else:
            z, eps = mu_enc, torch.zeros_like(mu_enc).to(self.device)

        out = self.decode(z)
        mu_out_real, mu_out_imag, log_var = out.chunk(3, dim=1)
        mu_out = (mu_out_real + 1j * mu_out_imag) @ self.F.conj()

        var_h = torch.exp(log_var) + 1e-6
        c_h_diag = torch.diag_embed(var_h).type(torch.cfloat).to(self.device)
        C_h = self.F.H @ c_h_diag @ self.F
        C_y = None
        C = (C_h, C_y)
        mu = mu_out

        return [mu_out, data, y, sigma, var_h, mu_enc, log_std_enc, z, None, mu, C]

    def loss_function(self, *args, **kwargs) -> dict:
        mu_out, data, y, sigma, var_h, mu_enc, log_std_enc, z, _, mu_h, C = args
        C_h = C[0]

        sigma = sigma.view(-1, 1, 1)
        C_n = (sigma**2) * torch.eye(self.A.shape[0], dtype=torch.cfloat, device=self.device).unsqueeze(0)
        y_min_mu = torch.unsqueeze(y - mu_h @ self.A.T, -1)
        C_y = self.A @ C_h @ self.A.H + C_n

        try:  # try calculation over Cholesky
            L = torch.linalg.cholesky(C_y)
            l_diag = torch.diagonal(L, dim1=1, dim2=2).real
            exp_arg = torch.linalg.norm(torch.linalg.solve_triangular(L, y_min_mu, upper=False), dim=1) ** 2
            rec_loss = -self.rf * torch.log(self.pi) - 2 * torch.sum(torch.log(l_diag), -1) - torch.squeeze(exp_arg)
        except torch.linalg.LinAlgError:  # calculate inverse and EVs directly otherwise
            print('LinAlgError!')
            C_eig = F.relu(torch.linalg.eigvalsh(C_y)) + 1e-6
            exp_arg = torch.real(y_min_mu.mH @ torch.linalg.solve(C_y, y_min_mu))
            rec_loss = -self.rf * torch.log(self.pi) - torch.sum(torch.log(C_eig), -1) - torch.squeeze(exp_arg)

        kld_loss = kl_div_diag_gauss(mu_enc, log_std_enc)

        loss = rec_loss - kwargs['alpha'] * kld_loss
        loss_back = rec_loss.mean() - kwargs['alpha'] * torch.maximum(kld_loss.mean(), self.lambda_z)

        return {'loss': loss, 'loss_back': loss_back, 'rec_loss': rec_loss, 'KLD': kld_loss}


class VAERealWideband(VAECircCovReal):

    def __init__(self,
                 in_channels: int,
                 stride: int,
                 kernel_szs: List,
                 latent_dim: int,
                 hidden_dims: List = None,
                 input_size: int = 16,
                 act: str = 'relu',
                 device: dev = 'cpu',
                 Np: int = None,
                 Nt: int = None,
                 Nc: int = None) -> None:
        super(VAERealWideband, self).__init__(in_channels, stride, kernel_szs, latent_dim,
                                              hidden_dims, input_size, act, device)

        self.Np = Np
        self.Nt = Nt
        self.Nc = Nc
        self.N = Nc * Nt

        Q_nt = dft_matrix(2 * Nt)[:, :Nt]
        Q_nc = dft_matrix(2 * Nc)[:, :Nc]
        self.Q = torch.kron(Q_nt, Q_nc).to(self.device)

        # create pilot pattern
        self.pilots_idx = None
        self.c_grid, self.t_grid = np.meshgrid(np.arange(self.Nc), np.arange(self.Nt), indexing='ij')

        self.register_buffer('A', torch.zeros((self.Np, self.N), device=self.device, dtype=torch.cfloat))
        self.update_Amat()

        self.final_layer = nn.Linear(3 * self.pre_out, 2 * self.N + 4 * self.Nc * self.Nt)

    def create_pilot_pattern(self):
        pilots_c = np.concatenate([np.random.permutation(self.Nc-2)[:2] + 1, [0, self.Nc-1]])
        pilots_t = np.concatenate([np.random.permutation(self.Nt-2)[:3] + 1, [0, self.Nt-1]])
        pilots = np.zeros((self.Nc, self.Nt))
        for i in pilots_c:
            for j in pilots_t:
        # for i in [0, 3, 7, 11]:
        #     for j in [0, 3, 6, 9, 13]:
                pilots[i, j] = 1
        pilots = np.reshape(pilots, (-1,), order='F')
        self.pilots_idx = np.nonzero(pilots)[0]

    def update_Amat(self):
        self.create_pilot_pattern()
        self.A = torch.zeros((self.Np, self.N), device=self.device, dtype=torch.cfloat)
        for i in range(self.Np):
            self.A[i, self.pilots_idx[i]] = 1

    def encode(self, data: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network and returns the latent codes.
        :param data: (Tensor) Input tensor to encoder [B x 2 x N]
        :return: (Tensor) List with mean and variance vector assuming a diagonal gaussian posterior
        """
        result = self.embed_data(data)
        result = self.encoder(result).flatten(start_dim=1)
        # Split the result into mu and var components of the latent Gaussian distribution
        [mu, log_std] = self.fc_mu_var(result).chunk(2, dim=1)
        return [mu, log_std]

    def decode(self, z: Tensor, **kwargs) -> Tensor:
        # shape result to fit dimensions after all conv layers of encoder and decoder
        result = self.decoder_input(z)
        result = result.view(len(result), -1, self.pre_latent)
        result = self.decoder(result)
        # flatten result and put into final layer to predict [mu_real, mu_imag, c_diag]
        result = torch.flatten(result, start_dim=1)
        result = self.final_layer(result)
        return result, z

    def forward(self, data: Tensor, **kwargs) -> List[Tensor]:
        data = data[:, 0, :] + 1j * data[:, 1, :]
        # self.update_Amat()
        y_nf = data @ self.A.T

        if 'n' in kwargs:
            y = y_nf + kwargs['n']
            sigma = kwargs['sigma']
        else:
            snr = -19 + 68 * torch.rand((len(data), 1), device=self.device)
            sigma = 1 / torch.sqrt(10 ** (snr / 10))
            y = y_nf + torch.randn_like(y_nf, dtype=torch.cfloat) * sigma
        y_red = y @ self.A
        encoder_input = y_red
        encoder_input = torch.unsqueeze(encoder_input, 1)
        encoder_input = torch.cat([encoder_input.real, encoder_input.imag], 1)

        mu_enc, log_std_enc = self.encode(encoder_input)

        if kwargs['train']:
            z_0, eps = reparameterize(mu_enc, log_std_enc, device=self.device)
        else:
            z_0, eps = mu_enc, torch.zeros_like(mu_enc).to(self.device)

        out, z = self.decode(z_0)
        # mu_out_real, mu_out_imag, log_prec = out.chunk(3, dim=1)
        # mu_out = mu_out_real + 1j * mu_out_imag
        #
        # if not kwargs['train']:
        #     c_diag = torch.diag_embed(torch.exp(-log_prec)).type(torch.cfloat).to(self.device)
        #     C = self.Q.H @ c_diag @ self.Q
        #     mu = mu_out @ self.Q.conj()
        # else:
        #     C, mu = None, None

        mu_out_real, mu_out_imag = out[:, :2*self.N].chunk(2, dim=1)
        mu_out = mu_out_real + 1j * mu_out_imag

        log_c = out[:, 2*self.N:]
        c_diag = torch.diag_embed(torch.exp(log_c)).type(torch.cfloat).to(self.device)
        mu = mu_out
        C_h = self.Q.H @ c_diag @ self.Q
        C_y = None
        C = (C_h, C_y)

        return [mu_out, data, y, sigma, log_c, mu_enc, log_std_enc, z_0, z, None, mu, C]

    def loss_function(self, *args, **kwargs) -> dict:
        mu_out, data, y, sigma, var_h, mu_enc, log_std_enc, z_0, z, _, mu_h, C = args
        C_h = C[0]
        C_h += 1e-3 * torch.eye(C_h.shape[-1], device=self.device).unsqueeze(0)  # add for numerical stability

        sigma = sigma.view(-1, 1, 1)
        C_n = (sigma**2) * torch.eye(self.A.shape[0], dtype=torch.cfloat, device=self.device).unsqueeze(0)
        C_y = self.A @ C_h @ self.A.H + C_n
        y_min_mu = torch.unsqueeze(y - mu_h @ self.A.T, -1)

        try:  # try calculation over Cholesky
            L = torch.linalg.cholesky(C_y)
            l_diag = torch.diagonal(L, dim1=1, dim2=2).real
            exp_arg = torch.linalg.norm(torch.linalg.solve_triangular(L, y_min_mu, upper=False), dim=1) ** 2
            rec_loss = -self.Np * torch.log(self.pi) - 2 * torch.sum(torch.log(l_diag), -1) - torch.squeeze(exp_arg)
        except torch.linalg.LinAlgError:  # calculate inverse and EVs directly otherwise
            print('LinAlgError!')
            C_eig = F.relu(torch.linalg.eigvalsh(C_y)) + 1e-6
            exp_arg = torch.real(y_min_mu.mH @ torch.linalg.solve(C_y, y_min_mu))
            rec_loss = -self.Np * torch.log(self.pi) - torch.sum(torch.log(C_eig), -1) - torch.squeeze(exp_arg)

        kld_loss = kl_div_diag_gauss(mu_enc, log_std_enc)

        loss = rec_loss - kwargs['alpha'] * kld_loss
        loss_back = rec_loss.mean() - kwargs['alpha'] * torch.maximum(kld_loss.mean(), self.lambda_z)

        return {'loss': loss, 'loss_back': loss_back, 'rec_loss': rec_loss, 'KLD': kld_loss}


def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) \
            or isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.ConvTranspose2d) \
            or isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(0.0, 0.05)
        m.bias.data.normal_(0.0, 0.05)
