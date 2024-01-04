import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class QuadrigaWideband(Dataset):
    def __init__(self, tc, p=20, train=True, eval=True, seed=1234, snr=None, losmixed='mixed'):
        """
        Input:
        :param tc (str)            : time-carrier resources (e.g. 12c-14t for 12 carriers, 14 time symbols)
        :param np (int)            : number of pilots
        :param train (bool)        : whether to use train or test dataset of same kind
        :param eval (bool)         : whether to use train or test dataset of same kind
        :param seed (int)          : seed for random number (for reproducibility)
        :param snr (float)         : signal-to-noise ratio in samples
        :param losmixed (str)      : use LOS or mixed LOS/NLOS data
        """

        self.rng = np.random.default_rng(seed)
        if train:
            file_end = '_train.npy'
        elif eval:
            file_end = '_eval.npy'
        else:
            file_end = '_test.npy'

        # load specified mat-files
        prefix = "./data/Quadriga_wideband/" + losmixed + "/" + tc + "/"
        self.data_raw = np.load(prefix + 'quadriga' + file_end)
        self.data_raw = self.data_raw[:1000]

        # assign parameters
        self.snr = snr
        if isinstance(snr, int) or isinstance(snr, float):
            self.snr_ar = snr * np.ones(len(self.data_raw))
        else:
            self.snr_ar = self.rng.uniform(self.snr[0] - 9, self.snr[1] + 9, len(self.data_raw))
        self.train = train

        # normalize data such that E[ ||h||^2 ] = N
        self.n = None
        self.sigma = None
        self.y = None
        self.nc = self.data_raw.shape[1]
        self.nt = self.data_raw.shape[2]
        self.p = p
        self.data_raw = np.reshape(self.data_raw, (len(self.data_raw), self.nc*self.nt), order='F')
        self.data_raw *= np.sqrt((self.nc*self.nt) / np.mean(np.linalg.norm(self.data_raw, axis=1)**2))

        # create pilot pattern
        self.pilots = np.zeros((self.nc, self.nt))
        for i in [0, 3, 7, 11]:
            for j in [0, 3, 6, 9, 13]:
                self.pilots[i, j] = 1
        self.pilots = np.reshape(self.pilots, (-1,), order='F')
        self.pilots_idx = np.nonzero(self.pilots)[0]
        self.A = np.zeros((self.p, self.nc*self.nt))
        for i in range(len(self.pilots_idx)):
            self.A[i, self.pilots_idx[i]] = 1

        # create observation based on snr
        self.create_observations()
        self.n_labels = self.y.shape[-1]

        self.data = self.data_raw[:, np.newaxis, ...]
        self.data = np.concatenate([self.data.real, self.data.imag], axis=1)
        self.n_dims = self.data.shape[-1]
        self.m = self.y.shape[-1]

    def __getitem__(self, index):

        y = self.y[index]
        h = self.data[index]
        n = self.n[index]

        # use noisy observation as condition
        cond = np.squeeze(np.copy(y))[np.newaxis, ...]
        cond = np.concatenate([cond.real, cond.imag], axis=0)

        # get noise level and label
        sigma = torch.tensor(self.sigma[index]).to(torch.float)

        # convert to tensors
        h_as_tensor = torch.tensor(h).to(torch.float)
        cond_as_tensor = torch.tensor(cond).to(torch.float)
        y_as_tensor = torch.tensor(y).to(torch.cfloat)
        n_as_tensor = torch.tensor(n).to(torch.cfloat)

        return h_as_tensor, cond_as_tensor, sigma, n_as_tensor, y_as_tensor, []

    def __len__(self):
        return len(self.data)

    def create_observations(self, snr=None):
        if snr is not None:
            self.snr_ar = snr * np.ones(len(self.data_raw))
        self.y, self.sigma, self.n = add_noise(self.data_raw, self.A, self.snr_ar, self.rng, get_both=True)


def add_noise(h, A, snr_dB, rng, get_sigmas=False, get_noise=False, get_both=False):
    r"""
    For every MxN-dimensional channel Hi of H, scale complex standard normal noise such that we have
        SNR = 1 / σ^2
    and compute the corresponding
        x_i + standard_gauss * σ.
    """
    # SNR = E[ || h_i ||^2 ] / E[ || n ||^2 ] * M/N = 1 / σ^2
    [M, N] = A.shape
    snr = 10 ** (snr_dB * 0.1)
    sigmas = 1 / np.sqrt(snr)
    sigmas = np.reshape(sigmas, (-1, 1))
    y_nf = h @ A.T
    n = crandn(y_nf.shape, rng) * sigmas
    if get_both:
        return y_nf + n, np.squeeze(sigmas), n
    elif get_sigmas:
        return y_nf + n, np.squeeze(sigmas)
    elif get_noise:
        return y_nf + n, n
    else:
        return y_nf + n


def crandn(shape, rng):
    real, imag = rng.normal(0, 1/np.sqrt(2), shape), rng.normal(0, 1/np.sqrt(2), shape)
    return real + 1j * imag
