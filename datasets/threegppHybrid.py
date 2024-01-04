from datasets.threegpp import ThreeGPP, crandn
from utils import dft_matrix
import numpy as np
import torch


class ThreeGPPHybrid(ThreeGPP):
    def __init__(self, paths, ant, rf, train=True, eval=True, seed=1234, dft=True, snr=None):
        """
        Input:
        :param paths (str)         : number of paths to consider (e.g. 1 for 1 path, 2-5 for two and five paths)
        :param ant (str)           : antenna array (e.g. 32rx for 32x1 channels)
        :param rf (int)            : number of RF chains
        :param train (bool)        : whether to use train dataset
        :param eval (bool)         : whether to use evaluation dataset
        :param seed (int)          : seed for random number (for reproducibility)
        :param dft (bool)          : bool to do dft
        :param snr (float)         : signal-to-noise ratio in samples
        """
        super().__init__(paths, ant, train, eval, seed, dft, snr)

        self.rf = rf
        self.A = None
        self.n = None

        self.rng_A = np.random.RandomState(seed)

        self.create_Amat()

        self.create_observations()

    def create_Amat(self):
        self.A = np.exp(1j*2*np.pi*self.rng_A.random((self.rf, self.rx))) / np.sqrt(self.rf)

    def create_observations(self, snr=None, A_perm=False):
        if snr is not None:
            self.snr_ar = snr * np.ones(len(self.data_raw))
        self.y, self.sigma, self.n = add_noise(self.data_raw, self.A, self.snr_ar, self.rng, get_both=True)

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


def add_noise(h, A, snr_dB, rng, get_sigmas=False, get_noise=False, get_both=False):
    r"""
    For every MxN-dimensional channel Hi of H, scale complex standard normal noise such that we have
        SNR = 1 / σ^2
    and compute the corresponding
        x_i + standard_gauss * σ.
    """
    # SNR = E[ || h_i ||^2 ] * / E[ || n ||^2 ] * M/N = 1 / σ^2
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
