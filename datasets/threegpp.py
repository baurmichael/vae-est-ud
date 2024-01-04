import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class ThreeGPP(Dataset):
    def __init__(self, paths, ant, train=True, eval=True, seed=1234, dft=True, snr=None):
        """
        Input:
        :param paths (str)         : number of paths to consider (e.g. 1-5 for 1 and 5 paths)
        :param ant (str)           : antenna array (e.g. 32rx for 32x1 channels)
        :param train (bool)        : whether to use train dataset
        :param eval (bool)         : whether to use evaluation dataset
        :param seed (int)          : seed for random number (for reproducibility)
        :param dft (bool)          : bool to do dft
        :param snr (float)         : signal-to-noise ratio in samples
        """

        self.rng = np.random.default_rng(seed)
        data_list = []
        if train:
            file_end = '-path-train.npy'
        elif eval:
            file_end = '-path-eval.npy'
        else:
            file_end = '-path-test.npy'

        # load specified mat-files
        prefix = "./data/3GPP/" + ant + "/"
        self.paths = paths.split('-')
        for i in range(len(self.paths)):
            file_load = prefix + 'scm3gpp_' + self.paths[i] + file_end
            data_list.append(np.load(file_load))

        # save fields
        self.data_raw = np.concatenate(data_list, axis=0)
        if (eval and not train) or (not eval and not train):
            self.data_raw = self.data_raw[:1000]
            snr = 20
        data_list.clear()
        self.paths = np.arange(len(self.paths))

        # normalize data such that E[ ||h||^2 ] = N
        self.rx = self.data_raw.shape[1]
        self.data_raw *= np.sqrt(self.rx / np.mean(np.linalg.norm(self.data_raw, axis=1)**2))

        # assign parameters
        self.snr = snr
        if isinstance(snr, int) or isinstance(snr, float):
            self.snr_ar = snr * np.ones(len(self.data_raw))
        else:
            self.snr_ar = self.rng.uniform(self.snr[0] - 9, self.snr[1] + 9, len(self.data_raw))
        self.dft = dft
        self.train = train

        # create observation based on snr
        self.y, self.sigma, self.n = add_noise(self.data_raw, self.snr_ar, self.rng, get_both=True)
        self.n_labels = self.y.shape[-1]

        # reshape real and imaginary part into different channels
        if self.dft:
            self.data = np.fft.fft(self.data_raw, axis=1) / np.sqrt(self.data_raw.shape[1])
            self.y = np.fft.fft(self.y, axis=-1) / np.sqrt(self.data_raw.shape[1])
        else:
            self.data = self.data_raw

        self.data = self.data[:, np.newaxis, ...]
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
        sigma = torch.tensor(self.sigma[index])
        # label = int(self.paths[index // int(len(self) / len(self.paths))])

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
        # else:
        #     self.snr_ar = self.rng.uniform(self.snr[0] - 9, self.snr[1] + 9, len(self.data_raw))
        self.y, self.sigma, self.n = add_noise(self.data_raw, self.snr_ar, self.rng, get_both=True)
        if self.dft:
            self.y = np.fft.fft(self.y, axis=-1) / np.sqrt(self.y.shape[-1])


def add_noise(h, snr_dB, rng, get_sigmas=False, get_noise=False, get_both=False):
    r"""
    For every MxN-dimensional channel Hi of H, scale complex standard normal noise such that we have
        SNR = 1 / σ^2
    and compute the corresponding
        x_i + standard_gauss * σ.
    """
    # SNR = E[ || h_i ||^2 ] / (M*σ^2)
    out_shape = h.shape
    snr = 10 ** (snr_dB * 0.1)
    sigmas = 1 / np.sqrt(snr)
    sigmas = np.reshape(sigmas, (-1, 1))
    n = crandn(out_shape, rng) * sigmas
    if get_both:
        return h + n, np.squeeze(sigmas), n
    elif get_sigmas:
        return h + n, np.squeeze(sigmas)
    elif get_noise:
        return h + n, n
    else:
        return h + n


def crandn(shape, rng):
    real, imag = rng.normal(0, 1/np.sqrt(2), shape), rng.normal(0, 1/np.sqrt(2), shape)
    return real + 1j * imag
