# import packages
import json
import numpy as np
import scipy as sp
import torch
import matplotlib.pyplot as plt
from datasets.threegppHybrid import ThreeGPPHybrid
from utils import rel_mse_np, compute_lmmse, dft_matrix, genie_omp
from models import VAENoisyHybrid
from models import VAERealHybrid


# set simulation parameters
ant = '128rx'  # 128 antennas at the BS
path = './models/'
path_t = './data/3GPP/' + ant + '/'
paths = '1'  # for 3GPP data
seed_train, seed_test = 585813, 54632
device = torch.device('cpu')
rf = 32
rf_str = str(rf) + 'rf'
fmt = ['%d', '%1.10f']
print("This is 3GPP hybrid with " + paths + " paths, " + ant + ", " + rf_str + '.')
snr_lim = [-10, 40]
snr_step = 5


# load training data
snr_ar = np.arange(snr_lim[0], snr_lim[1] + 1, snr_step)
data_train = ThreeGPPHybrid(paths, ant, rf, train=True, snr=10, dft=False, seed=seed_train)
data_test = ThreeGPPHybrid(paths, ant, rf, train=False, eval=False, dft=False, seed=seed_test, snr=0)
rel_mse_global_cov, rel_mse_omp, rel_mse_genie, rel_mse_noisy, rel_mse_real, rel_mse_genie_cov = [], [], [], [], [], []
rel_mse_ls, rel_mse_real_ign = [], []
A = data_test.A
A_tensor = torch.tensor(A, device=device).to(torch.cfloat)


# define global cov estimator
h_train = data_train.data_raw
C_global = 1 / len(h_train) * h_train.T @ h_train.conj()


# define genie-cov estimator if 3GPP data is used
t = np.load(path_t + 'scm3gpp_' + paths + '-path-cov-test.npy')
cov = [sp.linalg.toeplitz(t[i].conj()) for i in range(len(t))]
cov = torch.tensor(np.array(cov), device=device)[:1000]
mu = torch.zeros((len(cov), cov.shape[-1]), device=device).to(torch.cfloat)[:1000]


# define dictionary for genie-OMP
D = dft_matrix(2*data_test.rx, 'np')[:data_test.rx, :]


# define VAE-noisy
cf_name = 'config-vae_circ_hybrid_noisy-3gpp-fixA-' + ant + '-' + paths + 'p-' + rf_str + '.json'
with open(path + cf_name, "r") as f:
    cf = json.load(f)
kernel_szs = [cf['k'] for _ in range(cf['n_hid'])]
hidden_dims = []
ch_out = cf['ch']
for i in range(cf['n_hid']):
    hidden_dims.append(int(ch_out))
    ch_out *= cf['m']
input_size = h_train.shape[-1]
act = 'relu'
init = 'k_u'

vae_noisy = VAENoisyHybrid(in_channels=2, stride=cf['st'], kernel_szs=kernel_szs, latent_dim=cf['latent_dim'],
                           hidden_dims=hidden_dims, input_size=input_size, act=act, device=device,
                           N=input_size, rf=rf).eval()
m = 'best-vae_circ_hybrid-3gpp_hybrid-noisy-3gpp-fixA-' + ant + '-' + paths + 'p-' + rf_str + '.pt'
vae_noisy.load_state_dict(torch.load(path + m, map_location=device))


# define VAE-real that varies A during training
cf_name = 'config-vae_circ_real_hybrid_ignore_real-ignore-3gpp-varyA-' + ant + '-' + paths + 'p-' + rf_str + '.json'
with open(path + cf_name, "r") as f:
    cf = json.load(f)
hidden_dims.reverse()

vae_real = VAERealHybrid(in_channels=2, stride=cf['st'], kernel_szs=kernel_szs, latent_dim=cf['latent_dim'],
                         hidden_dims=hidden_dims, input_size=input_size, act=act, device=device,
                         N=input_size, rf=rf).eval()
m = 'best-vae_circ_real_hybrid_ignore-3gpp_hybrid-real-ignore-3gpp-varyA-' + ant + '-' + paths + 'p-' + rf_str + '.pt'
vae_real.load_state_dict(torch.load(path + m, map_location=device))


# define VAE-real that leaves A fixed during training
cf_name = 'config-vae_circ_real_hybrid_ignore_real-ignore-3gpp-fixA-' + ant + '-' + paths + 'p-' + rf_str + '.json'
with open(path + cf_name, "r") as f:
    cf = json.load(f)
hidden_dims.reverse()

vae_real_ign = VAERealHybrid(in_channels=2, stride=cf['st'], kernel_szs=kernel_szs, latent_dim=cf['latent_dim'],
                             hidden_dims=hidden_dims, input_size=input_size, act=act, device=device,
                             N=input_size, rf=rf).eval()
m = 'best-vae_circ_real_hybrid_ignore-3gpp_hybrid-real-ignore-3gpp-fixA-' + ant + '-' + paths + 'p-' + rf_str + '.pt'
vae_real_ign.load_state_dict(torch.load(path + m, map_location=device))


# iterate over SNR and evaluate every model
h_true = data_test.data_raw.reshape((len(data_test), -1), order='F')
h_true_tensor = torch.tensor(h_true, device=device).to(torch.cfloat)
h_tensor = torch.tensor(data_test.data, device=device).to(torch.float).to(device)
with torch.no_grad():
    for snr in snr_ar:
        print('Simulating SNR of %d dB.\n' % snr)

        # create new observations for current SNR
        data_test.create_observations(snr)
        sigma = data_test.sigma.copy()
        n = data_test.n.copy()
        y = h_true @ A.T + n
        y_tensor = torch.tensor(y, device=device).to(torch.cfloat)
        sigma_tensor = torch.tensor(sigma, device=device).to(torch.cfloat)
        n_tensor = torch.tensor(n, device=device).to(torch.cfloat)

        # calculate channel estimates with genie-OMP
        s_omp = genie_omp(A, D, y, h_true)
        h_omp = s_omp @ D.T
        rel_mse_omp.append(np.mean(rel_mse_np(h_true, h_omp)))

        # calculate channel estimates with global cov
        h_global_cov = np.zeros_like(h_true, dtype=complex)
        for i in range(len(h_true)):
            C_y = A @ C_global @ A.conj().T + (sigma[i] ** 2) * np.eye(A.shape[0], dtype=complex)
            h_global_cov[i] = C_global @ A.conj().T @ np.linalg.solve(C_y, y[i])
        rel_mse_global_cov.append(np.mean(rel_mse_np(h_true, h_global_cov)))

        # calculate channel estimates with VAE-noisy
        args_vae_noisy = vae_noisy(h_tensor, n=n_tensor, train=False)
        y_tensor_noisy = h_true_tensor @ vae_noisy.A.T + n_tensor
        mu_noisy, C_h_noisy = args_vae_noisy[-2], args_vae_noisy[-1]
        h_noisy = compute_lmmse(C_h_noisy, mu_noisy, y_tensor_noisy, sigma_tensor, None, vae_noisy.A, device).numpy()
        rel_mse_noisy.append(np.mean(rel_mse_np(h_true, h_noisy)))

        # calculate channel estimates with VAE-real (var A)
        args_vae_real = vae_real(h_tensor, n=n_tensor, sigma=sigma_tensor, train=False)
        y_tensor_real = h_true_tensor @ vae_real.A.T + n_tensor
        mu_real, C_h_real = args_vae_real[-2], args_vae_real[-1][0]
        h_real = compute_lmmse(C_h_real, mu_real, y_tensor_real, sigma_tensor, None, vae_real.A, device).numpy()
        rel_mse_real.append(np.mean(rel_mse_np(h_true, h_real)))

        # calculate channel estimates with VAE-real (fix A)
        args_vae_real_ign = vae_real_ign(h_tensor, n=n_tensor, sigma=sigma_tensor, train=False)
        y_tensor_real_ign = h_true_tensor @ vae_real_ign.A.T + n_tensor
        mu_real_ign, C_h_real_ign = args_vae_real_ign[-2], args_vae_real_ign[-1][0]
        h_real_ign = compute_lmmse(C_h_real_ign, mu_real_ign, y_tensor_real_ign, sigma_tensor, None,
                                   vae_real_ign.A, device).numpy()
        rel_mse_real_ign.append(np.mean(rel_mse_np(h_true, h_real_ign)))

        # calculate channel estimates with genie-cov
        h_genie_cov = compute_lmmse(cov, mu, y_tensor, sigma_tensor, None, A_tensor, device).numpy()
        rel_mse_genie_cov.append(np.mean(rel_mse_np(h_true, h_genie_cov)))


res_genie_cov = np.array([snr_ar, rel_mse_genie_cov]).T
print('genie-cov:\n', res_genie_cov)
plt.semilogy(snr_ar, rel_mse_genie_cov, 'x--b', label='CME')

res_global_cov = np.array([snr_ar, rel_mse_global_cov]).T
print('global-cov:\n', res_global_cov)
plt.semilogy(snr_ar, rel_mse_global_cov, '|--c', label='global')

res_omp = np.array([snr_ar, rel_mse_omp]).T
print('genie-OMP:\n', res_omp)
plt.semilogy(snr_ar, rel_mse_omp, '*--r', label='genie-OMP')

res_vae_noisy = np.array([snr_ar, rel_mse_noisy]).T
print('VAE-noisy:\n', res_vae_noisy)
plt.semilogy(snr_ar, rel_mse_noisy, '^-', color='orange', label='VAE-noisy')

res_vae_real = np.array([snr_ar, rel_mse_real]).T
print('VAE-real (var A):\n', res_vae_real)
plt.semilogy(snr_ar, rel_mse_real, '1-', color='purple', label='VAE-real (var A)')

res_vae_real_ign = np.array([snr_ar, rel_mse_real_ign]).T
print('VAE-real (fix A):\n', res_vae_real_ign)
plt.semilogy(snr_ar, rel_mse_real_ign, 's-k', markerfacecolor='none', label='VAE-real (fix A)')

plt.legend()
plt.title("Hybrid system with " + paths + " clusters, " + ant + ", " + rf_str)
plt.xlabel('SNR [dB]')
plt.ylabel('NMSE')
plt.tight_layout()
plt.grid(True)
plt.show()
