# import packages
import json
import numpy as np
import torch
from datasets.quadrigaWideband import QuadrigaWideband
from utils import rel_mse_np, compute_lmmse
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from models import VAENoisyWideband
from models import VAERealWideband


# set simulation parameters
tc = '12c-14t'  # 12 carriers and 14 time slots
Nc = int(tc[:2])
Nt = int(tc[-3:-1])
path = './models/'
seed_train, seed_test = 585813, 54632
device = torch.device('cpu')
Np = 20  # number of pilots
snr_lim = [-10, 40]
snr_step = 5
fmt = ['%d', '%1.10f']
print("This is Quadriga wideband mixed in " + tc + " config with " + str(Np) + " pilots.\n")


# load data
snr_ar = np.arange(snr_lim[0], snr_lim[1] + 1, snr_step)
data_train = QuadrigaWideband(tc, Np, train=True, snr=10, seed=seed_train)
data_test = QuadrigaWideband(tc, Np, train=False, eval=False, seed=seed_test, snr=0)
rel_mse_global_cov, rel_mse_omp, rel_mse_genie, rel_mse_noisy, rel_mse_real, rel_mse_lin_int = [], [], [], [], [], []
rel_mse_global_lin, rel_mse_real_ign = [], []
A = data_test.A
A_tensor = torch.tensor(A, device=device).to(torch.cfloat)


# define global cov estimator
h_train = data_train.data_raw.reshape((len(data_train), -1), order='F')
C_global = 1 / len(h_train) * h_train.T @ h_train.conj()


# prepare grid for interpolation
c_grid, t_grid = np.meshgrid(np.arange(Nc), np.arange(Nt), indexing='ij')
P = np.reshape(data_test.pilots, (Nc, Nt), order='F')
P_nz = np.nonzero(P)
grid = (np.unique(P_nz[0]), np.unique(P_nz[1]))


# define VAE-noisy
cf_name = 'config-vae_circ_wideband_noisy-quad-wide-fixA-' + tc + '-' + str(Np) + 'Np' + '.json'
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

vae_noisy = VAENoisyWideband(in_channels=2, stride=cf['st'], kernel_szs=kernel_szs, latent_dim=cf['latent_dim'],
                    hidden_dims=hidden_dims, input_size=input_size, act=act, device=device, Np=Np, Nt=Nt, Nc=Nc).eval()
m = 'best-vae_circ_wideband-wideband-noisy-quad-wide-fixA-' + tc + '-' + str(Np) + 'Np' + '.pt'
vae_noisy.load_state_dict(torch.load(path + m, map_location=device))
vae_noisy.A = torch.tensor(data_test.A, device=device, dtype=torch.cfloat)


# define VAE-real that varies A during training
cf_name = 'config-vae_circ_real_wideband_ignore_real-ignore-quad-wide-varyA-' + tc + '-' + str(Np) + 'Np' + '.json'
with open(path + cf_name, "r") as f:
    cf = json.load(f)
hidden_dims.reverse()

vae_real = VAERealWideband(in_channels=2, stride=cf['st'], kernel_szs=kernel_szs, latent_dim=cf['latent_dim'],
                           hidden_dims=hidden_dims, input_size=input_size, act=act, device=device,
                           Np=Np, Nt=Nt, Nc=Nc).eval()
m = 'best-vae_circ_real_wideband_ignore-wideband-real-ignore-quad-wide-varyA-' + tc + '-' + str(Np) + 'Np' + '.pt'
vae_real.load_state_dict(torch.load(path + m, map_location=device))


# define VAE-real that leaves A fixed during training
cf_name = 'config-vae_circ_real_wideband_ignore_real-ignore-quad-wide-fixA-' + tc + '-' + str(Np) + 'Np' + '.json'
with open(path + cf_name, "r") as f:
    cf = json.load(f)
hidden_dims.reverse()

vae_real_ign = VAERealWideband(in_channels=2, stride=cf['st'], kernel_szs=kernel_szs, latent_dim=cf['latent_dim'],
                               hidden_dims=hidden_dims, input_size=input_size, act=act, device=device,
                               Np=Np, Nt=Nt, Nc=Nc).eval()
m = 'best-vae_circ_real_wideband_ignore-wideband-real-ignore-quad-wide-fixA-' + tc + '-' + str(Np) + 'Np' + '.pt'
vae_real_ign.load_state_dict(torch.load(path + m, map_location=device))
vae_real_ign.A = torch.tensor(data_test.A, device=device, dtype=torch.cfloat)


# iterate over SNR and evaluate every model
h_true = data_test.data_raw.reshape((len(data_test), -1), order='F')
h_true_tensor = torch.tensor(h_true, device=device).to(torch.cfloat)
h_tensor = torch.tensor(data_test.data, device=device).to(torch.float).to(device)
with torch.no_grad():
    for snr in snr_ar:
        print('Simulating SNR of %d dB.\n' % snr)

        # create new observations for current SNR
        data_test.create_observations(snr)
        y = data_test.y.reshape((len(data_test), -1), order='F').copy()
        y_tensor = torch.tensor(y, device=device).to(torch.cfloat)
        sigma = data_test.sigma.copy()
        sigma_tensor = torch.tensor(sigma, device=device).to(torch.cfloat)
        n = data_test.n.copy()
        n_tensor = torch.tensor(n, device=device).to(torch.cfloat)

        # calculate estimate with linear interpolation
        Y_red = np.reshape(y @ A, (-1, Nc, Nt), order='F')
        y_red = np.reshape(y, (-1, len(grid[0]), len(grid[1])), order='F')
        Y_lin = np.zeros_like(Y_red)
        for i in range(len(Y_red)):
            interp = RegularGridInterpolator(grid, y_red[i].astype(np.complex128), method='linear', fill_value=None)
            Y_lin[i] = interp((c_grid, t_grid))
        y_lin = np.reshape(Y_lin, (-1, Nc*Nt), order='F')
        rel_mse_lin_int.append(np.mean(rel_mse_np(h_true, y_lin)))

        # calculate channel estimates with global cov
        h_global_cov = np.zeros_like(h_true, dtype=complex)
        for i in range(len(h_true)):
            C_y = A @ C_global @ A.conj().T + (sigma[i] ** 2) * np.eye(A.shape[0], dtype=complex)
            h_global_cov[i] = C_global @ A.conj().T @ np.linalg.solve(C_y, y[i])
        rel_mse_global_cov.append(np.mean(rel_mse_np(h_true, h_global_cov)))

        # calculate channel estimates with global cov based on linearly interpolated observations
        C_lin = 1 / len(y_lin) * y_lin.T @ y_lin.conj()
        e, U = np.linalg.eigh(C_lin - (sigma[i] ** 2) * np.eye(C_lin.shape[-1], dtype=complex))
        e[e < 0] = 0
        C_lin = U @ np.diag(e) @ U.conj().T
        h_global_lin = np.zeros_like(h_true, dtype=complex)
        for i in range(len(h_true)):
            C_y = A @ C_lin @ A.conj().T + (sigma[i] ** 2) * np.eye(A.shape[0], dtype=complex)
            h_global_lin[i] = C_lin @ A.conj().T @ np.linalg.solve(C_y, y[i])
        rel_mse_global_lin.append(np.mean(rel_mse_np(h_true, h_global_lin)))

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
        h_real_ign = compute_lmmse(C_h_real_ign, mu_real_ign, y_tensor_real_ign, sigma_tensor, None, vae_real_ign.A, device).numpy()
        rel_mse_real_ign.append(np.mean(rel_mse_np(h_true, h_real_ign)))



res_global_cov = np.array([snr_ar, rel_mse_global_cov]).T
print('global-cov:\n', res_global_cov)
plt.semilogy(snr_ar, rel_mse_global_cov, '|--g', label='global')

res_lin_int = np.array([snr_ar, rel_mse_lin_int]).T
print('lin-int:\n', res_lin_int)
plt.semilogy(snr_ar, rel_mse_lin_int, 'o--c', label='LI')

res_global_lin = np.array([snr_ar, rel_mse_global_lin]).T
print('global-cov-lin:\n', res_global_lin)
plt.semilogy(snr_ar, rel_mse_global_lin, 'o--', color='brown', markerfacecolor='none', label='global LI')

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
plt.title("Wideband system in " + tc + " config with " + str(Np) + " pilots")
plt.xlabel('SNR [dB]')
plt.ylabel('NMSE')
plt.tight_layout()
plt.grid(True)
plt.show()
