# Channel Estimation in Underdetermined Systems Utilizing Variational Autoencoders

This is the simulation code for the article:

M. Baur, N. Turan, B. Fesl, and W. Utschick, "Channel Estimation in Underdetermined Systems Utilizing Variational Autoencoders," *IEEE ICASSP,* 2024.

## Abstract
In this work, we propose to utilize a variational autoencoder (VAE) for channel estimation (CE) in underdetermined (UD) systems. The basis of the method forms a recently proposed concept in which a VAE is trained on channel state information (CSI) data and used to parameterize an approximation to the mean squared error (MSE)-optimal estimator. The contributions in this work extend the existing framework from fully-determined (FD) to UD systems, which are of high practical relevance. Particularly noteworthy is the extension of the estimator variant, which does not require perfect CSI during its offline training phase. This is a significant advantage compared to most other deep learning (DL)-based CE methods, where perfect CSI during the training phase is a crucial prerequisite. Numerical simulations for hybrid and wideband systems demonstrate the excellent performance of the proposed methods compared to related estimators.

## File Organization
Please download the data under this [link](https://syncandshare.lrz.de/getlink/fiNYD29zJxA6qnt2CRHdRZ/data). The password is `VAE-est-ud-2024!`. Afterward, place the `data` folder in the same directory as the `datasets` and `models` folders.
The executable files for reproducing the paper results are `eval_baselines_hybrid.py` and `eval_baselines_wideband.py`. The remaining files contain auxiliary functions and classes. The folder `models` contains the pre-trained model weights with corresponding config files.

## Implementation Notes
This code is written in _Python_. It uses the deep learning library _PyTorch_ and the _numpy_, _scipy_, _matplotlib_, and _json_ packages. The code was tested with the versions visible in the requirements file.

## Instructions
Run `eval_baselines_hybrid.py` to reproduce the hybrid system results from the paper or `eval_baselines_wideband.py` to reproduce the wideband system results. To this end, adapt the simulation parameters at the beginning of the file to your needs. Models are only available for the scenarios from the paper. Other scenario parameters will result in an error message.
