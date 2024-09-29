import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from scipy.fftpack import *
import cv2
import matplotlib.pyplot as plt
from torch.nn import functional as F
import os
import time
from fullarss_with_comAmp import fftshift,ifftshift

def propagation_ARSS_sfft(u_in, phaseh, phaseu, phasev,phaseh2 ,dtype=torch.complex64):
    """
    Propagate the input field u_in through the transfer function TF using FFT.
    通过使用FFT传播输入场u_in通过传递函数TF。
    """
    u = u_in * phaseu
    # u代表了输入场，phaseu代表了输入场的相位，这是一个虚拟汇聚光。

    U1 = fftshift(torch.fft.fftn(fftshift(u), dim=(-2, -1), norm='ortho'))

    Trans = fftshift(torch.fft.fftn(fftshift(phaseh), dim=(-2, -1), norm='ortho'))

    U2 = Trans * U1

    u1 = ifftshift(torch.fft.ifftn(ifftshift(U2), dim=(-2, -1), norm='ortho'))
    #下面 是我需要修改的
    #u_out = u1 * phasec
    u2=u1*phasev

    u_out=fftshift(torch.fft.fftn(fftshift(u2),dim=(-2,-1),norm='ortho'))
    u_out= u_out*phaseh2
    return u_out


def phase_generation_4(u_in, feature_size, wavelength, prop_dist, dtype=torch.complex64):
    """
    Propagate the input field u_in through the transfer function TF using FFT.
    """
    field_resolution = u_in.size()
    num_y, num_x = field_resolution[2], field_resolution[3]
    dy, dx = feature_size
    z1=prop_dist[0]
    z2=prop_dist[1]
    s = 1/3

    m = np.arange(-num_y / 2, num_y / 2)
    n = np.arange(-num_x / 2, num_x / 2)

    y = m * dy
    x = n * dx

    X, Y = np.meshgrid(x, y)

    # phaseh
    phaseh = np.exp(1j * np.pi / (wavelength * z1) * s * (X ** 2 + Y ** 2))
    phaseh = phaseh.reshape(1, 1, phaseh.shape[0], phaseh.shape[1])
    phaseh = torch.tensor(phaseh, dtype=dtype).to(u_in.device)
    
    # phaseu
    phaseu = np.exp(1j * np.pi / (wavelength * z1) * (s ** 2 - s) * (X ** 2 + Y ** 2))
    phaseu = phaseu.reshape(1, 1, phaseu.shape[0], phaseu.shape[1])
    phaseu = torch.tensor(phaseu, dtype=dtype).to(u_in.device)

    # phasec
    # phasec = np.exp(1j * wavelength * prop_dist + 1j * np.pi / (wavelength * prop_dist) * (1 - s) * (X ** 2 + Y ** 2)) / (1j * wavelength * prop_dist)
    #phasec = np.exp(1j * wavelength * prop_dist + 1j * np.pi / (wavelength * prop_dist) * (1 - s) * (X ** 2 + Y ** 2)) / (1j)
    #phasec = phasec.reshape(1, 1, phasec.shape[0], phasec.shape[1])
    #phasec = torch.tensor(phasec, dtype=dtype).to(u_in.device)
    phaseh2 = np.exp(1j * np.pi / (wavelength * z2) * s * (X ** 2 + Y ** 2))
    phaseh2 = phaseh2.reshape(1, 1, phaseh2.shape[0], phaseh2.shape[1])
    phaseh2 = torch.tensor(phaseh2, dtype=dtype).to(u_in.device)

    v= m*dy/3
    u= n*dx/3
    U,V = np.meshgrid(u,v)
    phasev =np.exp(1j*np.pi/(wavelength*z2)*s*(U**2+V**2))
    phasev = phasev.reshape(1,1,phasev.shape[0],phasev.shape[1])
    phasev = torch.tensor(phasev,dtype=dtype).to(u_in.device)




    return phaseh, phaseu,phasev,phaseh2

if __name__=="__main__":
    image_res = (1080, 1920)
    init_phase = (-0.5 + 1.0 * torch.rand(1, 1, *image_res))
    pad = torch.nn.ZeroPad2d((1920 // 2, 1920 // 2, 1080 // 2, 1080 // 2))
    field = pad(init_phase)

    feature_size = (0.000015, 0.000015)
    wavelength = 0.0005
    prop_dist = [0.1,0.1]
    phaseh, phaseu,phasev,phaseh2 = phase_generation_4(u_in=field, feature_size=feature_size, wavelength=wavelength, prop_dist=prop_dist)
    u_out = propagation_ARSS_sfft(field, phaseh, phaseu,phasev,phaseh2)
    print(u_out.shape)