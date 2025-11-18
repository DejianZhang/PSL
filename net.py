import os
import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import lil_matrix, spdiags
import matplotlib.pyplot as plt
from torchvision import transforms 
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_video

c = 3e8
z_trim = 512
z_offset = 0
width = 1
bin_resolution = 33e-12
# sigma = 25
# s_lamda_limit = 100 * 0.8 / 63
# sampling_coeff = 3
isbackprop = 0 
isdiffuse = 1
snr = 1
M = 512
N = 64
range_ = M * c * bin_resolution

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def definePsf(U, V, slope):
    x = torch.linspace(-1, 1, 2 * U)
    y = torch.linspace(-1, 1, 2 * U)
    z = torch.linspace(0, 2, 2 * V)
    grid_z, grid_y, grid_x = torch.meshgrid(z, y, x)
    psf = torch.abs((4 * slope) ** 2 * (grid_x**2 + grid_y**2) - grid_z)
    psf = torch.where(psf == torch.min(psf, dim=0).values, 1, 0)
    psf = psf / torch.sum(psf[:, U, U])
    psf = psf / torch.norm(psf)
    psf = torch.roll(psf, shifts=(U, U), dims=(1, 2))
    return psf


def resamplingOperator(M):
    mtx = lil_matrix((M**2, M))
    x = np.arange(1, M**2 + 1)
    mtx[x - 1, np.ceil(np.sqrt(x)).astype(int) - 1] = 1
    mtx = spdiags(1 / np.sqrt(x), 0, M**2, M**2) @ mtx
    mtxi = mtx.T
    K = np.log2(M)
    for k in range(int(K)):
        mtx = 0.5 * (mtx[::2, :] + mtx[1::2, :])
        mtxi = 0.5 * (mtxi[:, ::2] + mtxi[:, 1::2])
    return mtx, mtxi

psf = definePsf(N, M, width / range_)
fpsf = torch.fft.fftn(psf)
invpsf = (
    torch.conj(fpsf)
    if isbackprop
    else torch.conj(fpsf) / (torch.abs(fpsf) ** 2 + 1 / snr)
).to(device)

mtx, mtxi = resamplingOperator(M)
mtx = torch.from_numpy(mtx.toarray()).float().to(device)
mtxi = torch.from_numpy(mtxi.toarray()).float().to(device)
    
def double_conv(in_channels, out_channels):
    return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

def double_conv_2D(in_channels, out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

class net(nn.Module):
    def __init__(self):
        super().__init__()
        self.dconv_down1 = double_conv(1, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)
        self.maxpool = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dconv_up4 = double_conv(256 + 512, 256)
        self.dconv_up3 = double_conv(128 + 256, 128)
        self.dconv_up2 = double_conv(64 + 128, 64)
        self.dconv_up1 = double_conv(1 + 64, 1)
        self.conv_out1 = double_conv_2D(512,512)
        self.conv_out2 = double_conv_2D(512,512)
        self.conv_out3 = double_conv_2D(512,512)

    def forward(self, x):
        x0 = x.unsqueeze(1) # [1, 1, 512, 64, 64]

        x = self.dconv_down1(x0) # [1, 64, 512, 64, 64]
        x1 = self.maxpool(x) # [1, 64, 256, 32, 32]

        x = self.dconv_down2(x1) # [1, 128, 256, 32, 32]
        x2 = self.maxpool(x) # [1, 128, 128, 16, 16]

        x = self.dconv_down3(x2) # [1, 256, 128, 16, 16]
        x3 = self.maxpool(x) # [1, 256, 64, 8, 8]

        x = self.dconv_down4(x3) # [1, 512, 64, 8, 8]

        x = torch.cat([x, x3], dim=1) # [1, 768, 64, 8, 8]
        x = self.dconv_up4(x) # [1, 256, 64, 8, 8]
        x = self.upsample(x) # [1, 256, 128, 16, 16]

        x = torch.cat([x, x2], dim=1) # [1, 384, 128, 16, 16]
        x = self.dconv_up3(x) # [1, 128, 128, 16, 16]
        x = self.upsample(x) # [1, 128, 256, 32, 32]

        x = torch.cat([x, x1], dim=1) # [1, 192, 256, 32, 32]
        x = self.dconv_up2(x) # [1, 64, 256, 32, 32]
        x = self.upsample(x) # [1, 64, 512, 64, 64]

        x = torch.cat([x, x0], dim=1) # [1, 65, 512, 64, 64]
        x = self.dconv_up1(x) # [1, 1, 512, 64, 64]

        x = x.squeeze(1) # [1, 512, 64, 64]
        x = self.conv_out1(x)  # [1, 512, 64, 64]
        x = self.conv_out2(x)   # [1, 512, 64, 64]
        x = self.conv_out3(x)   # [1, 512, 64, 64]

        return x


def FP_T(vol):
    batch_size = vol.shape[0]
    N = vol.shape[2]  
    M = vol.shape[1]  

    tvol = torch.zeros((batch_size, 2 * M, 2 * N, 2 * N)).to(device)
    tvol[:, :M, :N, :N] = torch.reshape(mtx @ vol.reshape(batch_size, M, -1), (batch_size, M, N, N))


    # vol_real = torch.real(vol)
    # vol_real_padded_pre = torch.nn.functional.pad(vol_real, (N//2-1, N//2-1, N//2-1, N//2-1, M//2, M//2), mode='constant', value=0)
    # tvol = torch.nn.functional.pad(vol_real_padded_pre, (0, 2, 0, 2, 0, 0), mode='constant', value=0)


    tvol = torch.fft.fftn(tvol)


    tdata = torch.fft.ifftn(tvol * invpsf_1).real
    tdata = tdata[:, :M, :N, :N]
    # tdata = tdata[:, M//2:3*M//2, N//2:3*N//2, N//2:3*N//2]

    tof = torch.reshape(mtxi @ tdata.reshape(batch_size, M, -1), (batch_size, M, N, N))

    tof = torch.flip(tof, [1])

    tof = torch.roll(tof, shifts=-60, dims=1)

    # grid_z = torch.tile(
    #     torch.linspace(0.1, 1, M)[np.newaxis, :, np.newaxis, np.newaxis], (batch_size, 1, N, N)
    # ).to(device)
    # tof = tof / (grid_z**4)

    # tof[:, 0:180, :, :] = 0
    return tof

def kde(x, data, bandwidth):

    x = x[:, None]

    data = data[None, :]
    kernel = torch.ones(512, requires_grad=True).to(device)
    kernel = kernel.requires_grad_(True)
    kernel = torch.exp(-0.5 * ((x - data) / bandwidth) ** 2)

    kernel = kernel.mean(dim=1)
    return kernel

def FP(vol):
    b, _, H, W = vol.shape
    X_voxel, Y_voxel, Z_voxel = width*2 / H, width*2 / W, bin_resolution * c/2

    P, Z = vol.max(dim=1)

    P, Z = P.flatten(), Z.flatten()
    Z = Z.reshape(b, H * W, 1, 1)
    P = P.reshape(b, H * W, 1, 1)
    Z = Z.expand(b, H * W, H, W)
    # P = P.expand(b, H * W, H, W)

    X, Y = torch.meshgrid(torch.arange(H).to(device), torch.arange(W).to(device))
    X = X - X.reshape(H * W, 1, 1)
    Y = Y - Y.reshape(H * W, 1, 1)
    X = X[None, :, :, :].expand(b, H * W, H, W)
    Y = Y[None, :, :, :].expand(b, H * W, H, W)

    h = torch.sqrt((X_voxel * X) ** 2 + (Y_voxel * Y) ** 2 + (Z_voxel * Z) ** 2)
    
    # D = torch.round(P * 2e1 * (Z_voxel * Z / h) ** 4 / h**4)
    # D = D.clamp(0, 10)

    steps = 512
    bin = torch.linspace(0, steps * bin_resolution, steps).to(device)
    tof_data_FP = torch.zeros(b, steps, H, W, requires_grad=True).to(device)
    

    # D = torch.where(D == 0, 0, 1)
    h = h
    t = h * 2 / c
    for i in range(b):
        for x in range(H):
            for y in range(W):
                # optical_path = torch.repeat_interleave(h[i ,:, x, y], D[i ,:, x, y], dim=0)
                # optical_path = torch.where(D[i ,:, x, y]==0, 0, h[i ,:, x, y])
                tof_data_FP[i, :, x, y] = kde(bin, t[i ,:, x, y], bin_resolution)*(1/(bin*c + 0.001))** 2

    return tof_data_FP



class MyDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.mp4_files = []
        self.png_files = []

        for root, _, files in os.walk(directory):
            for file in files:
                if file == "video-confocal-gray-full.mp4":
                    self.mp4_files.append(os.path.join(root, file))
                    depth_files = [
                        f for f in files if f.startswith("confocal") and f.endswith(".png")
                    ]
                    self.png_files.append(os.path.join(root, depth_files[0]))

    def __len__(self):
        return len(self.mp4_files)

    def __getitem__(self, idx):
        video_file = self.mp4_files[idx]
        video_data, _, _ = read_video(video_file, pts_unit="sec")
        n = 256//N
        video_data = video_data[:, 0::n, 0::n, 0]

        transform = transforms.Compose([transforms.Grayscale(), transforms.Resize((N, N)), transforms.ToTensor()])

        png_file = self.png_files[idx]
        png_data = Image.open(png_file)
        png_data = transform(png_data)

        return video_data, png_data

def save_slices_as_images(matrix, folder="frames"):
    matrix = matrix / matrix.max()

    if not os.path.exists(folder):
        os.makedirs(folder)

    for i in range(matrix.shape[0]):
        plt.imsave(f"{folder}/{i:04d}.png", matrix[i], cmap="gray", vmin=0, vmax=1)

psf = definePsf(N, M, width / range_)
fpsf = torch.fft.fftn(psf)
invpsf = (
    torch.conj(fpsf)
    if isbackprop
    else torch.conj(fpsf) / (torch.abs(fpsf) ** 2 + 1 / snr)
).to(device)
invpsf_1 =torch.conj(fpsf).to(device)

mtx, mtxi = resamplingOperator(M)
mtx = torch.from_numpy(mtx.toarray()).float().to(device)
mtxi = torch.from_numpy(mtxi.toarray()).float().to(device)

# vol = torch.randn(1, 512, 64, 64, requires_grad=True).to(device)

# tof = torch.ones(1, 512, 64, 64, requires_grad=True).to(device)

# tof = FP_LCT(vol, isdiffuse, invpsf, mtxi)

# print(tof.shape)