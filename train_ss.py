from net import *
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import scipy.io
import cv2
import numpy as np
import torch.nn.functional as F
import scipy.io as sio
from pytorch_msssim import ssim

def threshold_cut(grayImage, threshold_value=20):
    grayImage = cv2.normalize(grayImage, None, 0, 255, cv2.NORM_MINMAX)
    grayImage = np.uint8(grayImage)
    _, mask = cv2.threshold(grayImage, threshold_value, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    return mask, mask_inv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x, _, _ = read_video('video-confocal-gray-full-10.mp4', pts_unit='sec')
x = x[:512, 0::4, 0::4, 0].float().to(device)
x1 = x.permute(1, 2, 0).cpu().detach().numpy()
sio.savemat('10_tof.mat', {'meas': x1})
x = x.unsqueeze(0)

file_path = r"10_vol.mat"
try:
    mat = scipy.io.loadmat(file_path)
    y2 = mat['vol']

except Exception as e:
    print(f"Error reading .mat file: {e}")

y2 = np.array(y2)
y2 = np.transpose(y2, (2, 0, 1))
y2 = torch.tensor(y2).float().to(device)
y_img = torch.max(y2, 0)[0]
y_img = y_img / y_img.max()
mask, mask_inv = threshold_cut(y_img.cpu().detach().numpy(),)
y2 = y2.unsqueeze(0)
mask = torch.tensor(mask).float().to(device)
mask = mask.unsqueeze(0).unsqueeze(0)  # 变为 [1, 1, 64, 64]
mask = mask.expand(1, 512, 64, 64)     # 变为 [1, 512, 64, 64]
y_img_1 = torch.max(y2, 2)[0].squeeze(0)
y_img_1 = y_img_1 / y_img_1.max()
mask_1 = y_img_1 > 0.05
mask_1 = mask_1.clone().detach().float().to(device)  # 512, 64
mask_1 = mask_1.unsqueeze(1).unsqueeze(0)  
mask_1 = mask_1.expand(1, 512, 64, 64)    
y1 = y2 *mask * mask_1
y1 = y1 / y1.max()

def tenengrad_3d(vol):
    sobel_x = torch.tensor([
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    ], dtype=torch.float32, device=vol.device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        [[-2, -4, -2], [0, 0, 0], [2, 4, 2]],
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    ], dtype=torch.float32, device=vol.device).unsqueeze(0).unsqueeze(0)
    sobel_z = torch.tensor([
        [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
    ], dtype=torch.float32, device=vol.device).unsqueeze(0).unsqueeze(0)
    grad_x = F.conv3d(vol, sobel_x, padding=1)
    grad_y = F.conv3d(vol, sobel_y, padding=1)
    grad_z = F.conv3d(vol, sobel_z, padding=1)
    grad_value = 1.3*grad_x**2 + 1.3*grad_y**2 + grad_z**2
    grad_value = grad_value.mean()
    return grad_value

model = net().to(device)
criterion = nn.MSELoss(reduction="sum")
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
writer = SummaryWriter()
for batch in range(5001):
    model.train()
    vol = model(x)
    img = torch.max(vol, 1)[0].squeeze(0)
    vol = vol / vol.max()
    tof = FP(vol)
    tof = tof / tof.max()
    img = img / img.max()
    loss = criterion(vol, y1)*5e1 - tenengrad_3d(vol)*7e4 - ssim(tof, x, data_range=1.0)*1e2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    writer.add_scalar("Loss", loss, batch)
    writer.add_image("images",make_grid(img, 1), batch)
    if batch % 1000 == 0:
        torch.save(model, f"model_EX_{batch}.pth")

vol = model(x)
img = torch.max(vol, 1)[0].squeeze(0)
plt.imshow(img.cpu().detach().numpy(),cmap='hot')
plt.show()
