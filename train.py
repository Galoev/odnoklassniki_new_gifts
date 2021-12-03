from dataset import DatasetNYUv2
from unet_model import UNet
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import time
from torch.autograd import Variable
import numpy as np
from dataset.constants import PATH_TO_NYU
import math

class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()
    # L1 norm
    def forward(self, grad_fake, grad_real):
        return torch.mean( torch.abs(grad_real-grad_fake) )

def bce_loss(y_real, y_pred):
  loss=y_pred-y_real*y_pred+torch.log(1+torch.exp(-y_pred))
  return torch.mean(loss)

def imgrad(img):
    img = torch.mean(img, 1, True)
    fx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv1.weight = nn.Parameter(weight)
    grad_x = conv1(img)
    # grad y
    fy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv2.weight = nn.Parameter(weight)
    grad_y = conv2(img)
    return grad_y, grad_x

def imgrad_yx(img):
    N,C,_,_ = img.size()
    grad_y, grad_x = imgrad(img)
    return torch.cat((grad_y.view(N,C,-1), grad_x.view(N,C,-1)), dim=1)

def train_net(  net,
                device,
                epochs: int = 5, 
                batch_size: int = 32,
                learning_rate = 0.0001,
                val_percent: float = 0.1):
    saveDir = PATH_TO_NYU / "saveDir"
    dataset = DatasetNYUv2()
    
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    optimizer = optim.Adam(net.parameters())
    grad_criterion = GradLoss()
    torch.cuda.empty_cache()

    for epoch in range(epochs):
        net.train()
        start = time.time()

        for img, depth in train_loader:
            img, depth = Variable(img.to(device)),Variable(depth.to(device))
            optimizer.zero_grad()
            depth_pred = net(img)
            grad_real, grad_fake = imgrad_yx(depth), imgrad_yx(depth_pred)
            loss = grad_criterion(grad_fake, grad_real)
            loss.backward()
            optimizer.step()

        torch.save(net.state_dict(), f"{saveDir}_unet_model_{epoch}.pth")
        end = time.time()

        print('model saved')
        print('time elapsed: %fs' % (end - start))

        net.eval()
        print('evaluating...')
        eval_loss = 0
        count = 0

        with torch.no_grad():
              for img, depth in val_loader:
                  img, depth = Variable(img.to(device)),Variable(depth.to(device))
                  depth_pred = net(img)
                  eval_loss += float(img.size(0)) * grad_criterion(depth_pred, depth).item()**2
                  count += float(img.size(0))
        print(f"[epoch {epoch}] RMSE_log: {math.sqrt(eval_loss/count)}")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(in_channel=3, out_channel=1)
    net.to(device=device)

    try:
        train_net(net=net,
                  device=device,
                  epochs=20,
                  batch_size=32,
                  learning_rate=0.0001,
                  val_percent= 0.1)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')

