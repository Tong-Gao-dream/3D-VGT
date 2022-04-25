import torch
from torch.utils.data import DataLoader

from model.VAE import VAE_3D
from part1.config import input_args
from train import train
from util.read_dataset import ShapeNet_voxel

args = input_args()
print('args', args)

dataset_vox = ShapeNet_voxel(args.vox_dir)

dataloader_vox = DataLoader(dataset_vox, batch_size=args.batch_size, shuffle=False)

vae = VAE_3D().cuda()

optim = torch.optim.Adam(vae.parameters(), lr=args.lr)

loss = torch.nn.MSELoss().cuda()

train(args, dataloader_vox, vae, optim, loss)
