import csv

import torch
from torch.utils.data import DataLoader
from part1.config import input_args
from model.VAE import VAE_3D
from util.read_dataset import ShapeNet_voxel

args = input_args()
print('args', args)

dataset_vox = ShapeNet_voxel(args.vox_dir)

dataloader_vox = DataLoader(dataset_vox, batch_size=args.batch_size, shuffle=False)

vae = VAE_3D().cuda()
vae.load_state_dict(torch.load('./model_save/part1.pth'))

for idx, sample in enumerate(dataloader_vox):
    sample_vox = sample.cuda()
    output = vae(sample_vox)

    output_feature = output.cpu().data[:1].squeeze().numpy()
    output_feature.tolist
    with open(r'../part1_feature/feature.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(output_feature)