import torch
from torch.utils.data import DataLoader

from model.Discriminator import Discriminator
from model.ViT import ViT
from part1.config import input_args
from train import train
from util.read_dataset import ShapeNet_img

args = input_args()
print('args', args)

dataset_img = ShapeNet_img(args.img_dir)

dataloader_img = DataLoader(dataset_img, batch_size=args.batch_size, shuffle=False)

encoder_2D = ViT(image_size=256,
                 patch_size=32,
                 dim=1024,
                 depth=6,
                 heads=16,
                 mlp_dim=2048,
                 dropout=0.1,
                 emb_dropout=0.1).cuda()

discriminator = Discriminator().cuda()
discriminator.load_state_dict(torch.load('./discriminator.pth'))

loss = torch.nn.BCELoss().cuda()

optimizer_G = torch.optim.Adam(encoder_2D.parameters(), lr=args.lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

train(args, dataloader_img, encoder_2D, discriminator, optimizer_G, optimizer_D, loss)
