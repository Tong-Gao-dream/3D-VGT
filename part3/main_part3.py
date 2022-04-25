import torch

from torch.utils.data import DataLoader
from part3.show_mesh import show_mesh
from model.ViT import ViT
from part1.config import input_args
from util.read_dataset import ShapeNet_img
from model.decoder_3D import decoder_3D

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
encoder_2D.load_state_dict(torch.load('../part2/model_save/encoder_2D/encoder_2D.pth'))

decoder_3d = decoder_3D()
decoder_3d.load_state_dict(torch.load('../part1/model_save/part1.pth'))

show_mesh(args, dataloader_img, encoder_2D, decoder_3d)