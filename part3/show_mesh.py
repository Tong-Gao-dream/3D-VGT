import torch
from util.mesh import *
import visdom

vis = visdom.Visdom()
def show_mesh(args, dataloader_img, encoder_2D, decoder_3d):
    for idx, img in enumerate(dataloader_img):

        img1, img2, img3, img4, img5, img6, img7, img8, img9, img10, img11, img12 = img
        temp = torch.cat([img1, img2])
        temp = torch.cat([temp, img3])
        temp = torch.cat([temp, img4])
        temp = torch.cat([temp, img5])
        temp = torch.cat([temp, img6])
        temp = torch.cat([temp, img7])
        temp = torch.cat([temp, img8])
        temp = torch.cat([temp, img9])
        temp = torch.cat([temp, img10])
        temp = torch.cat([temp, img11])
        img_12 = torch.cat([temp, img12])
        img_12 = img_12.reshape(1, 12, 192, 256).cuda()
        x = torch.randn(1, 1, 129, 129, 129).cuda()

        img_encoder_output = encoder_2D(img_12)
        # print('img_encoder_output.shape', img_encoder_output.shape)
        decoder_output = decoder_3d(x, img_encoder_output)

        # print(decoder_output.shape)
        samples = decoder_output.cpu().data[:1].squeeze().numpy()
        # output = int(output)
        # print('samples.shape', samples)
        plotVoxelVisdom(samples, vis, 'output')