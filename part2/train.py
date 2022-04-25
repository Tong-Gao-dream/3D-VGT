import csv

import numpy as np
import torch
from torch.autograd import Variable


def train(args, dataloader_img, encoder_2D, discriminator, optimizer_G, optimizer_D, loss):
    Tensor = torch.cuda.FloatTensor

    feature_list = []
    with open('./feature_1285.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            feature_list.append(row)
    feature_list_numpy = np.array(feature_list)
    j = 0

    # The code here is only to ensure the executable of the program
    for epoch in range(args.epoch):
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

            for i in range(args.numbers):
                feature = feature_list_numpy[j]
                j += 1
                if j == 1285:
                    j = 0
                targe_array = feature.astype(float)
                targe_tensor = torch.tensor(targe_array)
                real_tensor = targe_tensor.float().cuda()

                valid = Variable(Tensor(1, 1).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(1, 1).fill_(0.0), requires_grad=False)

                valid_ = real_tensor
                rgb = img_12

                optimizer_G.zero_grad()
                g_output = encoder_2D(rgb)
                g_output = g_output.reshape(1024)

                g_loss = loss(discriminator(g_output), valid)
                print('g_loss', g_loss)
                g_loss.backward()
                optimizer_G.step()

                # update discriminator
                optimizer_D.zero_grad()
                real_loss = loss(discriminator(valid_), valid)
                fake_loss = loss(discriminator(g_output.detach()), fake)

                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                optimizer_D.step()
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, args.epoch, idx, len(dataloader_img), d_loss.item(), g_loss.item())
                )
                break
    encoder_2D.eval()
    torch.save(encoder_2D.state_dict(),
               './encoder_2D/encoder_2D.pth')

    discriminator.eval()
    torch.save(discriminator.state_dict(),
               './discriminator/discriminator.pth')
