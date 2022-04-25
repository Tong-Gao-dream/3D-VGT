import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable

class decoder_3D(nn.Module):
    def __init__(self):
        super(decoder_3D, self).__init__()
        # Encoder
        self.conv1 = nn.Conv3d(1, 16, 3)
        self.conv2 = nn.Conv3d(16, 32, 3)
        self.conv3 = nn.Conv3d(32, 96, 2)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.pool2 = nn.MaxPool3d(kernel_size=3, stride=3, return_indices=True)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.enc_linear = nn.Linear(69984, 1024)
        self._enc_log_sigma = nn.Linear(69984, 1024)

        # Decoder
        self.deconv1 = nn.ConvTranspose3d(96, 32, 2)
        self.deconv2 = nn.ConvTranspose3d(32, 16, 3)
        self.deconv3 = nn.ConvTranspose3d(16, 1, 3)
        self.unpool1 = nn.MaxUnpool3d(kernel_size=2, stride=2)
        self.unpool2 = nn.MaxUnpool3d(kernel_size=3, stride=3)
        self.unpool3 = nn.MaxUnpool3d(kernel_size=2, stride=2)

        self.dec_linear = nn.Linear(1024, 69984)

    def encode(self, x, return_partials=True):
        # Encoder
        x = self.conv1(x)
        up3out_shape = x.shape
        x, indices1 = self.pool1(x)

        x = self.conv2(x)
        up2out_shape = x.shape
        x, indices2 = self.pool2(x)

        x = self.conv3(x)
        up1out_shape = x.shape
        x, indices3 = self.pool3(x)

        x = x.view((x.size(0), -1))

        x = x.reshape(1, 1, 69984)

        pool_par = {
            "P1": [indices1, up3out_shape],
            "P2": [indices2, up2out_shape],
            "P3": [indices3, up1out_shape]
        }

        if return_partials:
            return x, pool_par
        else:
            return x

    def decode(self, x, pool_par):
        x = self.dec_linear(x)
        x = x.view((x.size(0), 96, 9, 9, 9))
        x = self.unpool1(x, output_size=pool_par["P3"][1], indices=pool_par["P3"][0])
        x = self.deconv1(x)
        x = self.unpool2(x, output_size=pool_par["P2"][1], indices=pool_par["P2"][0])
        x = self.deconv2(x)
        x = self.unpool3(x, output_size=pool_par["P1"][1], indices=pool_par["P1"][0])
        x = self.deconv3(x)
        return x

    def _sample_latent(self,h_enc):

        mu = self.enc_linear(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()
        self.z_mean = mu
        self.z_sigma = sigma
        return mu + sigma * Variable(std_z, requires_grad=False).cuda()


    def forward(self, x ,y):
        feature, pool_par = self.encode(x)
        z = self._sample_latent(feature)
        out = self.decode(y, pool_par)
        return out
