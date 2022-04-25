import torch

from model.VAE import latent_loss

def train(args, dataloader_vox, vae, optim, loss):
    for epoch in range(args.epoch):
        for i_batch, sample in enumerate(dataloader_vox):
            sample_vox = sample.cuda()
            optim.zero_grad()
            output = vae(sample_vox)

            ll = latent_loss(vae.z_mean, vae.z_sigma)
            loss_ = loss(output, sample_vox) + ll

            optim.zero_grad()
            loss_.backward()
    vae.eval()
    torch.save(vae.state_dict(), './model_save/part1.pth')
