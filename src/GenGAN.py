import numpy as np
import cv2
import os
import pickle
import sys
import math
import matplotlib.pyplot as plt

from torchvision.io import read_image

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton
from GenVanillaNN import VideoSkeletonDataset, SkeToImageTransform, GenNNSkeImToImage, init_weights


class Discriminator(nn.Module):
    def __init__(self, use_cgan=True):
        super().__init__()
        # In a Conditional GAN (Pix2Pix), input is concatenation of real image (3)
        # and conditional image (skeleton, 3) => 6 channels.
        self.input_channels = 6 if use_cgan else 3
        ngf = 64

        self.model = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(self.input_channels, ngf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 -> 16x16
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 8x8
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8 -> 4x4
            nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # 4x4 -> 1x1 (scalar output)
            nn.Conv2d(ngf * 8, 1, 4, 1, 0, bias=False),
        )

        self.apply(init_weights)

    def forward(self, x):
        return self.model(x)


class GenGAN:
    """
    Conditional WGAN-GP GAN that generates a new image from a skeleton posture.
    """

    def __init__(self, videoSke, loadFromFile=False, use_cgan=True):
        self.use_cgan = use_cgan

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # MODELS
        # Generator: same architecture as in GenVanillaNN, but at 64x64 to match dataset
        self.netG = GenNNSkeImToImage(image_size=64)
        self.netD = Discriminator(use_cgan=use_cgan)

        self.netG.to(self.device)
        self.netD.to(self.device)

        # OPTIMIZERS
        self.optimizerD = torch.optim.Adam(self.netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizerG = torch.optim.Adam(self.netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # LOSSES
        self.criterionL1 = nn.L1Loss()
        self.lambda_L1 = 50
        self.lambda_gp = 10

        # CHECKPOINT FILES
        self.filename_G_checkpoint = '../data/Dance/DanceGenGAN_G_checkpoint.pth'
        self.filename_D_checkpoint = '../data/Dance/DanceGenGAN_D_checkpoint.pth'
        self.start_epoch = 0

        # DATASET AND DATALOADER (64x64 like in your code)
        image_size = 64

        src_transform = transforms.Compose([
            SkeToImageTransform(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5)),
        ])

        tgt_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5)),
        ])

        self.dataset = VideoSkeletonDataset(
            videoSke,
            ske_reduced=True,
            target_transform=tgt_transform,
            source_transform=src_transform
        )

        # Smaller batch size to make training faster / more stable
        self.dataloader = torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_size=8,
            shuffle=True,
            num_workers=0
        )

        # CHECKPOINT LOADING
        if loadFromFile and os.path.isfile(self.filename_G_checkpoint) and os.path.isfile(self.filename_D_checkpoint):
            print(f"GenGAN: Loading checkpoint from {self.filename_G_checkpoint}")
            try:
                checkpoint_G = torch.load(self.filename_G_checkpoint, map_location=self.device)
                checkpoint_D = torch.load(self.filename_D_checkpoint, map_location=self.device)

                self.netG.load_state_dict(checkpoint_G['model_state_dict'])
                self.netD.load_state_dict(checkpoint_D['model_state_dict'])

                self.optimizerG.load_state_dict(checkpoint_G['optimizer_state_dict'])
                self.optimizerD.load_state_dict(checkpoint_D['optimizer_state_dict'])

                self.start_epoch = checkpoint_G.get('epoch', 0)
                self.netG.train()
                self.netD.train()
                print(f"Checkpoint loaded. Resuming training from epoch {self.start_epoch}.")
            except Exception as e:
                print(f"Could not load checkpoint fully or correctly: {e}. Starting fresh.")
                self.start_epoch = 0

    def compute_gradient_penalty(self, real_samples, fake_samples, conditional_image):
        device = real_samples.device
        batch_size = real_samples.size(0)

        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        alpha = alpha.expand_as(real_samples)

        interpolated_image = alpha * real_samples + (1 - alpha) * fake_samples
        interpolated_input = torch.cat([conditional_image, interpolated_image], 1)
        interpolated_input.requires_grad_(True)

        d_interpolated = self.netD(interpolated_input)
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated_input,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train(self, n_epochs=200):
        lambda_gp = self.lambda_gp
        lambda_l1 = self.lambda_L1

        writer_dir_name = self.filename_G_checkpoint.split('/')[-1].split('.')[0]
        writer = SummaryWriter(f'runs/GenGAN_{writer_dir_name}')

        print(f"Starting Training from epoch {self.start_epoch} to {n_epochs} with WGAN-GP...")

        for epoch in range(self.start_epoch, n_epochs):
            for i, (ske_img, real_image) in enumerate(self.dataloader):
                ske_img = ske_img.to(self.device)
                real_image = real_image.to(self.device)
                ske_img_cond = ske_img

                # ====================
                # Train Discriminator
                # ====================
                self.netD.zero_grad()

                # Real
                real_pair = torch.cat((ske_img_cond, real_image), 1)
                output_real = self.netD(real_pair).mean()

                # Fake
                fake_image = self.netG(ske_img)
                fake_pair = torch.cat((ske_img_cond, fake_image.detach()), 1)
                output_fake = self.netD(fake_pair).mean()

                # WGAN-GP loss
                errD_Wasserstein = output_fake - output_real
                gradient_penalty = self.compute_gradient_penalty(real_image.data, fake_image.data, ske_img_cond.data)
                errD = errD_Wasserstein + lambda_gp * gradient_penalty

                errD.backward()
                self.optimizerD.step()

                # ====================
                # Train Generator
                # ====================
                self.netG.zero_grad()

                # Adversarial loss (maximize D(G(c)))
                output_G = self.netD(torch.cat((ske_img_cond, fake_image), 1)).mean()
                errG_GAN = -output_G

                # L1 loss
                errG_L1 = self.criterionL1(fake_image, real_image) * lambda_l1

                errG = errG_GAN + errG_L1
                errG.backward()
                self.optimizerG.step()

                # Logging
                if i % 10 == 0:
                    print(f'[{epoch+1}/{n_epochs}][{i}/{len(self.dataloader)}] '
                          f'Loss_D: {errD.item():.4f} | W_Loss: {errD_Wasserstein.item():.4f} | '
                          f'Loss_G: {errG.item():.4f} (GAN: {errG_GAN.item():.4f} L1: {errG_L1.item():.4f})')

                if (i % 50 == 0) and (i > 0):
                    global_step = epoch * len(self.dataloader) + i
                    writer.add_scalars('Loss_D', {
                        'W_Loss': errD_Wasserstein.item(),
                        'Total_D': errD.item()
                    }, global_step)
                    writer.add_scalars('Loss_G', {
                        'GAN': errG_GAN.item(),
                        'L1': errG_L1.item(),
                        'Total_G': errG.item()
                    }, global_step)

            # Periodic checkpoint
            if (epoch + 1) % 50 == 0 or (epoch + 1) == n_epochs:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.netG.state_dict(),
                    'optimizer_state_dict': self.optimizerG.state_dict(),
                }, self.filename_G_checkpoint)

                torch.save({
                    'model_state_dict': self.netD.state_dict(),
                    'optimizer_state_dict': self.optimizerD.state_dict(),
                }, self.filename_D_checkpoint)

                print(f"Checkpoint saved at epoch {epoch + 1}.")

        print("Training finished. Final models saved.")

    def generate(self, ske):
        self.netG.eval()
        with torch.no_grad():
            ske_t = self.dataset.preprocessSkeleton(ske)
            ske_t_batch = ske_t.unsqueeze(0).to(self.device)

            normalized_output = self.netG(ske_t_batch)
            output_cpu = normalized_output[0].cpu()
            res = self.dataset.tensor2image(output_cpu)
            res = (res * 255).astype(np.uint8)
        return res


if __name__ == '__main__':
    force = False

    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == "true"
    else:
        filename = "../data/taichi1.mp4"

    print("GenGAN: Current Working Directory=", os.getcwd())
    print("GenGAN: Filename=", filename)

    targetVideoSke = VideoSkeleton(filename)

    TRAIN_MODE = True
    # TRAIN_MODE = False

    if TRAIN_MODE:
        gen = GenGAN(targetVideoSke, loadFromFile=True)
        gen.train(n_epochs=300)
    else:
        gen = GenGAN(targetVideoSke, loadFromFile=True)

    for i in range(targetVideoSke.skeCount()):
        image = gen.generate(targetVideoSke.ske[i])
        new_size = (512, 512)
        image = cv2.resize(image, new_size)
        cv2.imshow('Image', image)
        key = cv2.waitKey(300)
