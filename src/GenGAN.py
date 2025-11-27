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
from GenVanillaNN import * 


class Discriminator(nn.Module):
    def __init__(self, use_cgan=True):
        super().__init__()
        # Dans un Conditional GAN (Pix2Pix), l'entrée est la concatenation de l'image (3) 
        # et de l'image conditionnelle (le squelette, 3). Total = 6 canaux.
        self.input_channels = 6 if use_cgan else 3 
        ngf = 64
        
        # Le discriminateur est une série de couches de convolution (encodeur)
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
            
            # 4x4 -> 1x1 (Sortie scalaire)
            nn.Conv2d(ngf * 8, 1, 4, 1, 0, bias=False), 
        )
        
        self.apply(init_weights)

    def forward(self, input):
        return self.model(input)
    

class GenGAN():
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
    """
    def __init__(self, videoSke, loadFromFile=False, use_cgan=True):
        self.use_cgan = use_cgan
        
        # MODIFICATION PRINCIPALE : Utiliser GenNNSkeImToImage au lieu de GenNNSke26ToImage
        self.netG = GenNNSkeImToImage()
        self.netD = Discriminator(use_cgan=use_cgan)

        # Optimiseurs
        self.optimizerD = torch.optim.Adam(self.netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizerG = torch.optim.Adam(self.netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Critères de perte
        self.criterionGAN = nn.BCEWithLogitsLoss() 
        self.criterionL1 = nn.L1Loss()
        self.lambda_L1 = 100
        self.lambda_gp = 10

        self.real_label = 1.
        self.fake_label = 0.
        self.filename = '../data/Dance/DanceGenGAN_SkeIm.pth'
        
        # MODIFICATION : Transformation source pour créer des images de squelette
        image_size = 64
        src_transform = transforms.Compose([
            SkeToImageTransform(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        tgt_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        # MODIFICATION : Passer source_transform au dataset
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, 
                                           target_transform=tgt_transform,
                                           source_transform=src_transform)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=32, shuffle=True)
        
        if loadFromFile and os.path.isfile(self.filename):
            print("GenGAN: Load=", self.filename, "   Current Working Directory=", os.getcwd())
            state_dict = torch.load(self.filename)
            self.netG.load_state_dict(state_dict)
            self.netG.eval()


    def compute_gradient_penalty(self, real_samples, fake_samples, conditional_image):
        """Compute gradient penalty for WGAN-GP using conditional input."""
        
        device = real_samples.device
        batch_size = real_samples.size(0)
        
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        alpha = alpha.expand_as(real_samples)

        # Interpolate between real and fake samples
        interpolated_image = alpha * real_samples + (1 - alpha) * fake_samples
        
        # Concaténer avec l'image conditionnelle
        interpolated_input = torch.cat([conditional_image, interpolated_image], 1)
        interpolated_input.requires_grad_(True)

        # Get discriminator output
        d_interpolated = self.netD(interpolated_input)

        # Calculate gradients
        gradients = torch.autograd.grad(outputs=d_interpolated, inputs=interpolated_input,
                                        grad_outputs=torch.ones_like(d_interpolated),
                                        create_graph=True, retain_graph=True)[0]

        # Calculate gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty
    

    def train(self, n_epochs=10):
        lambda_gp = self.lambda_gp
        lambda_l1 = self.lambda_L1
        
        writer = SummaryWriter(f'runs/GenGAN_{self.filename.split("/")[-1].split(".")[0]}')

        print(f"Starting Training for {n_epochs} epochs with WGAN-GP and SkeImToImage...")
        
        for epoch in range(n_epochs):
            # MODIFICATION : Le dataloader retourne maintenant directement les images de squelette
            for i, (ske_img, real_image) in enumerate(self.dataloader):
                batch_size = ske_img.size(0)
                
                # ske_img est déjà l'image conditionnelle (squelette dessiné)
                ske_img_cond = ske_img

                # =================================================================
                # (A) Entraînement du Discriminateur (D)
                # =================================================================
                self.netD.zero_grad()
                
                # Perte D sur les Échantillons Réels
                real_pair = torch.cat((ske_img_cond, real_image), 1)
                output_real = self.netD(real_pair).mean()
                
                # Perte D sur les Échantillons Faux
                fake_image = self.netG(ske_img)  # MODIFICATION : on passe ske_img au générateur
                fake_pair = torch.cat((ske_img_cond, fake_image.detach()), 1)
                output_fake = self.netD(fake_pair).mean()
                
                # Perte de Wasserstein
                errD_Wasserstein = output_fake - output_real

                # Gradient Penalty
                gradient_penalty = self.compute_gradient_penalty(real_image.data, fake_image.data, ske_img_cond.data)
                
                # Perte totale D
                errD = errD_Wasserstein + lambda_gp * gradient_penalty
                errD.backward()
                self.optimizerD.step()

                # =================================================================
                # (B) Entraînement du Générateur (G)
                # =================================================================
                self.netG.zero_grad()
                
                # Perte Adversaire
                output_G = self.netD(torch.cat((ske_img_cond, fake_image), 1)).mean()
                errG_GAN = -output_G
                
                # Perte de Reconstruction
                errG_L1 = self.criterionL1(fake_image, real_image) * lambda_l1
                
                # Perte totale G
                errG = errG_GAN + errG_L1
                errG.backward()
                self.optimizerG.step()
                
                # Logging
                if i % 10 == 0:
                    print(f'[{epoch+1}/{n_epochs}][{i}/{len(self.dataloader)}] Loss_D: {errD.item():.4f} | W_Loss: {errD_Wasserstein.item():.4f} | Loss_G: {errG.item():.4f} (GAN: {errG_GAN.item():.4f} L1: {errG_L1.item():.4f})')

                if (i % 50 == 0) and (i > 0):
                    writer.add_scalars('Loss_D', {'W_Loss': errD_Wasserstein.item(), 'Total_D': errD.item()}, epoch * len(self.dataloader) + i)
                    writer.add_scalars('Loss_G', {'GAN': errG_GAN.item(), 'L1': errG_L1.item(), 'Total_G': errG.item()}, epoch * len(self.dataloader) + i)

            # Sauvegarder périodiquement
            if (epoch + 1) % 100 == 0:
                torch.save(self.netG.state_dict(), self.filename.replace('.pth', '_G.pth'))
                torch.save(self.netD.state_dict(), self.filename.replace('.pth', '_D.pth'))

        # Sauvegarder le modèle final
        torch.save(self.netG.state_dict(), self.filename.replace('.pth', '_G.pth'))
        torch.save(self.netD.state_dict(), self.filename.replace('.pth', '_D.pth'))
        print(f"Training finished. Final models saved.")


    def generate(self, ske):
        """ generator of image from skeleton """
        self.netG.eval()
        
        with torch.no_grad():
            # MODIFICATION : Préparer le squelette en image
            ske_t = self.dataset.preprocessSkeleton(ske)
            ske_t_batch = ske_t.unsqueeze(0)
            
            # Générer l'image
            normalized_output = self.netG(ske_t_batch)
            
            # Convertir en image OpenCV
            res = self.dataset.tensor2image(normalized_output[0])
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

    if True:    # train or load
        # Train
        gen = GenGAN(targetVideoSke, False)
        gen.train(4)
    else:
        gen = GenGAN(targetVideoSke, loadFromFile=True)

    for i in range(targetVideoSke.skeCount()):
        image = gen.generate(targetVideoSke.ske[i])
        nouvelle_taille = (256, 256) 
        image = cv2.resize(image, nouvelle_taille)
        cv2.imshow('Image', image)
        key = cv2.waitKey(100)