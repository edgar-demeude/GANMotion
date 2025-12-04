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
        
        # self.apply(init_weights) 

    def forward(self, input):
        return self.model(input)
    

class GenGAN():
    """ class that Generate a new image from videoSke from a new skeleton posture
        Fonc generator(Skeleton)->Image
    """
    def __init__(self, videoSke, loadFromFile=False, use_cgan=True):
        self.use_cgan = use_cgan

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # MODÈLES
        self.netG = GenNNSkeImToImage()
        self.netD = Discriminator(use_cgan=use_cgan)

        self.netG.to(self.device)
        self.netD.to(self.device)

        # OPTIMISEURS
        self.optimizerD = torch.optim.Adam(self.netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizerG = torch.optim.Adam(self.netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # CRITÈRES DE PERTE
        self.criterionGAN = nn.BCEWithLogitsLoss() 
        self.criterionL1 = nn.L1Loss()
        self.lambda_L1 = 50 # Réduction de 100 à 50 pour potentiellement améliorer la couleur
        self.lambda_gp = 10

        self.real_label = 1.
        self.fake_label = 0.
        
        # FICHIERS DE CHECKPOINT
        self.filename_G_checkpoint = '../data/Dance/DanceGenGAN_G_checkpoint.pth'
        self.filename_D_checkpoint = '../data/Dance/DanceGenGAN_D_checkpoint.pth'
        self.start_epoch = 0 # Époque de départ

        # TRANSFORMATIONS ET DATASET
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
        
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, 
                                            target_transform=tgt_transform,
                                            source_transform=src_transform)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=32, shuffle=True)
        
        # CHARGEMENT DU CHECKPOINT
        if loadFromFile and os.path.isfile(self.filename_G_checkpoint):
            print(f"GenGAN: Loading checkpoint from {self.filename_G_checkpoint}")
            
            # Chargement des checkpoints
            try:
                checkpoint_G = torch.load(self.filename_G_checkpoint, map_location=self.device)
                checkpoint_D = torch.load(self.filename_D_checkpoint, map_location=self.device)

                # 1. Charger les poids des modèles
                self.netG.load_state_dict(checkpoint_G['model_state_dict'])
                self.netD.load_state_dict(checkpoint_D['model_state_dict'])

                # 2. Charger l'état des optimiseurs
                self.optimizerG.load_state_dict(checkpoint_G['optimizer_state_dict'])
                self.optimizerD.load_state_dict(checkpoint_D['optimizer_state_dict'])

                # 3. Définir l'époque de départ
                self.start_epoch = checkpoint_G['epoch']
                
                # 4. S'assurer que les modèles sont en mode 'train' pour continuer
                self.netG.train()
                self.netD.train()
                print(f"Checkpoint loaded. Resuming training from epoch {self.start_epoch}.")
            except Exception as e:
                 print(f"Could not load checkpoint fully or correctly: {e}. Starting fresh.")
                 self.start_epoch = 0 # Recommencer si le chargement échoue


    def compute_gradient_penalty(self, real_samples, fake_samples, conditional_image):
        device = real_samples.device
        batch_size = real_samples.size(0)
        
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        alpha = alpha.expand_as(real_samples)

        interpolated_image = alpha * real_samples + (1 - alpha) * fake_samples
        interpolated_input = torch.cat([conditional_image, interpolated_image], 1)
        interpolated_input.requires_grad_(True)

        d_interpolated = self.netD(interpolated_input)

        gradients = torch.autograd.grad(outputs=d_interpolated, inputs=interpolated_input,
                                         grad_outputs=torch.ones_like(d_interpolated),
                                         create_graph=True, retain_graph=True)[0]

        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty
    

    def train(self, n_epochs=5000):
        lambda_gp = self.lambda_gp
        lambda_l1 = self.lambda_L1
        
        # Utiliser un nom de répertoire basé sur le nom du fichier checkpoint pour SummaryWriter
        writer_dir_name = self.filename_G_checkpoint.split('/')[-1].split('.')[0]
        writer = SummaryWriter(f'runs/GenGAN_{writer_dir_name}')

        print(f"Starting Training from epoch {self.start_epoch} to {n_epochs} with WGAN-GP...")
        
        # Utiliser self.start_epoch
        for epoch in range(self.start_epoch, n_epochs):
            for i, (ske_img, real_image) in enumerate(self.dataloader):
                ske_img = ske_img.to(self.device)
                real_image = real_image.to(self.device)
                
                ske_img_cond = ske_img
                self.netD.zero_grad()
                real_pair = torch.cat((ske_img_cond, real_image), 1)
                output_real = self.netD(real_pair).mean()
                fake_image = self.netG(ske_img)
                fake_pair = torch.cat((ske_img_cond, fake_image.detach()), 1)
                output_fake = self.netD(fake_pair).mean()
                errD_Wasserstein = output_fake - output_real
                gradient_penalty = self.compute_gradient_penalty(real_image.data, fake_image.data, ske_img_cond.data)
                errD = errD_Wasserstein + lambda_gp * gradient_penalty
                errD.backward()
                self.optimizerD.step()

                self.netG.zero_grad()
                output_G = self.netD(torch.cat((ske_img_cond, fake_image), 1)).mean()
                errG_GAN = -output_G
                errG_L1 = self.criterionL1(fake_image, real_image) * lambda_l1
                errG = errG_GAN + errG_L1
                errG.backward()
                self.optimizerG.step()
                
                # Logging
                if i % 10 == 0:
                    print(f'[{epoch+1}/{n_epochs}][{i}/{len(self.dataloader)}] Loss_D: {errD.item():.4f} | W_Loss: {errD_Wasserstein.item():.4f} | Loss_G: {errG.item():.4f} (GAN: {errG_GAN.item():.4f} L1: {errG_L1.item():.4f})')

                if (i % 50 == 0) and (i > 0):
                    global_step = epoch * len(self.dataloader) + i
                    writer.add_scalars('Loss_D', {'W_Loss': errD_Wasserstein.item(), 'Total_D': errD.item()}, global_step)
                    writer.add_scalars('Loss_G', {'GAN': errG_GAN.item(), 'L1': errG_L1.item(), 'Total_G': errG.item()}, global_step)

            # SAUVEGARDE PÉRIODIQUE DU CHECKPOINT
            if (epoch + 1) % 100 == 0 or (epoch + 1) == n_epochs:
                
                # Sauvegarde du Checkpoint (Générateur)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.netG.state_dict(),
                    'optimizer_state_dict': self.optimizerG.state_dict(),
                }, self.filename_G_checkpoint)

                # Sauvegarde du Checkpoint (Discriminateur)
                torch.save({
                    'model_state_dict': self.netD.state_dict(),
                    'optimizer_state_dict': self.optimizerD.state_dict(),
                }, self.filename_D_checkpoint)
                print(f"Checkpoint saved at epoch {epoch + 1}.")
        
        print(f"Training finished. Final models saved.")


    def generate(self, ske):
        self.netG.eval()
        
        with torch.no_grad():
            ske_t = self.dataset.preprocessSkeleton(ske)
            ske_t_batch = ske_t.unsqueeze(0)

            ske_t_batch = ske_t_batch.to(self.device)
            
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

    # ===================================================
    # Utilisation du Checkpointing pour l'entraînement
    # Si True: Tentative de reprise ou nouvel entraînement
    # Si False: Chargement du dernier modèle entraîné (pour inférence)
    # ===================================================
    
    # TRAIN_MODE = True 
    TRAIN_MODE = False 

    if TRAIN_MODE:
        # Tente de charger un checkpoint si loadFromFile=True
        gen = GenGAN(targetVideoSke, loadFromFile=True) 
        # Si un checkpoint est chargé, il repartira de self.start_epoch
        # sinon il repartira de 0.
        gen.train(n_epochs=10000)
    else:
        # Pour l'inférence, charge le dernier checkpoint trouvé 
        # et le met en mode évaluation (génération seulement)
        gen = GenGAN(targetVideoSke, loadFromFile=True)

    # Visualisation
    for i in range(targetVideoSke.skeCount()):
        image = gen.generate(targetVideoSke.ske[i])
        nouvelle_taille = (256, 256) 
        image = cv2.resize(image, nouvelle_taille)
        cv2.imshow('Image', image)
        key = cv2.waitKey(300)