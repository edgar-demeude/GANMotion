
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
            # Cette couche donne la prédiction finale (Réel/Faux)
            nn.Conv2d(ngf * 8, 1, 4, 1, 0, bias=False), 
            # Note: Pas de Sigmoid ici, car la Loss BCEWithLogitsLoss ou la WGAN-GP n'en ont pas besoin.
        )
        
        self.apply(init_weights)

    def forward(self, input):
        return self.model(input)
    

class GenGAN():
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
    """
    def __init__(self, videoSke, loadFromFile=False, use_cgan=True): # Ajout de use_cgan
        self.use_cgan = use_cgan # <--- Fix : use_cgan doit être un argument ou initialisé
        self.netG = GenNNSke26ToImage()
        self.netD = Discriminator(use_cgan=use_cgan)

        # 3. Optimiseurs (doivent être initialisés ici)
        self.optimizerD = torch.optim.Adam(self.netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizerG = torch.optim.Adam(self.netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # 4. Critère de perte (BCE pour le GAN, L1 pour la reconstruction)
        self.criterionGAN = nn.BCEWithLogitsLoss() 
        self.criterionL1 = nn.L1Loss()
        self.lambda_L1 = 100 # Poids de la perte L1 (Pix2Pix utilise 100)

        # Si vous utilisez WGAN-GP:
        # self.criterionGAN = None # Pas de BCE loss dans WGAN
        self.lambda_gp = 10 # Poids du gradient penalty

        self.real_label = 1.
        self.fake_label = 0.
        self.filename = '../data/Dance/DanceGenGAN.pth'
        tgt_transform = transforms.Compose(
                            [transforms.Resize((64, 64)),
                            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            transforms.CenterCrop(64),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, target_transform=tgt_transform)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=32, shuffle=True)
        if loadFromFile and os.path.isfile(self.filename):
            print("GenGAN: Load=", self.filename, "   Current Working Directory=", os.getcwd())
            self.netG = torch.load(self.filename)


    # Mettez à jour la définition de la fonction dans GenGAN
    def compute_gradient_penalty(self, real_samples, fake_samples, conditional_image):
        """Compute gradient penalty for WGAN-GP using conditional input."""
        
        device = real_samples.device
        batch_size = real_samples.size(0)
        
        # Étape 1: Concaténer l'image conditionnelle (Squelette) avec les échantillons
        # Nous interpoleons sur l'espace des images (3 canaux), pas sur l'espace concaténé.
        # L'image conditionnelle reste la même pour tous les points interpolés issus du même échantillon.
        
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        alpha = alpha.expand_as(real_samples)

        # Interpolate between real and fake samples (3 canaux)
        interpolated_image = alpha * real_samples + (1 - alpha) * fake_samples
        
        # Étape 2: Concaténer l'image interpolée avec l'image conditionnelle pour le Discriminateur (6 canaux)
        interpolated_input = torch.cat([conditional_image, interpolated_image], 1)
        
        # Le Discriminateur doit voir la paire complète
        interpolated_input.requires_grad_(True)

        # Get discriminator output for interpolated images
        d_interpolated = self.netD(interpolated_input)

        # Calculate gradients of probabilities with respect to examples
        # NOTE: Ici, les 'inputs' pour l'autograd sont les 6 canaux concaténés.
        gradients = torch.autograd.grad(outputs=d_interpolated, inputs=interpolated_input,
                                        grad_outputs=torch.ones_like(d_interpolated),
                                        create_graph=True, retain_graph=True)[0]

        # Calculate gradient penalty
        gradients = gradients.view(batch_size, -1)
        # L'erreur est la distance à la norme 1 (WGAN-GP)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty
    
    def create_ske_image_batch(self, batch_ske_objects):
        """
        Convertit un batch d'objets Skeleton en un tenseur d'images squelette normalisées [-1, 1].
        """
        image_size = 64
        # Assurez-vous que cette transformation est correcte dans votre code
        ske_im_transform = transforms.Compose([ 
            SkeToImageTransform(image_size), # Supposons que ceci convertit l'objet Skeleton en PIL/Tensor
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        # Le dataloader retourne l'index 'i' du batch, nous devons obtenir les objets Skeleton
        # correspondants. Ceci nécessite une adaptation de l'itération ou du dataloader.
        # Pour le moment, nous allons supposer que nous recevons les objets Skeleton dans `batch_ske_objects`.
        
        ske_images = [ske_im_transform(ske) for ske in batch_ske_objects]
        return torch.stack(ske_images).to(self.netG.parameters().__next__().device) # Met sur le bon device


    def train(self, n_epochs=10):
        lambda_gp = self.lambda_gp # 10.0
        lambda_l1 = self.lambda_L1 # 100
        
        # Utilisez un writer si vous voulez suivre les métriques
        writer = SummaryWriter(f'runs/GenGAN_{self.filename.split("/")[-1].split(".")[0]}')
        
        # Définition des éléments externes nécessaires (comme les objets Skeleton)
        ske_objects = self.dataset.videoSke.ske # Assurez-vous que c'est une liste d'objets Skeleton

        print(f"Starting Training for {n_epochs} epochs with WGAN-GP...")
        
        for epoch in range(n_epochs):
            for i, (ske_vec, real_image) in enumerate(self.dataloader):
                batch_size = ske_vec.size(0)
                
                # --- Préparation de l'Image Conditionnelle (Squelette Dessiné) ---
                # Récupérer les objets Skeleton originaux pour créer l'image conditionnelle
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                batch_ske_objects = ske_objects[start_idx:end_idx]
                ske_img_cond = self.create_ske_image_batch(batch_ske_objects)
                
                # Mettre les données sur le device (si vous utilisez CUDA)
                # ske_vec, real_image, ske_img_cond = ske_vec.to(device), real_image.to(device), ske_img_cond.to(device)

                # =================================================================
                # (A) Entraînement du Discriminateur (D) : Min_G Max_D [ D(x) - D(G(z)) ] + GP
                # =================================================================
                self.netD.zero_grad()
                
                # --- 1. Perte D sur les Échantillons Réels ---
                # D(Squelette + Image Réelle)
                real_pair = torch.cat((ske_img_cond, real_image), 1)
                output_real = self.netD(real_pair).mean()
                
                # --- 2. Perte D sur les Échantillons Faux ---
                # D(Squelette + Image Fausse). NOTE: fake_image.detach()
                fake_image = self.netG(ske_vec)
                fake_pair = torch.cat((ske_img_cond, fake_image.detach()), 1)
                output_fake = self.netD(fake_pair).mean()
                errD_fake = output_fake # Moyenne de la sortie (D(Fake))
                
                # Perte de Wasserstein : D(Fake) - D(Real) (pour minimiser la distance)
                errD_Wasserstein = errD_fake - output_real # output_real est calculé plus haut

                # 3. Gradient Penalty (GP)
                # CORRECTION ici : on passe ske_img_cond comme troisième argument
                gradient_penalty = self.compute_gradient_penalty(real_image.data, fake_image.data, ske_img_cond.data) 
                # Note : L'utilisation de .data est souvent découragée; il est préférable d'utiliser .detach() si nécessaire.
                # Ici, real_image et fake_image sont les images 3 canaux. ske_img_cond est l'image 3 canaux.
                
                # Perte totale D = W Loss + GP * lambda_gp
                errD = errD_Wasserstein + lambda_gp * gradient_penalty
                errD.backward()
                self.optimizerD.step()

                # =================================================================
                # (B) Entraînement du Générateur (G) : Min_G [ -D(G(z)) + lambda_L1 * L1 ]
                # =================================================================
                
                # On ne met à jour G que tous les 1 ou 5 cycles de D (souvent 1 cycle de G pour 5 de D dans WGAN)
                # Pour l'instant, faisons 1:1 pour la simplicité, mais optimisez si l'entraînement est instable.
                
                self.netG.zero_grad()
                
                # 1. Perte Adversaire (GAN Loss) : -D(G(z))
                # On réutilise fake_image SANS detach()
                output_G = self.netD(torch.cat((ske_img_cond, fake_image), 1)).mean()
                errG_GAN = -output_G # Minimiser D(G(z)) équivaut à maximiser -D(G(z))
                
                # 2. Perte de Reconstruction (L1 Loss)
                errG_L1 = self.criterionL1(fake_image, real_image) * lambda_l1
                
                # Perte totale G
                errG = errG_GAN + errG_L1
                errG.backward()
                self.optimizerG.step()
                
                # -----------------------------------------------
                # 3. Logging et Sauvegarde
                # -----------------------------------------------
                if i % 10 == 0:
                    print(f'[{epoch+1}/{n_epochs}][{i}/{len(self.dataloader)}] Loss_D: {errD.item():.4f} | W_Loss: {errD_Wasserstein.item():.4f} | Loss_G: {errG.item():.4f} (GAN: {errG_GAN.item():.4f} L1: {errG_L1.item():.4f})')

                # Sauvegarde du modèle périodiquement (Utilisez des noms différents pour G et D)
                if (i % 50 == 0) and (i > 0):
                    writer.add_scalars('Loss_D', {'W_Loss': errD_Wasserstein.item(), 'Total_D': errD.item()}, epoch * len(self.dataloader) + i)
                    writer.add_scalars('Loss_G', {'GAN': errG_GAN.item(), 'L1': errG_L1.item(), 'Total_G': errG.item()}, epoch * len(self.dataloader) + i)

                if (epoch + 1) % 100 == 0 and i == 0:
                    torch.save(self.netG.state_dict(), self.filename.replace('.pth', '_G.pth'))
                    torch.save(self.netD.state_dict(), self.filename.replace('.pth', '_D.pth'))


            # Sauvegarder le modèle final
            torch.save(self.netG.state_dict(), self.filename.replace('.pth', '_G.pth'))
            torch.save(self.netD.state_dict(), self.filename.replace('.pth', '_D.pth'))
            print(f"Training finished. Final models saved.")


    def generate(self, ske):
        """ generator of image from skeleton """
        # TP-TODO: Fonction de génération
        self.netG.eval()  # Mettre le modèle en mode évaluation
        
        with torch.no_grad():
            # Préparer le squelette: appliquer les transformations et créer un batch de taille 1
            ske_t = self.dataset.preprocessSkeleton(ske)
            ske_t_batch = ske_t.unsqueeze(0)        # make a batch
            
            # Générer l'image normalisée [-1, 1]
            normalized_output = self.netG(ske_t_batch)
            
            # Convertir le tenseur normalisé en image OpenCV (dénormalisée)
            res = self.dataset.tensor2image(normalized_output[0])      # get image 0 from the batch
            
            # Remettre l'image en [0, 255] et en uint8 pour cv2.imshow
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

    #if False:
    if True:    # train or load
        # Train
        gen = GenGAN(targetVideoSke, False)
        gen.train(10) #5) #200)
    else:
        gen = GenGAN(targetVideoSke, loadFromFile=True)    # load from file        


    for i in range(targetVideoSke.skeCount()):
        image = gen.generate(targetVideoSke.ske[i])
        #image = image*255
        nouvelle_taille = (256, 256) 
        image = cv2.resize(image, nouvelle_taille)
        cv2.imshow('Image', image)
        key = cv2.waitKey(100)
