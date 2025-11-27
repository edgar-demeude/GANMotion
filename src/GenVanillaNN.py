import numpy as np
import cv2
import os
import pickle
import sys
import math

from PIL import Image
import matplotlib.pyplot as plt
from torchvision.io import read_image

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from torchvision import transforms

#from tensorboardX import SummaryWriter

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton

torch.set_default_dtype(torch.float32)


class SkeToImageTransform:
    def __init__(self, image_size):
        self.imsize = image_size

    def __call__(self, ske):
        #image = Image.new('RGB', (self.imsize, self.imsize), (255, 255, 255))
        image = white_image = np.ones((self.imsize, self.imsize, 3), dtype=np.uint8) * 255
        ske.draw(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imshow('Image', image)
        # key = cv2.waitKey(-1)
        return image


class VideoSkeletonDataset(Dataset):
    def __init__(self, videoSke, ske_reduced, source_transform=None, target_transform=None):
        """ videoSkeleton dataset: 
                videoske(VideoSkeleton): video skeleton that associate a video and a skeleton for each frame
                ske_reduced(bool): use reduced skeleton (13 joints x 2 dim=26) or not (33 joints x 3 dim = 99)
        """
        self.videoSke = videoSke
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.ske_reduced = ske_reduced
        print("VideoSkeletonDataset: ",
              "ske_reduced=", ske_reduced, "=(", Skeleton.reduced_dim, " or ",Skeleton.full_dim,")" )


    def __len__(self):
        return self.videoSke.skeCount()


    def __getitem__(self, idx):
        # prepreocess skeleton (input)
        reduced = True
        ske = self.videoSke.ske[idx]
        ske = self.preprocessSkeleton(ske)
        # prepreocess image (output)
        image = Image.open(self.videoSke.imagePath(idx))
        if self.target_transform:
            image = self.target_transform(image)
        return ske, image

    
    def preprocessSkeleton(self, ske):
        if self.source_transform:
            ske = self.source_transform(ske)
        else:
            ske = torch.from_numpy( ske.__array__(reduced=self.ske_reduced).flatten() )
            ske = ske.to(torch.float32)
            ske = ske.reshape( ske.shape[0],1,1)
        return ske


    def tensor2image(self, normalized_image):
        numpy_image = normalized_image.detach().numpy()
        # Réorganiser les dimensions (C, H, W) en (H, W, C)
        numpy_image = np.transpose(numpy_image, (1, 2, 0))
        # passage a des images cv2 pour affichage
        numpy_image = cv2.cvtColor(np.array(numpy_image), cv2.COLOR_RGB2BGR)
        denormalized_image = numpy_image * np.array([0.5, 0.5, 0.5]) + np.array([0.5, 0.5, 0.5])
        denormalized_output = denormalized_image * 1
        return denormalized_output




def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class GenNNSke26ToImage(nn.Module):
    """ class that Generate a new image from videoSke from a new skeleton posture
            Fonc generator(Skeleton_dim26)->Image
    """
    def __init__(self):
        super().__init__()
        self.input_dim = Skeleton.reduced_dim # 26
        self.output_dim = 3 * 64 * 64        # 3 canaux * 64 * 64 = 12288
        
        # Définir une dimension cachée pour ajouter de la profondeur
        hidden_dim = 1024 
        
        self.model = nn.Sequential(
            # 1. Couche Linéaire Squelette -> Espace Latent (26 -> 1024)
            nn.Linear(self.input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), # Ajout d'une Batch Norm pour stabiliser l'entraînement
            nn.LeakyReLU(0.2, inplace=True), # LeakyReLU est souvent meilleur que ReLU pour les générateurs
            
            # 2. Couche Linéaire Espace Latent -> Image (1024 -> 12288)
            nn.Linear(hidden_dim, self.output_dim),
            
            # Note: La fonction d'activation finale (Tanh) sera appliquée dans le forward après le reshape.
        )
        self.apply(init_weights) # Initialisation des poids
        print(self.model)

    def forward(self, z):
        # L'entrée est (batch_size, 26, 1, 1). On doit la rendre (batch_size, 26) pour nn.Linear
        z = z.squeeze(-1).squeeze(-1) # -> (batch_size, 26)
        
        # Passer par les couches linéaires
        img = self.model(z) # -> (batch_size, 12288)

        # Reshape l'output en (batch_size, 3, 64, 64)
        img = img.view(img.size(0), 3, 64, 64) # -> (batch_size, 3, 64, 64)
        
        # Appliquer la Tanh finale pour mettre l'image dans l'intervalle [-1, 1]
        img = torch.tanh(img)
        
        return img


class SelfAttention(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim):
        super(SelfAttention,self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_dim , in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_dim , in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_dim , in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
        """
        m_batchsize, C, width, height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize, -1, width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query, proj_key) 
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height) # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        
        out = self.gamma*out + x
        return out


class GenNNSkeImToImage(nn.Module):
    """
    Générateur de traduction d'image (Skeleton Image -> Real Image)
    Utilise une architecture de type U-Net avec des Skip Connections pour des résultats optimaux.
    """
    def __init__(self):
        super().__init__()
        self.input_channels = 3
        ngf = 64 # Nombre de filtres de base
        
        # === ENCODEUR (Downsampling) ===
        # C1 (64x64)
        self.enc1 = nn.Sequential(
            nn.Conv2d(self.input_channels, ngf, 4, 2, 1, bias=False), # -> 32x32
            nn.LeakyReLU(0.2, inplace=True)
        )
        # C2 (32x32)
        self.enc2 = self._make_down_block(ngf, ngf * 2) # -> 16x16
        # C3 (16x16)
        self.enc3 = self._make_down_block(ngf * 2, ngf * 4) # -> 8x8
        # C4 (8x8)
        self.enc4 = self._make_down_block(ngf * 4, ngf * 8) # -> 4x4 (Espace Latent)

        # === DÉCODEUR (Upsampling) ===
        # D1 (4x4 -> 8x8)
        # Le Décodeur reçoit 512 (du ConvT) + 512 (du skip C4) = 1024.
        self.dec1 = self._make_up_block(ngf * 8 + ngf * 8, ngf * 4)
        # D2 (8x8 -> 16x16)
        # Reçoit 256 (du ConvT) + 256 (du skip C3) = 512.
        self.dec2 = self._make_up_block(ngf * 4 + ngf * 4, ngf * 2)
        # D3 (16x16 -> 32x32)
        # Reçoit 128 (du ConvT) + 128 (du skip C2) = 256.
        self.dec3 = self._make_up_block(ngf * 2 + ngf * 2, ngf)
        
        # D4 (32x32 -> 64x64)
        # Reçoit 64 (du ConvT) + 64 (du skip C1) = 128.
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(ngf + ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh() # Sortie finale dans [-1, 1]
        )
        
        self.apply(init_weights)
        print("Modèle optimisé (U-Net Style) :")
        print(self)

    def _make_down_block(self, in_channels, out_channels):
        """ Crée un bloc de convolution pour l'encodeur (Conv + BatchNorm + LeakyReLU) """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def _make_up_block(self, in_channels, out_channels, activation=nn.ReLU(True)):
        """ Crée un bloc de déconvolution pour le décodeur (ConvT + BatchNorm + ReLU) """
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            activation
        )

    def forward(self, x):
        # === Encodeur (Sauvegarde des sorties pour les Skip Connections) ===
        e1 = self.enc1(x) 
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3) # Espace Latent (N, 512, 4, 4)

        # === Décodeur avec Skip Connections ===
        # D1: Combine E4 et E4 (pas de véritable skip car on est au centre, mais on le traite comme tel)
        d1 = self.dec1(torch.cat([e4, e4], 1)) # Concatenation (N, 512+512, 4, 4) -> (N, 256, 8, 8)
        
        # D2: Combine D1 et E3 (Skip Connection)
        d2 = self.dec2(torch.cat([d1, e3], 1)) # Concatenation (N, 256+256, 8, 8) -> (N, 128, 16, 16)
        
        # D3: Combine D2 et E2 (Skip Connection)
        d3 = self.dec3(torch.cat([d2, e2], 1)) # Concatenation (N, 128+128, 16, 16) -> (N, 64, 32, 32)
        
        # D4: Combine D3 et E1 (Skip Connection)
        # La dernière couche est spéciale (pas de BatchNorm, Tanh à la fin)
        img = self.dec4(torch.cat([d3, e1], 1)) # Concatenation (N, 64+64, 32, 32) -> (N, 3, 64, 64)
        
        return img


class GenVanillaNN():
    """ class that Generate a new image from a new skeleton posture
        Fonc generator(Skeleton)->Image
    """
    def __init__(self, videoSke, loadFromFile=False, optSkeOrImage=1):
        image_size = 64
        if optSkeOrImage==1:        # skeleton_dim26 to image
            self.netG = GenNNSke26ToImage()
            # src_transform = transforms.Compose([ transforms.ToTensor(),
            #                                      ])
            src_transform = None
            self.filename = '../data/Dance/DanceGenVanillaFromSke26.pth'
        else:                       # skeleton_image to image
            self.netG = GenNNSkeImToImage()
            src_transform = transforms.Compose([ SkeToImageTransform(image_size),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                 ])
            self.filename = '../data/Dance/DanceGenVanillaFromSkeim.pth'



        tgt_transform = transforms.Compose([
                            transforms.Resize(image_size),
                            transforms.CenterCrop(image_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            # [transforms.Resize((64, 64)),
                            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])
                            # ouput image (target) are in the range [-1,1] after normalization
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, target_transform=tgt_transform, source_transform=src_transform)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=16, shuffle=True)
        if loadFromFile and os.path.isfile(self.filename):
            print("GenVanillaNN: Load=", self.filename)
            print("GenVanillaNN: Current Working Directory: ", os.getcwd())
            
            # MODIFICATION ICI : Charger le state_dict
            # 1. Charger les poids depuis le fichier
            state_dict = torch.load(self.filename)
            # 2. Les appliquer à l'instance self.netG déjà créée
            self.netG.load_state_dict(state_dict) 
            self.netG.eval() # Optionnel mais bonne pratique pour un modèle chargé pour inférence


    def train(self, n_epochs=20):
        # TP-TODO: Fonction d'entraînement
        
        # 1. Définir l'optimiseur et la fonction de perte
        learning_rate = 0.0002
        beta1 = 0.5
        optimizerG = torch.optim.Adam(self.netG.parameters(), lr=learning_rate, betas=(beta1, 0.999))
        
        # L1 Loss pour la régression d'image (MAE), souvent plus adaptée que MSE pour la netteté.
        criterion = nn.L1Loss()
        
        print(f"Starting Training for {n_epochs} epochs...")
        #writer = SummaryWriter(f'runs/GenVanillaNN_{self.filename.split("/")[-1].split(".")[0]}')
        
        for epoch in range(n_epochs):
            for i, (ske, real_image) in enumerate(self.dataloader):
                
                # Mettre à jour le Générateur
                self.netG.zero_grad()
                
                # Générer l'image
                fake_image = self.netG(ske)
                
                # Calculer la perte du générateur
                errG = criterion(fake_image, real_image)
                
                # Backpropagation et optimisation
                errG.backward()
                optimizerG.step()
                
                # Affichage des statistiques (optionnel)
                if i % 10 == 0:
                    print(f'[{epoch}/{n_epochs}][{i}/{len(self.dataloader)}] Loss_G: {errG.item():.4f}')
                    # if i == 0:
                    #     # Sauvegarder un exemple de sortie pour TensorBoard
                    #     writer.add_image('Real Image', self.dataset.tensor2image(real_image[0]), epoch)
                    #     writer.add_image('Fake Image', self.dataset.tensor2image(fake_image[0]), epoch)
            
            # Sauvegarder le modèle périodiquement
            if (epoch + 1) % 100 == 0:
                # MODIFICATION : Sauvegarder le state_dict
                torch.save(self.netG.state_dict(), self.filename) 
                print(f"Model saved to {self.filename} after epoch {epoch+1}")

        # Sauvegarder le modèle final
        # MODIFICATION : Sauvegarder le state_dict
        torch.save(self.netG.state_dict(), self.filename)
        print(f"Training finished. Final model saved to {self.filename}")


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
    optSkeOrImage = 2         # use as input a skeleton (1) or an image with a skeleton drawed (2)
    n_epoch = 10  # 200
    # train = False
    train = True

    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == "true"
    else:
        filename = "../data/taichi1.mp4"
    print("GenVanillaNN: Current Working Directory=", os.getcwd())
    print("GenVanillaNN: Filename=", filename)
    print("GenVanillaNN: Filename=", filename)

    targetVideoSke = VideoSkeleton(filename)

    if train:
        # Train
        gen = GenVanillaNN(targetVideoSke, loadFromFile=False, optSkeOrImage=optSkeOrImage)
        gen.train(n_epoch)
    else:
        gen = GenVanillaNN(targetVideoSke, loadFromFile=True, optSkeOrImage=optSkeOrImage)    # load from file        

    # Test with a second video
    for i in range(targetVideoSke.skeCount()):
        image = gen.generate( targetVideoSke.ske[i] )
        #image = image*255
        nouvelle_taille = (256, 256) 
        image = cv2.resize(image, nouvelle_taille)
        cv2.imshow('Image', image)
        key = cv2.waitKey(100)
