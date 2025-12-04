import numpy as np
import cv2
import os
import pickle
import sys
import math

from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter # Use standard SummaryWriter

# Assuming VideoSkeleton, VideoReader, and Skeleton are implemented correctly
from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton

# Set default tensor type (good practice, but not strictly necessary for GPU)
torch.set_default_dtype(torch.float32)

# --- Utility Classes ---

class SkeToImageTransform:
    """Transforms a Skeleton object into a rendered image (3-channel numpy array)."""
    def __init__(self, image_size):
        self.imsize = image_size

    def __call__(self, ske):
        # Create a white canvas (H, W, 3)
        image = np.ones((self.imsize, self.imsize, 3), dtype=np.uint8) * 255
        # Draw the skeleton onto the image
        ske.draw(image)
        # Convert BGR (OpenCV default) to RGB for consistency with PIL/Torch
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


class VideoSkeletonDataset(Dataset):
    """
    Dataset class to pair skeleton data (input) with real video frames (target).
    Input can be a skeleton vector or a rendered skeleton image.
    """
    def __init__(self, videoSke, ske_reduced, source_transform=None, target_transform=None):
        self.videoSke = videoSke
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.ske_reduced = ske_reduced
        print(f"VideoSkeletonDataset: reduced_skeleton={ske_reduced} (Dim: {Skeleton.reduced_dim} or {Skeleton.full_dim})")


    def __len__(self):
        return self.videoSke.skeCount()


    def __getitem__(self, idx):
        # 1. Preprocess skeleton (source)
        ske = self.videoSke.ske[idx]
        ske_input = self.preprocessSkeleton(ske)
        
        # 2. Preprocess image (target)
        image = Image.open(self.videoSke.imagePath(idx)).convert('RGB') # Ensure 3 channels
        if self.target_transform:
            image = self.target_transform(image)
            
        return ske_input, image

    
    def preprocessSkeleton(self, ske):
        """Applies source_transform or converts skeleton to a 26-dim tensor."""
        if self.source_transform:
            # If source_transform is provided (e.g., SkeToImageTransform), use it
            ske = self.source_transform(ske)
        else:
            # Otherwise, convert the vector to the format (N_dim, 1, 1) for the MLP
            ske = torch.from_numpy( ske.__array__(reduced=self.ske_reduced).flatten() )
            ske = ske.to(torch.float32)
            ske = ske.reshape( ske.shape[0],1,1)
        return ske


    def tensor2image(self, normalized_image_tensor):
        """Converts a PyTorch tensor (C, H, W) in [-1, 1] to a denormalized OpenCV image."""
        
        # 1. Move to CPU and convert to numpy array
        numpy_image = normalized_image_tensor.cpu().numpy()
        
        # 2. Transpose dimensions (C, H, W) to (H, W, C)
        numpy_image = np.transpose(numpy_image, (1, 2, 0))
        
        # 3. Denormalize from [-1, 1] to [0, 1]
        denormalized_image = numpy_image * 0.5 + 0.5
        
        # 4. Convert RGB (normalized) to BGR (OpenCV format)
        denormalized_output_bgr = cv2.cvtColor(denormalized_image, cv2.COLOR_RGB2BGR)
        
        return denormalized_output_bgr


class SelfAttention(nn.Module):
    """Implémentation du Non-local Block pour l'auto-attention."""
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        
        # Réduction de dimension pour Q et K pour l'efficacité (C/8)
        self.f = nn.Conv2d(in_channels, in_channels // 8, 1) # Query
        self.g = nn.Conv2d(in_channels, in_channels // 8, 1) # Key
        
        # Value (pas de réduction)
        self.h = nn.Conv2d(in_channels, in_channels, 1) # Value
        
        # Paramètre d'échelle (initialisé à zéro, pour commencer comme une fonction identité)
        self.gamma = nn.Parameter(torch.zeros(1)) 
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, H, W = x.size()
        N = H * W # Nombre total de positions (pixels)
        
        # 1. Calcul de Query, Key, Value
        # Query: (B, N, C/8)
        proj_query = self.f(x).view(batch_size, -1, N).permute(0, 2, 1) 
        # Key: (B, C/8, N)
        proj_key = self.g(x).view(batch_size, -1, N) 
        # Value: (B, C, N)
        proj_value = self.h(x).view(batch_size, -1, N) 

        # 2. Carte d'Attention (Energy): S = Q * K_T
        # energy: (B, N, N)
        energy = torch.bmm(proj_query, proj_key) 
        
        # 3. Normalisation (Softmax)
        attention = self.softmax(energy) # Matrice des poids d'attention

        # 4. Contexte Pondéré: O = V * Attention_T
        # out: (B, C, N)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1)) # Transpose attention pour le bmm
        
        # 5. Reshape et combinaison avec l'entrée
        out = out.view(batch_size, C, H, W)
        
        # Output avec connexion résiduelle: y = gamma * out + x
        out = self.gamma * out + x
        return out


# --- Model Weights Initialization ---

def init_weights(m):
    """Initializes weights for Conv and BatchNorm layers."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# --- Generator Architectures ---

class GenNNSke26ToImage(nn.Module):
    """Simple MLP Generator: Skeleton Vector (26-dim) -> Image (3x64x64)."""
    def __init__(self):
        super().__init__()
        self.input_dim = Skeleton.reduced_dim # 26
        self.output_dim = 3 * 64 * 64 
        
        hidden_dim = 1024 
        
        self.model = nn.Sequential(
            # 1. Skeleton Vector -> Latent Space
            nn.Linear(self.input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 2. Latent Space -> Flattened Image
            nn.Linear(hidden_dim, self.output_dim),
        )
        self.apply(init_weights)

    def forward(self, z):
        # Input: (batch_size, 26, 1, 1) -> Squeeze to (batch_size, 26) for nn.Linear
        z = z.squeeze(-1).squeeze(-1) 
        
        # Pass through linear layers
        img = self.model(z) 

        # Reshape output to image format (batch_size, 3, 64, 64)
        img = img.view(img.size(0), 3, 64, 64)
        
        # Apply Tanh to normalize output to [-1, 1]
        img = torch.tanh(img)
        
        return img

# --- Generator Architecture Modifiée ---
class GenNNSkeImToImage(nn.Module):
    """
    U-Net Generator pour la traduction Image-to-Image (Skeleton Image -> Real Image)
    avec Self-Attention dans le Bottleneck.
    """
    def __init__(self):
        super().__init__()
        self.input_channels = 3
        ngf = 64 # Base number of filters
        
        # === ENCODER (Downsampling) ===
        # C1, C2, C3 restent inchangés
        self.enc1 = nn.Sequential(nn.Conv2d(self.input_channels, ngf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True))
        self.enc2 = self._make_down_block(ngf, ngf * 2) 
        self.enc3 = self._make_down_block(ngf * 2, ngf * 4) 
        
        # C4 (8x8 -> 4x4) (Latent Space)
        self.enc4 = self._make_down_block(ngf * 4, ngf * 8) 
        
        # === NOUVEAU: MODULE D'ATTENTION ===
        self.attention = SelfAttention(ngf * 8) # Appliqué sur le feature map 4x4x512

        # === DECODER (Upsampling) ===
        # D1 (4x4 -> 8x8). L'entrée est doublée comme dans votre architecture originale.
        self.dec1 = self._make_up_block(ngf * 8 + ngf * 8, ngf * 4)
        # D2, D3, D4 restent inchangés (gestion des Skip Connections)
        self.dec2 = self._make_up_block(ngf * 4 + ngf * 4, ngf * 2)
        self.dec3 = self._make_up_block(ngf * 2 + ngf * 2, ngf)
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(ngf + ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
        self.apply(init_weights)

    def _make_down_block(self, in_channels, out_channels):
        """Creates a standard encoder block (Conv + BatchNorm + LeakyReLU)"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def _make_up_block(self, in_channels, out_channels, activation=nn.ReLU(True)):
        """Creates a standard decoder block (ConvT + BatchNorm + ReLU)"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            activation
        )

    def forward(self, x):
        # === Encoder ===
        e1 = self.enc1(x) 
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4_conv = self.enc4(e3) # Output de la convolution (4x4x512)
        
        # === Application de l'Attention ===
        e4_attn = self.attention(e4_conv) 

        # === Decoder avec Skip Connections ===
        # D1: Utilise l'output attentif (e4_attn)
        # On concatène e4_attn avec lui-même pour l'entrée de d1 (selon votre design original)
        d1 = self.dec1(torch.cat([e4_attn, e4_attn], 1)) 
        
        # D2, D3, D4 (Unchanged Skip Connections)
        d2 = self.dec2(torch.cat([d1, e3], 1)) 
        d3 = self.dec3(torch.cat([d2, e2], 1)) 
        img = self.dec4(torch.cat([d3, e1], 1))
        
        return img




# --- Main GAN Class ---

class GenVanillaNN():
    """ 
    A simple image regression generator (not a GAN) that maps skeleton data to an image.
    Supports both skeleton vector and skeleton image as input.
    """
    def __init__(self, videoSke, loadFromFile=False, optSkeOrImage=2):
        
        # 🌟 GPU INTEGRATION 1: Device Detection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        image_size = 64
        
        # 1. Model and Transform Selection based on input type
        if optSkeOrImage == 1: # skeleton_dim26 to image (MLP)
            self.netG = GenNNSke26ToImage()
            src_transform = None
            self.filename = '../data/Dance/DanceGenVanillaFromSke26.pth'
        else: # skeleton_image to image (U-Net)
            self.netG = GenNNSkeImToImage()
            src_transform = transforms.Compose([ 
                SkeToImageTransform(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            self.filename = '../data/Dance/DanceGenVanillaFromSkeim.pth'

        # 2. Target Image Transform
        tgt_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # Normalize to [-1, 1]
        ])
        
        # 🌟 GPU INTEGRATION 2: Move Model to Device
        self.netG.to(self.device)
            
        # 3. Dataset and DataLoader
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, target_transform=tgt_transform, source_transform=src_transform)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=16, shuffle=True)
        
        # 4. Model Loading
        if loadFromFile and os.path.isfile(self.filename):
            print(f"GenVanillaNN: Loading model from {self.filename}")
            # Ensure the weights are loaded to the correct device
            state_dict = torch.load(self.filename, map_location=self.device, weights_only=True)
            self.netG.load_state_dict(state_dict) 
            self.netG.eval()


    def train(self, n_epochs=20):
        
        # 1. Define Optimizer and Loss Function
        learning_rate = 0.0002
        beta1 = 0.5
        # The optimizer parameters already point to the model on self.device
        optimizerG = torch.optim.Adam(self.netG.parameters(), lr=learning_rate, betas=(beta1, 0.999))
        criterion = nn.L1Loss() # L1 Loss (MAE) is often preferred for image regression
        
        print(f"Starting Training for {n_epochs} epochs on {self.device}...")
        # writer = SummaryWriter(f'runs/GenVanillaNN_{self.filename.split("/")[-1].split(".")[0]}') # Uncomment if SummaryWriter is available
        
        self.netG.train() # Set generator to training mode
        
        for epoch in range(n_epochs):
            for i, (ske, real_image) in enumerate(self.dataloader):
                
                # 🌟 GPU INTEGRATION 3: Move Data to Device
                ske = ske.to(self.device)
                real_image = real_image.to(self.device)
                
                # Zero gradients
                self.netG.zero_grad()
                
                # Forward pass
                fake_image = self.netG(ske)
                
                # Calculate loss
                errG = criterion(fake_image, real_image)
                
                # Backward pass and optimization
                errG.backward()
                optimizerG.step()
                
                # Logging
                if i % 10 == 0:
                    print(f'[{epoch}/{n_epochs}][{i}/{len(self.dataloader)}] Loss_G: {errG.item():.4f}')
            
            # Save the model periodically
            if (epoch + 1) % 10 == 0:
                torch.save(self.netG.state_dict(), self.filename) 
                print(f"Model saved to {self.filename} after epoch {epoch+1}")

        # Save the final model
        torch.save(self.netG.state_dict(), self.filename)
        print(f"Training finished. Final model saved to {self.filename}")


    def generate(self, ske):
        """Generates an image from a skeleton posture."""
        
        self.netG.eval() # Set model to evaluation mode (crucial for Batch Norm/Dropout)
        
        with torch.no_grad():
            # Prepare skeleton: apply transforms and create a batch of size 1
            ske_t = self.dataset.preprocessSkeleton(ske)
            ske_t_batch = ske_t.unsqueeze(0) 
            
            # 🌟 GPU INTEGRATION 4: Move Input to Device
            ske_t_batch = ske_t_batch.to(self.device)
            
            # Generate the normalized image [-1, 1]
            normalized_output = self.netG(ske_t_batch)
            
            # Convert the normalized tensor to a denormalized OpenCV image
            # Note: tensor2image handles moving the tensor back to CPU for numpy conversion
            res = self.dataset.tensor2image(normalized_output[0])
            
            # Convert image to [0, 255] and uint8 for cv2.imshow
            res = (res * 255).astype(np.uint8)
            
            return res


if __name__ == '__main__':
    force = False
    optSkeOrImage = 1 # 1: skeleton vector (MLP), 2: skeleton image (U-Net)
    n_epoch = 1000
    train = True
    # train = False

    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == "true"
    else:
        filename = "../data/taichi1.mp4"
        
    print(f"--- GenVanillaNN Execution ---")
    print(f"Input File: {filename}")
    print(f"Training Mode: {train}")
    print(f"Input Type: {'Skeleton Vector' if optSkeOrImage == 1 else 'Skeleton Image'}")
    
    targetVideoSke = VideoSkeleton(filename)

    if train:
        gen = GenVanillaNN(targetVideoSke, loadFromFile=False, optSkeOrImage=optSkeOrImage)
        gen.train(n_epoch)
    else:
        gen = GenVanillaNN(targetVideoSke, loadFromFile=True, optSkeOrImage=optSkeOrImage)

    # Test/Generation Loop
    for i in range(targetVideoSke.skeCount()):
        image = gen.generate( targetVideoSke.ske[i] )
        new_size = (256, 256) 
        image = cv2.resize(image, new_size)
        cv2.imshow('Generated Image', image)
        key = cv2.waitKey(300)