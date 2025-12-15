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
from torchvision import models
from torch.utils.tensorboard import SummaryWriter  # Use standard SummaryWriter

# Assuming VideoSkeleton, VideoReader, and Skeleton are implemented correctly
from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton

# Set default tensor type
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
        image = Image.open(self.videoSke.imagePath(idx)).convert('RGB')  # Ensure 3 channels

        if self.target_transform:
            image = self.target_transform(image)

        return ske_input, image

    def preprocessSkeleton(self, ske):
        """Applies source_transform or converts skeleton to a 26-dim tensor."""
        if self.source_transform:
            # source_transform (e.g. SkeToImageTransform + ToTensor + Normalize)
            ske = self.source_transform(ske)  # must return a Tensor (C,H,W)
        else:
            # Skeleton vector -> (N_dim,1,1) for the MLP
            ske = torch.from_numpy(ske.__array__(reduced=self.ske_reduced).flatten())
            ske = ske.to(torch.float32)
            ske = ske.reshape(ske.shape[0], 1, 1)
        return ske

    def tensor2image(self, normalized_image_tensor):
        """Converts a PyTorch tensor (C, H, W) in [-1, 1] to a denormalized OpenCV image."""
        # 1. Move to CPU and convert to numpy array
        numpy_image = normalized_image_tensor.cpu().numpy()
        # 2. Transpose dimensions (C, H, W) to (H, W, C)
        numpy_image = np.transpose(numpy_image, (1, 2, 0))
        # 3. Denormalize from [-1, 1] to [0, 1]
        denormalized_image = numpy_image * 0.5 + 0.5
        denormalized_image = np.clip(denormalized_image, 0.0, 1.0)
        # 4. Convert RGB (normalized) to BGR (OpenCV format)
        denormalized_output_bgr = cv2.cvtColor(denormalized_image, cv2.COLOR_RGB2BGR)
        return denormalized_output_bgr


class SelfAttention(nn.Module):
    """Non-local Self-Attention block."""

    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels

        # Reduced dimension for Q and K for efficiency (C/8)
        self.f = nn.Conv2d(in_channels, in_channels // 8, 1)  # Query
        self.g = nn.Conv2d(in_channels, in_channels // 8, 1)  # Key
        # Value (no reduction)
        self.h = nn.Conv2d(in_channels, in_channels, 1)       # Value

        # Learnable scale parameter
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, H, W = x.size()
        N = H * W  # Number of positions

        # 1. Query, Key, Value
        # Query: (B, N, C/8)
        proj_query = self.f(x).view(batch_size, -1, N).permute(0, 2, 1)
        # Key: (B, C/8, N)
        proj_key = self.g(x).view(batch_size, -1, N)
        # Value: (B, C, N)
        proj_value = self.h(x).view(batch_size, -1, N)

        # 2. Energy (attention scores)
        # energy: (B, N, N)
        energy = torch.bmm(proj_query, proj_key)

        # 3. Softmax
        attention = self.softmax(energy)

        # 4. Weighted context
        # out: (B, C, N)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))

        # 5. Reshape and residual connection
        out = out.view(batch_size, C, H, W)
        out = self.gamma * out + x
        return out


# --- Model Weights Initialization ---

def init_weights(m):
    """Initializes weights for Conv and BatchNorm layers."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


# --- Generator Architectures ---

class GenNNSke26ToImage(nn.Module):
    """
    Simple MLP Generator: Skeleton Vector (26-dim) -> Image (3 x image_size x image_size).
    """

    def __init__(self, image_size=64):
        super().__init__()
        self.input_dim = Skeleton.reduced_dim  # 26
        self.image_size = image_size
        self.output_dim = 3 * image_size * image_size

        hidden_dim = 1024

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, self.output_dim),
        )

        self.apply(init_weights)

    def forward(self, z):
        # Input: (batch_size, 26, 1, 1) -> (batch_size, 26)
        z = z.squeeze(-1).squeeze(-1)
        img = self.model(z)
        img = img.view(img.size(0), 3, self.image_size, self.image_size)
        img = torch.tanh(img)
        return img


class GenNNSkeImToImage(nn.Module):
    """
    U-Net Generator for Image-to-Image translation (Skeleton Image -> Real Image)
    with an optional Self-Attention block at the bottleneck.
    """

    def __init__(self, image_size, base_filters=64):
        super().__init__()
        self.input_channels = 3
        ngf = base_filters

        # ENCODER
        self.enc1 = nn.Sequential(
            nn.Conv2d(self.input_channels, ngf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc2 = self._make_down_block(ngf, ngf * 2)
        self.enc3 = self._make_down_block(ngf * 2, ngf * 4)
        self.enc4 = self._make_down_block(ngf * 4, ngf * 8)

        # Self-Attention at bottleneck
        self.attention = SelfAttention(ngf * 8)

        # DECODER
        self.dec1 = self._make_up_block(ngf * 8 + ngf * 8, ngf * 4)
        self.dec2 = self._make_up_block(ngf * 4 + ngf * 4, ngf * 2)
        self.dec3 = self._make_up_block(ngf * 2 + ngf * 2, ngf)
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(ngf + ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

        self.apply(init_weights)

    def _make_down_block(self, in_channels, out_channels):
        """Creates a standard encoder block (Conv + BatchNorm + LeakyReLU)."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def _make_up_block(self, in_channels, out_channels, activation=nn.ReLU(True)):
        """Creates a standard decoder block (ConvT + BatchNorm + ReLU)."""
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            activation,
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4_conv = self.enc4(e3)

        # Attention
        e4_attn = self.attention(e4_conv)

        # Decoder with skip connections
        d1 = self.dec1(torch.cat([e4_attn, e4_attn], 1))
        d2 = self.dec2(torch.cat([d1, e3], 1))
        d3 = self.dec3(torch.cat([d2, e2], 1))
        img = self.dec4(torch.cat([d3, e1], 1))

        return img


# --- Main Generator Wrapper ---

class GenVanillaNN:
    """
    Simple image regression generator that maps skeleton data to an image.
    Supports both skeleton vector and skeleton image as input.
    """

    def __init__(self, videoSke, loadFromFile=False, optSkeOrImage=2):
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        image_size = 512  # U-Net resolution
        batch_size = 2    # 2 for high-res stability

        # Model + source transform
        if optSkeOrImage == 1:
            self.netG = GenNNSke26ToImage(image_size)
            src_transform = None
            self.filename = '../data/Dance/DanceGenVanillaFromSke26.pth'
        else:
            self.netG = GenNNSkeImToImage(image_size)
            # ensure source transform returns a tensor with same normalization as targets
            src_transform = transforms.Compose([
                SkeToImageTransform(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            self.filename = '../data/Dance/DanceGenVanillaFromSkeim.pth'

        # Target image transform
        tgt_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5)),
        ])

        # Move model to device
        self.netG.to(self.device)

        # optional pretrained VGG for perceptual loss
        try:
            vgg = models.vgg16(pretrained=True).features
            vgg.to(self.device).eval()
            for p in vgg.parameters():
                p.requires_grad = False
            self.vgg = vgg
            print("VGG16 loaded for perceptual loss.")
        except Exception as e:
            self.vgg = None
            print(f"VGG16 unavailable, perceptual loss disabled: {e}")

        # Dataset and DataLoader
        self.dataset = VideoSkeletonDataset(
            videoSke,
            ske_reduced=True,
            target_transform=tgt_transform,
            source_transform=src_transform
        )
        self.dataloader = torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )

        # Load model if requested
        if loadFromFile and os.path.isfile(self.filename):
            print(f"GenVanillaNN: Loading model from {self.filename}")
            try:
                # Accept several checkpoint key conventions
                state = torch.load(self.filename, map_location=self.device)

                # common keys: 'model_state_dict', 'state_dict', or the file may already be a state_dict
                if isinstance(state, dict):
                    if 'model_state_dict' in state:
                        state_dict = state['model_state_dict']
                    elif 'state_dict' in state:
                        state_dict = state['state_dict']
                    else:
                        # assume dict is a raw state_dict
                        state_dict = state
                else:
                    state_dict = state

                # Normalize keys (remove 'module.' prefix from DataParallel if present)
                new_state = {}
                for k, v in state_dict.items():
                    new_key = k[len('module.'):] if k.startswith('module.') else k
                    new_state[new_key] = v

                # Load with strict=False to allow partial mismatch (e.g., BatchNorm -> InstanceNorm change)
                load_result = self.netG.load_state_dict(new_state, strict=False)
                self.netG.to(self.device)
                self.netG.eval()
                print("Model loaded (partial load allowed).")
                if getattr(load_result, 'missing_keys', None):
                    print(f"  Missing keys ({len(load_result.missing_keys)}): {load_result.missing_keys[:10]}")
                if getattr(load_result, 'unexpected_keys', None):
                    print(f"  Unexpected keys ({len(load_result.unexpected_keys)}): {load_result.unexpected_keys[:10]}")
                # Warn about common incompatibility
                if any(('running_mean' in k or 'running_var' in k) for k in new_state.keys()):
                    print("Note: checkpoint contains running stats (BatchNorm); if you switched to InstanceNorm weights may not map exactly and colors may change. Consider retraining or using a checkpoint trained with the same normalization.")
            except Exception as e:
                print(f"Warning: failed to load model '{self.filename}': {e}")

    def train(self, n_epochs=20):
        learning_rate = 0.0002
        beta1 = 0.5
        optimizer = torch.optim.Adam(self.netG.parameters(), lr=learning_rate, betas=(beta1, 0.999), weight_decay=0.0)
        criterion = nn.L1Loss()
        lambda_perc = 0.1

        print(f"Starting Training for {n_epochs} epochs on {self.device}...")
        self.netG.train()

        for epoch in range(n_epochs):
            for i, (ske, real_image) in enumerate(self.dataloader):
                ske = ske.to(self.device)
                real_image = real_image.to(self.device)

                self.netG.zero_grad()

                fake_image = self.netG(ske)

                l1 = criterion(fake_image, real_image)
                perc = self._perceptual_loss(real_image, fake_image) if self.vgg is not None else torch.tensor(0.0, device=self.device)
                lossG = l1 + lambda_perc * perc

                lossG.backward()
                optimizer.step()

                if i % 10 == 0:
                    print(f'[{epoch}/{n_epochs}][{i}/{len(self.dataloader)}] L1: {l1.item():.4f} Perc: {perc.item() if isinstance(perc, torch.Tensor) else 0.0:.4f}')

            if (epoch + 1) % 10 == 0:
                ckpt = {'epoch': epoch + 1, 'model_state_dict': self.netG.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
                save_path = f"{self.filename}.epoch{epoch+1}.pth"
                torch.save(ckpt, save_path)
                # Also save plain state_dict to the canonical filename for compatibility
                try:
                    torch.save(self.netG.state_dict(), self.filename)
                except Exception:
                    pass
                print(f"Checkpoint saved to {save_path}")

        ckpt = {'epoch': n_epochs, 'model_state_dict': self.netG.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
        save_path = f"{self.filename}.epoch{n_epochs}.pth"
        torch.save(ckpt, save_path)
        # Also write plain state_dict for backward compatibility with older loaders
        try:
            torch.save(self.netG.state_dict(), self.filename)
        except Exception:
            pass
        print(f"Training finished. Final checkpoint saved to {save_path}")

    def _vgg_preprocess(self, x):
        """Normalize tensor in [ -1, 1 ] to ImageNet stats for VGG."""
        x = (x + 1.0) * 0.5
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std

    def _perceptual_loss(self, real, fake, layers=(3, 8, 15)):
        if self.vgg is None:
            return torch.tensor(0.0, device=real.device)
        xr = self._vgg_preprocess(real)
        xf = self._vgg_preprocess(fake)
        loss = 0.0
        x_r = xr
        x_f = xf
        for i, layer in enumerate(self.vgg):
            x_r = layer(x_r)
            x_f = layer(x_f)
            if i in layers:
                loss = loss + F.mse_loss(x_r, x_f)
        return loss

    def generate(self, ske):
        """Generates an image from a skeleton posture."""
        self.netG.eval()  # Set model to evaluation mode
        with torch.no_grad():
            # Use dataset preprocessing to obtain the model input (tensor)
            ske_t = self.dataset.preprocessSkeleton(ske)
            # If preprocess returned a numpy image, convert and normalize
            if isinstance(ske_t, np.ndarray):
                ske_t = transforms.ToTensor()(ske_t)
                ske_t = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(ske_t)

            # Add batch dimension and send to device
            ske_t_batch = ske_t.unsqueeze(0).to(self.device)

            # Generate normalized image [-1, 1]
            normalized_output = self.netG(ske_t_batch)

            # Convert to OpenCV BGR image
            res = self.dataset.tensor2image(normalized_output[0])
            res = (res * 255).astype(np.uint8)
        return res


if __name__ == '__main__':
    force = False
    optSkeOrImage = 2  # 1: skeleton vector (MLP), 2: skeleton image (U-Net)
    n_epoch = 50
    # train = True
    train = False

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

    for i in range(targetVideoSke.skeCount()):
        image = gen.generate(targetVideoSke.ske[i])
        new_size = (512, 512)
        image = cv2.resize(image, new_size)
        cv2.imshow('Generated Image', image)
        key = cv2.waitKey(300)
