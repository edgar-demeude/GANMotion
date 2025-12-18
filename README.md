
# GANMotion — Motion transfer demo

##### Github link: https://github.com/edgar-demeude/GANMotion?tab=readme-ov-file

This repository contains code and models for a skeleton-to-image system used in a practical tutorial for deep learning and image synthesis. The pipeline extracts skeleton poses from a source video and generates images of a target person performing the same poses.

This README documents:
- 2 minute demo video;
- how to run inference with trained networks (Vanilla generator and GAN);
- how to train the networks from scratch (basic commands and recommended hyperparameters);
- a concise list of the code changes and model improvements made to improve background and texture quality.

[See the course main page with the description of this tutorial/TP](http://alexandre.meyer.pages.univ-lyon1.fr/m2-apprentissage-profond-image/am/tp_dance/)

## Video running the demo

[![](https://markdown-videos-api.jorgenkh.no/youtube/oIhg813cOwE)](https://youtu.be/oIhg813cOwE)

## Contents
- `src/GenVanillaNN.py` — simple regression generator (skeleton -> image) and dataset utilities.
- `src/GenGAN.py` — conditional WGAN-GP implementation with a Patch-style discriminator (skeleton conditioned).
- `src/DanceDemo.py` — utility to run a demo combining source motion and generator output.
- `src/DisplaySkeleton.py` — helper to visualize rendered skeleton inputs.
- `data/` — example videos and saved checkpoints (check for `.pth` files in `data/Dance/`).
- `exportedDances/` — generated result videos (examples for Nearest, GenVanilla, GenGAN outputs).

**⚠️ Important:** Model weight files (`.pth` checkpoints in `data/Dance/`) are not included in this repository as they exceed 100 MB. You will need to train the models yourself using the provided training scripts or obtain them separately before running inference.

## Requirements
- Python 3.8+ (tested with 3.8/3.9)
- PyTorch (match your CUDA version) and torchvision
- OpenCV (`opencv-python`)
- Pillow
- tensorboard (optional, for logging)
- ffmpeg (optional, for screen capture or combining frames into a video)

Install via pip (example):

```powershell
python -m pip install torch torchvision opencv-python pillow tensorboard
```

## Training the networks

Files to look at:
- `src/GenVanillaNN.py` — `GenVanillaNN.train()` implements L1 (and optional perceptual) training for the generator.
- `src/GenGAN.py` — `GenGAN.train()` implements conditional WGAN-GP training (discriminator + generator updates).

Basic training commands (PowerShell examples):

Vanilla generator (quick test, few epochs):

```powershell
cd src
python GenVanillaNN.py # set train=True in GenVanillaNN.main
```

GAN (WGAN-GP):

```powershell
cd src
python GenGAN.py # set train=True in GenGAN.main
```

Recommended hyperparameters (starting point):
- Learning rate: 2e-4 (Adam)
- Batch size: 4 or 2 for 512×512 images (use accumulation if GPU memory is low)
- WGAN-GP lambda_gp: 10
- L1 weight for generator: 50 (in GenGAN)
- Perceptual loss weight (if enabled): 0.05–0.1

Checkpoints:
- The training scripts save checkpoints periodically into `data/Dance/` (filenames depend on the script). The code saves model state_dict and optimizer state when periodic checkpointing is enabled.

## Inference with a trained network

Pretrained checkpoints (if available) are expected under `data/Dance/` with names like `DanceGenGAN_G_checkpoint.pth` or `DanceGenVanillaFromSkeim.pth`.

To run inference (no training) with the Vanilla generator (skeleton-image -> image):

```powershell
cd src
python GenVanillaNN.py # the script will run in inference if train=False in main
```

To run inference (no training) with the GAN generator:

```powershell
cd src
python GenGAN.py # the script will run in inference if train=False in main
```

Or run the `DanceDemo.py` selecting the generator type that points to the pretrained model (set `typeOfGen` to the right value in the demo or pass arguments where implemented).

Notes:
- On Windows set DataLoader `num_workers=0` (already set in the code).
- For high-resolution inference (512×512), use a GPU with sufficient VRAM. Batch size in `GenVanillaNN` is set small for that reason.

## How to run the final result

The `DanceDemo.py` script runs the pipeline: it reads a source video (motion), extracts skeletons, and for each pose uses the selected generator to produce a target-frame image. By default the demo displays the output live.

Run the demo (example):

```powershell
cd src
python DanceDemo.py
```

Generated demo recordings and exported result videos are saved to the `exportedDances/` folder (when the demo or recording helper is used). Check that folder to compare outputs from different models (Nearest, GenVanilla, GenGAN).

## Implemented features (summary)

Pipeline summary: from a source video, skeleton poses are detected and extracted frame-by-frame. These rendered skeletons are fed to a generator network (U-Net regression or a conditional GAN generator) that learns to transfer the pose onto a target person's appearance and produces synthetic images. The improvements listed below focus on background and texture quality and on training stability.

To improve background and texture quality and to make training robust, the following things were implemented in the codebase:

1. Self-Attention block
- A non-local self-attention module was added at the U-Net bottleneck to allow spatially global interactions and improve coherent background reconstruction.

2. Safer upsampling
- with Upsample (bilinear) + Conv2d patterns to reduce checkerboard artifacts and preserve texture.

3. Normalization changes
- BatchNorm in the generator and discriminator to improve stability when training with small batch sizes (common for high-resolution frames).

4. Perceptual loss (VGG)
- An optional perceptual loss using pretrained VGG features was added and combined with L1 to encourage perceptually plausible textures and details in the background.

5. Robust checkpoint loading/saving
- Checkpoint loader accepts multiple formats, strips `module.` prefixes (DataParallel), and loads with `strict=False` to ease experiments with architecture changes. Periodic checkpoint saving now stores epoch and optimizer state in a single dict.

6. Fixed dataset alignment and shapes
- Ensured skeleton rendering (`SkeToImageTransform`) and target transforms use the same image size and cropping so the network receives correctly aligned pairs. Power-of-two spatial sizes (e.g., 64, 128, 256, 512) are used to avoid fractional-downsample shape mismatches in the U-Net.

7. Utility: skeleton visualization
- A small utility `DisplaySkeleton.py` was added to visualize skeleton renderings and verify input alignment before training.

## Troubleshooting
- If you see tensor size mismatch errors in U-Net (e.g. 134 vs 135), check that `image_size` is a power of two consistent with the number of downsampling layers. Use 64, 128, 256, 512, etc.
- If training is unstable with GAN, reduce the generator learning rate, use spectral normalization on discriminator, or decrease the discriminator updates per generator step.
