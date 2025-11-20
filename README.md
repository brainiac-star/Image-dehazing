Inception-Pix2Pix Image Dehazing

This repository contains an implementation of an image-to-image translation model for single-image dehazing based on the Pix2Pix framework. The generator follows a U-Net architecture enhanced with Inception-style parallel convolution branches, while the discriminator is a PatchGAN that evaluates local image patches. This setup allows the model to learn effective haze removal while preserving structure and fine details.

The approach is related to recent work on generative dehazing methods, such as the method presented here:
IEEE Paper: https://ieeexplore.ieee.org/abstract/document/10899674

Features

• U-Net generator with Inception-style modules
• PatchGAN discriminator
• Paired hazy/clear dataset support
• Adversarial and L1 reconstruction losses
• SSIM and PSNR evaluation
• Automatic sample output saving during training

Dataset Format

Your dataset should include paired hazy and clear images organized in the following way:

train/input – hazy images
train/target – corresponding clear images

Similarly for test sets such as test_thin, test_moderate, and test_thick.
Each hazy image must have a corresponding clear image with the same filename.

Model Overview

The generator downsamples the input image using convolutional layers, applies a bottleneck, and upsamples it back to full resolution while using skip connections to retain spatial details. Inception-style parallel convolutions increase the model’s representational power. The discriminator predicts real or fake labels on small overlapping patches, encouraging realistic local structure in the generated output.

Training

The model is trained on paired hazy and clear images using adversarial loss (GAN loss) together with an L1 reconstruction loss that encourages structural fidelity. Training logs and generated sample images are saved periodically.

Evaluation

The model’s dehazing performance is evaluated using SSIM and PSNR.
Example performance:

• Moderate haze: SSIM ≈ 0.88, PSNR ≈ 19 dB
• Thick haze: SSIM ≈ 0.78, PSNR ≈ 14 dB

Inference

Once trained, the generator takes a single hazy image as input and outputs its dehazed version. Input images are normalized into the range [-1, 1], and model outputs are converted back to the original image range.

Saved Model

The trained generator is saved in the folder named “generator_model”, and can be reloaded to run inference on new images.
