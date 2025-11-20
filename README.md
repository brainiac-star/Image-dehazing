Inception-Pix2Pix for Single Image Dehazing

A GAN-based Approach Using an Inception-Enhanced U-Net Generator

This repository contains an end-to-end implementation of a conditional GAN (Pix2Pix) for single-image dehazing. The model uses a U-Net generator enhanced with Inception-style parallel convolution blocks and a PatchGAN discriminator. The system is trained on paired hazy/clear image datasets and evaluated using SSIM and PSNR.

1. Overview

Image dehazing is an important task in outdoor vision systems, autonomous navigation, and remote sensing. Haze reduces contrast, visibility, and feature clarity. Traditional methods often struggle to generalize across varying haze densities.

This project adopts a GAN-based approach, where:

The Generator learns a mapping from hazy images to clear images.

The Discriminator evaluates whether the generated clear image is realistic.

A supervised paired dataset guides the model toward accurate restoration.

2. Architecture
2.1 Generator – Inception U-Net

The generator is a U-Net encoder–decoder with the following characteristics:

Multi-scale feature extraction using Inception-style blocks (1x1, 3x3, 5x5 convs + pooling).

Skip connections to preserve spatial information.

Upsampling in the decoder path.

Final output layer produces 3-channel RGB images.

Advantages:

Robust multi-scale haze removal.

Strong structure preservation.

High detail recovery.

2.2 Discriminator – PatchGAN

The discriminator is a 70x70 PatchGAN, which classifies local image patches instead of the entire image.

Advantages:

Enforces high-frequency realism.

More stable GAN training.

Fewer parameters than full-image discriminator.

3. Dataset Structure

Organize your dataset as follows:

dataset/
    train/
        input/       # hazy images
        target/      # corresponding clear images
    test_thin/
        input/
        target/
    test_moderate/
        input/
        target/
    test_thick/
        input/
        target/


Each hazy image must have a matching ground truth clear image with the same filename.

Images are resized to 256x256 and normalized to [-1, 1].

4. Training Pipeline
4.1 Loss Functions

Generator Loss:

GAN Loss (Binary Cross Entropy)

L1 Loss (pixel-level reconstruction)

Weighted combination:

L_G = L_GAN + 100 * L1


Discriminator Loss:

BCE(real → 1)

BCE(fake → 0)

4.2 Optimizer

Both networks use:

Adam(lr=2e-4, beta1=0.5)


This is the standard stable configuration for GAN training.

4.3 Training Loop Description

For each training batch:

Load hazy and clear image pair.

Generator produces predicted clear image.

Discriminator evaluates (hazy, clear) as real and (hazy, generated) as fake.

Compute generator and discriminator losses.

Backpropagate using two gradient tapes.

Save logs and example outputs every few epochs.

The model is trained for 25 epochs.

5. Evaluation Metrics

Two quantitative metrics are used:

SSIM (Structural Similarity Index)

Measures perceptual similarity.

PSNR (Peak Signal-to-Noise Ratio)

Measures reconstruction quality in decibels.

Example Results
Dataset	SSIM	PSNR (dB)
Moderate	~0.88	~19.31
Thick	~0.78	~14.40
Thin	Highest performance among all	

These values show strong results on moderate haze and reasonable performance under denser haze.

6. Inference

Load a trained generator and run prediction:

loaded_generator = tf.keras.models.load_model("generator_model")
pred = loaded_generator(test_image)

Preprocessing:
img = (img / 127.5) - 1.0

Postprocessing:
img = (img + 1) / 2.0

7. Saving Models

To save the generator:

generator.save("generator_model")


This saves the architecture, weights, and optimizer state.

8. Project Structure
├── README.md
├── generator_model/          # saved TF model
├── training_res.csv          # training logs
├── results/                  # generated samples
└── dehaze.ipynb              # main notebook / script

9. Key Contributions

Implemented a Pix2Pix-style conditional GAN specifically for dehazing.

Created a custom U-Net generator enhanced with Inception blocks.

Trained on multiple haze levels and achieved strong SSIM/PSNR results.

Added complete evaluation pipeline and visualization system.

Provided reproducible training and inference workflow.

10. Future Work

Planned extensions include:

Attention mechanisms (CBAM, Self-Attention)

Perceptual Loss (VGG-based)

Multi-scale discriminators

Unpaired training using CycleGAN

Higher-resolution dehazing (512x512+)
