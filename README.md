# Image-dehazing
using pix2pix gan with inception module
Inception-Pix2Pix for Single Image Dehazing

A GAN-based Approach Using an Inception-Enhanced U-Net Generator

This repository implements a conditional GAN (cGAN) for single-image dehazing, inspired by the Pix2Pix framework and enhanced with Inception-style feature extraction blocks. The goal is to learn a mapping from hazy images to their corresponding clear images using paired training data.

The model is trained and evaluated on different haze levels (Thin, Moderate, Thick), and the pipeline includes preprocessing, training, inference, and quantitative evaluation using SSIM and PSNR.

📚 Overview

Image dehazing is a challenging low-level vision task due to the complexity of atmospheric scattering. This project uses a GAN-based supervised learning approach, where:

The Generator learns to translate hazy images into clean counterparts.

The Discriminator ensures generated images appear realistic and structurally consistent.

An L1 reconstruction loss stabilizes the training and encourages fidelity to the ground truth.

This model is particularly effective for remote sensing, autonomous drones, outdoor perception, and degraded-scene restoration.

1. Architecture
1.1 Generator – Inception U-Net

The generator follows a U-Net encoder–decoder structure but enhances feature extraction using Inception-style parallel convolutions.

Key elements:

Multi-scale feature extraction (1×1, 3×3, 5×5 convs, + pooling)

Skip connections for fine detail preservation

Upsampling in the decoder for reconstruction

Final 3-channel output (linear activation)

This design enables:

Better modeling of haze structures at multiple scales

Improved learning stability

Stronger high-frequency detail recovery

1.2 Discriminator – PatchGAN

A 70×70 PatchGAN is used to classify if local patches are real or fake.

Advantages:

Encourages local realism

Reduces number of parameters

Stabilizes adversarial training

Better suited for dehazing than a global discriminator

2. Dataset Format

Place data in the following directory structure:

dataset/
    train/
        input/        # hazy images
        target/       # corresponding clear images
    test_thin/
        input/
        target/
    test_moderate/
        input/
        target/
    test_thick/
        input/
        target/


Each hazy image must have a one-to-one paired ground truth image.

Images are automatically:

Loaded

Decoded

Resized to 256×256

Normalized to [-1, 1]

3. Training Pipeline
3.1 Loss Functions
Generator Loss
L_G = L_GAN + λ * L1
λ = 100


Where:

L_GAN = BCE loss with labels = 1

L1 loss encourages pixel-level accuracy and structural correctness

Discriminator Loss
L_D = BCE(real → 1) + BCE(fake → 0)

3.2 Optimizer

Both networks use:

Adam(learning_rate = 2e-4, beta1 = 0.5)


This configuration is standard for stable GAN training.

3.3 Training Loop

The model is trained for 25 epochs using custom Keras train_step logic:

Fetch batch of (hazy, clear)

Generator produces predicted output

Discriminator evaluates both real and generated pairs

Compute individual losses

Backpropagate via separate gradient tapes

Save training logs and sample outputs

A callback generates dehazed outputs every 5 epochs for qualitative comparison.

4. Evaluation Metrics
4.1 SSIM (Structural Similarity Index)

Measures perceptual similarity between predicted and target images.

4.2 PSNR (Peak Signal-to-Noise Ratio)

Measures reconstruction fidelity.

Both metrics are computed per image and averaged per dataset (Thin, Moderate, Thick).

Sample Final Results
Dataset	SSIM	PSNR (dB)
Moderate	~0.88	~19.31
Thick	~0.78	~14.40
Thin	Higher, stable performance	

These results demonstrate strong performance in moderate haze and reasonable generalization in thick haze.

5. Inference

Use a previously saved generator:

loaded_generator = tf.keras.models.load_model("generator_model")
pred = loaded_generator(test_image)

Preprocessing
img = (img / 127.5) - 1.0

Postprocessing
img = (img + 1) / 2.0


Save or display prediction using matplotlib or cv2.

6. Saving Models
generator.save("generator_model")


This stores:

Architecture

Weights

Optimizer state

7. Project Structure
├── README.md
├── generator_model/         # saved TensorFlow model
├── training_res.csv         # training log
├── results/                 # generated sample outputs
└── dehaze.ipynb             # main codebase / notebook

8. Key Contributions

Implemented a Pix2Pix-based cGAN specifically adapted for image dehazing

Designed a custom Inception-U-Net generator for superior multi-scale feature extraction

Achieved high-quality dehazing on multiple haze levels

Added comprehensive evaluation (SSIM, PSNR)

Provided reproducible pipeline with visualization and model export

9. Future Work

Potential improvements include:

Adding self-attention (e.g., SAGAN, CBAM)

Using Perceptual Loss (VGG-based)

Training with unpaired datasets using CycleGAN

Multi-scale discriminators

Higher-resolution training (512×512 or 1024×1024)
