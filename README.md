Inception-Pix2Pix Image Dehazing

This repository contains a conditional GAN (Pix2Pix) for single-image dehazing, using an Inception-enhanced U-Net generator and a PatchGAN discriminator. The model is trained on paired hazy/clear datasets and evaluated using SSIM and PSNR.

Features

Inception-U-Net generator

PatchGAN discriminator

Paired hazy → clear training

SSIM & PSNR evaluation

tf.data pipeline

Sample results saved during training

Dataset Structure
dataset/
  train/
    input/
    target/
  test_thin/
    input/
    target/
  test_moderate/
    input/
    target/
  test_thick/
    input/
    target/


Each hazy image must have a matching clear image with the same filename.

Model Architecture
Generator (Inception U-Net)

Encoder–decoder U-Net

Parallel 1x1, 3x3, 5x5 convolutions

Skip connections

Final Conv2D producing RGB output

Discriminator (PatchGAN)

70x70 PatchGAN

Input: concatenated (hazy, clear/generated)

Output: real/fake patch map

Training
pix2pix_gan.fit(
    train_ds,
    epochs=25,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

Losses

Generator Loss = GAN Loss + 100 * L1 Loss
Discriminator Loss = BCE(real) + BCE(fake)

Optimizer
Adam(lr=2e-4, beta1=0.5)

Evaluation

Metrics used:

SSIM

PSNR

Example Results
Dataset	SSIM	PSNR (dB)
Moderate	0.88	19.31
Thick	0.78	14.40
Thin	Best performance	
Inference
loaded_generator = tf.keras.models.load_model("generator_model")
pred = loaded_generator(test_image)

Preprocess
img = (img / 127.5) - 1.0

Postprocess
img = (img + 1) / 2.0

Saving the Model
generator.save("generator_model")

Project Structure
├── README.md
├── generator_model/
├── training_res.csv
├── results/
└── dehaze.ipynb

Future Improvements

Add attention mechanisms

Add perceptual loss (VGG)

Multi-scale discriminator

Train with unpaired data (CycleGAN)

Higher-resolution training
