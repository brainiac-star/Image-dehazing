Inception-Pix2Pix Image Dehazing

This repository contains a conditional GAN (Pix2Pix) model for single-image dehazing, using
an Inception-enhanced U-Net generator and a PatchGAN discriminator.
The model is trained on paired hazy/clear datasets and evaluated using SSIM and PSNR.

📌 Features

Inception-U-Net generator

PatchGAN discriminator

Paired training on haze datasets

SSIM + PSNR evaluation

tf.data pipeline

Saves sample results every few epochs

Fully reproducible training pipeline

📁 Dataset Structure
dataset/
  train/
    input/      # hazy images
    target/     # clear images
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

🧠 Model Architecture
Generator (Inception U-Net)

Encoder–decoder U-Net

Inception-style multi-scale feature blocks

Skip connections

Final Conv2D → outputs clean image

Discriminator (PatchGAN)

Operates on 70×70 patches

Takes (hazy, clean/generated) concatenated pair

Outputs patch-level real/fake map

⚙️ Training
pix2pix_gan.fit(
    train_ds,
    epochs=25,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

Losses

Generator Loss = GAN Loss + 100 × L1 Loss

Discriminator: BCE(real) + BCE(fake)

Optimizer
Adam(lr=2e-4, beta1=0.5)

📊 Evaluation

Metrics used:

SSIM

PSNR

Example results:

Dataset	SSIM	PSNR (dB)
Moderate	~0.88	~19.31
Thick	~0.78	~14.40
Thin	Higher performance	
▶️ Inference
loaded_generator = tf.keras.models.load_model("generator_model")
pred = loaded_generator(test_image)


Preprocess:

img = (img / 127.5) - 1.0


Postprocess:

img = (img + 1) / 2.0

💾 Saving
generator.save("generator_model")

📂 Project Structure
├── README.md
├── generator_model/
├── training_res.csv
├── results/
└── dehaze.ipynb

🔮 Future Improvements

Add attention blocks

Use perceptual/VGG loss

Train with unpaired data (CycleGAN)

Multi-scale discriminator

Higher-resolution training
