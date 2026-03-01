# Bird Image Generation - Master's Thesis

Comparative study of deep generative models for fine-grained bird image synthesis using the **CUB-200-2011** dataset (11,788 images, 200 species).

---

## Models

| # | Model | Type | Resolution |
|---|-------|------|------------|
| 1 | DCGAN | Unconditional GAN | 64×64 |
| 2 | cDCGAN | Class-conditional GAN | 64×64 |
| 3 | Stable Diffusion v1.5 + LoRA | Text-guided diffusion | 256×256 |

---

## Dataset

[CUB-200-2011](http://www.vision.caltech.edu/datasets/cub_200_2011/) - Caltech-UCSD Birds-200-2011  
11,788 images across 200 fine-grained bird species.

## Methods

### 1. DCGAN - Unconditional
Standard Deep Convolutional GAN without class conditioning. Uses soft label smoothing and evaluates FID / IS every 10 epochs over 150 epochs of training.

**Architecture:**
- Generator: 5-layer transposed convolution stack, input noise dim = 100, output 64×64×3
- Discriminator: 5-layer convolution stack with LeakyReLU

### 2. Conditional DCGAN (cDCGAN)
Class-conditional extension. The Generator concatenates a 100-dim noise vector with a learned 200-dim class embedding. The Discriminator projects the class label as an extra spatial channel (1×64×64) concatenated to the image input.

### 3. Stable Diffusion v1.5 + LoRA
Fine-tunes Stable Diffusion v1.5 on CUB-200-2011 using **LoRA** adapters (rank 4) applied to UNet cross-attention layers (`to_q`, `to_k`, `to_v`, `to_out`). Training uses HuggingFace PEFT + Accelerate with FP16 mixed precision over 20 epochs.

---

## Evaluation

Metrics are computed using [torchmetrics](https://torchmetrics.readthedocs.io/):

| Metric | Description |
|--------|-------------|
| **FID** | Fréchet Inception Distance - measures distributional similarity between real and generated images (lower is better) |
| **IS** | Inception Score - measures quality and diversity of generated images (higher is better) |

For statistical analysis, each model is evaluated over **20 independent runs** (500–1000 images per run). Results are saved to CSV for downstream hypothesis testing.

---

## Requirements

```bash
pip install torch torchvision torchmetrics
pip install diffusers transformers peft accelerate
pip install pytorch-fid torch-fidelity torchviz
```

---

## Usage

Open `main.ipynb` in Google Colab (GPU recommended). The notebook is organized into the following sections:

1. **Setup** - Mount Google Drive, extract dataset, install dependencies
2. **DCGAN** - Unconditional model training
3. **cDCGAN** - Conditional model training
4. **Stable Diffusion + LoRA** - Fine-tuning, inference, and epoch-by-epoch evaluation
5. **Statistical Evaluation** - 20-run FID & IS measurement for all models

---

## Results

Trained model checkpoints and metric logs are saved during training. CSV output files:

| File | Contents |
|------|----------|
| `metrics_log.csv` | Per-epoch G/D loss, FID, IS, duration (GAN models) |
| `lora-sd15-cub200/training_metrics_SD.csv` | Per-epoch loss, FID, IS (Stable Diffusion) |
| `fid_and_is_scores_dcgan.csv` | 20-run FID & IS for DCGAN |
| `fid_and_is_scores_cdcgan.csv` | 20-run FID & IS for cDCGAN |
| `fid_and_is_scores_stablediffusion.csv` | 20-run FID & IS for SD + LoRA |

---

## Would you like to see my master's thesis?
Feel free to contact me. The thesis is written in Polish.