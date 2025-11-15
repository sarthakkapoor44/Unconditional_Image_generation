# Unconditional Image Generation with Diffusion Models

A from-scratch implementation of **Denoising Diffusion Probabilistic Models (DDPM)** for unconditional image generation. This project demonstrates how neural networks can learn to generate high-quality images by iteratively denoising random Gaussian noise, trained on the Oxford Flowers dataset.

## 🌟 Features

- **Two Model Architectures**: Compare a simple U-Net vs. an improved U-Net with attention mechanisms
- **Interactive Streamlit UI**: Generate images with a single button click
- **Training Pipeline**: Complete training script with periodic sampling visualization
- **Modular Design**: Clean separation of diffusion math, model architecture, and inference logic
- **From-Scratch Implementation**: Educational codebase implementing core DDPM concepts

## 📚 What are Diffusion Models?

Diffusion models work by learning to reverse a gradual noising process:

1. **Forward Diffusion** (Training): Progressively add Gaussian noise to real images over T timesteps until they become pure noise
2. **Reverse Diffusion** (Sampling): Train a neural network to predict and remove noise at each timestep
3. **Generation**: Start from random noise and iteratively denoise using the trained model to create new images

This implementation uses 300 timesteps and a linear beta schedule for the noise schedule.

## 🏗️ Architecture

### Project Structure

```
├── app.py                    # Streamlit web interface for image generation
├── train.py                  # Training script for the simple U-Net model
├── inference.py              # Sampling/generation utilities (reverse diffusion)
├── diffusion.py              # Forward diffusion process and noise schedules
├── model_architecture.py     # Simple U-Net implementation
├── improved_model_arch.py    # Enhanced U-Net with attention and ResNet blocks
├── data_preprocessing.py     # Data loading and visualization utilities
├── imports.py                # Shared imports and global constants
├── requirements.txt          # Python dependencies
├── new_linear_model_1090.pt  # Pretrained simple U-Net weights
├── model_400pt               # Pretrained attention U-Net weights
└── Notebooks/                # Jupyter notebooks for experimentation
```

### Model Architectures

#### 1. Simple U-Net (`model_architecture.py`)

- Basic encoder-decoder architecture with skip connections
- Time embeddings via sinusoidal position encodings
- Conv2d blocks with BatchNorm and SiLU activation
- Down-sampling: 32 → 64 → 128 → 256 → 512 channels
- Up-sampling: 512 → 256 → 128 → 64 → 32 channels

#### 2. Improved U-Net with Attention (`improved_model_arch.py`)

- ResNet-style blocks with group normalization
- **Linear self-attention** mechanisms for better global context
- Residual connections throughout
- More sophisticated time embedding MLP
- Better suited for high-quality generation

### Key Components

**`diffusion.py`**: Core diffusion mathematics

- `linear_beta_schedule()`: Defines noise schedule (β₁ to βₜ)
- `forward_diffusion_sample()`: Adds noise to images (q(xₜ | x₀))
- `get_loss()`: Computes L1 loss between true and predicted noise
- Precomputed constants: αₜ, ᾱₜ, √ᾱₜ, √(1-ᾱₜ), etc.

**`inference.py`**: Reverse diffusion sampling

- `p_sample()`: Single denoising step (p(xₜ₋₁ | xₜ))
- `p_sample_loop()`: Full reverse process from noise to image
- `sample()`: High-level API for batch generation

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- CUDA (optional, for GPU acceleration)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/himanshu-skid19/Unconditional-Image-Generation-Using-a-Diffusion-model.git
   cd Unconditional-Image-Generation-Using-a-Diffusion-model
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Demo

**Launch the Streamlit app:**

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the Interface

1. **Select a Model**: Choose between "Model without Attention" (faster) or "Model with Attention" (higher quality)
2. **Generate Images**: Click "Click to generate image" button
3. **Wait for Results**: Progress bar shows sampling progress (300 timesteps)
4. **View Output**: 8 generated flower images will be displayed in a grid

![Diffusion Model Interface](https://github.com/himanshu-skid19/Unconditional-Image-Generation-Using-a-Diffusion-model/assets/114365148/8a8c2813-8609-40e9-b14b-038326dd76c0)

### Final Result
![Diffusion Model Output](https://github.com/sarthakkapoor44/Unconditional_Image_generation/blob/master/images/result.jpeg)

### Performance Expectations

| Hardware                 | Batch Size  | Generation Time | Notes                  |
| ------------------------ | ----------- | --------------- | ---------------------- |
| CPU (Intel/AMD)          | 8 images    | 3-5 minutes     | Default setting        |
| GPU (CUDA)               | 16 images   | < 1 minute      | Significantly faster   |
| Apple Silicon (M1/M2/M3) | 8-16 images | 1-2 minutes     | Requires MPS backend\* |

\*For Apple Silicon acceleration, modify `imports.py`:

```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

## 🎓 Training Your Own Model

The `train.py` script trains the simple U-Net from scratch:

```bash
python train.py
```

**Training Configuration:**

- Dataset: Oxford Flowers (first 1000 images)
- Image size: 64×64
- Batch size: 64
- Epochs: 25
- Optimizer: Adam (lr=0.001)
- Loss: L1 loss between true and predicted noise
- Timesteps: 600

**What happens during training:**

- Loads flower images from Hugging Face datasets
- Transforms images to [-1, 1] range
- For each batch:
  - Randomly samples timestep t
  - Applies forward diffusion to add noise
  - Model predicts the noise
  - Computes loss and updates weights
- Every 5 epochs: generates sample images to visualize progress

**Saving checkpoints:**
To save your trained model:

```python
torch.save(model.state_dict(), "my_model.pt")
```

## 🔧 Technical Details

### Hyperparameters

| Parameter    | Value  | Description               |
| ------------ | ------ | ------------------------- |
| `timesteps`  | 300    | Number of diffusion steps |
| `beta_start` | 0.0001 | Initial noise level       |
| `beta_end`   | 0.02   | Final noise level         |
| `img_size`   | 64     | Image resolution (64×64)  |
| `BATCH_SIZE` | 64     | Training batch size       |

### Diffusion Schedule

The linear schedule computes:

```
βₜ = linspace(0.0001, 0.02, 300)
αₜ = 1 - βₜ
ᾱₜ = ∏(α₁ to αₜ)
```

### Forward Process (Training)

Given clean image x₀ and timestep t:

```
q(xₜ | x₀) = √ᾱₜ · x₀ + √(1-ᾱₜ) · ε,  where ε ~ N(0, I)
```

### Reverse Process (Sampling)

Model learns p(xₜ₋₁ | xₜ):

```
xₜ₋₁ = 1/√αₜ · (xₜ - βₜ/√(1-ᾱₜ) · εθ(xₜ, t)) + σₜ · z
```

where εθ is the neural network and z ~ N(0, I)

## 🐛 Troubleshooting

### Common Issues

**1. `torch.classes` warning on startup**

```
Tried to instantiate class '__path__._path', but it does not exist!
```

- **Solution**: This is a harmless warning from PyTorch/Streamlit interaction. Ignore it.

**2. Generation appears stuck or very slow**

- **Cause**: Running on CPU with default settings
- **Solution**:
  - Use GPU if available
  - Reduce batch size in `app.py` (already optimized to 8 for CPU)
  - Enable MPS on Apple Silicon (see instructions above)

**3. Out of memory errors**

- **Solution**: Reduce `batch_size` parameter in `app.py` or `train.py`
- For training: Lower `BATCH_SIZE` in `imports.py`

**4. Model checkpoint not loading**

- **Cause**: Missing or corrupted `.pt` files
- **Solution**: Ensure `new_linear_model_1090.pt` and `model_400pt` exist in root directory
- Re-download from repository or retrain using `train.py`

**5. Poor quality generations**

- **Try**: Use "Model with Attention" instead of simple model
- **Note**: Quality depends on training epochs and dataset size

## 📊 Results & Observations

- **Simple U-Net**: Faster inference, decent quality for 64×64 flowers
- **Attention U-Net**: Better global coherence, sharper details, smoother color transitions
- **Training**: Model shows progressive improvement over 25 epochs
- **Dataset**: Oxford Flowers provides diverse floral structures and colors

## 🛠️ Future Improvements

- [ ] Add DDIM sampling for faster generation (fewer steps)
- [ ] Implement classifier-free guidance for controllable generation
- [ ] Support higher resolutions (128×128, 256×256)
- [ ] Add more noise schedules (cosine, quadratic)
- [ ] Integrate Weights & Biases for training monitoring
- [ ] Implement FID score evaluation
- [ ] Add conditional generation (text-to-image, class-conditional)
- [ ] Multi-GPU training support

## 📖 References

- [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)
- [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)
- [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests for:

- Bug fixes
- Performance improvements
- New features
- Documentation enhancements

## 💬 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This is an educational implementation designed for learning diffusion models. For production use cases, consider established libraries like:

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- [DALL-E 2](https://github.com/openai/dall-e-2)
