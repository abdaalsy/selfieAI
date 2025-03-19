# SelfieAI
This app allows users to generate selfies with ideally 1-3 other people while only needing a prompt, and individual photos of each person as references. The base model being used
is Stable Diffusion XL, with RealisticVision fine-tuning the model for photorealistic images. Since photorealistic images is a wide category, DreamBooth will be used to train
the model to recreate the style of selfies taken on a typical mobile phone camera in both portrait and landscape.

## System Requirements
SelfieAI as of March 2025 runs locally thus requiring at least 8GB of VRAM to generate images.
### Minimum
| **GPU Model**      | **VRAM** | **SDXL Capability**                           |
|-------------------|----------|-----------------------------------------------|
| **RTX 3050**      | 8 GB      | Can run SDXL at low res (~512x512)            |
| **RTX 3060**      | 12 GB     | Best budget option for SDXL at 1024x1024      |
| **RTX 4060**      | 8 GB      | Runs SDXL well at medium resolution           |
| **RTX 2070**      | 8 GB      | Handles SDXL at lower steps and resolution    |
| **Apple M2 Ultra** | Unified Memory | Can handle SDXL natively (using Metal)        |
### Recommended
| **GPU Model**         | **Time to Generate 1024x1024 Image** |
|----------------------|---------------------------------------|
| **GTX 1080 Ti**        | ~20 seconds                          |
| **RTX 2070**           | ~16 seconds                          |
| **RTX 2080 Ti**        | ~14 seconds                          |
| **RTX 3060 (12 GB)**   | ~12 seconds                          |
| **RTX 3060 Ti**        | ~10 seconds                          |
| **RTX 3080**           | ~8 seconds                           |
| **RTX 3090**           | ~6 seconds                           |
| **RTX 4070 Ti**        | ~5 seconds                           |
| **RTX 3090 Ti**        | ~4.5 seconds                         |
| **RTX 4080**           | ~3.5 seconds                         |
| **RTX 4090**           | ~2.8 seconds                         |
This data was provided by ChatGPT, it might not be the most reliable.