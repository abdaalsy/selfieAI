# SelfieAI
This app allows users to generate selfies with ideally 1-3 other people while only needing a prompt, and individual photos of each person as references. The base model being used
is Stable Diffusion XL, with RealisticVision fine-tuning the model for photorealistic images. Since photorealistic images is a wide category, Low Rank Adaptation (LoRA) will be used to train the model to recreate the style of selfies taken on a typical mobile phone camera in both portrait and landscape.

## System Requirements
SelfieAI as of March 2025 runs locally thus requiring at least 8GB of VRAM to generate images.

### Minimum
| **GPU Model**      | **VRAM** | **Notes**                           |
|-------------------|----------|-----------------------------------------------|
| **RTX 3050**      | 8 GB      | Can run SDXL at low res (~512x512)            |

### Recommended
| **GPU Model**         | **Time to Generate 1024x1024 Image** |
|----------------------|---------------------------------------|
| **RTX 3050**           | ~27 seconds                          |
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

## Plans for the Future
I hope to deploy this to the web someday. I'd likely use an AWS cloud GPU to handle all the heavy processing so that any device can use selfieAI. I'm completely new to AI/ML, so for now I'm going to focus on training this model to give some good results. Good thing SDXL is free cus my minimum wage job ain't paying for no subscription.