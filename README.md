# SelfieAI
This future web app uses individual photos of two people, and can combine them into a selfie. The base model being used is Stable Diffusion 1.5, with RealisticVision fine-tuning the model for photorealistic images. Since photorealistic images is a wide category, Low Rank Adaptation (LoRA) will be used to train the model to recreate the style of selfies taken on a typical mobile phone camera in landscape.

## System Requirements
| **GPU Model**        | **Approximate Time per Image (512x512)** |
|------------------|--------------------------------------|
| GTX 1660 Ti      | ~25 seconds                          |
| RTX 2060         | ~20 seconds                          |
| GTX 1080 Ti      | ~15 seconds                          |
| RTX 2080         | ~10 seconds                          |
| RTX 3060 Ti      | ~1.8 seconds                         |
| RTX 3060 12GB    | ~1.8 seconds                         |
| RTX 3070         | ~1.5 seconds                         |
| RTX 3080         | ~1.2 seconds                         |
| RTX 3090         | ~0.9 seconds                         |
| RTX 4070         | ~2.7 seconds                         |
| RTX 4070 SUPER   | ~0.9 seconds                         |
| RTX 4080         | ~0.7 seconds                         |
| RTX 4090         | ~0.6 seconds                         |

This data was provided by ChatGPT, so it might not be reliable.

## Plans for the Future
I hope to deploy this to the web someday. I'd likely use a RunPod cloud GPU to handle all the heavy processing so that any device can use selfieAI. I'm completely new to AI/ML, so for now I'm going to focus on training this model to give some good results. Good thing SD is open-source cus my minimum wage job ain't paying for no subscription.