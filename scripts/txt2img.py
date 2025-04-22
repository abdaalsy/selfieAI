"""
Need to refactor this file to be more modular. Some changes:
- Use mandatory arguments for prompt, output
- Use optional arguments (have default values) for model weights path, lora weights path, output width, output height, seed, cuda/cpu  
"""

import os
import argparse
from PIL import Image
from tqdm.auto import tqdm
import torch
from diffusers import UNet2DConditionModel, UniPCMultistepScheduler, AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel

def check_positive(value):
    ivalue= int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("Image dimensions must be positive and greater than zero.")
    return ivalue

def check_dir(directory):
    dir = str(directory)
    if not os.path.isdir(dir):
        raise argparse.ArgumentError("The specified folder path was not found!")
    else:
        return dir

def check_lora(path):
    str_path = str(path)
    if not os.path.isfile(str_path):
        raise argparse.ArgumentError(f"The specified LoRA weights could not be located in \"{str_path}\"")
    elif not str_path.endswith(".safetensors"):
        raise argparse.ArgumentTypeError("LoRA weights must be stored in a .safetensors file.")
    else:
        return str_path

def check_device(device):
    device = str(device)
    if not (device == "cpu" or device == "cuda"):
        raise argparse.ArgumentError("Possible choices of device are \"cuda\" and \"cpu\".")
    if device=="cuda" and torch.cuda.is_available()==False:
        raise argparse.ArgumentError("The GPU could not be accessed by PyTorch.")
    return device

def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for text to image inference")
    parser.add_argument(
        "-p", "--prompt",
        type=str,
        default=None,
        required=True,
        help="Text prompt for conditional image generation."
    )
    parser.add_argument(
        "-o", "--output_path",
        type=check_dir,
        default=".",
        required=True,
        help="Folder where the generated image will be stored."
    )
    parser.add_argument(
        "--width",
        type=check_positive,
        default=512,
        required=False,
        help="Output image width."
    )
    parser.add_argument(
        "--height",
        type=check_positive,
        default=512,
        required=False,
        help="Output image height."
    )
    parser.add_argument(
        "-m", "--model_path",
        type=check_dir,
        default="./RealisticVision",
        required=False,
        help="Path to the folder containing the image generation model."
    )
    # parser.add_argument(
    #     "-sl", "--selfie_lora_path",
    #     type=check_lora,
    #     default="../lora/selfie_lora/selfie_lora.safetensors",
    #     required=False,
    #     help="The path to the .safetensors file containing the LoRA weights for selfie_lora"
    # )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=0,
        required=False,
        help="Seed used for image generation. The same seed with the same prompt produces the same image."
    )
    parser.add_argument(
        "-d", "--device",
        type=check_device,
        default="cpu",
        required=False,
        help="(cuda/cpu) Device that tensors will be stored on. Use \"cuda\" for faster image generation if you have a GPU with cuda 11.8 or 12.1"
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    torch_device = args.device
    unet = UNet2DConditionModel.from_pretrained(args.model_path, subfolder="unet").to(torch_device)
    vae = AutoencoderKL.from_pretrained(args.model_path, subfolder="vae").to(torch_device)
    scheduler = UniPCMultistepScheduler.from_pretrained(args.model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_path, subfolder="text_encoder").to(torch_device)

    prompt = [args.prompt]
    height = args.height
    width = args.width
    num_inference_steps = 25
    guidance_scale = 2
    generator = torch.Generator(torch_device).manual_seed(args.seed)
    batch_size = len(prompt)

    tokens = tokenizer(
        prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )

    with torch.no_grad():
        text_embeddings = text_encoder(tokens.input_ids.to(torch_device))[0]
    uncond_tokens = tokenizer(
        [""]*batch_size, padding="max_length", max_length=tokens.input_ids.shape[-1], return_tensors="pt" 
    )
    uncond_embeddings = text_encoder(uncond_tokens.input_ids.to(torch_device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    latents = torch.randn((1, unet.config.in_channels, height // 8, width // 8), generator=generator, device=torch_device)
    scheduler.set_timesteps(num_inference_steps)
    with tqdm(scheduler.timesteps) as pbar:
        for t in scheduler.timesteps:
            latent_model_input = torch.cat([latents]*2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            uncond_pred, cond_pred = noise_pred.chunk(2)
            noise_pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
            latents = scheduler.step(noise_pred, t, latents).prev_sample
            pbar.update()

    latents = 1/0.18215 * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image/2 + 0.5).clamp(0, 1).squeeze().permute(1, 2, 0) * 255
    image = image.to(torch.uint8).cpu().numpy()
    Image.fromarray(image).save(f"{args.output_path}/generated.png")
    print(f"Saved generated image to {args.output_path}/generated.png")

if __name__ == "__main__":
    main()

