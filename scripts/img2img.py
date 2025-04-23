#Relic from when I thought the original idea was still possible

from PIL import Image
from PIL.ImageOps import fit
from tqdm.auto import tqdm
import torch
from diffusers import UNet2DConditionModel, UniPCMultistepScheduler, AutoencoderKL
from diffusers.utils import load_image
from transformers import CLIPTokenizer, CLIPTextModel
import numpy as np


torch_device = "cuda"
print("CUDA is enabled: " + str(torch.cuda.is_available()))  # Should be True
print("GPU being used by torch: " + torch.cuda.get_device_name(0))  # Should show your GPU
unet = UNet2DConditionModel.from_pretrained("./RealisticVision", subfolder="unet").to(torch_device)
vae = AutoencoderKL.from_pretrained("./RealisticVision", subfolder="vae").to(torch_device)
scheduler = UniPCMultistepScheduler.from_pretrained("./RealisticVision", subfolder="scheduler")
tokenizer = CLIPTokenizer.from_pretrained("./RealisticVision", subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained("./RealisticVision", subfolder="text_encoder").to(torch_device)

img_filepath = ".\\img2imgtest.jpg"
prompt = ["A cat as an astronaut on the moon"]
height = 512  # default height of Stable Diffusion
width = 512  # default width of Stable Diffusion
num_inference_steps = 25  # Number of denoising steps
guidance_scale = 2  # Scale for classifier-free guidance
strength = 0.2
generator = torch.Generator(torch_device).manual_seed(0) # Seed generator to create the initial latent noise
batch_size = len(prompt)

def create_latent_image_tensor(image: Image) -> torch.Tensor:
    image = fit(image, (512, 512))
    image_tensor = torch.tensor(data=np.array(image).reshape(1, 3, image.height, image.width), dtype=torch.float32, device=torch_device)
    image_tensor = image_tensor/127.5 - 1
    with torch.no_grad():
        latent = vae.encode(image_tensor).latent_dist.sample() * 0.18215
    return latent

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

latents = create_latent_image_tensor(load_image(img_filepath))
noise = torch.randn_like(latents)
scheduler.set_timesteps(num_inference_steps)
timestep_index = int(strength * num_inference_steps)
latents = scheduler.add_noise(latents, noise, timesteps=torch.tensor([scheduler.timesteps[timestep_index]]).to(torch_device))
with tqdm(scheduler.timesteps) as pbar:
    for t in scheduler.timesteps[timestep_index:]:
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
Image.fromarray(image).save(f"generated.png")
print(f"Saved generated image to ./generated.png")


    
    