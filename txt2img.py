from PIL import Image
from tqdm.auto import tqdm
import torch
from diffusers import UNet2DConditionModel, UniPCMultistepScheduler, AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel

torch_device = "cuda"
print("CUDA is enabled: " + str(torch.cuda.is_available()))  # Should be True
print("GPU being used by torch: " + torch.cuda.get_device_name(0))  # Should show your GPU
unet = UNet2DConditionModel.from_pretrained("./RealisticVision", subfolder="unet").to(torch_device)
vae = AutoencoderKL.from_pretrained("./RealisticVision", subfolder="vae").to(torch_device)
scheduler = UniPCMultistepScheduler.from_pretrained("./RealisticVision", subfolder="scheduler")
tokenizer = CLIPTokenizer.from_pretrained("./RealisticVision", subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained("./RealisticVision", subfolder="text_encoder").to(torch_device)

prompt = ["A photo-realistic portrait of an 18 year old Pakistani male. He weighs 140lbs, height of 180cm, is slightly lean, palish-tan complexion, has dark brown curly hair that reaches eyebrows, is clean shaven, has a slightly large nose and slightly close together eyes, and has slightly larger lips."]
height = 512  # default height of Stable Diffusion
width = 512  # default width of Stable Diffusion
num_inference_steps = 25  # Number of denoising steps
guidance_scale = 7  # Scale for classifier-free guidance
generator = torch.Generator(torch_device).manual_seed(0) # Seed generator to create the initial latent noise
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
Image.fromarray(image).save(f"generated.png")
print(f"Saved generated image to ./generated.png")

