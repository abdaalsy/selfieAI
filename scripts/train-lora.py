#Adapted from https://github.com/huggingface/diffusers/blob/dd9a5caf61f04d11c0fa9f3947b69ab0010c9a0f/examples/text_to_image/train_text_to_image_lora.py
#Shed ~600 lines of code and learned a ton about diffusion model architecture, training, and lora in the process.

import argparse
import logging
import datasets
import transformers
import torch
import os
import numpy
import random
import math
import shutil

from tqdm.auto import tqdm
from pathlib import Path
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.logging import get_logger
from transformers import CLIPTokenizer, CLIPTextModel
from datasets import load_dataset
from torchvision import transforms
from diffusers.optimization import get_constant_schedule_with_warmup
from diffusers.models.attention_processor import LoRAAttnProcessor2_0
import torch.nn.functional as F

import diffusers
from diffusers import UNet2DConditionModel, UniPCMultistepScheduler, AutoencoderKL
from diffusers.loaders import AttnProcsLayers

logger = get_logger(__name__, log_level="INFO")
torch_device = "cuda"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        required=True,
        help="Path to the folder containing the model. This folder should contain subfolders for the unet, vae, etc.."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True,
        help="Path to the folder containing the training data, should be in a format that HuggingFace's datasets library can understand. See https://huggingface.co/docs/datasets/image_dataset#imagefolder for more details."
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        default=None,
        help="Path to the folder where the newly created LoRA weights will be stored."
    )
    parser.add_argument(        #Need this cus I'm on a 30 series gpu
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--save_checkpoint_steps",
        default=500,
        type=int,
        required=False,
        help="Save a checkpoint of the current training state every X update. Used for resuming training from checkpoint."
    )
    parser.add_argument(
        "--limit_saved_checkpoints",
        type=int,
        default=10,
        required=False,
        help="Limit for number of saved checkpoints."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help=(
            "For debugging purposes decrease the number of training steps to this value if set."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "\"latest\" to automatically select the last available checkpoint."
        ),
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    accelerator_project_config = ProjectConfiguration(args.output_dir, args.logging_dir)

    seed = 0
    height = 384
    width = 512
    learning_rate = 1e-4
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-08
    weight_decay = 1e-2
    batch_size = 16
    gradient_accumulation_steps = 4
    num_epochs = 20
    warmup_steps = 500
    max_grad_norm = 1.0
    rank = 4

    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="fp16",
        project_config=accelerator_project_config
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    datasets.utils.logging.set_verbosity_warning()  #datasets will only log warnings or higher
    transformers.utils.logging.set_verbosity_warning()  #transformers will only log warnings or higher
    diffusers.utils.logging.set_verbosity_info()    #diffusers will log info and higher

    set_seed(seed)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_path, subfolder="unet")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
    noise_scheduler = UniPCMultistepScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    unet.requires_grad_(False)      # free up memory by freezing model parameters. The model itself is not being trained with LoRA
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    #Cast all non trainable weights to half precision to save VRAM as these weights are only being used for inference during evaluation
    weight_dtype = torch.float16
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Set correct lora layers
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRAAttnProcessor2_0(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=rank,
        )
    unet.set_attn_processor(lora_attn_procs)
    lora_layers = AttnProcsLayers(unet.attn_processors)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        lora_layers.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=epsilon,
        weight_decay=weight_decay
    )

    dataset = load_dataset(
        "imagefolder",
        data_dir=args.dataset_path,
    )
    column_names = dataset["train"].column_names
    image_column = column_names[0]
    caption_column = column_names[1]

    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, numpy.ndarray)):
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError("Caption columns can be str, or sequence")
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids
    
    #Preprocess dataset
    train_transforms = transforms.Compose([
        transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop((height, width)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = [token for token in tokenize_captions(examples[caption_column])]
        return examples
    train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])   #Combine all image tensors into one per batch
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])     #Combine all caption tensors into one per batch
        return {"pixel_values": pixel_values, "input_ids": input_ids}
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn   #Called on each batch when enumerating through dataloader
    )

    num_update_steps_in_epoch = math.ceil(int(len(train_dataloader)) / gradient_accumulation_steps)
    max_train_steps = num_update_steps_in_epoch * num_epochs

    lr_scheduler = get_constant_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps
    )

    lora_layers, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        lora_layers, optimizer, train_dataloader, lr_scheduler
    )

    accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    #Finally time to train
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Batch size = {batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total steps where weights will be changed = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    #Load from an earlier save if possible
    if args.resume_from_checkpoint:
        dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path_latest = dirs[-1] if len(dirs) > 0 else None

        if path_latest is not None:
            accelerator.print(
                f"There is no latest checkpoint, starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path_latest}")
            accelerator.load_state(os.path.join(args.output_dir, path_latest))
            global_step = int(path_latest.split("-")[1])

            initial_global_step = global_step   #constant
            first_epoch = global_step // num_update_steps_in_epoch  #constant
    else:
        initial_global_step = 0
    
    #start up progress bar
    progress_bar = tqdm(
        range(max_train_steps),
        initial=initial_global_step,
        desc="Steps"
    )

    #training loop
    unet.train()
    for epoch in range(first_epoch, num_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * 0.18215
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                noise = torch.randn_like(latents)
                bat_sz = latents.shape[0]   #This might not always be the batch size we set if the length of our dataset is not perfectly divisible
                timesteps = torch.randint(low=0, high=noise_scheduler.config.num_train_timesteps, size=(bat_sz,), device=torch_device)
                latents = noise_scheduler.add_noise(latents, noise, timesteps)
                target = noise
                unet_pred = unet(latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(unet_pred.float(), target.float(), reduction="mean")
                avg_loss = loss.repeat(batch_size).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps
                accelerator.backward(loss)
                params_to_clip = lora_layers.parameters()
                accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            global_step += 1
            accelerator.log({"training loss": train_loss}, step=global_step)
            train_loss = 0.0

            #Save checkpoint if applicable for training step
            if global_step % args.save_checkpoint_steps == 0:
                if args.limit_saved_checkpoints is not None:
                    checkpoints = os.listdir(args.output_dir)
                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda name: int(name.split("-")[1]))
                    if len(checkpoints) >= args.limit_saved_checkpoints:
                        index_remove_until = len(checkpoints) - args.limit_saved_checkpoints + 1
                        remove_checkpoints = checkpoints[0:index_remove_until]
                        
                        logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(remove_checkpoints)} checkpoints"
                                )
                        logger.info(f"removing checkpoints: {', '.join(remove_checkpoints)}")

                        for removing_checkpoint in remove_checkpoints:
                            removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                            shutil.rmtree(removing_checkpoint)

                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}")
            
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    #Save LoRA Layers
    unet = unet.to(torch.float32)
    unet.save_attn_procs(args.output_dir)

    accelerator.end_training()    

if __name__ == "__main__":
    main()         