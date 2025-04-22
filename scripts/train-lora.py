#Adapted from https://github.com/huggingface/diffusers/blob/dd9a5caf61f04d11c0fa9f3947b69ab0010c9a0f/examples/text_to_image/train_text_to_image_lora.py

import argparse
import logging
import datasets
import transformers
import torch
import os
import numpy
import random
import math

from tqdm.auto import tqdm
from pathlib import Path
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.logging import get_logger
from transformers import CLIPTokenizer, CLIPTextModel
from datasets import load_dataset
from torchvision import transforms
from diffusers.optimization import get_constant_schedule_with_warmup

import diffusers
from diffusers import UNet2DConditionModel, UniPCMultistepScheduler, AutoencoderKL
from diffusers.models.attention_processor import LoRAAttnProcessor
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
    parser.add_argument(        #Need this since I'm on an RTX 3060 ti
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
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
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
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
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(args.output_dir, args.logging_dir)

    seed = 0
    learning_rate = 1e-4
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-08
    weight_decay = 1e-2
    batch_size = 16
    gradient_accumulation_steps = 8
    num_epochs = 20
    warmup_steps = 500

    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="fp16",
        log_with="tensorboard",
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
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    noise_scheduler = UniPCMultistepScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    unet.required_grad_(False)      # free up memory by freezing model parameters. The model itself is not being trained with LoRA
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

        lora_attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=args.rank,
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

    data_files = {}
    if args.dataset_path is not None:
        data_files["train"] = os.path.join(args.dataset_path, "**")
    dataset = load_dataset(
        "imagefolder",
        data_files=data_files
    )
    column_names = dataset["train"].column_names
    image_column = column_names[0]
    caption_column = column_names[1]

    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
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
    train_transforms = transforms.Compose(
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop((512, 512)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = [token for token in tokenize_captions(examples[caption_column])]
        return examples
    
    if args.max_train_samples is not None:
        dataset["train"] = dataset["train"].shuffle(seed=seed).select(range(args.max_train_samples))    #Selecting based on number of training rows we are using
    train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    num_update_steps_in_epoch = math.ceil(int(len(train_dataloader)) / gradient_accumulation_steps)
    max_train_steps = num_update_steps_in_epoch * num_epochs

    lr_scheduler = get_constant_schedule_with_warmup(
        optimizer=optimizer,
        num_training_steps=max_train_steps,
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
    