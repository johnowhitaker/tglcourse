import wandb
import torch
import torchvision
import math
import torch.nn.functional as F
import numpy as np
from torch import nn
from datasets import load_dataset
from diffusers import DDPMPipeline, DDPMScheduler, DDIMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler
from fastcore.script import *
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)
from tqdm.auto import tqdm

## Adapted from https://github.com/huggingface/diffusers/tree/main/examples/unconditional_image_generation
## Check that out to see some extra nifty things like EMA and multi-GPU training with accelerate


@call_parse
def main(dataset_name = 'huggan/smithsonian_butterflies_subset', # Dataset name
         img_size = 32, # Image size in pixels
         batch_size=64,
         num_epochs = 100, 
         job_type = 'train', # For W&B
         comments = '',  # For W&B
         wandb_entity = 'tglcourse', # Use your own account if you don't want to log to our team
         wandb_project = 'lesson12_diffusers_training',
         learning_rate = 1e-4,
         n_sampling_steps = 40,
         save_images_epochs = 10, # How frequenty to save images
         log_every=10, # How often to log loss etc to W&B
         use_device = 'cuda:0', # Which device should we use? Usually cuda:0
         ema=False, # Use EMA?
         ema_beta=0.999, # EMA factor
        ):
    device = torch.device(use_device if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}') 
    
    model = UNet2DModel(
        sample_size=img_size,              # the target image resolution
        in_channels=3,                     # the number of input channels, 3 for RGB images
        out_channels=3,                    # the number of output channels
        layers_per_block=2,                # how many ResNet layers to use per UNet block
        block_out_channels=(64, 128, 256), # <<< Can adjust
        down_block_types=( 
            "DownBlock2D",      # a regular ResNet downsampling block
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "AttnDownBlock2D",  # Add more here if needed - must match block_out_channels. 
        ), 
        up_block_types=(
            "AttnUpBlock2D",  # Attention blocks use more memory, only use at lower layers if training high res
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",      # a regular ResNet upsampling block
          ),
    )
    model.to(device)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )
    
    
    augmentations = Compose(
        [
            Resize(img_size, interpolation=InterpolationMode.BILINEAR),
            CenterCrop(img_size),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize([0.5], [0.5]),
        ]
    )

    if dataset_name is not None:
        dataset = load_dataset(dataset_name, split="train")
        
    def transforms(examples):
        images = [augmentations(image.convert("RGB")) for image in examples["image"]]
        return {"input": images}

    dataset.set_transform(transforms)
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    
    lr_scheduler = get_scheduler(
        'cosine',
        optimizer=optimizer,
        num_warmup_steps=400,
        num_training_steps=(len(train_dataloader) * num_epochs) #// args.gradient_accumulation_steps,
    )
    
    num_update_steps_per_epoch = math.ceil(len(train_dataloader)) #/ args.gradient_accumulation_steps)
    global_step=0
    
    # Config with all the settings
    cfg = dict(model.config)
    cfg['num_epochs'] = num_epochs
    cfg['learning_rate'] = learning_rate
    cfg['comments'] = comments
    cfg['dataset'] = dataset_name

    # Training!
    wandb.init(project=wandb_project, job_type=job_type, entity=wandb_entity, config=cfg)
    
    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            clean_images = batch["input"].to(device)
            # Sample noise that we'll add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bsz = clean_images.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # Predict the noise residual
            noise_pred = model(noisy_images, timesteps).sample
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            global_step += 1

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "global_step": global_step}
            progress_bar.set_postfix(**logs)
            logs['epoch'] = epoch
            if global_step%log_every==0: wandb.log(logs, step=global_step)
        progress_bar.close()

        # Generate sample images for visual inspection
        if epoch % save_images_epochs == 0 or epoch == num_epochs - 1:
            sampling_scheduler = DDIMScheduler(num_train_timesteps=1000)
            sampling_scheduler.set_timesteps(n_sampling_steps)
            generator = torch.manual_seed(0)
            sample = torch.randn(batch_size, 3, 32, 32, generator=generator).to(device)
            for i, t in enumerate(sampling_scheduler.timesteps):
                with torch.no_grad():
                    residual = model(sample, t).sample
                sample = sampling_scheduler.step(residual, t, sample).prev_sample
            sample = sample.clip(-1, 1) * 0.5 + 0.5
            preview_im = torchvision.utils.make_grid(sample, nrow=int(batch_size**0.5)).permute(1, 2, 0).cpu() 
            preview_im = (np.array(preview_im) * 255).round().astype("uint8")
            
            # Turn them into a grid
            wandb.log({'Image':wandb.Image(preview_im)}, step=global_step)
            

        # if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
        #     # save the model
        #     pipeline.save_pretrained(args.output_dir)
        #     if args.push_to_hub:
        #         repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=False)
    
    wandb.finish()
    # TODO save model
    # TODO push to hub?
    # TODO smaller model or model variants?
    # TODO wnab save code
    # TODO log epoch
    # TODO use global step when logging