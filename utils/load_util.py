import torch
import torch
from diffusers import UNet2DConditionModel, LCMScheduler, StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, DDIMScheduler, TCDScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import sys
sys.path.append('.')
from .utils import call_sdxl

StableDiffusionXLPipeline.__call__ = call_sdxl

def load_model(distillation_type=None, weights_dtype=torch.bfloat16, device='cuda:0'):
    """
    Load SDXL models with specified distillation type.
    
    Args:
        distillation_type: Model type to load. Options: 'base', 'dmd', 'lightning', 'turbo', 
                          'lcm', 'hyper', 'hyper_1step', 'pcm', 'tcd', 'flash'
        weights_dtype: Data type for model weights
        device: Device to load model on
    
    Returns:
        For 'base': (pipe, base_unet, base_scheduler)
        For others: (pipe, base_unet, base_scheduler, distilled_unet, distilled_scheduler)
    """
    basemodel_id = "stabilityai/stable-diffusion-xl-base-1.0"

    # Load base model and scheduler
    base_unet = UNet2DConditionModel.from_pretrained(basemodel_id, subfolder="unet").to(device, weights_dtype)
    pipe = StableDiffusionXLPipeline.from_pretrained(basemodel_id, unet=base_unet, torch_dtype=weights_dtype, use_safetensors=True)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    base_scheduler = pipe.scheduler
    
    # Return early for base model
    if distillation_type == None:
        pipe.to(device).to(weights_dtype)
        return pipe, base_unet, base_scheduler
    
    # Load distilled model based on type
    if distillation_type == 'dmd':
        
        repo_name = "tianweiy/DMD2"
        ckpt_name = "dmd2_sdxl_4step_unet_fp16.bin"

        distilled_unet = UNet2DConditionModel.from_config(basemodel_id, subfolder="unet").to(device, weights_dtype)
        distilled_unet.load_state_dict(torch.load(hf_hub_download(repo_name, ckpt_name), weights_only=True))
        distilled_scheduler =  LCMScheduler.from_config(pipe.scheduler.config)


    elif distillation_type == 'lightning':
        
        repo = "ByteDance/SDXL-Lightning"
        ckpt = "sdxl_lightning_4step_unet.safetensors" # Use the correct ckpt for your step setting!
        
        # Load model.
        distilled_unet = UNet2DConditionModel.from_config(basemodel_id, subfolder="unet").to(device, weights_dtype)
        distilled_unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))
        # Ensure sampler uses "trailing" timesteps.
        distilled_scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        base_scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    elif distillation_type == 'turbo':
        
        distilled_unet = UNet2DConditionModel.from_pretrained("stabilityai/sdxl-turbo", subfolder="unet", torch_dtype=weights_dtype).to(device, weights_dtype)
        distilled_scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        base_scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        
    elif distillation_type == 'lcm':
        
        distilled_unet = UNet2DConditionModel.from_pretrained("latent-consistency/lcm-sdxl", torch_dtype=weights_dtype).to(device, weights_dtype)
        distilled_scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    
    elif distillation_type == 'hyper':
        # Hyper-SDXL 8-step CFG-preserved LoRA (supports typical guidance scales)
        # Note: For 1-step, use hyper_1step which requires very low guidance
        from diffusers import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(basemodel_id, torch_dtype=weights_dtype)
        pipe.load_lora_weights("ByteDance/Hyper-SD", weight_name="Hyper-SDXL-8steps-CFG-lora.safetensors", adapter_name="hyper-sdxl-8step")
        pipe.set_adapters(["hyper-sdxl-8step"], adapter_weights=[1.0])
        
        distilled_unet = pipe.unet
        distilled_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    elif distillation_type == 'hyper_1step':
        # Hyper-SDXL 1-step unified LoRA (very low guidance, negatives have limited effect)
        from diffusers import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(basemodel_id, torch_dtype=weights_dtype)
        pipe.load_lora_weights("ByteDance/Hyper-SD", weight_name="Hyper-SDXL-1step-lora.safetensors", adapter_name="hyper-sdxl-1step")
        pipe.set_adapters(["hyper-sdxl-1step"], adapter_weights=[1.0])
        
        distilled_unet = pipe.unet
        distilled_scheduler = TCDScheduler.from_config(pipe.scheduler.config)
    
    elif distillation_type == 'pcm':
        # PCM-SDXL - Phased Consistency Models (good for 1-16 steps)
        # Uses DDIM scheduler with specific settings as recommended by PCM repo
        from diffusers import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(basemodel_id, torch_dtype=weights_dtype)
        pipe.load_lora_weights("wangfuyun/PCM_Weights", weight_name="pcm_sdxl_smallcfg_4step_converted.safetensors", subfolder="sdxl", adapter_name="pcm-lora")
        pipe.set_adapters(["pcm-lora"], adapter_weights=[1.0])
        
        distilled_unet = pipe.unet
        # PCM recommends DDIM with these specific settings
        distilled_scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config,
            timestep_spacing="trailing",
            clip_sample=False,
            set_alpha_to_one=False
        )
    
    elif distillation_type == 'tcd':
        # TCD-SDXL - Trajectory Consistency Distillation (2-8 steps, uses standard CFG)
        # Includes stochasticity parameter (gamma) for quality/stability tradeoff
        from diffusers import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(basemodel_id, torch_dtype=weights_dtype)
        pipe.load_lora_weights("h1t/TCD-SDXL-LoRA", adapter_name="tcd-lora")
        pipe.set_adapters(["tcd-lora"], adapter_weights=[1.0])
        
        distilled_unet = pipe.unet
        distilled_scheduler = TCDScheduler.from_config(pipe.scheduler.config)
    
    elif distillation_type == 'flash':
        # Flash Diffusion - Light adapters with broad compatibility
        # Works with ControlNet, IP-Adapter, etc.
        repo = "jasperai/flash-sdxl"
        ckpt = "pytorch_lora_weights.safetensors"  # Correct filename in the repo
        
        from diffusers import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(basemodel_id, torch_dtype=weights_dtype)
        pipe.load_lora_weights(repo, weight_name=ckpt, adapter_name="flash-sdxl")
        pipe.set_adapters(["flash-sdxl"], adapter_weights=[1.0])
        pipe.fuse_lora()
        
        distilled_unet = pipe.unet
        distilled_scheduler = LCMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    
    else:
        raise ValueError(f"Distillation type '{distillation_type}' is not recognized. "
                        "Available options: 'base', 'dmd', 'lightning', 'turbo', 'lcm', 'hyper', "
                        "'hyper_1step', 'pcm', 'tcd', 'flash'")
    
    pipe.to(device).to(weights_dtype)
    return pipe, base_unet, base_scheduler, distilled_unet, distilled_scheduler



def load_pipe(distillation_type=None, weights_dtype=torch.bfloat16, device='cuda:0'):
    """
    Load and return a configured SDXL pipeline for direct use.
    
    Args:
        distillation_type: Model type to load. Options: 'base', 'dmd', 'lightning', 'turbo', 
                          'lcm', 'hyper', 'hyper_1step', 'pcm', 'tcd', 'flash', or None for base
        weights_dtype: Data type for model weights
        device: Device to load model on
    
    Returns:
        Configured pipeline ready for inference
    """
    
    # Load model components
    result = load_model(distillation_type, weights_dtype, device)
    
    # Extract pipe and scheduler based on distillation type
    if distillation_type == None:
        pipe, _, _ = result
    else:
        pipe, _, _, _, distilled_scheduler = result
        pipe.scheduler = distilled_scheduler
    
    return pipe