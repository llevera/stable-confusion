#!/usr/bin/env python3
"""
Quick test to verify PCM model loading fix
"""
import torch
import sys
sys.path.append('.')

from utils.load_util import load_sdxl_models

print("="*80)
print("Testing PCM Model Loading Fix")
print("="*80)
print("\nThis test verifies that PCM model loads from the correct subfolder path")
print("Old path (404): wangfuyun/PCM_Weights/pcm_sdxl_smallcfg_4step_converted.safetensors")
print("New path (correct): wangfuyun/PCM_Weights/sdxl/pcm_sdxl_smallcfg_4step_converted.safetensors")
print("\n" + "="*80)

try:
    print("\nLoading PCM model...")
    pipe, base_unet, base_scheduler, distilled_unet, distilled_scheduler = load_sdxl_models(
        distillation_type='pcm',
        weights_dtype=torch.bfloat16,
        device='cuda:0'
    )
    
    print("✓ PCM model loaded successfully!")
    print(f"\nScheduler type: {type(distilled_scheduler).__name__}")
    print(f"Scheduler config:")
    print(f"  - timestep_spacing: {distilled_scheduler.config.timestep_spacing}")
    print(f"  - clip_sample: {distilled_scheduler.config.clip_sample}")
    print(f"  - set_alpha_to_one: {distilled_scheduler.config.set_alpha_to_one}")
    
    print("\n" + "="*80)
    print("✅ PCM Fix Verified Successfully!")
    print("="*80)
    print("\nPCM is now ready to use with the recommended DDIM scheduler settings:")
    print("  - timestep_spacing='trailing'")
    print("  - clip_sample=False")
    print("  - set_alpha_to_one=False")
    
    # Clean up
    del pipe, base_unet, base_scheduler, distilled_unet, distilled_scheduler
    torch.cuda.empty_cache()
    
except Exception as e:
    print(f"\n❌ Error loading PCM model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
