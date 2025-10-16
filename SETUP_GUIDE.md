# Azure ML A100 Setup Guide for Diversity Distillation

This guide provides complete instructions for setting up and running the Diversity Distillation solution on an Azure ML A100 instance from a blank VM.

## Quick Start (TL;DR)

```bash
cd /home/azureuser/cloudfiles/code/Users/Andrew.Revell/distillation
bash setup_azure_a100.sh
source .venv/bin/activate  # Activate the virtual environment
```

That's it! The script handles everything automatically and creates a local `.venv` virtual environment.

## What the Setup Script Does

The `setup_azure_a100.sh` script is a comprehensive, automated setup tool that:

### 1. **System Verification** ✓
- Checks OS, Python version, and CUDA availability
- Displays GPU information (A100 specs)
- Verifies disk space

### 2. **Virtual Environment Setup** 🐍
- Creates a local `.venv` virtual environment
- Activates the environment automatically during setup
- Isolates project dependencies from system packages

### 3. **Package Installation** 📦
- Upgrades pip, setuptools, and wheel
- Installs all required packages from `requirements-fixed.txt`:
  - PyTorch 2.4.0+ (already installed on Azure ML)
  - Diffusers 0.33.1
  - Transformers 4.48.0
  - Hugging Face Hub 0.29.3
  - SafeTensors, NumPy, Pandas, Scikit-learn, Scikit-image
  - Pillow, OpenCV (headless)
  - Matplotlib, LPIPS
  - tqdm, requests, packaging, submitit

### 4. **CUDA Verification** 🔥
- Verifies PyTorch can use CUDA
- Tests GPU computation
- Displays GPU memory and capabilities

### 5. **Package Verification** ✅
- Checks all required Python packages are importable
- Reports missing dependencies

### 6. **Project Structure** 📁
- Verifies all required directories exist
- Checks for required Python files and data
- Creates output directories

### 7. **Model Caching** 💾
- Downloads and caches required models from Hugging Face:
  - Stable Diffusion XL Base 1.0
  - DMD2 distilled UNET
  - (Optional: Lightning, Turbo, LCM models)
- Models are cached in `~/.cache/huggingface/`
- First run takes ~5-10 minutes for downloads

### 8. **Test Generation** 🎨
- Runs a complete test image generation
- Verifies the entire pipeline works
- Saves test image to `test-output/test_generation.png`

## Prerequisites

- Azure ML A100 compute instance (already provisioned)
- Ubuntu 22.04 (already installed)
- Python 3.10+ (already installed)
- CUDA 12.2+ (already installed)
- Internet connection for downloading models

**Note:** The Azure ML kernels (Python 3.10 - AzureML, Python 3.10 - Pytorch and Tensorflow, etc.) are already available, but we'll create a local `.venv` environment for better dependency isolation.

## Manual Setup (Alternative)

If you prefer to set up manually or troubleshoot issues:

### Step 1: Create Virtual Environment

```bash
cd /home/azureuser/cloudfiles/code/Users/Andrew.Revell/distillation
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements-fixed.txt
```

### Step 3: Verify PyTorch and CUDA

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Step 4: Test Model Loading

```python
from utils.load_util import load_sdxl_models
import torch

pipe, base_unet, base_scheduler, distilled_unet, distilled_scheduler = load_sdxl_models(
    distillation_type='dmd',
    weights_dtype=torch.bfloat16,
    device='cuda:0'
)
print("Models loaded successfully!")
```

## Usage Examples

**Important:** Always activate the virtual environment first:
```bash
source .venv/bin/activate
```

### 1. Basic Image Generation

Generate a single image with diversity distillation:

```bash
source .venv/bin/activate  # Always activate first!

python evalscripts/diversity_distillation_sdxl.py \
    --distillation_type 'dmd' \
    --prompts_path 'data/coco_30k.csv' \
    --exp_name 'my_first_test' \
    --device 'cuda:0' \
    --from_case 0 \
    --till_case 1
```

### 2. Batch Generation

Generate 100 images from the COCO dataset:

```bash
python evalscripts/diversity_distillation_sdxl.py \
    --distillation_type 'dmd' \
    --prompts_path 'data/coco_30k.csv' \
    --exp_name 'dmd_batch_100' \
    --device 'cuda:0' \
    --from_case 0 \
    --till_case 100
```

### 3. Different Distillation Types

Try different distillation models:

```bash
# SDXL-Lightning (4-step)
python evalscripts/diversity_distillation_sdxl.py \
    --distillation_type 'lightning' \
    --exp_name 'lightning_test'

# SDXL-Turbo
python evalscripts/diversity_distillation_sdxl.py \
    --distillation_type 'turbo' \
    --exp_name 'turbo_test'

# LCM-SDXL
python evalscripts/diversity_distillation_sdxl.py \
    --distillation_type 'lcm' \
    --exp_name 'lcm_test'
```

### 4. Custom Parameters

Fine-tune generation parameters:

```bash
python evalscripts/diversity_distillation_sdxl.py \
    --distillation_type 'dmd' \
    --prompts_path 'data/coco_30k.csv' \
    --exp_name 'custom_params' \
    --device 'cuda:0' \
    --base_guidance_scale 7.0 \
    --distilled_guidance_scale 0 \
    --base_num_inference_steps 20 \
    --distilled_num_inference_steps 4 \
    --run_distilled_from_timestep 1 \
    --num_images_per_prompt 4
```

## Command-Line Arguments Reference

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--distillation_type` | str | 'dmd' | Model type: 'dmd', 'lightning', 'turbo', 'lcm' |
| `--prompts_path` | str | 'data/coco_30k.csv' | Path to CSV file with prompts |
| `--exp_name` | str | None | Custom name for output folder |
| `--device` | str | 'cuda:0' | CUDA device to use |
| `--from_case` | int | 0 | Start case number |
| `--till_case` | int | 9999 | End case number |
| `--base_guidance_scale` | float | 5.0 | CFG scale for base model |
| `--distilled_guidance_scale` | float | 0.0 | CFG scale for distilled model |
| `--base_num_inference_steps` | int | 4 | Steps for base model |
| `--distilled_num_inference_steps` | int | 4 | Steps for distilled model |
| `--run_base_till_timestep` | int | None | Timestep to stop base model |
| `--run_distilled_from_timestep` | int | 1 | Timestep to start distilled model |
| `--dtype` | str | 'bf16' | Data type: 'fp16', 'fp32', 'bf16' |
| `--num_images_per_prompt` | int | 1 | Images to generate per prompt |

## Jupyter Notebooks

The project includes several Jupyter notebooks. When opening them in Azure ML, **select the `.venv` kernel**:

1. **`diversity_distillation_2.ipynb`** - Main diversity distillation notebook
2. **`dt-visualization.ipynb`** - DT-Visualization technique
3. **`notebooks/lora-transfer.ipynb`** - LoRA transfer experiments
4. **`notebooks/slider_transfer.ipynb`** - Concept Sliders experiments
5. **`notebooks/teaser.ipynb`** - Teaser examples

**To use the .venv kernel in Jupyter:**
1. Open a notebook in Azure ML
2. Click on the kernel selector (top right)
3. Choose "Python 3 (ipykernel)" or manually select `.venv/bin/python`
4. If `.venv` doesn't appear, you may need to install ipykernel:
   ```bash
   source .venv/bin/activate
   pip install ipykernel
```
distillation/
├── .venv/                        # Virtual environment (created by setup)
├── setup_azure_a100.sh          # Automated setup script
├── SETUP_GUIDE.md                # This file
├── README.md                     # Project README
```
distillation/
├── setup_azure_a100.sh          # Automated setup script
├── SETUP_GUIDE.md                # This file
├── README.md                     # Project README
├── requirements-fixed.txt        # Python dependencies
├── requirements-azure.txt        # Azure ML specific deps
├── utils/
│   ├── load_util.py             # Model loading utilities
│   ├── utils.py                 # Custom pipeline functions
│   ├── lora.py                  # LoRA utilities
│   └── sdv14.py                 # SD v1.4 utilities
├── evalscripts/
│   ├── diversity_distillation_sdxl.py  # Main evaluation script
│   └── sdxl_inference.py        # SDXL inference script
├── data/
│   └── coco_30k.csv             # COCO prompts dataset
├── notebooks/                    # Jupyter notebooks
├── images/                       # Project images
### Issue: CUDA Out of Memory

**Solution**: Reduce batch size or use fp16 instead of bf16:
```bash
source .venv/bin/activate
python evalscripts/diversity_distillation_sdxl.py \

### Issue: CUDA Out of Memory

**Solution**: Reduce batch size or use fp16 instead of bf16:
```bash
python3 evalscripts/diversity_distillation_sdxl.py \
    --dtype 'fp16' \
    --num_images_per_prompt 1
```

### Issue: Package import errors

**Solution**: Reinstall packages in the virtual environment:
```bash
source .venv/bin/activate
pip install --force-reinstall -r requirements-fixed.txt
```

### Issue: Virtual environment not activated

**Symptoms**: Import errors, command not found
**Solution**: Always activate before running commands:
```bash
source .venv/bin/activate
```

### Issue: Package import errors

**Solution**: Reinstall packages:
```bash
pip3 install --force-reinstall -r requirements-fixed.txt
```

### Issue: Slow generation

**First run is slow** due to model compilation. Subsequent runs will be faster.

### Issue: Permission denied on setup script

**Solution**:
```bash
chmod +x setup_azure_a100.sh
bash setup_azure_a100.sh
```

## Performance Notes
## Next Steps

1. **Activate the environment**: `source .venv/bin/activate`
2. **Explore the notebooks**: Start with `diversity_distillation_2.ipynb`
3. **Generate images**: Use the evaluation scripts
4. **Experiment**: Try different distillation types and parameters
5. **Read the paper**: Check out the [ArXiv preprint](https://arxiv.org/pdf/2503.10637.pdf)

## Virtual Environment Tips

- **Activate**: `source .venv/bin/activate`
- **Deactivate**: `deactivate`
- **Check if active**: `echo $VIRTUAL_ENV` (should show path to .venv)
- **Reinstall packages**: `pip install -r requirements-fixed.txt`
- **Remove and recreate**: `rm -rf .venv && bash setup_azure_a100.sh`
## Expected Output

After running the setup script, you should see:

1. ✓ All packages installed
2. ✓ PyTorch CUDA verified
3. ✓ Models downloaded and cached
4. ✓ Test image generated
5. ✓ Ready for production use

## Next Steps

1. **Explore the notebooks**: Start with `diversity_distillation_2.ipynb`
2. **Generate images**: Use the evaluation scripts
3. **Experiment**: Try different distillation types and parameters
4. **Read the paper**: Check out the [ArXiv preprint](https://arxiv.org/pdf/2503.10637.pdf)

## Support

For issues specific to this setup:
- Check the troubleshooting section above
- Review the original README.md
- Check the Azure ML documentation

For project-specific questions:
- Visit the [project website](https://distillation.baulab.info)
- Read the paper for methodology details

## Credits

Original implementation by Rohit Gandikota and David Bau.

Setup automation for Azure ML A100 by Andrew Revell.

Last updated: October 15, 2025
