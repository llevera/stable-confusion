#!/bin/bash
################################################################################
# A100 Setup Script for Diversity Distillation Project
# 
# This script sets up the complete environment for the Diversity Distillation
# solution on an A100 GPU instance.
#
# Usage: bash setup_azure_a100.sh
################################################################################

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

################################################################################
# STEP 0: Environment Information
################################################################################
print_status "======================================================================"
print_status "A100 Environment Setup for Diversity Distillation"
print_status "======================================================================"
echo ""

print_status "Checking system information..."
echo "OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d '=' -f2 | tr -d '\"')"
echo "Python: $(python3 --version)"
echo "CUDA: $(nvidia-smi | grep "CUDA Version" | awk '{print $9}')"
echo ""

print_status "GPU Information:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo ""

################################################################################
# STEP 1: Set Working Directory
################################################################################
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"
print_status "Working directory: $SCRIPT_DIR"
echo ""

################################################################################
# STEP 2: Create and activate virtual environment
################################################################################
print_status "Setting up Python virtual environment..."

if [ ! -d ".venv" ]; then
    print_status "Creating .venv virtual environment..."
    python3 -m venv .venv
    print_success "Virtual environment created"
else
    print_status "Virtual environment .venv already exists"
fi

print_status "Activating virtual environment..."
source .venv/bin/activate

# Verify we're in the venv
if [[ "$VIRTUAL_ENV" != "" ]]; then
    print_success "Virtual environment activated: $VIRTUAL_ENV"
else
    print_error "Failed to activate virtual environment"
    exit 1
fi
echo ""

################################################################################
# STEP 3: Upgrade pip and install build tools
################################################################################
print_status "Upgrading pip and installing build tools..."
pip install --upgrade pip setuptools wheel
print_success "Build tools updated"
echo ""

################################################################################
# STEP 3.5: Install problematic packages with pre-built wheels
################################################################################
print_status "Installing packages that require pre-built wheels..."
# Install ruamel.yaml with a known working version that has pre-built wheels
# Skip source builds to avoid compilation errors
pip install "ruamel.yaml>=0.17.0,<0.18.0" --only-binary=:all: 2>/dev/null || \
pip install "ruamel.yaml==0.17.21" 2>/dev/null || \
print_warning "Could not install ruamel.yaml from wheel, will try later"

# Fix urllib3/botocore compatibility issue
# Upgrade botocore to a version compatible with urllib3 1.26.x
pip install --upgrade "botocore>=1.31.0" 2>/dev/null || true
pip install --upgrade "boto3>=1.28.0" 2>/dev/null || true

print_success "Pre-built wheels step completed"
echo ""

################################################################################
# STEP 4: Install missing core packages
################################################################################
print_status "Checking and installing missing core packages..."

# Check if diffusers is installed
if ! pip show diffusers &> /dev/null; then
    print_warning "diffusers not found. Installing..."
    pip install diffusers==0.33.1
fi

# Install all required packages from requirements.txt
if [ -f "requirements.txt" ]; then
    print_status "Installing packages from requirements.txt..."
    # Use --prefer-binary to avoid building from source when possible
    pip install -r requirements.txt --prefer-binary
    print_success "All packages from requirements.txt installed"
else
    print_warning "requirements.txt not found. Installing packages manually..."
    
    # Core ML packages (torch already installed)
    print_status "Installing core ML packages..."
    pip install \
        diffusers==0.33.1 \
        transformers==4.48.0 \
        safetensors==0.5.3 \
        huggingface_hub==0.29.3 \
        peft>=0.13.1
    
    # Data science packages
    print_status "Installing data science packages..."
    pip install \
        numpy==2.2.5 \
        pandas==2.2.3 \
        scikit-learn==1.6.1 \
        scikit-image
    
    # Image processing
    print_status "Installing image processing packages..."
    pip install \
        Pillow==11.2.1 \
        opencv-python-headless==4.10.0.84
    
    # Visualization and metrics
    print_status "Installing visualization packages..."
    pip install \
        matplotlib==3.10.1 \
        lpips==0.1.4
    
    # Utilities
    print_status "Installing utility packages..."
    pip install \
        tqdm==4.66.5 \
        requests==2.32.3 \
        packaging==25.0 \
        submitit==1.5.2
    
    print_success "All core packages installed"
fi
echo ""

################################################################################
# STEP 5: Verify PyTorch and CUDA
################################################################################
print_status "Verifying PyTorch and CUDA setup..."
python << 'EOF'
import torch
import sys

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")
    
    # Enable performance optimizations
    print("\n✓ Enabling PyTorch performance optimizations...")
    torch.backends.cudnn.benchmark = True  # Auto-tune kernels
    torch.backends.cuda.matmul.allow_tf32 = True  # Use TF32 for matmul on Ampere
    torch.backends.cudnn.allow_tf32 = True  # Use TF32 for cuDNN on Ampere
    print("  - cuDNN benchmark mode: enabled")
    print("  - TF32 precision: enabled (faster on A100)")
    
    # Test CUDA with performance check
    try:
        import time
        x = torch.randn(4096, 4096).cuda()
        y = torch.randn(4096, 4096).cuda()
        
        # Warmup
        for _ in range(3):
            _ = torch.matmul(x, y)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        for _ in range(10):
            z = torch.matmul(x, y)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        tflops = (2 * 4096**3 * 10) / elapsed / 1e12
        print(f"  - Performance: {tflops:.2f} TFLOPS (4096x4096 matmul)")
        print("✓ CUDA computation test passed")
    except Exception as e:
        print(f"✗ CUDA computation test failed: {e}", file=sys.stderr)
        sys.exit(1)
else:
    print("✗ CUDA not available! This will not work properly.", file=sys.stderr)
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    print_success "PyTorch and CUDA verified successfully with optimizations enabled"
else
    print_error "PyTorch CUDA verification failed!"
    exit 1
fi
echo ""

################################################################################
# STEP 6: Verify all required packages
################################################################################
print_status "Verifying all required packages..."
python << 'EOF'
import sys

packages_to_check = {
    'torch': 'PyTorch',
    'torchvision': 'TorchVision',
    'diffusers': 'Diffusers',
    'transformers': 'Transformers',
    'safetensors': 'SafeTensors',
    'huggingface_hub': 'Hugging Face Hub',
    'numpy': 'NumPy',
    'pandas': 'Pandas',
    'sklearn': 'Scikit-learn',
    'skimage': 'Scikit-image',
    'PIL': 'Pillow',
    'cv2': 'OpenCV',
    'matplotlib': 'Matplotlib',
    'lpips': 'LPIPS',
    'tqdm': 'tqdm',
    'requests': 'Requests',
}

missing_packages = []
for package, name in packages_to_check.items():
    try:
        __import__(package)
        print(f"✓ {name}")
    except ImportError:
        print(f"✗ {name} - MISSING")
        missing_packages.append(name)

if missing_packages:
    print(f"\nMissing packages: {', '.join(missing_packages)}", file=sys.stderr)
    sys.exit(1)
else:
    print("\n✓ All required packages are installed")
EOF

if [ $? -eq 0 ]; then
    print_success "All package dependencies verified"
else
    print_error "Some packages are missing. Please check the output above."
    exit 1
fi
echo ""

################################################################################
# STEP 6.5: Setup Jupyter Notebook Support
################################################################################
print_status "Setting up Jupyter notebook support..."

# Install Jupyter and ipykernel if not already installed
if ! pip show jupyter &> /dev/null || ! pip show ipykernel &> /dev/null; then
    print_status "Installing Jupyter and ipykernel..."
    pip install jupyter ipykernel jupyter_contrib_nbextensions --prefer-binary
    print_success "Jupyter packages installed"
else
    print_status "Jupyter packages already installed"
fi

# Register the virtual environment as a Jupyter kernel
print_status "Registering virtual environment as Jupyter kernel..."
python -m ipykernel install --user --name=diversity-distillation --display-name="Python (Diversity Distillation)"

if [ $? -eq 0 ]; then
    print_success "Jupyter kernel 'Python (Diversity Distillation)' registered"
    print_status "You can now select this kernel in Jupyter notebooks"
else
    print_warning "Kernel registration had issues, but continuing..."
fi
echo ""

################################################################################
# STEP 7: Verify project structure
################################################################################
print_status "Verifying project structure..."

required_dirs=("utils" "evalscripts" "data" "notebooks")
required_files=(
    "utils/load_util.py"
    "utils/utils.py"
    "evalscripts/diversity_distillation_sdxl.py"
    "data/coco_30k.csv"
)

missing_items=()

for dir in "${required_dirs[@]}"; do
    if [ ! -d "$dir" ]; then
        print_warning "Directory missing: $dir"
        missing_items+=("$dir")
    else
        echo "✓ Directory exists: $dir"
    fi
done

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        print_warning "File missing: $file"
        missing_items+=("$file")
    else
        echo "✓ File exists: $file"
    fi
done

if [ ${#missing_items[@]} -gt 0 ]; then
    print_warning "Some project files are missing, but continuing..."
else
    print_success "All project files verified"
fi
echo ""

################################################################################
# STEP 8: Create output directories
################################################################################
print_status "Creating output directories..."
mkdir -p eval-images
mkdir -p images
print_success "Output directories created"
echo ""

################################################################################
# STEP 9: Test model loading and cache models
################################################################################
print_status "Testing model loading and caching models..."
print_warning "This will download several GB of models from Hugging Face Hub"
print_warning "Models will be cached in ~/.cache/huggingface/"
echo ""

# Set HuggingFace cache to a location with enough space
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
print_status "HuggingFace cache directory: $HF_HOME"
echo ""

# Test basic model loading
python << 'EOF'
import torch
from diffusers import StableDiffusionXLPipeline
from utils.load_util import load_sdxl_models
import sys
import time

# Enable performance optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')  # Use TF32

print("Testing model loading functionality...")
print("This may take several minutes on first run (downloading models)...\n")
print("✓ PyTorch performance optimizations enabled")

try:
    # Test loading DMD models (default)
    print("Loading DMD distillation models...")
    pipe, base_unet, base_scheduler, distilled_unet, distilled_scheduler = load_sdxl_models(
        distillation_type='dmd',
        weights_dtype=torch.bfloat16,
        device='cuda:0'
    )
    print("✓ DMD models loaded successfully")
    
    # Clean up
    del pipe, base_unet, base_scheduler, distilled_unet, distilled_scheduler
    torch.cuda.empty_cache()
    
    print("\n✓ Model loading test passed!")
    print("All models are now cached for future use.")
    
except Exception as e:
    print(f"\n✗ Model loading test failed: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    print_success "Model loading test passed and models cached"
else
    print_error "Model loading test failed!"
    exit 1
fi
echo ""

################################################################################
# STEP 10: Run a quick test generation with performance benchmark
################################################################################
print_status "Running a quick test image generation with performance benchmark..."
python << 'EOF'
import torch
import sys
import os
import time
sys.path.append('.')

from utils.load_util import load_sdxl_models

# Enable performance optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

try:
    print("Loading models with performance optimizations enabled...")
    start_load = time.time()
    pipe, base_unet, base_scheduler, distilled_unet, distilled_scheduler = load_sdxl_models(
        distillation_type='dmd',
        weights_dtype=torch.bfloat16,
        device='cuda:0'
    )
    load_time = time.time() - start_load
    print(f"✓ Models loaded in {load_time:.2f}s")
    
    # Set schedulers
    base_scheduler.set_timesteps(4)
    distilled_scheduler.set_timesteps(4)
    
    print("\nGenerating test image with timing...")
    prompt = ["A beautiful sunset over mountains"]
    
    # Warmup run
    print("Warmup pass...")
    pipe.unet = base_unet
    pipe.scheduler = base_scheduler
    
    _ = pipe(
        prompt=prompt,
        from_timestep=0,
        till_timestep=0,
        guidance_scale=5.0,
        num_inference_steps=4,
        output_type='latent'
    )
    torch.cuda.synchronize()
    
    # Timed generation
    print("Timed generation pass...")
    start_gen = time.time()
    
    # Use base model for first step
    pipe.unet = base_unet
    pipe.scheduler = base_scheduler
    
    base_latents = pipe(
        prompt=prompt,
        from_timestep=0,
        till_timestep=0,
        guidance_scale=5.0,
        num_inference_steps=4,
        output_type='latent'
    )
    
    # Switch to distilled model
    pipe.unet = distilled_unet
    pipe.scheduler = distilled_scheduler
    
    pil_images = pipe(
        prompt=prompt,
        start_latents=base_latents,
        guidance_scale=0,
        from_timestep=1,
        till_timestep=None,
        num_inference_steps=4
    )
    
    torch.cuda.synchronize()
    gen_time = time.time() - start_gen
    
    # Save test image
    os.makedirs('test-output', exist_ok=True)
    pil_images[0].save('test-output/test_generation.png')
    
    print(f"\n✓ Test image generated successfully: test-output/test_generation.png")
    print(f"✓ Generation time: {gen_time:.2f}s")
    print(f"✓ Images/second: {1.0/gen_time:.2f}")
    
    # Memory stats
    mem_allocated = torch.cuda.max_memory_allocated() / 1024**3
    mem_reserved = torch.cuda.max_memory_reserved() / 1024**3
    print(f"✓ Peak memory: {mem_allocated:.2f} GB allocated, {mem_reserved:.2f} GB reserved")
    
    # Cleanup
    del pipe, base_unet, base_scheduler, distilled_unet, distilled_scheduler
    torch.cuda.empty_cache()
    
except Exception as e:
    print(f"✗ Test generation failed: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    print_success "Test image generation successful with performance metrics"
else
    print_error "Test image generation failed!"
    exit 1
fi
echo ""

################################################################################
# STEP 11: Display usage information
################################################################################
print_success "======================================================================"
print_success "Setup Complete! ✓"
print_success "======================================================================"
echo ""
print_status "Virtual environment: $VIRTUAL_ENV"
echo ""
print_status "Environment is ready to use. Here's how to get started:"
echo ""
echo "IMPORTANT: Always activate the virtual environment first:"
echo "   source .venv/bin/activate"
echo ""
echo "1. Generate images with diversity distillation:"
echo "   python evalscripts/diversity_distillation_sdxl.py \\"
echo "       --distillation_type 'dmd' \\"
echo "       --prompts_path 'data/coco_30k.csv' \\"
echo "       --exp_name 'dmd_diversity_distillation' \\"
echo "       --device 'cuda:0' \\"
echo "       --from_case 0 \\"
echo "       --till_case 100"
echo ""
echo "2. Available distillation types:"
echo "   - dmd (default)"
echo "   - lightning"
echo "   - turbo"
echo "   - lcm"
echo ""
echo "3. Run Jupyter notebooks:"
echo "   - Open VS Code or Jupyter Lab"
echo "   - Navigate to notebooks/ directory (e.g., diversity_distillation_2.ipynb)"
echo "   - Select kernel: 'Python (Diversity Distillation)'"
echo "   - The notebook is ready to run with all dependencies installed"
echo ""
echo "4. Test image saved to: test-output/test_generation.png"
echo ""
echo "5. To deactivate the virtual environment when done:"
echo "   deactivate"
echo ""
print_status "GPU Status:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
echo ""
print_status "Virtual environment: .venv/"
print_status "Cached models location: ~/.cache/huggingface/"
echo ""
print_success "Happy experimenting! 🚀"
