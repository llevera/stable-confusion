#!/usr/bin/env python3
"""
Validation script for Diversity Distillation setup on Azure ML A100
This script performs comprehensive checks to ensure the environment is ready.
"""

import sys
import os
from pathlib import Path

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def print_success(text):
    print(f"✓ {text}")

def print_error(text):
    print(f"✗ {text}", file=sys.stderr)

def print_warning(text):
    print(f"⚠ {text}")

def check_python_version():
    """Check Python version"""
    print_header("Checking Python Version")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major >= 3 and version.minor >= 9:
        print_success("Python version is compatible (>=3.9)")
        return True
    else:
        print_error("Python version must be 3.9 or higher")
        return False

def check_cuda():
    """Check CUDA availability"""
    print_header("Checking CUDA and GPU")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print_success(f"CUDA is available (version {torch.version.cuda})")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / 1024**3
                print(f"  GPU {i}: {name}")
                print(f"    Memory: {memory_gb:.2f} GB")
                print(f"    Compute Capability: {props.major}.{props.minor}")
            
            # Test CUDA computation
            try:
                x = torch.randn(100, 100).cuda()
                y = torch.randn(100, 100).cuda()
                z = torch.matmul(x, y)
                print_success("CUDA computation test passed")
                return True
            except Exception as e:
                print_error(f"CUDA computation test failed: {e}")
                return False
        else:
            print_error("CUDA is not available")
            return False
    except ImportError as e:
        print_error(f"PyTorch not installed: {e}")
        return False

def check_packages():
    """Check required packages"""
    print_header("Checking Required Packages")
    
    packages = {
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
    
    all_ok = True
    for package, name in packages.items():
        try:
            mod = __import__(package)
            version = getattr(mod, '__version__', 'unknown')
            print_success(f"{name:20s} (version {version})")
        except ImportError:
            print_error(f"{name:20s} - NOT INSTALLED")
            all_ok = False
    
    return all_ok

def check_project_structure():
    """Check project files and directories"""
    print_header("Checking Project Structure")
    
    base_dir = Path.cwd()
    
    required_dirs = [
        'utils',
        'evalscripts',
        'data',
        'notebooks',
    ]
    
    required_files = [
        'utils/load_util.py',
        'utils/utils.py',
        'evalscripts/diversity_distillation_sdxl.py',
        'data/coco_30k.csv',
    ]
    
    all_ok = True
    
    print("\nDirectories:")
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists() and dir_path.is_dir():
            print_success(f"{dir_name}/")
        else:
            print_error(f"{dir_name}/ - MISSING")
            all_ok = False
    
    print("\nFiles:")
    for file_name in required_files:
        file_path = base_dir / file_name
        if file_path.exists() and file_path.is_file():
            size = file_path.stat().st_size
            size_str = f"{size:,} bytes" if size < 1024 else f"{size/1024:.1f} KB"
            print_success(f"{file_name} ({size_str})")
        else:
            print_error(f"{file_name} - MISSING")
            all_ok = False
    
    return all_ok

def check_model_loading():
    """Test model loading"""
    print_header("Testing Model Loading")
    
    try:
        import torch
        from utils.load_util import load_sdxl_models
        
        print("Attempting to load DMD models...")
        print("(This may take a while on first run - downloading models)")
        
        pipe, base_unet, base_scheduler, distilled_unet, distilled_scheduler = load_sdxl_models(
            distillation_type='dmd',
            weights_dtype=torch.bfloat16,
            device='cuda:0'
        )
        
        print_success("DMD models loaded successfully")
        
        # Check model components
        print(f"  Base UNET: {type(base_unet).__name__}")
        print(f"  Distilled UNET: {type(distilled_unet).__name__}")
        print(f"  Pipeline: {type(pipe).__name__}")
        
        # Cleanup
        del pipe, base_unet, base_scheduler, distilled_unet, distilled_scheduler
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print_error(f"Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_disk_space():
    """Check available disk space"""
    print_header("Checking Disk Space")
    
    try:
        import shutil
        
        # Check current directory
        total, used, free = shutil.disk_usage("/")
        
        print(f"Total: {total / 1024**3:.1f} GB")
        print(f"Used:  {used / 1024**3:.1f} GB")
        print(f"Free:  {free / 1024**3:.1f} GB")
        
        # Check if we have enough space (at least 50GB free recommended)
        if free / 1024**3 > 50:
            print_success("Sufficient disk space available")
            return True
        else:
            print_warning("Low disk space (< 50 GB free)")
            return True  # Don't fail, just warn
            
    except Exception as e:
        print_warning(f"Could not check disk space: {e}")
        return True  # Don't fail on this

def check_huggingface_cache():
    """Check HuggingFace cache"""
    print_header("Checking HuggingFace Cache")
    
    import os
    from pathlib import Path
    
    hf_home = os.environ.get('HF_HOME', Path.home() / '.cache' / 'huggingface')
    hf_path = Path(hf_home)
    
    print(f"HuggingFace cache directory: {hf_path}")
    
    if hf_path.exists():
        print_success("HuggingFace cache directory exists")
        
        # Try to estimate cache size
        try:
            total_size = sum(f.stat().st_size for f in hf_path.rglob('*') if f.is_file())
            print(f"Cache size: {total_size / 1024**3:.2f} GB")
        except Exception as e:
            print_warning(f"Could not calculate cache size: {e}")
    else:
        print_warning("HuggingFace cache directory doesn't exist yet (will be created)")
    
    return True

def main():
    """Run all checks"""
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*15 + "DIVERSITY DISTILLATION VALIDATION" + " "*20 + "║")
    print("║" + " "*18 + "Azure ML A100 Environment Check" + " "*18 + "║")
    print("╚" + "="*68 + "╝")
    
    checks = [
        ("Python Version", check_python_version),
        ("CUDA & GPU", check_cuda),
        ("Required Packages", check_packages),
        ("Project Structure", check_project_structure),
        ("Disk Space", check_disk_space),
        ("HuggingFace Cache", check_huggingface_cache),
        ("Model Loading", check_model_loading),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print_error(f"Check '{name}' failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    print_header("VALIDATION SUMMARY")
    
    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8s} - {name}")
        if not result:
            all_passed = False
    
    print("\n" + "="*70)
    
    if all_passed:
        print("\n🎉 All checks passed! The environment is ready to use.")
        print("\nNext steps:")
        print("  1. Review SETUP_GUIDE.md for usage examples")
        print("  2. Try generating images with evalscripts/diversity_distillation_sdxl.py")
        print("  3. Explore Jupyter notebooks in notebooks/")
        return 0
    else:
        print("\n❌ Some checks failed. Please review the errors above.")
        print("\nTroubleshooting:")
        print("  1. Re-run setup_azure_a100.sh")
        print("  2. Check SETUP_GUIDE.md for manual setup steps")
        print("  3. Verify all requirements are installed")
        return 1

if __name__ == '__main__':
    sys.exit(main())
