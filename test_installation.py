#!/usr/bin/env python3
"""
Quick test script to verify the installation and imports
Run this after installing requirements.txt to ensure everything is set up correctly
"""

import sys

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...")
    print("-" * 60)
    
    tests = [
        ("TensorFlow", "import tensorflow as tf; print(f'  Version: {tf.__version__}')"),
        ("Keras", "from tensorflow import keras; print('  Keras included in TensorFlow')"),
        ("NumPy", "import numpy as np; print(f'  Version: {np.__version__}')"),
        ("PIL/Pillow", "from PIL import Image; print(f'  Version: {Image.__version__}')"),
        ("Matplotlib", "import matplotlib; print(f'  Version: {matplotlib.__version__}')"),
        ("tqdm", "import tqdm; print(f'  Version: {tqdm.__version__}')"),
        ("kagglehub", "import kagglehub; print('  kagglehub imported successfully')"),
    ]
    
    failed = []
    
    for name, code in tests:
        try:
            print(f"\n✓ {name}")
            exec(code)
        except Exception as e:
            print(f"\n✗ {name}")
            print(f"  Error: {e}")
            failed.append(name)
    
    print("\n" + "=" * 60)
    
    if not failed:
        print("✓ All packages imported successfully!")
        print("\nYou can now run:")
        print("  python image_captioning.py")
        return True
    else:
        print(f"✗ Failed to import: {', '.join(failed)}")
        print("\nPlease install missing packages:")
        print("  pip install -r requirements.txt")
        return False


def test_gpu():
    """Test if GPU is available for TensorFlow"""
    print("\n" + "=" * 60)
    print("Checking GPU availability...")
    print("-" * 60)
    
    try:
        import tensorflow as tf
        
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"✓ Found {len(gpus)} GPU(s):")
            for gpu in gpus:
                print(f"  - {gpu.name}")
            print("\nTraining will use GPU acceleration (much faster!)")
        else:
            print("✗ No GPU found")
            print("\nTraining will use CPU (slower but still works)")
            print("Consider using Google Colab or a GPU instance for faster training")
    
    except Exception as e:
        print(f"Error checking GPU: {e}")


def test_model_loading():
    """Test if the main module can be imported"""
    print("\n" + "=" * 60)
    print("Testing image_captioning module...")
    print("-" * 60)
    
    try:
        from image_captioning import ImageCaptioningModel
        print("✓ ImageCaptioningModel imported successfully")
        
        model = ImageCaptioningModel()
        print(f"✓ Model initialized with:")
        print(f"  - max_length: {model.max_length}")
        print(f"  - embedding_dim: {model.embedding_dim}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Image Captioning System - Installation Test")
    print("=" * 60)
    
    # Test imports
    imports_ok = test_imports()
    
    if not imports_ok:
        sys.exit(1)
    
    # Test GPU
    test_gpu()
    
    # Test model loading
    model_ok = test_model_loading()
    
    if not model_ok:
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run: python image_captioning.py")
    print("2. Follow the interactive prompts")
    print("3. See QUICKSTART.md for detailed instructions")
    print("\nHappy captioning! 🖼️ ➡️ 📝")


if __name__ == "__main__":
    main()
