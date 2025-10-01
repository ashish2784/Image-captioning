# Quick Start Guide

## Installation Steps

1. **Clone the repository:**
```bash
git clone https://github.com/ashish2784/Image-captioning.git
cd Image-captioning
```

2. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

Note: This will install TensorFlow, NumPy, PIL, matplotlib, tqdm, and kagglehub.

## Running the Image Captioning System

### Option 1: Interactive Mode (Recommended for Beginners)

Simply run:
```bash
python image_captioning.py
```

The script will guide you through:
1. Downloading the dataset (if needed)
2. Training the model or loading an existing one
3. Generating captions for your images

### Option 2: Programmatic Usage

Use the example script:
```bash
python example_usage.py
```

Or import in your own code:
```python
from image_captioning import ImageCaptioningModel, predict_caption

# Load pre-trained model
model = ImageCaptioningModel()
model.load_model()

# Generate caption
caption = predict_caption('path/to/image.jpg', model)
```

## First Time Setup

When you run the script for the first time:

1. **Dataset Download**: The script will automatically download the Mini-COCO 2014 dataset from Kaggle
   - Dataset size: ~500MB - 1GB
   - This only happens once

2. **Training Configuration**: You'll be asked for:
   - Number of epochs (recommended: 20-30 for good results, 5 for quick testing)
   - Batch size (recommended: 32 for most systems, 16 for low memory)

3. **Training Time**: 
   - With 5 epochs: ~30-60 minutes
   - With 20 epochs: ~2-4 hours
   - Depends on your hardware (GPU highly recommended)

4. **Model Saving**: After training, model files are saved:
   - `image_captioning_model.h5` (main model)
   - `tokenizer.pkl` (vocabulary)
   - `model_config.pkl` (configuration)

## Generating Captions

After training (or loading a pre-trained model):

1. Select option 1: "Generate caption for an image"
2. Enter the full path to your image file
3. The script will:
   - Generate a caption
   - Display the image with caption
   - Save as `captioned_image.png`

## Troubleshooting

### Memory Issues
If you get memory errors during training:
- Reduce batch size (try 16 or 8)
- Close other applications
- Use a machine with more RAM

### Kaggle Dataset Download
If dataset download fails:
- Ensure you have kagglehub installed: `pip install kagglehub`
- Check your internet connection
- The first download may require Kaggle authentication

### GPU Acceleration
For faster training, ensure TensorFlow can use your GPU:
```bash
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

## Example Workflow

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run the main script
python image_captioning.py

# Step 3: Choose to train a new model (first time)
# Enter: 1

# Step 4: Configure training
# Epochs: 20
# Batch size: 32

# Step 5: Wait for training to complete (~2-4 hours)

# Step 6: Generate captions
# Choose option 1
# Enter path to your image: /path/to/your/photo.jpg

# Step 7: View results
# The script displays the image and caption
# Result is saved as captioned_image.png
```

## Tips for Best Results

1. **Training Data**: The model is trained on COCO dataset, so it works best with:
   - Natural everyday scenes
   - Common objects and activities
   - Indoor and outdoor photos

2. **Image Quality**: 
   - Use clear, well-lit images
   - Avoid heavily edited or artistic images
   - Standard formats: JPG, JPEG, PNG

3. **Model Improvement**:
   - Train for more epochs (30-50) for better results
   - Use larger batch size if you have enough memory
   - Fine-tune on your specific domain if needed

## Next Steps

- Try the example usage script for batch processing
- Experiment with different images
- Adjust training parameters for your use case
- Consider transfer learning for specialized domains
