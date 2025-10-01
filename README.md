# Image Captioning with CNN+LSTM

An end-to-end deep learning system for generating captions for images using Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks.

## Features

- **CNN Feature Extraction**: Uses pre-trained ResNet50 to extract image features
- **LSTM Caption Generation**: Generates natural language descriptions using LSTM
- **Pre-trained Dataset Support**: Automatically downloads and trains on Mini-COCO 2014 dataset
- **User-Friendly Interface**: Simple command-line interface for training and inference
- **Custom Image Captioning**: Generate captions for your own images

## Architecture

The system uses an encoder-decoder architecture:
1. **Encoder (CNN)**: ResNet50 pre-trained on ImageNet extracts 2048-dimensional feature vectors from images
2. **Decoder (LSTM)**: LSTM network generates captions word-by-word based on image features

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ashish2784/Image-captioning.git
cd Image-captioning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation (optional):
```bash
python test_installation.py
```

## Usage

### Training the Model

Run the main script:
```bash
python image_captioning.py
```

The script will:
1. Download the Mini-COCO 2014 dataset from Kaggle using kagglehub
2. Ask for training configuration (epochs, batch size)
3. Extract features from images using ResNet50
4. Build vocabulary from captions
5. Train the CNN+LSTM model
6. Save the trained model

### Generating Captions for Your Images

After training (or if you have a pre-trained model), you can generate captions:

1. Run the script:
```bash
python image_captioning.py
```

2. Choose option 2 to use the existing model
3. Choose option 1 to generate a caption
4. Enter the path to your image file

The system will:
- Display the image with the generated caption
- Save the result as `captioned_image.png`
- Print the caption to the console

## Dataset

The model uses the Mini-COCO 2014 dataset, which is automatically downloaded via kagglehub:
```python
import kagglehub
path = kagglehub.dataset_download("nagasai524/mini-coco2014-dataset-for-image-captioning")
```

## Model Files

After training, the following files are saved:
- `image_captioning_model.h5`: Trained model weights
- `tokenizer.pkl`: Vocabulary tokenizer for caption generation
- `model_config.pkl`: Model configuration (max length, vocab size, etc.)

## Example

```python
from image_captioning import ImageCaptioningModel, predict_caption

# Load pre-trained model
model = ImageCaptioningModel()
model.load_model()

# Generate caption for an image
caption = predict_caption('path/to/your/image.jpg', model)
print(f"Generated Caption: {caption}")
```

## Requirements

- Python 3.7+
- TensorFlow 2.10+
- See `requirements.txt` for full list

## Model Parameters

Default parameters:
- **Max Caption Length**: 34 words
- **Embedding Dimension**: 256
- **Epochs**: 20
- **Batch Size**: 32
- **Image Input Size**: 224x224
- **CNN Features**: 2048 dimensions (ResNet50)

## License

MIT License - see LICENSE file for details

## Acknowledgments

- COCO Dataset creators
- Mini-COCO 2014 dataset on Kaggle
- TensorFlow and Keras teams