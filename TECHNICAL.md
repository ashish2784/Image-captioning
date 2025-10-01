# Technical Documentation

## Architecture Overview

This image captioning system uses an **Encoder-Decoder** architecture combining Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks.

### High-Level Architecture

```
Image → CNN Encoder → Feature Vector → LSTM Decoder → Caption
                    (2048-dim)         (word by word)
```

## Components

### 1. CNN Encoder (Feature Extraction)

**Model**: ResNet50 pre-trained on ImageNet

- **Input**: RGB image (224×224×3)
- **Output**: 2048-dimensional feature vector
- **Architecture**: 
  - ResNet50 without top classification layers
  - Global average pooling applied
  - Weights frozen (transfer learning)

```python
resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')
```

**Why ResNet50?**
- Pre-trained on ImageNet (1.2M images, 1000 classes)
- Excellent feature extraction capabilities
- Residual connections help with gradient flow
- Reasonable model size (~98MB)

### 2. LSTM Decoder (Caption Generation)

**Model**: Single-layer LSTM with embedding

- **Input**: Image features + partial caption sequence
- **Output**: Next word probability distribution
- **Architecture**:
  - Embedding layer (vocab_size → 256)
  - LSTM layer (256 units)
  - Dense layers (256 → vocab_size)
  - Softmax activation

```python
# Simplified architecture
image_features → Dense(256)
caption_seq → Embedding(256) → LSTM(256)
merged → Dense(256) → Dense(vocab_size, softmax)
```

### 3. Training Process

**Dataset**: Mini-COCO 2014
- Subset of COCO 2014 dataset
- Multiple captions per image (typically 5)
- JSON format with annotations

**Data Preprocessing**:

1. **Image Processing**:
   - Resize to 224×224
   - Convert to RGB
   - Apply ResNet preprocessing (normalize)
   - Extract features once (offline)

2. **Caption Processing**:
   - Lowercase conversion
   - Add start token: `startseq`
   - Add end token: `endseq`
   - Tokenization using Keras Tokenizer
   - Padding to max_length

**Training Strategy**:

1. **Teacher Forcing**: During training, use ground truth previous words
2. **Data Generation**: Create (image, partial_caption) → next_word pairs
3. **Loss**: Categorical cross-entropy
4. **Optimizer**: Adam
5. **Batch Processing**: Efficient batch generation

Example training sequence for caption "a dog plays with a ball":
```
Input: [image_features, "startseq"] → Output: "a"
Input: [image_features, "startseq a"] → Output: "dog"
Input: [image_features, "startseq a dog"] → Output: "plays"
...
```

### 4. Inference Process

**Greedy Search Decoding**:

1. Start with `startseq` token
2. Generate next word using current sequence
3. Append generated word to sequence
4. Repeat until `endseq` or max_length reached

```python
current_sequence = "startseq"
for i in range(max_length):
    next_word = predict(image_features, current_sequence)
    if next_word == "endseq":
        break
    current_sequence += " " + next_word
```

**Alternative**: Could implement beam search for better results (not currently implemented)

## Model Parameters

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Image Size | 224×224 | Input to ResNet50 |
| CNN Features | 2048 | ResNet50 output dimension |
| Embedding Dim | 256 | Word embedding size |
| LSTM Units | 256 | LSTM hidden state size |
| Max Caption Length | 34 | Maximum words in caption |
| Dropout Rate | 0.5 | Regularization |
| Batch Size | 32 | Training batch size |
| Epochs | 20 | Default training epochs |
| Optimizer | Adam | Adaptive learning rate |
| Loss Function | Categorical Cross-Entropy | Multi-class classification |

### Model Size

- **ResNet50**: ~98 MB (weights frozen)
- **LSTM + Dense layers**: ~50-100 MB (depends on vocab size)
- **Total trained model**: ~150-200 MB
- **Tokenizer**: ~1-5 MB
- **Total disk space**: ~200 MB

## Data Flow

### Training Phase

```
1. Load Dataset
   ├── Read JSON annotations
   ├── Map image_id to captions
   └── Create image_name → captions dictionary

2. Preprocess Captions
   ├── Lowercase
   ├── Add start/end tokens
   └── Build vocabulary (tokenizer)

3. Extract Features
   ├── Load each image
   ├── Preprocess for ResNet50
   ├── Extract 2048-dim features
   └── Cache features (avoid recomputation)

4. Create Training Batches
   ├── For each image-caption pair
   ├── Create input sequences (partial captions)
   ├── Create output (next word)
   └── Yield batches

5. Train Model
   ├── Forward pass
   ├── Calculate loss
   ├── Backpropagation
   └── Update weights

6. Save Model
   ├── Save Keras model (.h5)
   ├── Save tokenizer (.pkl)
   └── Save config (.pkl)
```

### Inference Phase

```
1. Load Model
   ├── Load model weights
   ├── Load tokenizer
   ├── Load config
   └── Initialize ResNet50

2. Process Input Image
   ├── Load image
   ├── Resize to 224×224
   ├── Preprocess
   └── Extract features

3. Generate Caption
   ├── Initialize with "startseq"
   ├── Loop until "endseq" or max_length:
   │   ├── Encode current sequence
   │   ├── Predict next word
   │   └── Append to sequence
   └── Clean up tokens

4. Display Results
   ├── Show image
   ├── Show caption
   └── Save annotated image
```

## Performance Considerations

### Memory Usage

- **Feature Extraction**: Stores all image features in memory
  - ~2048 × 4 bytes × num_images
  - For 5000 images: ~40 MB
- **Training**: Batch processing reduces memory
- **Inference**: Minimal memory (single image)

### Speed Optimization

1. **Offline Feature Extraction**: Extract CNN features once before training
2. **Batch Processing**: Process multiple samples simultaneously
3. **Frozen ResNet**: Don't update ResNet weights (faster)
4. **Generator Pattern**: Load data on-demand during training

### GPU Acceleration

- ResNet50 inference: ~10-50ms per image (GPU) vs ~500ms (CPU)
- Training: 10-20x faster with GPU
- Inference: 5-10x faster with GPU

## Evaluation Metrics

While not implemented in the current version, standard metrics for image captioning include:

- **BLEU** (Bilingual Evaluation Understudy): Measures n-gram overlap
- **METEOR**: Considers synonyms and stemming
- **CIDEr**: Consensus-based metric
- **ROUGE-L**: Longest common subsequence

## Limitations

1. **Vocabulary Constraints**: Limited to words in training data
2. **Greedy Decoding**: May not produce optimal captions
3. **Single Caption**: Generates one caption (no diversity)
4. **Domain Specific**: Best for COCO-like images
5. **No Attention**: Doesn't use attention mechanism

## Future Improvements

1. **Attention Mechanism**: Focus on relevant image regions
2. **Beam Search**: Better caption generation
3. **Fine-tuning ResNet**: Adapt CNN to captioning task
4. **Transformer Architecture**: Replace LSTM with transformers
5. **Multi-caption Generation**: Generate diverse captions
6. **Evaluation Metrics**: Add BLEU, METEOR, CIDEr scores
7. **Data Augmentation**: Improve generalization

## Code Structure

```
image_captioning.py
├── ImageCaptioningModel (main class)
│   ├── build_feature_extractor()
│   ├── extract_features()
│   ├── load_captions()
│   ├── preprocess_captions()
│   ├── build_tokenizer()
│   ├── build_model()
│   ├── data_generator()
│   ├── train()
│   ├── generate_caption()
│   ├── save_model()
│   └── load_model()
├── download_dataset()
├── extract_all_features()
├── train_model()
├── predict_caption()
└── main()
```

## Dependencies

- **TensorFlow/Keras**: Deep learning framework
- **NumPy**: Numerical computations
- **PIL**: Image loading and processing
- **Matplotlib**: Visualization
- **tqdm**: Progress bars
- **kagglehub**: Dataset downloading

## References

1. Vinyals et al., "Show and Tell: A Neural Image Caption Generator" (2015)
2. He et al., "Deep Residual Learning for Image Recognition" (2015)
3. COCO Dataset: https://cocodataset.org/
4. ResNet: https://arxiv.org/abs/1512.03385
