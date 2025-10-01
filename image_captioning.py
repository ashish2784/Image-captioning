import os
import json
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import kagglehub

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add


class ImageCaptioningModel:
    """CNN+LSTM based image captioning model"""
    
    def __init__(self, max_length=34, vocab_size=None, embedding_dim=256):
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.tokenizer = None
        self.model = None
        self.feature_extractor = None
        
    def build_feature_extractor(self):
        """Build CNN feature extractor using pre-trained ResNet50"""
        print("Building feature extractor (ResNet50)...")
        resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        resnet.trainable = False
        self.feature_extractor = resnet
        return resnet
    
    def extract_features(self, image_path):
        """Extract features from a single image"""
        img = load_img(image_path, target_size=(224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        features = self.feature_extractor.predict(img, verbose=0)
        return features
    
    def load_captions(self, annotations_file):
        """Load captions from COCO annotations file"""
        print("Loading captions...")
        with open(annotations_file, 'r') as f:
            data = json.load(f)
        
        captions_dict = {}
        for annotation in data['annotations']:
            image_id = annotation['image_id']
            caption = annotation['caption']
            
            if image_id not in captions_dict:
                captions_dict[image_id] = []
            captions_dict[image_id].append(caption)
        
        # Create mapping from filename to captions
        image_captions = {}
        for image in data['images']:
            image_id = image['id']
            file_name = image['file_name']
            if image_id in captions_dict:
                image_captions[file_name] = captions_dict[image_id]
        
        return image_captions
    
    def preprocess_captions(self, captions_dict):
        """Preprocess captions: lowercase, add start/end tokens"""
        print("Preprocessing captions...")
        processed_captions = {}
        
        for image_name, captions in captions_dict.items():
            processed_list = []
            for caption in captions:
                caption = caption.lower()
                caption = 'startseq ' + caption + ' endseq'
                processed_list.append(caption)
            processed_captions[image_name] = processed_list
        
        return processed_captions
    
    def build_tokenizer(self, captions_dict):
        """Build tokenizer from captions"""
        print("Building tokenizer...")
        all_captions = []
        for captions in captions_dict.values():
            all_captions.extend(captions)
        
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(all_captions)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        
        print(f"Vocabulary size: {self.vocab_size}")
        return self.tokenizer
    
    def calculate_max_length(self, captions_dict):
        """Calculate maximum caption length"""
        all_captions = []
        for captions in captions_dict.values():
            all_captions.extend(captions)
        
        max_len = max(len(caption.split()) for caption in all_captions)
        print(f"Maximum caption length: {max_len}")
        self.max_length = max_len
        return max_len
    
    def build_model(self):
        """Build CNN+LSTM model for image captioning"""
        print("Building CNN+LSTM model...")
        
        # Image feature input (from CNN)
        image_input = Input(shape=(2048,))
        image_dense = Dropout(0.5)(image_input)
        image_dense = Dense(256, activation='relu')(image_dense)
        
        # Caption sequence input (for LSTM)
        caption_input = Input(shape=(self.max_length,))
        caption_embedding = Embedding(self.vocab_size, self.embedding_dim, mask_zero=True)(caption_input)
        caption_dropout = Dropout(0.5)(caption_embedding)
        caption_lstm = LSTM(256)(caption_dropout)
        
        # Combine image and caption features
        decoder = add([image_dense, caption_lstm])
        decoder = Dense(256, activation='relu')(decoder)
        
        # Output layer
        output = Dense(self.vocab_size, activation='softmax')(decoder)
        
        # Create model
        model = Model(inputs=[image_input, caption_input], outputs=output)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        self.model = model
        print(model.summary())
        return model
    
    def data_generator(self, image_features, captions_dict, batch_size=32):
        """Generate batches of training data"""
        image_names = list(captions_dict.keys())
        
        while True:
            for i in range(0, len(image_names), batch_size):
                batch_images = image_names[i:i+batch_size]
                
                X1, X2, y = [], [], []
                
                for image_name in batch_images:
                    if image_name not in image_features:
                        continue
                    
                    feature = image_features[image_name][0]
                    captions = captions_dict[image_name]
                    
                    for caption in captions:
                        seq = self.tokenizer.texts_to_sequences([caption])[0]
                        
                        for j in range(1, len(seq)):
                            in_seq = seq[:j]
                            out_seq = seq[j]
                            
                            in_seq = pad_sequences([in_seq], maxlen=self.max_length)[0]
                            out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]
                            
                            X1.append(feature)
                            X2.append(in_seq)
                            y.append(out_seq)
                
                if len(X1) > 0:
                    yield [np.array(X1), np.array(X2)], np.array(y)
    
    def train(self, image_features, captions_dict, epochs=20, batch_size=32):
        """Train the model"""
        print("Starting training...")
        
        steps = len(captions_dict) // batch_size
        if steps == 0:
            steps = 1
        
        generator = self.data_generator(image_features, captions_dict, batch_size)
        
        history = self.model.fit(
            generator,
            epochs=epochs,
            steps_per_epoch=steps,
            verbose=1
        )
        
        return history
    
    def generate_caption(self, image_path):
        """Generate caption for a given image"""
        # Extract features
        feature = self.extract_features(image_path)
        
        # Start with 'startseq'
        in_text = 'startseq'
        
        for i in range(self.max_length):
            # Encode input sequence
            sequence = self.tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=self.max_length)
            
            # Predict next word
            yhat = self.model.predict([feature, sequence], verbose=0)
            yhat = np.argmax(yhat)
            
            # Map integer to word
            word = None
            for w, idx in self.tokenizer.word_index.items():
                if idx == yhat:
                    word = w
                    break
            
            if word is None:
                break
            
            # Append word to input sequence
            in_text += ' ' + word
            
            # Stop if we reach end token
            if word == 'endseq':
                break
        
        # Remove start and end tokens
        caption = in_text.replace('startseq', '').replace('endseq', '').strip()
        return caption
    
    def save_model(self, model_path='image_captioning_model.h5', 
                   tokenizer_path='tokenizer.pkl',
                   config_path='model_config.pkl'):
        """Save model and tokenizer"""
        print(f"Saving model to {model_path}...")
        self.model.save(model_path)
        
        print(f"Saving tokenizer to {tokenizer_path}...")
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        print(f"Saving config to {config_path}...")
        config = {
            'max_length': self.max_length,
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim
        }
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)
    
    def load_model(self, model_path='image_captioning_model.h5',
                   tokenizer_path='tokenizer.pkl',
                   config_path='model_config.pkl'):
        """Load trained model and tokenizer"""
        print(f"Loading config from {config_path}...")
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        
        self.max_length = config['max_length']
        self.vocab_size = config['vocab_size']
        self.embedding_dim = config['embedding_dim']
        
        print(f"Loading tokenizer from {tokenizer_path}...")
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        print(f"Loading model from {model_path}...")
        self.model = tf.keras.models.load_model(model_path)
        
        # Build feature extractor
        self.build_feature_extractor()


def download_dataset():
    """Download dataset using kagglehub"""
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("nagasai524/mini-coco2014-dataset-for-image-captioning")
    print("Path to dataset files:", path)
    return path


def extract_all_features(dataset_path, captions_dict, feature_extractor):
    """Extract features for all images in the dataset"""
    print("Extracting features from images...")
    
    # Find images directory
    images_dir = None
    for root, dirs, files in os.walk(dataset_path):
        if 'train2014' in dirs:
            images_dir = os.path.join(root, 'train2014')
            break
        elif any(f.endswith('.jpg') or f.endswith('.png') for f in files):
            images_dir = root
            break
    
    if images_dir is None:
        print("Searching for images in dataset...")
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    images_dir = root
                    break
            if images_dir:
                break
    
    print(f"Images directory: {images_dir}")
    
    features = {}
    for image_name in tqdm(captions_dict.keys(), desc="Extracting features"):
        image_path = os.path.join(images_dir, image_name)
        
        if os.path.exists(image_path):
            try:
                img = load_img(image_path, target_size=(224, 224))
                img = img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img = preprocess_input(img)
                feature = feature_extractor.predict(img, verbose=0)
                features[image_name] = feature
            except Exception as e:
                print(f"Error processing {image_name}: {e}")
    
    print(f"Extracted features for {len(features)} images")
    return features


def train_model(dataset_path, epochs=20, batch_size=32):
    """Complete training pipeline"""
    # Find annotations file
    annotations_file = None
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if 'annotations' in file.lower() and file.endswith('.json'):
                annotations_file = os.path.join(root, file)
                break
        if annotations_file:
            break
    
    if annotations_file is None:
        print("Searching for JSON files...")
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.json'):
                    annotations_file = os.path.join(root, file)
                    print(f"Found JSON file: {annotations_file}")
                    break
            if annotations_file:
                break
    
    if annotations_file is None:
        raise FileNotFoundError("Could not find annotations file in dataset")
    
    print(f"Using annotations file: {annotations_file}")
    
    # Initialize model
    model = ImageCaptioningModel()
    
    # Build feature extractor
    feature_extractor = model.build_feature_extractor()
    
    # Load and preprocess captions
    captions_dict = model.load_captions(annotations_file)
    captions_dict = model.preprocess_captions(captions_dict)
    
    # Build tokenizer
    model.build_tokenizer(captions_dict)
    model.calculate_max_length(captions_dict)
    
    # Extract features from all images
    image_features = extract_all_features(dataset_path, captions_dict, feature_extractor)
    
    # Filter captions to only include images with extracted features
    captions_dict = {k: v for k, v in captions_dict.items() if k in image_features}
    
    print(f"Training on {len(captions_dict)} images")
    
    # Build model
    model.build_model()
    
    # Train
    model.train(image_features, captions_dict, epochs=epochs, batch_size=batch_size)
    
    # Save model
    model.save_model()
    
    return model


def predict_caption(image_path, model=None):
    """Generate caption for user-provided image"""
    if model is None:
        model = ImageCaptioningModel()
        model.load_model()
    
    caption = model.generate_caption(image_path)
    
    # Display image and caption
    img = Image.open(image_path)
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Caption: {caption}", fontsize=14, wrap=True)
    plt.tight_layout()
    plt.savefig('captioned_image.png', bbox_inches='tight', dpi=150)
    plt.show()
    
    print(f"\nGenerated Caption: {caption}")
    return caption


def main():
    """Main function with user interface"""
    print("=" * 60)
    print("Image Captioning System - CNN+LSTM")
    print("=" * 60)
    
    # Check if trained model exists
    if os.path.exists('image_captioning_model.h5'):
        print("\nPre-trained model found!")
        choice = input("Do you want to (1) Train new model or (2) Use existing model? [1/2]: ")
        
        if choice == '1':
            train_new = True
        else:
            train_new = False
    else:
        print("\nNo pre-trained model found. Will train a new model.")
        train_new = True
    
    if train_new:
        # Download dataset
        dataset_path = download_dataset()
        
        # Ask for training parameters
        print("\nTraining Configuration:")
        epochs_input = input("Enter number of epochs (default 20): ")
        epochs = int(epochs_input) if epochs_input else 20
        
        batch_size_input = input("Enter batch size (default 32): ")
        batch_size = int(batch_size_input) if batch_size_input else 32
        
        # Train model
        model = train_model(dataset_path, epochs=epochs, batch_size=batch_size)
        print("\nTraining complete!")
    else:
        # Load existing model
        model = ImageCaptioningModel()
        model.load_model()
        print("\nModel loaded successfully!")
    
    # Prediction loop
    print("\n" + "=" * 60)
    print("Image Caption Generation")
    print("=" * 60)
    
    while True:
        print("\nOptions:")
        print("1. Generate caption for an image")
        print("2. Exit")
        
        choice = input("\nEnter your choice [1/2]: ")
        
        if choice == '1':
            image_path = input("Enter the path to your image: ")
            
            if not os.path.exists(image_path):
                print(f"Error: File '{image_path}' not found!")
                continue
            
            try:
                caption = predict_caption(image_path, model)
                print(f"\nCaption saved to 'captioned_image.png'")
            except Exception as e:
                print(f"Error generating caption: {e}")
        
        elif choice == '2':
            print("\nThank you for using Image Captioning System!")
            break
        
        else:
            print("Invalid choice. Please enter 1 or 2.")


if __name__ == "__main__":
    main()
