"""
Example script demonstrating how to use the image captioning system programmatically
"""

from image_captioning import ImageCaptioningModel, download_dataset, train_model, predict_caption
import os


def example_training():
    """Example: Train a new model"""
    print("Example 1: Training a new model")
    print("-" * 60)
    
    # Download dataset
    dataset_path = download_dataset()
    
    # Train model with custom parameters
    model = train_model(
        dataset_path=dataset_path,
        epochs=5,  # Use fewer epochs for quick testing
        batch_size=32
    )
    
    print("Training completed!")
    return model


def example_loading_and_prediction():
    """Example: Load existing model and generate captions"""
    print("\nExample 2: Loading model and generating captions")
    print("-" * 60)
    
    # Check if model exists
    if not os.path.exists('image_captioning_model.h5'):
        print("No trained model found. Please train a model first.")
        return
    
    # Load model
    model = ImageCaptioningModel()
    model.load_model()
    
    print("Model loaded successfully!")
    
    # Generate caption for an image
    # Replace 'test_image.jpg' with your image path
    image_path = input("Enter path to your image: ")
    
    if os.path.exists(image_path):
        caption = predict_caption(image_path, model)
        print(f"\nGenerated Caption: {caption}")
    else:
        print(f"Image not found: {image_path}")


def example_batch_prediction():
    """Example: Generate captions for multiple images"""
    print("\nExample 3: Batch caption generation")
    print("-" * 60)
    
    if not os.path.exists('image_captioning_model.h5'):
        print("No trained model found. Please train a model first.")
        return
    
    # Load model
    model = ImageCaptioningModel()
    model.load_model()
    
    # List of image paths
    image_directory = input("Enter directory containing images: ")
    
    if not os.path.exists(image_directory):
        print(f"Directory not found: {image_directory}")
        return
    
    # Get all image files
    image_files = [f for f in os.listdir(image_directory) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"\nFound {len(image_files)} images")
    
    # Generate captions
    results = {}
    for image_file in image_files:
        image_path = os.path.join(image_directory, image_file)
        try:
            caption = model.generate_caption(image_path)
            results[image_file] = caption
            print(f"{image_file}: {caption}")
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Image Captioning - Usage Examples")
    print("=" * 60)
    
    print("\nAvailable examples:")
    print("1. Train a new model")
    print("2. Load existing model and generate caption")
    print("3. Batch caption generation for multiple images")
    
    choice = input("\nSelect an example (1-3): ")
    
    if choice == '1':
        example_training()
    elif choice == '2':
        example_loading_and_prediction()
    elif choice == '3':
        example_batch_prediction()
    else:
        print("Invalid choice")
