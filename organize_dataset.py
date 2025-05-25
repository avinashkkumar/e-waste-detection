import os
import shutil
import random
from PIL import Image
import argparse

def organize_dataset(source_dir, output_dir, test_split=0.2):
    """
    Organize e-waste dataset into train and test folders with proper structure
    
    Args:
        source_dir: Directory containing e-waste images
        output_dir: Output directory to create organized dataset
        test_split: Fraction of images to use for testing (default: 0.2)
    """
    # Create necessary directories
    train_ewaste_dir = os.path.join(output_dir, 'train', 'ewaste')
    train_non_ewaste_dir = os.path.join(output_dir, 'train', 'non_ewaste')
    test_ewaste_dir = os.path.join(output_dir, 'test', 'ewaste')
    test_non_ewaste_dir = os.path.join(output_dir, 'test', 'non_ewaste')
    
    for directory in [train_ewaste_dir, train_non_ewaste_dir, test_ewaste_dir, test_non_ewaste_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Process source directory
    ewaste_files = []
    non_ewaste_files = []
    
    # Find all image files in source directory
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                file_path = os.path.join(root, file)
                # Classify files based on filename or folder structure
                if 'e-waste' in file.lower() or 'ewaste' in file.lower() or 'e-waste' in root.lower():
                    ewaste_files.append(file_path)
                else:
                    # If you have non-ewaste images, add them here
                    # For now, I'm assuming all images in dataset are e-waste
                    non_ewaste_files.append(file_path)
    
    # If we didn't find any explicit non-ewaste files, print a warning
    if not non_ewaste_files:
        print("Warning: No non-ewaste images found. For a proper classification model, you need both classes.")
        print("All images will be treated as e-waste.")
    
    # Split the dataset
    random.shuffle(ewaste_files)
    random.shuffle(non_ewaste_files)
    
    ewaste_split_idx = int(len(ewaste_files) * (1 - test_split))
    train_ewaste = ewaste_files[:ewaste_split_idx]
    test_ewaste = ewaste_files[ewaste_split_idx:]
    
    non_ewaste_split_idx = int(len(non_ewaste_files) * (1 - test_split))
    train_non_ewaste = non_ewaste_files[:non_ewaste_split_idx]
    test_non_ewaste = non_ewaste_files[non_ewaste_split_idx:]
    
    # Copy files to respective directories
    def copy_files(file_list, target_dir, class_name):
        print(f"Copying {len(file_list)} {class_name} files to {target_dir}...")
        for i, file_path in enumerate(file_list):
            # Create a new filename to avoid conflicts
            ext = os.path.splitext(file_path)[1]
            new_name = f"{class_name}_{i+1:04d}{ext}"
            target_path = os.path.join(target_dir, new_name)
            
            try:
                # Check if it's a valid image before copying
                with Image.open(file_path) as img:
                    pass
                shutil.copy2(file_path, target_path)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    copy_files(train_ewaste, train_ewaste_dir, "ewaste")
    copy_files(test_ewaste, test_ewaste_dir, "ewaste")
    copy_files(train_non_ewaste, train_non_ewaste_dir, "non_ewaste")
    copy_files(test_non_ewaste, test_non_ewaste_dir, "non_ewaste")
    
    print("\nDataset organization complete!")
    print(f"Training set: {len(train_ewaste)} e-waste images, {len(train_non_ewaste)} non-e-waste images")
    print(f"Testing set: {len(test_ewaste)} e-waste images, {len(test_non_ewaste)} non-e-waste images")
    
    # Create dummy non-e-waste images if none found
    if len(non_ewaste_files) == 0:
        print("\nCreating dummy non-e-waste images for demonstration purposes...")
        create_dummy_non_ewaste(train_non_ewaste_dir, 100)  
        create_dummy_non_ewaste(test_non_ewaste_dir, 20)
        print("Dummy images created. Replace these with real non-e-waste images for a proper model!")

def create_dummy_non_ewaste(target_dir, num_images):
    """
    Create dummy non-e-waste images (simple colored rectangles)
    This is just for demonstration when no non-e-waste images are available
    """
    for i in range(num_images):
        img_size = (224, 224)
        color = (
            random.randint(100, 255),
            random.randint(100, 255),
            random.randint(100, 255)
        )
        
        # Create a colored image
        img = Image.new('RGB', img_size, color)
        
        # Add some shapes to make it look different
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        
        # Draw random shapes
        for _ in range(3):
            shape_color = (
                random.randint(0, 100),
                random.randint(0, 100),
                random.randint(0, 100)
            )
            x1 = random.randint(0, img_size[0] - 50)
            y1 = random.randint(0, img_size[1] - 50)
            x2 = x1 + random.randint(40, 100)
            y2 = y1 + random.randint(40, 100)
            shape_type = random.choice(['rectangle', 'ellipse'])
            
            if shape_type == 'rectangle':
                draw.rectangle([x1, y1, x2, y2], fill=shape_color)
            else:
                draw.ellipse([x1, y1, x2, y2], fill=shape_color)
        
        # Save the image
        img_path = os.path.join(target_dir, f"non_ewaste_{i+1:04d}.jpg")
        img.save(img_path, quality=95)

def main():
    parser = argparse.ArgumentParser(description='Organize e-waste dataset')
    parser.add_argument('--source', type=str, default='dataset', 
                        help='Source directory containing e-waste images')
    parser.add_argument('--output', type=str, default='organized_dataset',
                        help='Output directory for organized dataset')
    parser.add_argument('--test_split', type=float, default=0.2,
                        help='Fraction of data to use for testing (default: 0.2)')
    
    args = parser.parse_args()
    
    print(f"Organizing dataset from {args.source} into {args.output}")
    organize_dataset(args.source, args.output, args.test_split)

if __name__ == "__main__":
    main() 