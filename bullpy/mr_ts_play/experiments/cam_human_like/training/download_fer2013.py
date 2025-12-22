#!/usr/bin/env python3
"""
Download and set up FER2013 dataset for fine-tuning.

FER2013 is available on Kaggle and needs to be:
1. Downloaded (requires Kaggle API)
2. Converted from CSV (pixel strings) to images
3. Organized into train/test/val folders with emotion subfolders

Usage:
    # Option 1: Using Kaggle API (recommended)
    python download_fer2013.py --output_dir data/fer2013 --use_kaggle

    # Option 2: Manual download
    # 1. Download fer2013.zip from Kaggle manually
    # 2. Extract to data/fer2013_raw/
    # 3. Run: python download_fer2013.py --input_dir data/fer2013_raw --output_dir data/fer2013
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import zipfile
import shutil
from tqdm import tqdm

# Try to import kaggle API
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
    print("Warning: Kaggle API not installed. Install with: pip install kaggle")
    print("You can still use manual download option.")


def download_from_kaggle(output_dir: Path):
    """Download FER2013 from Kaggle using API."""
    if not KAGGLE_AVAILABLE:
        raise ImportError("Kaggle API not available. Install with: pip install kaggle")
    
    print("Authenticating with Kaggle API...")
    api = KaggleApi()
    api.authenticate()
    
    print("Downloading FER2013 dataset from Kaggle...")
    dataset = "msambare/fer2013"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download dataset
    api.dataset_download_files(dataset, path=str(output_dir), unzip=True)
    
    print(f"Downloaded to {output_dir}")
    return output_dir


def convert_csv_to_images(csv_path: Path, output_dir: Path):
    """
    Convert FER2013 CSV to organized image folders.
    
    FER2013 CSV format:
    - emotion: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
    - pixels: space-separated pixel values (48x48 = 2304 values)
    - Usage: Training, PublicTest, PrivateTest
    """
    print(f"Reading FER2013 CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Emotion mapping
    emotion_map = {
        0: 'angry',
        1: 'disgust',
        2: 'fear',
        3: 'happy',
        4: 'sad',
        5: 'surprise',
        6: 'neutral'
    }
    
    # Usage mapping (Training -> train, PublicTest -> test, PrivateTest -> val)
    usage_map = {
        'Training': 'train',
        'PublicTest': 'test',
        'PrivateTest': 'val'
    }
    
    # Create directory structure
    for split in ['train', 'test', 'val']:
        for emotion in emotion_map.values():
            (output_dir / split / emotion).mkdir(parents=True, exist_ok=True)
    
    print("Converting pixel strings to images...")
    
    # Process each row
    counts = {'train': 0, 'test': 0, 'val': 0}
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting images"):
        emotion = emotion_map[row['emotion']]
        usage = usage_map.get(row['Usage'], 'train')  # Default to train if unknown
        
        # Parse pixel string
        pixels = np.array([int(x) for x in row['pixels'].split()], dtype=np.uint8)
        pixels = pixels.reshape(48, 48)  # FER2013 images are 48x48
        
        # Convert to PIL Image (grayscale)
        img = Image.fromarray(pixels, mode='L')
        
        # Convert to RGB (CLIP expects RGB)
        img = img.convert('RGB')
        
        # Resize to 224x224 (CLIP input size)
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Save image
        img_path = output_dir / usage / emotion / f"{idx:06d}.jpg"
        img.save(img_path, 'JPEG', quality=95)
        
        counts[usage] += 1
    
    print("\nConversion complete!")
    print(f"Train images: {counts['train']}")
    print(f"Test images: {counts['test']}")
    print(f"Val images: {counts['val']}")
    print(f"\nDataset saved to: {output_dir}")


def setup_fer2013(
    output_dir: str = "data/fer2013",
    input_dir: str = None,
    use_kaggle: bool = False,
):
    """
    Set up FER2013 dataset.
    
    Args:
        output_dir: Where to save organized FER2013 dataset
        input_dir: If provided, use this directory (manual download)
        use_kaggle: If True, download from Kaggle using API
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find fer2013.csv
    csv_path = None
    
    if use_kaggle:
        # Download from Kaggle
        print("Downloading from Kaggle...")
        download_dir = output_dir.parent / "fer2013_raw"
        download_from_kaggle(download_dir)
        
        # Find CSV in downloaded files
        csv_path = download_dir / "fer2013.csv"
        if not csv_path.exists():
            # Try alternative locations
            for possible_path in download_dir.rglob("fer2013.csv"):
                csv_path = possible_path
                break
        
        if not csv_path or not csv_path.exists():
            raise FileNotFoundError(f"Could not find fer2013.csv in {download_dir}")
    
    elif input_dir:
        # Use provided input directory
        input_dir = Path(input_dir)
        csv_path = input_dir / "fer2013.csv"
        
        if not csv_path.exists():
            # Search recursively
            for possible_path in input_dir.rglob("fer2013.csv"):
                csv_path = possible_path
                break
        
        if not csv_path or not csv_path.exists():
            raise FileNotFoundError(f"Could not find fer2013.csv in {input_dir}")
    
    else:
        # Check if already exists in output_dir
        csv_path = output_dir / "fer2013.csv"
        if not csv_path.exists():
            raise ValueError(
                "No input provided. Either:\n"
                "  1. Use --use_kaggle to download from Kaggle\n"
                "  2. Use --input_dir to specify directory with fer2013.csv\n"
                "  3. Place fer2013.csv in output_dir manually"
            )
    
    # Convert CSV to images
    convert_csv_to_images(csv_path, output_dir)
    
    print("\n" + "="*60)
    print("FER2013 dataset setup complete!")
    print("="*60)
    print(f"Dataset location: {output_dir}")
    print(f"Structure:")
    print(f"  {output_dir}/")
    print(f"    train/ (angry, disgust, fear, happy, neutral, sad, surprise)")
    print(f"    test/  (angry, disgust, fear, happy, neutral, sad, surprise)")
    print(f"    val/   (angry, disgust, fear, happy, neutral, sad, surprise)")
    print("\nYou can now use this dataset for fine-tuning:")
    print(f'  python finetune_clip_emotions.py --fer2013_dir "{output_dir}"')


def main():
    parser = argparse.ArgumentParser(
        description="Download and set up FER2013 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download from Kaggle (requires Kaggle API setup)
  python download_fer2013.py --output_dir data/fer2013 --use_kaggle

  # Use manually downloaded dataset
  python download_fer2013.py --input_dir /path/to/fer2013_raw --output_dir data/fer2013

  # If fer2013.csv is already in output_dir
  python download_fer2013.py --output_dir data/fer2013
        """
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/fer2013',
        help='Output directory for organized FER2013 dataset'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default=None,
        help='Input directory containing fer2013.csv (for manual download)'
    )
    parser.add_argument(
        '--use_kaggle',
        action='store_true',
        help='Download from Kaggle using API (requires kaggle package and authentication)'
    )
    
    args = parser.parse_args()
    
    setup_fer2013(
        output_dir=args.output_dir,
        input_dir=args.input_dir,
        use_kaggle=args.use_kaggle,
    )


if __name__ == "__main__":
    main()





