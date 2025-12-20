#!/usr/bin/env python3
"""
Test script to verify EU-Emotion dataset loader works with your data.

This script will:
1. Try to load the EU-Emotion dataset
2. Show dataset statistics
3. Display sample images/emotions
4. Verify compatibility with fine-tuning pipeline

Usage:
    python experiments/cam_human_like/training/test_eu_emotion.py --eu_emotion_dir /path/to/eu_emotion
"""

import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from eu_emotion_dataset import EUEmotionDataset
from collections import Counter


def main():
    parser = argparse.ArgumentParser(description="Test EU-Emotion dataset loader")
    parser.add_argument('--eu_emotion_dir', type=str, required=True, help='Path to EU-Emotion dataset')
    parser.add_argument('--modality', type=str, default='face', choices=['face', 'voice', 'body', 'all'], help='Modality to test')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to display')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("EU-Emotion Dataset Loader Test")
    print("=" * 60)
    print()
    
    # Test train split
    print(f"Testing train split (modality: {args.modality})...")
    try:
        train_dataset = EUEmotionDataset(
            args.eu_emotion_dir,
            split='train',
            modality=args.modality,
            num_frames=8,
        )
        print(f"✅ Successfully loaded train split!")
        print(f"   Samples: {len(train_dataset)}")
        print(f"   Emotions: {len(train_dataset.emotions)}")
        print(f"   Emotion classes: {', '.join(train_dataset.emotions)}")
        print()
        
        # Show emotion distribution
        emotion_counter = Counter([emotion for _, emotion in train_dataset.samples])
        print("Emotion distribution (top 10):")
        for emotion, count in emotion_counter.most_common(10):
            print(f"   {emotion}: {count}")
        print()
        
        # Test sample loading
        print(f"Testing sample loading (showing {args.num_samples} samples)...")
        for i in range(min(args.num_samples, len(train_dataset))):
            try:
                image, emotion = train_dataset[i]
                print(f"   Sample {i+1}: emotion='{emotion}', image_size={image.size}, image_mode={image.mode}")
            except Exception as e:
                print(f"   Sample {i+1}: ❌ Error - {e}")
        print()
        
    except Exception as e:
        print(f"❌ Error loading train split: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test test/val split
    for split_name in ['test', 'val']:
        print(f"Testing {split_name} split...")
        try:
            split_dataset = EUEmotionDataset(
                args.eu_emotion_dir,
                split=split_name,
                modality=args.modality,
                num_frames=8,
            )
            print(f"   ✅ Successfully loaded {split_name} split!")
            print(f"   Samples: {len(split_dataset)}")
            print(f"   Emotions: {len(split_dataset.emotions)}")
        except Exception as e:
            print(f"   ⚠️  {split_name} split not found or error: {e}")
        print()
    
    # Test compatibility with fine-tuning pipeline
    print("Testing compatibility with fine-tuning pipeline...")
    try:
        from finetune_clip_emotions import collate_pil_images
        
        # Create a small batch
        batch = [train_dataset[i] for i in range(min(2, len(train_dataset)))]
        images, emotions = collate_pil_images(batch)
        
        print(f"   ✅ Batch collation works!")
        print(f"   Batch size: {len(images)}")
        print(f"   Image types: {[type(img).__name__ for img in images]}")
        print(f"   Emotions: {emotions}")
    except Exception as e:
        print(f"   ⚠️  Compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 60)
    print("Test Complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. If test passed, you can run fine-tuning:")
    print(f"   python experiments/cam_human_like/training/finetune_clip_emotions.py \\")
    print(f"       --eu_emotion_dir {args.eu_emotion_dir} \\")
    print(f"       --output_dir models/clip_eu_emotion_test \\")
    print(f"       --num_epochs 1 \\")
    print(f"       --batch_size 4")
    print()
    print("2. For two-stage fine-tuning (EU-Emotion → CAM):")
    print("   Stage 1: Fine-tune on EU-Emotion (external dataset)")
    print("   Stage 2: Fine-tune on CAM using EU-Emotion model as starting point")


if __name__ == "__main__":
    main()

