#!/usr/bin/env python3
"""
Inspect V and T video modalities to understand the difference.
Extracts sample frames from both types of videos.
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

DATASET_ROOT = Path(
    "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/mindreading_transporter_files/Mindreading emotions library/Emotions"
)

def extract_frames(video_path, num_frames=3):
    """Extract frames from a video."""
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"  Warning: Could not open {video_path.name}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frames = []
    # Sample frames uniformly
    if total_frames > 0:
        indices = np.linspace(0, total_frames - 1, min(num_frames, total_frames), dtype=int)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
    
    cap.release()
    
    return {
        'frames': frames,
        'total_frames': total_frames,
        'fps': fps,
        'duration': duration,
        'width': width,
        'height': height,
    }

def main():
    # Find example V and T videos from the same scenario
    scenario_dir = DATASET_ROOT / "01" / "0100104"
    
    v_videos = []
    t_videos = []
    
    for video_file in scenario_dir.glob("*.mov"):
        if "V" in video_file.name and "T" not in video_file.name:
            v_videos.append(video_file)
        elif "T" in video_file.name and "V" not in video_file.name:
            t_videos.append(video_file)
    
    if not v_videos:
        print("Could not find V videos")
        return
    
    if not t_videos:
        print("Could not find T videos")
        return
    
    # Try each T video until we find one that works
    v_video = v_videos[0]
    t_video = None
    t_info = None
    
    print(f"Trying V video: {v_video.name}")
    for t_vid in t_videos:
        print(f"Trying T video: {t_vid.name}")
        t_info = extract_frames(t_vid)
        if t_info and t_info['frames']:
            t_video = t_vid
            break
    
    print(f"\nV video: {v_video.name}")
    if t_video:
        print(f"T video: {t_video.name}")
    else:
        print("T video: Could not find readable T video (codec issue)")
    print()
    
    # Extract frames from V video
    print("Extracting frames from V video...")
    v_info = extract_frames(v_video)
    
    if not v_info:
        print("Could not extract frames from V video")
        return
    
    # Try to get T video info even if we can't read frames
    t_info = None
    if t_video:
        print("Attempting to extract frames from T video...")
        t_info = extract_frames(t_video)
    
    # If T video can't be read, at least show file properties
    if not t_info or not t_info['frames']:
        print("\nNote: T videos appear to use a codec that OpenCV cannot read directly.")
        print("This is common with older QuickTime formats.")
        print("File properties:")
        print(f"  T video file size: {t_videos[0].stat().st_size / 1024:.1f} KB")
        print(f"  V video file size: {v_video.stat().st_size / 1024:.1f} KB")
        print(f"  Size ratio: {t_videos[0].stat().st_size / v_video.stat().st_size:.2f}x smaller")
    
    # Print video properties
    print("=" * 60)
    print("V (Visual) Video Properties:")
    print(f"  Total frames: {v_info['total_frames']}")
    print(f"  FPS: {v_info['fps']:.2f}")
    print(f"  Duration: {v_info['duration']:.2f} seconds")
    print(f"  Resolution: {v_info.get('width', 'unknown')}x{v_info.get('height', 'unknown')}")
    print(f"  File size: {v_video.stat().st_size / 1024:.1f} KB")
    
    if t_info and t_info['frames']:
        print("\n" + "=" * 60)
        print("T (Textual) Video Properties:")
        print(f"  Total frames: {t_info['total_frames']}")
        print(f"  FPS: {t_info['fps']:.2f}")
        print(f"  Duration: {t_info['duration']:.2f} seconds")
        print(f"  Resolution: {t_info.get('width', 'unknown')}x{t_info.get('height', 'unknown')}")
        print(f"  File size: {t_video.stat().st_size / 1024:.1f} KB")
    else:
        print("\n" + "=" * 60)
        print("T (Textual) Video Properties:")
        print("  Could not read video (codec issue)")
        if t_videos:
            print(f"  File size: {t_videos[0].stat().st_size / 1024:.1f} KB")
            print("  Note: T videos are much smaller than V videos")
    
    # Save frames for visual inspection
    output_dir = Path("video_inspection")
    output_dir.mkdir(exist_ok=True)
    
    # Save V video frames
    print(f"\n{'=' * 60}")
    print("Saving V video frames...")
    for i, frame in enumerate(v_info['frames']):
        output_path = output_dir / f"V_frame_{i+1}.png"
        plt.imsave(output_path, frame)
        print(f"  Saved: {output_path}")
    
    # Save T video frames (if available)
    if t_info and t_info['frames']:
        print(f"\n{'=' * 60}")
        print("Saving T video frames...")
        for i, frame in enumerate(t_info['frames']):
            output_path = output_dir / f"T_frame_{i+1}.png"
            plt.imsave(output_path, frame)
            print(f"  Saved: {output_path}")
    
    # Create side-by-side comparison (if both available)
    if v_info['frames'] and t_info and t_info['frames']:
        fig, axes = plt.subplots(max(len(v_info['frames']), len(t_info['frames'])), 2, 
                                figsize=(12, 6 * max(len(v_info['frames']), len(t_info['frames']))))
        
        if len(v_info['frames']) == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(len(v_info['frames'])):
            axes[i, 0].imshow(v_info['frames'][i])
            axes[i, 0].set_title(f"V Video - Frame {i+1}")
            axes[i, 0].axis('off')
        
        for i in range(len(t_info['frames'])):
            axes[i, 1].imshow(t_info['frames'][i])
            axes[i, 1].set_title(f"T Video - Frame {i+1}")
            axes[i, 1].axis('off')
        
        # Hide unused subplots
        for i in range(len(v_info['frames']), axes.shape[0]):
            axes[i, 0].axis('off')
        for i in range(len(t_info['frames']), axes.shape[0]):
            axes[i, 1].axis('off')
        
        comparison_path = output_dir / "V_vs_T_comparison.png"
        plt.tight_layout()
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        print(f"\n{'=' * 60}")
        print(f"Saved comparison: {comparison_path}")
        plt.close()
    
    print(f"\n{'=' * 60}")
    print("Inspection complete! Check the 'video_inspection' directory for saved frames.")
    print("You can visually compare V and T videos to understand the difference.")

if __name__ == "__main__":
    main()

