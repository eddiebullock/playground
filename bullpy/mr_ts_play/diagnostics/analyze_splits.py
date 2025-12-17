#!/usr/bin/env python3
"""
Diagnostic script to analyze data splits and identify fundamental issues.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


def analyze_splits(splits_dir: str = "data/splits"):
    """Comprehensive analysis of train/val/test splits."""
    
    splits_dir = Path(splits_dir)
    
    print("=" * 80)
    print("DATASET SPLIT DIAGNOSTICS")
    print("=" * 80)
    
    # Load splits
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")
    
    print(f"\n📊 OVERALL STATISTICS")
    print("-" * 80)
    print(f"Train: {len(train_df)} samples")
    print(f"Val:   {len(val_df)} samples")
    print(f"Test:  {len(test_df)} samples")
    print(f"Total: {len(train_df) + len(val_df) + len(test_df)} samples")
    
    # Unique classes
    train_classes = set(train_df['emotion'].unique())
    val_classes = set(val_df['emotion'].unique())
    test_classes = set(test_df['emotion'].unique())
    all_classes = train_classes | val_classes | test_classes
    
    print(f"\n📚 CLASS STATISTICS")
    print("-" * 80)
    print(f"Total unique classes: {len(all_classes)}")
    print(f"Classes in train: {len(train_classes)}")
    print(f"Classes in val:   {len(val_classes)}")
    print(f"Classes in test:  {len(test_classes)}")
    print(f"Classes only in train: {len(train_classes - val_classes - test_classes)}")
    print(f"Classes only in val:   {len(val_classes - train_classes - test_classes)}")
    print(f"Classes only in test:  {len(test_classes - train_classes - val_classes)}")
    
    # Samples per class
    print(f"\n📈 SAMPLES PER CLASS")
    print("-" * 80)
    
    train_counts = train_df['emotion'].value_counts()
    val_counts = val_df['emotion'].value_counts()
    test_counts = test_df['emotion'].value_counts()
    
    print(f"\nTrain set:")
    print(f"  Min samples per class:  {train_counts.min()}")
    print(f"  Max samples per class:  {train_counts.max()}")
    print(f"  Mean samples per class: {train_counts.mean():.2f}")
    print(f"  Median samples per class: {train_counts.median():.1f}")
    print(f"  Classes with 1 sample:   {(train_counts == 1).sum()}")
    print(f"  Classes with ≤2 samples: {(train_counts <= 2).sum()}")
    print(f"  Classes with ≤3 samples: {(train_counts <= 3).sum()}")
    
    print(f"\nVal set:")
    print(f"  Min samples per class:  {val_counts.min()}")
    print(f"  Max samples per class:  {val_counts.max()}")
    print(f"  Mean samples per class: {val_counts.mean():.2f}")
    print(f"  Median samples per class: {val_counts.median():.1f}")
    
    print(f"\nTest set:")
    print(f"  Min samples per class:  {test_counts.min()}")
    print(f"  Max samples per class:  {test_counts.max()}")
    print(f"  Mean samples per class: {test_counts.mean():.2f}")
    print(f"  Median samples per class: {test_counts.median():.1f}")
    print(f"  Classes with 1 sample:   {(test_counts == 1).sum()}")
    print(f"  Classes with ≤2 samples: {(test_counts <= 2).sum()}")
    print(f"  Classes with ≤3 samples: {(test_counts <= 3).sum()}")
    
    # Actor distribution
    print(f"\n👥 ACTOR DISTRIBUTION")
    print("-" * 80)
    
    train_actors = train_df['actor'].value_counts()
    val_actors = val_df['actor'].value_counts()
    test_actors = test_df['actor'].value_counts()
    
    print(f"\nTrain actors: {sorted(train_actors.index.tolist())}")
    print(f"  Total: {train_actors.sum()} samples")
    print(f"  Distribution:\n{train_actors}")
    
    print(f"\nVal actors: {sorted(val_actors.index.tolist())}")
    print(f"  Total: {val_actors.sum()} samples")
    print(f"  Distribution:\n{val_actors}")
    
    print(f"\nTest actors: {sorted(test_actors.index.tolist())}")
    print(f"  Total: {test_actors.sum()} samples")
    print(f"  Distribution:\n{test_actors}")
    
    # Check actor overlap (should be none)
    train_actor_set = set(train_actors.index)
    val_actor_set = set(val_actors.index)
    test_actor_set = set(test_actors.index)
    
    print(f"\n🔒 ACTOR INDEPENDENCE CHECK")
    print("-" * 80)
    print(f"Train-Val overlap:   {train_actor_set & val_actor_set}")
    print(f"Train-Test overlap:   {train_actor_set & test_actor_set}")
    print(f"Val-Test overlap:     {val_actor_set & test_actor_set}")
    
    if train_actor_set & val_actor_set or train_actor_set & test_actor_set or val_actor_set & test_actor_set:
        print("⚠️  WARNING: Actor overlap detected! Data leakage risk!")
    else:
        print("✅ Actor independence maintained")
    
    # Class distribution by actor
    print(f"\n📊 CLASS DISTRIBUTION BY ACTOR")
    print("-" * 80)
    
    # For each actor, how many unique classes do they have?
    actor_class_counts = {}
    for actor in all_classes:  # Wait, this should be actors, not classes
        pass
    
    # Actually, let's check: for each actor, how many classes do they appear in?
    for split_name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        actor_classes = df.groupby('actor')['emotion'].nunique()
        print(f"\n{split_name} set - Classes per actor:")
        print(actor_classes)
    
    # Random baseline calculation
    print(f"\n🎲 BASELINE PERFORMANCE ESTIMATES")
    print("-" * 80)
    random_baseline = 1.0 / len(all_classes)
    print(f"Random guessing accuracy: {random_baseline:.4f} ({random_baseline*100:.2f}%)")
    print(f"Majority class baseline: {train_counts.max() / len(train_df):.4f} ({train_counts.max() / len(train_df)*100:.2f}%)")
    
    # Expected performance with current data
    print(f"\n⚠️  FEW-SHOT LEARNING ASSESSMENT")
    print("-" * 80)
    avg_train_samples = train_counts.mean()
    print(f"Average samples per class in train: {avg_train_samples:.2f}")
    
    if avg_train_samples < 5:
        print("🔴 CRITICAL: This is a FEW-SHOT learning problem!")
        print("   Standard supervised learning will likely fail.")
        print("   Consider:")
        print("   - Few-shot learning methods (prototypical networks, etc.)")
        print("   - Metric learning approaches")
        print("   - Hierarchical classification")
        print("   - Reducing class granularity")
    elif avg_train_samples < 10:
        print("🟡 WARNING: Very limited data per class.")
        print("   Standard methods may struggle. Consider data augmentation,")
        print("   transfer learning, or few-shot approaches.")
    else:
        print("🟢 OK: Sufficient samples for standard supervised learning.")
    
    # Save detailed report
    output_dir = Path("diagnostics/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save class distribution
    train_counts.to_csv(output_dir / "train_class_distribution.csv")
    val_counts.to_csv(output_dir / "val_class_distribution.csv")
    test_counts.to_csv(output_dir / "test_class_distribution.csv")
    
    # Plot distributions
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].hist(train_counts.values, bins=20, edgecolor='black')
    axes[0].set_title('Train: Samples per Class')
    axes[0].set_xlabel('Samples per Class')
    axes[0].set_ylabel('Number of Classes')
    axes[0].axvline(train_counts.mean(), color='r', linestyle='--', label=f'Mean: {train_counts.mean():.1f}')
    axes[0].legend()
    
    axes[1].hist(val_counts.values, bins=20, edgecolor='black', color='orange')
    axes[1].set_title('Val: Samples per Class')
    axes[1].set_xlabel('Samples per Class')
    axes[1].set_ylabel('Number of Classes')
    axes[1].axvline(val_counts.mean(), color='r', linestyle='--', label=f'Mean: {val_counts.mean():.1f}')
    axes[1].legend()
    
    axes[2].hist(test_counts.values, bins=20, edgecolor='black', color='green')
    axes[2].set_title('Test: Samples per Class')
    axes[2].set_xlabel('Samples per Class')
    axes[2].set_ylabel('Number of Classes')
    axes[2].axvline(test_counts.mean(), color='r', linestyle='--', label=f'Mean: {test_counts.mean():.1f}')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "samples_per_class_distribution.png", dpi=150)
    plt.close()
    
    print(f"\n💾 Saved detailed reports to {output_dir}/")
    
    return {
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df,
        'train_counts': train_counts,
        'val_counts': val_counts,
        'test_counts': test_counts,
    }


def analyze_without_actor_stratification(splits_dir: str = "data/splits"):
    """Analyze what the distribution would look like WITHOUT actor stratification."""
    
    splits_dir = Path(splits_dir)
    
    print("\n" + "=" * 80)
    print("ANALYSIS: WITHOUT ACTOR STRATIFICATION")
    print("=" * 80)
    print("\nThis shows what the data distribution would look like if we")
    print("allowed actors to appear in multiple splits (standard random split).")
    print("\n⚠️  WARNING: This would introduce data leakage risk!")
    
    # Load all data
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")
    
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    # Simulate random split (70/15/15) without actor constraints
    np.random.seed(42)
    n_total = len(all_df)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    
    indices = np.random.permutation(n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train+n_val]
    test_indices = indices[n_train+n_val:]
    
    train_random = all_df.iloc[train_indices]
    val_random = all_df.iloc[val_indices]
    test_random = all_df.iloc[test_indices]
    
    print(f"\n📊 RANDOM SPLIT (NO ACTOR CONSTRAINTS)")
    print("-" * 80)
    print(f"Train: {len(train_random)} samples")
    print(f"Val:   {len(val_random)} samples")
    print(f"Test:  {len(test_random)} samples")
    
    train_random_counts = train_random['emotion'].value_counts()
    val_random_counts = val_random['emotion'].value_counts()
    test_random_counts = test_random['emotion'].value_counts()
    
    print(f"\nTrain set (random split):")
    print(f"  Min samples per class:  {train_random_counts.min()}")
    print(f"  Max samples per class:  {train_random_counts.max()}")
    print(f"  Mean samples per class: {train_random_counts.mean():.2f}")
    print(f"  Median samples per class: {train_random_counts.median():.1f}")
    print(f"  Classes with 1 sample:   {(train_random_counts == 1).sum()}")
    print(f"  Classes with ≤2 samples: {(train_random_counts <= 2).sum()}")
    print(f"  Classes with ≤3 samples: {(train_random_counts <= 3).sum()}")
    
    print(f"\nTest set (random split):")
    print(f"  Min samples per class:  {test_random_counts.min()}")
    print(f"  Max samples per class:  {test_random_counts.max()}")
    print(f"  Mean samples per class: {test_random_counts.mean():.2f}")
    print(f"  Median samples per class: {test_random_counts.median():.1f}")
    print(f"  Classes with 1 sample:   {(test_random_counts == 1).sum()}")
    print(f"  Classes with ≤2 samples: {(test_random_counts <= 2).sum()}")
    print(f"  Classes with ≤3 samples: {(test_random_counts <= 3).sum()}")
    
    # Actor overlap in random split
    train_actors_random = set(train_random['actor'].unique())
    val_actors_random = set(val_random['actor'].unique())
    test_actors_random = set(test_random['actor'].unique())
    
    print(f"\n🔓 ACTOR OVERLAP IN RANDOM SPLIT")
    print("-" * 80)
    print(f"Train-Val overlap:   {len(train_actors_random & val_actors_random)} actors")
    print(f"Train-Test overlap:   {len(train_actors_random & test_actors_random)} actors")
    print(f"Val-Test overlap:     {len(val_actors_random & test_actors_random)} actors")
    print(f"⚠️  All actors appear in multiple splits - DATA LEAKAGE RISK!")
    
    # Comparison
    print(f"\n📈 COMPARISON: Actor-Stratified vs Random Split")
    print("-" * 80)
    
    # Load original stratified counts
    train_df_orig = pd.read_csv(splits_dir / "train.csv")
    train_counts_orig = train_df_orig['emotion'].value_counts()
    
    print(f"\n{'Metric':<30} {'Actor-Stratified':<20} {'Random Split':<20} {'Improvement':<15}")
    print("-" * 85)
    print(f"{'Mean samples/class (train)':<30} {train_counts_orig.mean():<20.2f} {train_random_counts.mean():<20.2f} {train_random_counts.mean() - train_counts_orig.mean():+.2f}")
    print(f"{'Median samples/class (train)':<30} {train_counts_orig.median():<20.1f} {train_random_counts.median():<20.1f} {train_random_counts.median() - train_counts_orig.median():+.1f}")
    print(f"{'Classes with ≤2 samples':<30} {(train_counts_orig <= 2).sum():<20} {(train_random_counts <= 2).sum():<20} {(train_random_counts <= 2).sum() - (train_counts_orig <= 2).sum():+d}")
    print(f"{'Classes with ≤3 samples':<30} {(train_counts_orig <= 3).sum():<20} {(train_random_counts <= 3).sum():<20} {(train_random_counts <= 3).sum() - (train_counts_orig <= 3).sum():+d}")
    
    improvement_factor = train_random_counts.mean() / train_counts_orig.mean()
    print(f"\n💡 Random split provides {improvement_factor:.1f}x more samples per class on average")
    
    return {
        'train_random': train_random,
        'val_random': val_random,
        'test_random': test_random,
        'train_random_counts': train_random_counts,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze dataset splits")
    parser.add_argument(
        "--splits_dir",
        type=str,
        default="data/splits",
        help="Directory containing splits"
    )
    parser.add_argument(
        "--compare_random",
        action="store_true",
        help="Also analyze random split (no actor stratification)"
    )
    
    args = parser.parse_args()
    
    # Main analysis
    results = analyze_splits(args.splits_dir)
    
    # Optional comparison
    if args.compare_random:
        random_results = analyze_without_actor_stratification(args.splits_dir)
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)

