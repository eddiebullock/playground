"""
CAM Face-Voice Battery Dataset Module

This module implements the dataset loading and trial parsing for the computational
replication of the Cambridge Mindreading (CAM) Face-Voice Battery (Golan et al., 2006).

Methodology mapping:
- Each trial consists of: one video (+ audio), four candidate mental-state labels,
  and one correct label
- Actor-independent train/val/test splits ensure no actor appears in multiple splits
- Supports both face (visual) and voice (audio) modalities
- Original CAM: 20 concepts with 5 items each (100 total trials)
- Each concept has either 3 face + 2 voice items OR 3 voice + 2 face items

The original CAM validation criteria:
- Item valid if >50% control group selected target and <33% selected any foil
- 5 concepts excluded, 8 had invalid items removed, then balanced to 5 items/concept
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class CAMTrial:
    """
    Represents a single CAM trial following the original methodology.
    
    Each trial corresponds to one test question from the CAM battery:
    - One stimulus (video with audio)
    - Four candidate labels (one correct + three foils)
    - Correct answer index
    """
    trial_id: str
    stimulus_path: str  # Path to video file
    modality: str  # "face" or "voice"
    correct_label: str  # The correct emotion label
    candidate_labels: List[str]  # Four labels: [correct, foil1, foil2, foil3]
    correct_idx: int  # Index of correct label in candidate_labels (0-3)
    actor: str
    scenario_id: str
    concept: Optional[str] = None  # The emotion concept this trial belongs to
    
    def __post_init__(self):
        """Validate trial structure."""
        if len(self.candidate_labels) != 4:
            raise ValueError(f"Trial {self.trial_id} must have exactly 4 candidate labels")
        if self.correct_idx not in range(4):
            raise ValueError(f"Trial {self.trial_id} correct_idx must be 0-3")
        if self.candidate_labels[self.correct_idx] != self.correct_label:
            raise ValueError(f"Trial {self.trial_id} correct_label mismatch")


class CAMDataset:
    """
    Dataset loader for CAM Face-Voice Battery trials.
    
    Handles:
    - Loading trial definitions from configuration or generating from dataset
    - Enforcing actor-independent train/val/test splits
    - Providing trial-level access (not just video-level)
    
    The original CAM methodology uses actor-independent evaluation to ensure
    models generalize across different actors, matching the human evaluation protocol.
    """
    
    def __init__(
        self,
        data_root: str,
        trial_definitions_file: Optional[str] = None,
        splits_dir: Optional[str] = None,
        split_name: str = "test",
        seed: int = 42,
        use_actor_filtering: bool = True,
    ):
        """
        Initialize CAM dataset loader.
        
        Args:
            data_root: Root directory of CAM stimuli (video files)
            trial_definitions_file: Optional JSON file with trial definitions.
                If None, trials will be generated from dataset structure.
                Format: {
                    "trials": [
                        {
                            "trial_id": "trial_001",
                            "stimulus_path": "relative/path/to/video.mov",
                            "modality": "face" or "voice",
                            "correct_label": "emotion_name",
                            "candidate_labels": ["correct", "foil1", "foil2", "foil3"],
                            "correct_idx": 0,
                            "actor": "M",
                            "scenario_id": "0100104",
                            "concept": "humiliating"
                        }
                    ]
                }
            splits_dir: Directory containing actor-independent split files.
                Expected files: train_actors.txt, val_actors.txt, test_actors.txt
                OR: train.csv, val.csv, test.csv with actor column
            split_name: Which split to load ("train", "val", or "test")
                Use "all" to load all trials without actor filtering (matches original CAM)
            seed: Random seed for reproducibility
            use_actor_filtering: If False, load all trials regardless of actor splits
                (matches original CAM methodology where all participants saw same trials)
        """
        self.data_root = Path(data_root)
        self.splits_dir = Path(splits_dir) if splits_dir else None
        self.split_name = split_name
        self.seed = seed
        self.use_actor_filtering = use_actor_filtering
        np.random.seed(seed)
        
        # Load trial definitions
        if trial_definitions_file:
            self.trials = self._load_trials_from_file(trial_definitions_file)
        else:
            # Generate trials from dataset structure using CAM taxonomy
            # Note: Use create_trial_definitions.py script to generate trial definitions file
            # This avoids circular imports and keeps code modular
            raise ValueError(
                "trial_definitions_file is required. "
                "Generate trial definitions using: "
                "python experiments/cam_human_like/create_trial_definitions.py"
            )
        
        # Apply actor-independent split filtering (unless disabled)
        if self.use_actor_filtering and self.splits_dir and split_name != "all":
            self.trials = self._filter_by_split(self.trials, split_name)
            print(f"Loaded {len(self.trials)} trials for {split_name} split (actor-filtered)")
        else:
            print(f"Loaded {len(self.trials)} trials (no actor filtering - matches original CAM)")
    
    def _load_trials_from_file(self, trial_file: str) -> List[CAMTrial]:
        """Load trial definitions from JSON file."""
        with open(trial_file, 'r') as f:
            data = json.load(f)
        
        trials = []
        for trial_data in data['trials']:
            # Resolve stimulus path
            stimulus_path = trial_data['stimulus_path']
            if not Path(stimulus_path).is_absolute():
                stimulus_path = str(self.data_root / stimulus_path)
            
            trial = CAMTrial(
                trial_id=trial_data['trial_id'],
                stimulus_path=stimulus_path,
                modality=trial_data['modality'],
                correct_label=trial_data['correct_label'],
                candidate_labels=trial_data['candidate_labels'],
                correct_idx=trial_data['correct_idx'],
                actor=trial_data['actor'],
                scenario_id=trial_data['scenario_id'],
                concept=trial_data.get('concept'),
            )
            trials.append(trial)
        
        return trials
    
    def _filter_by_split(self, trials: List[CAMTrial], split_name: str) -> List[CAMTrial]:
        """
        Filter trials by actor-independent split.
        
        Ensures no actor from this split appears in other splits, matching
        the original CAM methodology for actor-independent evaluation.
        """
        # Try loading actor list file first
        actor_file = self.splits_dir / f"{split_name}_actors.txt"
        if actor_file.exists():
            with open(actor_file, 'r') as f:
                allowed_actors = set(line.strip() for line in f if line.strip())
        else:
            # Try loading from CSV split file
            split_file = self.splits_dir / f"{split_name}.csv"
            if split_file.exists():
                df = pd.read_csv(split_file)
                allowed_actors = set(df['actor'].unique())
            else:
                raise FileNotFoundError(
                    f"Could not find split definition: {actor_file} or {split_file}"
                )
        
        filtered_trials = [
            trial for trial in trials
            if trial.actor in allowed_actors
        ]
        
        print(f"Filtered to {len(filtered_trials)} trials with actors: {sorted(allowed_actors)}")
        return filtered_trials
    
    def __len__(self) -> int:
        """Return number of trials in this split."""
        return len(self.trials)
    
    def __getitem__(self, idx: int) -> CAMTrial:
        """Get a trial by index."""
        return self.trials[idx]
    
    def get_trial(self, trial_id: str) -> Optional[CAMTrial]:
        """Get a trial by ID."""
        for trial in self.trials:
            if trial.trial_id == trial_id:
                return trial
        return None
    
    def get_all_trials(self) -> List[CAMTrial]:
        """Get all trials in this split."""
        return self.trials.copy()
    
    def get_concepts(self) -> List[str]:
        """Get unique emotion concepts in this split."""
        concepts = [trial.concept for trial in self.trials if trial.concept]
        return sorted(set(concepts))
    
    def get_actors(self) -> List[str]:
        """Get unique actors in this split."""
        actors = [trial.actor for trial in self.trials]
        return sorted(set(actors))


def create_actor_splits(
    data_root: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    """
    Create actor-independent train/val/test splits.
    
    This function ensures that actors are split across train/val/test sets
    such that no actor appears in multiple splits. This matches the original
    CAM methodology for actor-independent evaluation.
    
    Args:
        data_root: Root directory containing video files
        output_dir: Directory to save split files
        train_ratio: Proportion of actors for training
        val_ratio: Proportion of actors for validation
        test_ratio: Proportion of actors for testing
        seed: Random seed for reproducibility
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(seed)
    
    # Discover all unique actors from dataset
    data_path = Path(data_root)
    actors = set()
    
    for video_file in data_path.rglob("*.mov"):
        parsed = _parse_filename(video_file.name)
        if parsed and parsed.get('actor'):
            actors.add(parsed['actor'])
    
    actors = sorted(list(actors))
    print(f"Found {len(actors)} unique actors: {actors}")
    
    # Shuffle actors
    np.random.shuffle(actors)
    
    # Split actors
    n_total = len(actors)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    
    train_actors = actors[:n_train]
    val_actors = actors[n_train:n_train + n_val]
    test_actors = actors[n_train + n_val:]
    
    print(f"Train actors ({len(train_actors)}): {train_actors}")
    print(f"Val actors ({len(val_actors)}): {val_actors}")
    print(f"Test actors ({len(test_actors)}): {test_actors}")
    
    # Save actor lists
    for split_name, actor_list in [
        ("train", train_actors),
        ("val", val_actors),
        ("test", test_actors),
    ]:
        output_file = output_dir / f"{split_name}_actors.txt"
        with open(output_file, 'w') as f:
            for actor in actor_list:
                f.write(f"{actor}\n")
        print(f"Saved {split_name} actors to {output_file}")


def _parse_filename(filename: str) -> Optional[Dict]:
    """Parse CAM video filename to extract metadata."""
    base = filename.replace(".mov", "")
    
    # Extract emotion (last part after V/T, may contain hyphens)
    match = re.search(r'([VT])([a-z]+(?:-[a-z]+)*)$', base)
    if not match:
        return None
    
    modality_code = match.group(1)  # V or T
    emotion = match.group(2)
    prefix = base[:match.start()]
    
    # Extract scenario ID (first 7 digits)
    scenario_match = re.match(r'^(\d{7})', prefix)
    if not scenario_match:
        return None
    
    scenario_id = scenario_match.group(1)
    actor_part = prefix[7:]
    
    # Extract actor and number
    actor_match = re.match(r'^([A-Z]+)(\d+)', actor_part)
    if actor_match:
        actor = actor_match.group(1)
        instance_num = actor_match.group(2)
    else:
        actor = actor_part[0] if actor_part else "?"
        instance_num = actor_part[1:] if len(actor_part) > 1 else "?"
    
    # Map modality code to CAM terminology
    modality = "face" if modality_code == "V" else "voice"
    
    return {
        "scenario_id": scenario_id,
        "actor": actor,
        "instance_num": instance_num,
        "modality": modality,
        "modality_code": modality_code,
        "emotion": emotion,
    }

