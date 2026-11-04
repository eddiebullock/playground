"""
Human response predictor from Snoek AU data, rule 5 of RESEARCH_QUESTIONS.md.

It maps 33 AU amplitudes to P(label) by the one-vs-rest ridge model of
research/scripts/human_au_effects.py. It is a model of humans and never counts as human
data. Its output is always labelled a prediction.

Validation, with subjects and stimuli both unseen:
  within group  Five random splits. Fit on half the subjects and half the stimuli at the
                lambda stored in human_au_effects.json, then test on the other subjects
                viewing the other stimuli.
  across group  Slopes from human_au_effects.json for one group (Western, East Asian),
                applied to the other group's subjects. Those slopes never saw that group.
                Intercepts are fit on half of its stimuli and the score uses the other half.
Scores: log loss against the training base rate, top-1 agreement, and, over test stimuli
seen by at least 3 test subjects, the correlation between predicted and observed P(label).

The zip is read in place. Limits stated in the output: avatars not video, AUs only, 8 of
21 labels. Validation on EU-Emotion and CREMA-D is not done here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "scripts"))
import human_au_effects as hae  # noqa: E402

N_SPLITS = 5
SEED = 20261104
PAIRS = {"basic_WC": "basic_EA", "basic_EA": "basic_WC", "conversational_WC": "conversational_EA", "conversational_EA": "conversational_WC"}


def predict(au: np.ndarray, gender: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """P(label), trials by labels. coefficients: labels by (intercept, gender, 33 AUs)."""
    design = np.column_stack([np.ones(len(au)), gender, au])
    probabilities = hae.expit(design @ coefficients.T)
    return probabilities / probabilities.sum(1, keepdims=True)


def scores(probabilities: np.ndarray, y: np.ndarray, base: np.ndarray, stimulus: np.ndarray, subject_n: int) -> Dict[str, float]:
    n = len(y)
    chosen = probabilities[np.arange(n), y]
    loss = float(-np.log(np.clip(chosen, 1e-9, 1)).mean())
    null = float(-np.log(np.clip(base[y], 1e-9, 1)).mean())
    top = float((probabilities.argmax(1) == y).mean())
    predicted, observed = [], []
    for stim in np.unique(stimulus):
        keep = stimulus == stim
        if keep.sum() >= 3:
            predicted.append(probabilities[keep].mean(0))
            observed.append(np.bincount(y[keep], minlength=probabilities.shape[1]) / keep.sum())
    r = float(np.corrcoef(np.ravel(predicted), np.ravel(observed))[0, 1]) if predicted else float("nan")
    return {
        "n_test_trials": n,
        "n_test_subjects": subject_n,
        "n_test_stimuli_scored": len(predicted),
        "log_loss": round(loss, 4),
        "base_rate_log_loss": round(null, 4),
        "top1": round(top, 4),
        "base_rate_top1": round(float(base.max()), 4),
        "r_predicted_observed_by_stimulus": round(r, 4),
    }


def design_parts(frame: Any, labels: Sequence[str], label_col: str) -> Dict[str, np.ndarray]:
    stim = (frame["cfg"].astype(str) + "|" + frame["face_gender"].astype(str) + "|" + frame["face_id"].astype(str)).to_numpy()
    return {
        "au": frame[hae.AU_COLS].to_numpy(float),
        "gender": (frame["face_gender"].to_numpy() == "M").astype(float) - 0.5,
        "y": frame[label_col].map({k: i for i, k in enumerate(labels)}).to_numpy(int),
        "sub": frame["sub"].to_numpy(),
        "stim": stim,
    }


def fit_mean_subject(parts: Dict[str, np.ndarray], mask: np.ndarray, n_labels: int, lam: float) -> np.ndarray:
    subjects = np.unique(parts["sub"][mask])
    pen = np.r_[0.0, 0.0, np.full(len(hae.AU_COLS), lam)]
    total = np.zeros((n_labels, 2 + len(hae.AU_COLS)))
    for subject in subjects:
        keep = mask & (parts["sub"] == subject)
        X = np.column_stack([np.ones(keep.sum()), parts["gender"][keep], parts["au"][keep]])
        for label in range(n_labels):
            total[label] += hae.fit_ridge_logit(X, (parts["y"][keep] == label).astype(float), pen=pen)
    return total / len(subjects)


def within_group(frame: Any, labels: Sequence[str], label_col: str, lam: float) -> Dict[str, Any]:
    parts = design_parts(frame, labels, label_col)
    rng = np.random.default_rng(SEED)
    rows = []
    subjects = np.unique(parts["sub"])
    stimuli = np.unique(parts["stim"])
    for _split in range(N_SPLITS):
        train_subjects = set(rng.permutation(subjects)[: len(subjects) // 2].tolist())
        train_stimuli = set(rng.permutation(stimuli)[: len(stimuli) // 2].tolist())
        in_subject = np.array([s in train_subjects for s in parts["sub"]])
        in_stimulus = np.array([s in train_stimuli for s in parts["stim"]])
        train = in_subject & in_stimulus
        test = ~in_subject & ~in_stimulus
        coefficients = fit_mean_subject(parts, train, len(labels), lam)
        base = np.bincount(parts["y"][train], minlength=len(labels)) / train.sum()
        rows.append(
            scores(
                predict(parts["au"][test], parts["gender"][test], coefficients),
                parts["y"][test],
                base,
                parts["stim"][test],
                len(np.unique(parts["sub"][test])),
            )
        )
    keys = [k for k in rows[0] if isinstance(rows[0][k], float)]
    return {"mean_over_splits": {k: round(float(np.mean([r[k] for r in rows])), 4) for k in keys}, "splits": rows}


def across_group(frame: Any, labels: Sequence[str], label_col: str, slopes: np.ndarray) -> Dict[str, Any]:
    """Slopes from the other group; intercepts and gender refit on half this group's stimuli."""
    parts = design_parts(frame, labels, label_col)
    rng = np.random.default_rng(SEED)
    stimuli = np.unique(parts["stim"])
    calibrate = set(rng.permutation(stimuli)[: len(stimuli) // 2].tolist())
    mask = np.array([s in calibrate for s in parts["stim"]])
    coefficients = np.zeros((len(labels), 2 + len(hae.AU_COLS)))
    coefficients[:, 2:] = slopes
    for label in range(len(labels)):
        offset = parts["au"][mask] @ slopes[label]
        y = (parts["y"][mask] == label).astype(float)
        a, b = 0.0, 0.0
        X = np.column_stack([np.ones(mask.sum()), parts["gender"][mask]])
        for _ in range(25):
            mu = hae.expit(a + b * X[:, 1] + offset)
            v = np.maximum(mu * (1 - mu), 1e-9)
            step = np.linalg.solve((X.T * v) @ X, X.T @ (y - mu))
            a, b = a + step[0], b + step[1]
        coefficients[label, :2] = a, b
    test = ~mask
    base = np.bincount(parts["y"][mask], minlength=len(labels)) / mask.sum()
    return scores(
        predict(parts["au"][test], parts["gender"][test], coefficients),
        parts["y"][test],
        base,
        parts["stim"][test],
        len(np.unique(parts["sub"][test])),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--zip", type=Path, default=Path("data/external/snoek_au/au_emotion_data.zip"))
    parser.add_argument("--effects", type=Path, default=Path("research/reproduction/human_au_effects.json"))
    parser.add_argument("--output", type=Path, default=Path("research/reproduction/human_response_predictor.json"))
    args = parser.parse_args()
    effects = json.loads(args.effects.read_text(encoding="utf-8"))["datasets"]
    frames, _ = hae.load_zip(args.zip)
    out: Dict[str, Any] = {
        "status": "A model of human judgement. Prediction, never human data.",
        "limits": [
            "Glasgow generative avatars, not video.",
            "AU amplitudes only: no gaze, head or timing.",
            "Covers about 8 of 21 EU-Emotion labels.",
            "Slopes in the source json were fit on all subjects of a group, so they are validated only across groups here.",
            "Not yet validated on EU-Emotion or CREMA-D, as rule 5 asks. Snoek amplitudes and detector intensities do not share units.",
        ],
        "datasets": {},
    }
    for name, (frame, label_col) in frames.items():
        record = effects[name]
        labels = record["labels"]
        lam = float(record["ridge_lambda_selected"])
        other = effects[PAIRS[name]]
        slopes = np.array(other["beta_log_odds_per_unit_amplitude"])[:, : len(hae.AU_COLS)]
        if other["labels"] != labels:
            raise ValueError("label order differs across groups")
        out["datasets"][name] = {
            "labels": labels,
            "lambda": lam,
            "within_group_unseen_subjects_and_stimuli": within_group(frame, labels, label_col, lam),
            "across_group_slopes_from": PAIRS[name],
            "across_group": across_group(frame, labels, label_col, slopes),
        }
        print(name, json.dumps(out["datasets"][name]["within_group_unseen_subjects_and_stimuli"]["mean_over_splits"]), json.dumps(out["datasets"][name]["across_group"]), flush=True)
    args.output.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
