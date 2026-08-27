## Does AI Have a Theory of Mind? Evaluating Mental State Recognition of Large Language Models

**Authors**: _TBD_

## Abstract
We evaluate large language models on two mental state recognition datasets—EU-Emotion (27 mental states) and Mindreading (357 mental states)—using a four-alternative forced-choice paradigm. **Gemini 3 Flash** provides audio-capable modality ablations; additional models are evaluated video-only where applicable. Gemini 3 Pro was omitted from the main study after pilot evaluation (see `analysis/study_config.py`).

## Repository structure
- **`models/`**: Model wrappers/adapters and shared inference utilities for each LLM provider.
- **`experiments/`**: Experiment runners and configuration for the forced-choice evaluation pipeline.
- **`analysis/`**: Statistical analyses, figure/table generation scripts, and any post-processing code.
- **`prompts/`**: Prompt templates and prompt variants used for each model and condition.
- **`data/`**: Local dataset mount point (not included in this repository); expected input file layouts and helper scripts.
- **`results/`**: Generated outputs (metrics, logs, intermediate artifacts). The directory is tracked, but large files are ignored by git.
- **`cache/`**: Local caches (e.g., downloaded assets, preprocessed features, API response caches). Not intended for version control.

## Installation
Create a Python environment (recommended) and install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the repository root (this directory) with your API keys, for example:

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
```

## Usage
Run the evaluation entrypoint:

```bash
python experiments/run_evaluation.py
```

Run analyses on completed results (excludes `gemini-3-pro` by default):

```bash
cd publication_repo
python analysis/run_study_analysis.py --results-dir results/full_run
```

Outputs land in `analysis_outputs/` (`statistical_analysis.json`, `per_emotion_breakdown.csv`).

## Data availability
The EU-Emotion and Mindreading stimuli cannot be included in this repository due to licensing restrictions. Please obtain the datasets from the original sources:

- EU-Emotion dataset: _citation/source link TBD_
- Mindreading dataset: _citation/source link TBD_

## Citation
_Journal/conference TBD._

```bibtex
@article{tbd2026_theory_of_mind_llms,
  title   = {Does AI Have a Theory of Mind? Evaluating Mental State Recognition of Large Language Models},
  author  = {TBD},
  journal = {TBD},
  year    = {2026}
}
```

## License
MIT
