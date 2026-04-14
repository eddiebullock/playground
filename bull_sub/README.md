# bull_sub — Grey Matters content engine

Local-first Substack content pipeline for **Grey Matters** (neuroscience, AI, mental health).

## What this does
- Imports Substack CSV exports into a local SQLite database
- Pulls Google Trends signals (no API key) and scores topic opportunities
- Generates article drafts + title variants via Gemini (human approves; never auto-publishes)
- Generates Substack Notes (copy/paste queue) and social threads
- Can auto-post **Bluesky** threads only when you click a button
- Detects engagement spikes and drafts a paid CTA email

## Setup
1. Create a virtualenv (recommended) and install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Create `.env` from the example:

```bash
cp .env.example .env
```

3. Put Substack CSV exports into `data/exports/` (they are gitignored).

## Run
- Run the full pipeline once:

```bash
python main.py          # same as run-once (default)
python main.py run-once
```

- Run the scheduler (daily 9am by default):

```bash
python main.py schedule
```

- Run the dashboard (from the `bull_sub` folder; filename is `dashboard.py`):

```bash
streamlit run dashboard.py
```

## Notes
- The database is stored at `data/bull_sub.db` (gitignored).
- Articles are saved as drafts with `status=pending` until approved in the dashboard.
- If Google Trends is blocked (proxy/VPN), the pipeline uses a **config keyword fallback** (neutral scores) so you can still generate drafts.
- Set `GEMINI_MODEL_NAME` in `.env` if the default model id is not enabled for your API key (see `.env.example`).
- Each draft can include a **cover image prompt** (text only): paste it into DALL-E, Midjourney, etc., then upload the image in Substack.

