---
title: SumVid.ai
emoji: 🎬
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
---

# SumVid.ai — Long-to-Short Video Summariser

Converts a 60-minute MP4 into a 4-minute highlight reel using:

- **Whisper** for transcription
- **Claude claude-sonnet-4-6** for intelligent segment selection
- **FFmpeg** for lossless cutting and stitching

## Quick Start

### 1. Prerequisites

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your API key

```bash
cp .env.example .env
# edit .env and paste your Anthropic API key
```

### 4. Run

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Sidebar Options

| Setting | Default | Description |
|---|---|---|
| Target duration | 240 s | Final highlight reel length |
| Whisper model | base | `tiny` fastest, `medium` most accurate |
| Anthropic API Key | — | Can also be set via `.env` |

## Pipeline Stages

```
Upload MP4
   │
   ▼
[Stage 1] FFmpeg → extract mono 16 kHz WAV
   │
   ▼
[Stage 2] Whisper → timestamped transcript (JSON)
   │
   ▼
[Stage 3] Claude claude-sonnet-4-6 → select segments (Beginning/Middle/End arc)
   │
   ▼
[Stage 4] FFmpeg → cut clips → concat → summary.mp4
   │
   ▼
Download button
```

## Notes

- All intermediate files are written to Python `tempfile` directories and deleted automatically after each run.
- FFmpeg uses **stream copy** (no re-encode) for maximum speed and quality. A re-encode fallback triggers automatically if the source has variable keyframes.
- The Claude prompt enforces a narrative arc and enforces minimum/maximum clip length to avoid jarring edits.
