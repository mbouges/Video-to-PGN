# Video-to-PGN

Convert YouTube chess videos into PGN (Portable Game Notation) files that can be imported into Chess.com or Lichess.

## Features

- 🎥 **YouTube Integration** — Paste a YouTube URL and extract chess games automatically
- ♟️ **AI-Powered Recognition** — Uses Gemini Vision to recognize board positions
- 📝 **PGN Generation** — Creates standard PGN files compatible with Chess.com and Lichess
- 🎮 **Multi-Game Support** — Handles videos with multiple games
- 🌐 **Web Interface** — Beautiful dark-mode UI with real-time progress tracking

## Prerequisites

- Python 3.11+
- FFmpeg (for video processing)
- Google Gemini API key ([Get one here](https://aistudio.google.com/apikey))

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set your Gemini API key
set GEMINI_API_KEY=your_api_key_here

# Run the application
python -m uvicorn app.main:app --reload
```

Then open http://localhost:8000 in your browser.

## Usage

1. Paste a YouTube video URL containing chess games
2. Click "Convert"
3. Wait for processing (download → game detection → position analysis → PGN generation)
4. Download individual or combined PGN files
5. Import into Chess.com or Lichess

## Tech Stack

- **Backend:** FastAPI + Uvicorn
- **Video Processing:** yt-dlp + OpenCV
- **Chess Logic:** python-chess
- **Position Recognition:** Google Gemini Vision API
- **Frontend:** Vanilla HTML/CSS/JS
