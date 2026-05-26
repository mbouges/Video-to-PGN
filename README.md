# Video-to-PGN

Convert YouTube chess videos (such as speedruns, lessons, or streamers' games) into PGN (Portable Game Notation) files that can be imported directly into Chess.com or Lichess for analysis.

---

## Features

- 🎥 **YouTube Integration** — Paste any YouTube URL (standard, live, or shorts) and process it automatically.
- ⚙️ **Optimized Frame Sampling** — OpenCV samples frames at fixed intervals and deduplicates them using **cropped board perceptual hashing** (`dedup_threshold = 12`) to ignore mouse hovers, tick clocks, and colored arrow overlays while capturing every actual chess move.
- 👁️ **Local Grid Pre-Filter** — Uses standard-deviation projection of Canny edges to locally detect chess board presence in `< 5ms`. Lobbies, facecams, and queue screens are skipped instantly without calling the Gemini API, saving API quota.
- ♟️ **AI-Powered Recognition** — Utilizes Google Gemini Vision (`gemini-2.5-flash`) with structured JSON schemas to recognize board positions row-by-row, supporting standard and flipped (Black at bottom) board orientations.
- 🧹 **FEN Debouncing** — A 3-frame rolling window filters out transient vision misclassifications to keep game segments continuous.
- 🔀 **Fuzzy Move Resolution** — Matches consecutive FEN states using a legal-move difference tree. It resolves classification noise, toggles active turn to handle skipped frames, and supports custom mid-game starting positions.
- 🗃️ **Review and Noise Filtering** — Discards short segments (fewer than 4 moves) and error-heavy reviews, re-indexing remaining games consecutively.
- 🌐 **Modern Dark-Mode UI** — Beautiful dark-mode dashboard with real-time stage progress updates via Server-Sent Events (SSE).

---

## Prerequisites

- **Python 3.11+**
- **FFmpeg** (must be installed and added to your system's PATH)
- **Google Gemini API Key** (Get one at [Google AI Studio](https://aistudio.google.com/apikey))

---

## Setup & Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mbouges/Video-to-PGN.git
   cd Video-to-PGN
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Create a `.env` file in the root directory (this file is excluded from git via `.gitignore`):
   ```env
   VTP_GEMINI_API_KEY=your_gemini_api_key_here
   ```

4. **Run the Application**:
   ```bash
   python -m uvicorn app.main:app
   ```
   *For hot-reload in development, append the `--reload` flag:*
   ```bash
   python -m uvicorn app.main:app --reload
   ```

5. Open your browser and navigate to `http://localhost:8000`.

---

## Project Structure

```text
├── app/
│   ├── config.py           # Configuration settings loaded from .env
│   ├── main.py             # FastAPI entrypoint
│   ├── tasks.py            # Background job orchestrator (download, extract, segment)
│   ├── models/
│   │   └── schemas.py      # Request, response, and job data schemas
│   ├── routes/
│   │   ├── api.py          # Process, status stream, and download endpoints
│   │   └── pages.py        # Web UI pages
│   ├── services/
│   │   ├── downloader.py   # yt-dlp downloader helper
│   │   ├── frame_extractor.py # Perceptual hashing and frame extraction
│   │   ├── piece_classifier_vision.py # Local CV grid check & Gemini FEN parsing
│   │   ├── move_detector.py # Fuzzy move solver
│   │   └── pgn_builder.py  # PGN metadata and string formatting
│   ├── static/
│   │   ├── css/            # Vanilla CSS styling files
│   │   └── js/             # app.js frontend logic
│   └── templates/
│       └── index.html      # UI page template
├── temp/                   # Temporary directory for video downloads and frames
├── output/                 # Output directory for generated PGN files
├── .gitignore              # Git ignored files (including temp, output, and .env)
├── README.md               # Project documentation
└── requirements.txt        # Project dependencies
```

---

## Configuration Parameter Reference

Settings can be customized in the `.env` file or directly in `app/config.py`:
* `VTP_GEMINI_API_KEY`: Your Gemini API key.
* `VTP_SAMPLE_INTERVAL_SECONDS` (default `3.0`): Time between sampled frames.
* `VTP_DEDUP_THRESHOLD` (default `12`): Perceptual hash Hamming distance. Set lower (e.g. `5`) to increase sensitivity, or higher (e.g. `20`) to skip more static states.
* `VTP_API_CALL_DELAY` (default `1.0`): Delay in seconds between consecutive Gemini API calls to prevent free-tier Rate Limit errors.
