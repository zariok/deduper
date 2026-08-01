# Deduper

Ever back up multiple folders of your own pictures and need to try and clean them up?

Ever have media scrapers running in the background and some influncers post the same image over and over?

I created this tool to help consolidate the images by symlinking the similar or matching files to the best one so the scrapers would then ignore previously acquired images.

# !!!! NOTICE !!!!
As this was developed for a linux-based system, **symlink** implies the media is deleted and a symlink is pointed to the "best" file available saving space and allowing the scrapers to not redownload media.  THIS IS **DESTRUCTIVE** and will **DELETE** media.

## How it works

### Images
- Uses perceptual hashing to detect simlar images
- Groups images with similar hash values
- Compares resolution, file size and other metadata to determine the "best" file

### Videos
- Extracts thumbnail frame at 1s mark
- Groups videos with simlar thumbnails and metadata
- Uses perceptual hashing to detect simlilar videos based on thumbnail
- Compares video resolution, duration and file size to determine the "best" file

### Buttons Actions
❤️ - Sets selected item as new "best", symlinks all other files to selection

❌ - Sets only this selection as a symlink to the "best" file.

💛 - symlinks all duplicates to selection

## Requirements

- Python 3.10 or higher (tested with Python 3.12.3)
- FFmpeg (for video processing)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/zariok/deduper.git
cd deduper
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install the package:
```bash
pip install -e .
```

## Usage

### Upgrading from Previous Version

If you're upgrading from a version that used JSON cache files (`.deduper`), you **must** run the migration script **before** starting the new version:

```bash
# Dry-run to see what would be migrated
python scripts/migrate_deduper_to_sqlite.py --dry-run

# Migrate all .deduper files to .deduper.db (and remove .deduper files)
python scripts/migrate_deduper_to_sqlite.py --root /path/to/data

# Or use DEDUPER_DATA_DIR environment variable
python scripts/migrate_deduper_to_sqlite.py
```

The migration script converts all existing `.deduper` JSON cache files to the new SQLite format (`.deduper.db`) for 10-20x faster performance.

### Basic Usage
1.  Create a `data` directory and put your folders of images within it or set the environmental variable `DEDUPER_DATA_DIR`

2. Start the application. A `SECRET_KEY` is required, otherwise the app prints an
   error and exits:
```bash
SECRET_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(32))')" python3 -m deduper
```

For local use you can instead set `DEDUPER_DEV=true`, which falls back to an
insecure development key:
```bash
DEDUPER_DEV=true python3 -m deduper
```

3. Open your web browser and navigate to `http://localhost:5000`

4. Use the web interface to scan for duplicates and manage them

### What happens on first run

- Every scanned folder gets a hidden `.deduper.db` SQLite database holding
  perceptual hashes, media dimensions and the groups found. Deleting it is safe -
  it just forces a full rescan.
- Videos get a `thumb-deduper.<name>.jpg` thumbnail alongside them, used for
  hashing and for the web interface.
- A background scanner pre-scans your folders so results load instantly when you
  open them. It waits 5 minutes after a folder stops changing before rescanning,
  so ongoing downloads or transfers are not scanned mid-flight.
- Exact duplicates (identical hash, resolution *and* file size) are symlinked
  automatically during a scan. Everything else waits for you to choose.

### Configuration Options

The application can be configured using environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `DEDUPER_PORT` | Port to run the server on | 5000 |
| `DEDUPER_HOST` | Host to bind the server to | 127.0.0.1 |
| `DEDUPER_DATA_DIR` | Directory to store user data | ./data |
| `DEDUPER_LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) | INFO |
| `SECRET_KEY` | Flask secret key. **Required** - the app exits if it is unset and `DEDUPER_DEV`/`FLASK_ENV` are not set | _(none)_ |
| `DEDUPER_DEV` | Set to `true` to allow an insecure `'dev'` secret key for local use | unset |

### FFmpeg Installation

The application requires FFmpeg for video processing. Install it using:

- macOS: `brew install ffmpeg`
- Linux: `sudo apt-get update && sudo apt-get install ffmpeg`
- Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

## Development

Install the development dependencies and run the test suite:

```bash
pip install -r requirements-dev.txt
pytest
```

The suite generates real images and ffmpeg videos, then scans them end to end.
Tests needing video skip automatically when `ffmpeg`/`ffprobe` are not on PATH:

```bash
pytest -k "not Video"     # skip everything requiring ffmpeg
pytest --no-cov           # skip coverage reporting
```

Scanning is destructive, so every test works on a throwaway copy of its fixtures
and writes nothing to your `data` directory.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 