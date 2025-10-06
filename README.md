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

- Python 3.6 or higher
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

### Basic Usage
1.  Create a `data` directory and put your folders of images within it or set the environmental variable `DEDUPER_DATA_DIR`

2. Start the application:
```bash
python3 -m deduper
```

3. Open your web browser and navigate to `http://localhost:5000`

4. Use the web interface to scan for duplicates and manage them

### Configuration Options

The application can be configured using environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `DEDUPER_PORT` | Port to run the server on | 5000 |
| `DEDUPER_HOST` | Host to bind the server to | 127.0.0.1 |
| `DEDUPER_DATA_DIR` | Directory to store user data | ./data |
| `DEDUPER_LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) | INFO |
| `SECRET_KEY` | Flask secret key | 'dev' |

### FFmpeg Installation

The application requires FFmpeg for video processing. Install it using:

- macOS: `brew install ffmpeg`
- Linux: `sudo apt-get update && sudo apt-get install ffmpeg`
- Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 