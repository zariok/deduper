import os
from pathlib import Path
from ..config import Config

def create_example_folder():
    """Create an example folder in the data directory."""
    example_folder = Config.DATA_DIR / 'example'
    example_folder.mkdir(exist_ok=True)
    os.chmod(example_folder, 0o755)
    
    # Create a README file in the example folder
    readme_path = example_folder / 'README.txt'
    readme_content = """This is an example folder for the Media Deduplicator.

To use this application:
1. Add your images and videos to this folder
2. Go back to the web interface
3. Click on this folder to scan for duplicates

Supported file types:
- Images: JPG, PNG, GIF, BMP
- Videos: MP4, AVI, MOV, MKV
"""
    readme_path.write_text(readme_content)
    return str(example_folder) 