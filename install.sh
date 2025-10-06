#!/bin/bash

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to detect OS and package manager
detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        PKG_MANAGER="brew"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        if command -v apt-get &> /dev/null; then
            PKG_MANAGER="apt"
        elif command -v yum &> /dev/null; then
            PKG_MANAGER="yum"
        elif command -v dnf &> /dev/null; then
            PKG_MANAGER="dnf"
        elif command -v pacman &> /dev/null; then
            PKG_MANAGER="pacman"
        else
            PKG_MANAGER="unknown"
        fi
    else
        OS="unknown"
        PKG_MANAGER="unknown"
    fi
}

# Function to install system dependencies
install_system_deps() {
    print_status "Installing system dependencies..."
    
    case $OS in
        "macos")
            if ! command -v brew &> /dev/null; then
                print_error "Homebrew is not installed. Please install it first:"
                echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
                exit 1
            fi
            print_status "Installing ffmpeg and other dependencies via Homebrew..."
            brew install ffmpeg libjpeg zlib
            ;;
        "linux")
            case $PKG_MANAGER in
                "apt")
                    print_status "Installing dependencies via apt..."
                    sudo apt-get update
                    sudo apt-get install -y ffmpeg libjpeg-dev zlib1g-dev python3-dev python3-pip
                    ;;
                "yum")
                    print_status "Installing dependencies via yum..."
                    sudo yum install -y ffmpeg libjpeg-devel zlib-devel python3-devel python3-pip
                    ;;
                "dnf")
                    print_status "Installing dependencies via dnf..."
                    sudo dnf install -y ffmpeg libjpeg-devel zlib-devel python3-devel python3-pip
                    ;;
                "pacman")
                    print_status "Installing dependencies via pacman..."
                    sudo pacman -S --noconfirm ffmpeg libjpeg-turbo zlib python
                    ;;
                *)
                    print_warning "Unknown package manager. Please install ffmpeg, libjpeg, and zlib manually."
                    ;;
            esac
            ;;
        *)
            print_warning "Unknown operating system. Please install ffmpeg, libjpeg, and zlib manually."
            ;;
    esac
}

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed"
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 is not installed"
    exit 1
fi

# Detect OS and package manager
detect_os

# Install system dependencies
install_system_deps

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_status "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install requirements
print_status "Installing Python dependencies..."
pip install -r requirements.txt

# Check if user wants to install dev requirements
if [ "$1" = "--dev" ] || [ "$1" = "-d" ]; then
    print_status "Installing development dependencies..."
    pip install -r requirements-dev.txt
fi

# Verify ffmpeg installation
print_status "Verifying ffmpeg installation..."
if command -v ffmpeg &> /dev/null; then
    FFMPEG_VERSION=$(ffmpeg -version 2>&1 | head -n1 | cut -d' ' -f3)
    print_success "ffmpeg is installed (version: $FFMPEG_VERSION)"
else
    print_error "ffmpeg installation failed or not found in PATH"
    exit 1
fi

# Verify ffprobe installation
if command -v ffprobe &> /dev/null; then
    FFPROBE_VERSION=$(ffprobe -version 2>&1 | head -n1 | cut -d' ' -f3)
    print_success "ffprobe is installed (version: $FFPROBE_VERSION)"
else
    print_warning "ffprobe not found, but ffmpeg should include it"
fi

# Test ffmpeg functionality
print_status "Testing ffmpeg functionality..."
if ffmpeg -f lavfi -i testsrc=duration=1:size=320x240:rate=1 -f null - 2>/dev/null; then
    print_success "ffmpeg is working correctly"
else
    print_warning "ffmpeg test failed, but installation may still be functional"
fi

# Create data directory if it doesn't exist
if [ ! -d "data" ]; then
    print_status "Creating data directory..."
    mkdir data
fi

# Test Python imports
print_status "Testing Python dependencies..."
python3 -c "
import sys
try:
    import flask
    import PIL
    import imagehash
    import werkzeug
    print('✓ All Python dependencies imported successfully')
except ImportError as e:
    print(f'✗ Failed to import: {e}')
    sys.exit(1)
"

print_success "Installation complete!"
print_status "To activate the virtual environment, run: source venv/bin/activate"
print_status "To run the application, run: python -m deduper"
print_status "To run with development dependencies, use: ./install.sh --dev"
