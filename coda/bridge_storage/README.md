# Bridge Dataset Storage System

A simple and efficient system for downloading and analyzing the Bridge v2 dataset.

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Download and Inspect Files

Download a specific range of files and inspect their contents:
```bash
# Download and inspect first 2 training files
python download.py --split train --end-index 1 --inspect

# Download and inspect first 5 validation files
python download.py --split val --end-index 4 --inspect
```

### Analyze Files

Analyze downloaded files:
```bash
# Analyze first 2 training files
python analyze.py --split train --end-index 1

# Analyze first 5 validation files
python analyze.py --split val --end-index 4
```

## Detailed Usage

### Download Script

The download script (`download.py`) provides the following options:
```bash
python download.py --help
```

Options:
- `--local-dir`: Directory to store downloaded files (default: data/bridge)
- `--split`: Dataset split to download (choices: train, val)
- `--start-index`: Starting shard index (default: 0)
- `--end-index`: Ending shard index (inclusive)
- `--inspect`: Inspect the first downloaded file

Examples:
```bash
# Download first 2 training files
python download.py --split train --end-index 1

# Download with custom directory
python download.py --local-dir my_data --split train --end-index 1

# Download and inspect files
python download.py --split train --end-index 1 --inspect
```

### Analyze Script

The analyze script (`analyze.py`) provides the following options:
```bash
python analyze.py --help
```

Options:
- `--local-dir`: Directory containing downloaded files (default: data/bridge)
- `--split`: Dataset split to analyze (choices: train, val)
- `--start-index`: Starting shard index (default: 0)
- `--end-index`: Ending shard index (inclusive)

Examples:
```bash
# Analyze first 2 training files
python analyze.py --split train --end-index 1

# Analyze with custom directory
python analyze.py --local-dir my_data --split train --end-index 1
```

### Programmatic Usage

```python
from loader import BridgeDatasetLoader

# Initialize loader
loader = BridgeDatasetLoader()

# Download files
files = loader.download_dataset(split="train", start_index=0, end_index=1)

# Get dataset info
info = loader.get_dataset_info()

# Inspect a file
loader.inspect_file("data/bridge/train/bridge_dataset-train.tfrecord-00000-of-01024")
```

## Features

- Simple file-based caching (skips re-downloading existing files)
- Progress tracking during downloads
- Basic file verification
- Dataset statistics and analysis
- Feature inspection with human-readable output

## Directory Structure

- `data/bridge/`: Default directory for downloaded files
  - `train/`: Training split files
  - `val/`: Validation split files

## Requirements

- Python 3.7+
- tensorflow>=2.12.0
- requests>=2.28.0
- tqdm>=4.65.0
- pathlib>=1.0.1
- rich>=13.3.0

## Troubleshooting

1. If you see "URL not found" errors:
   - Check your internet connection
   - Verify the dataset URL is accessible

2. If you see binary data in the inspection:
   - This is normal for certain fields like `language_embedding` and `observation`
   - The inspection will show the data type and size

3. If files fail verification:
   - Try downloading again
   - Check your disk space
   - Verify file permissions 