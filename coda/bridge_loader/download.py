import argparse
from loader import BridgeDatasetLoader

# To run from the Coda dir: python bridge_loader/download.py (no specifiaciton you run everything)
# If you want to specify which type of data, use the --split flag (takes train or val)


def main():
    parser = argparse.ArgumentParser(description='Download and inspect Bridge Dataset files')
    parser.add_argument('--local-dir', type=str, default='~/reasoning-embodied-test/data/bridge',
                        help='Local directory to store downloaded files')
    parser.add_argument('--split', type=str, choices=['train', 'val'], default='train',
                        help='Dataset split to download')
    parser.add_argument('--start-index', type=int, default=0,
                        help='Starting shard index')
    parser.add_argument('--end-index', type=int, default=None,
                        help='Ending shard index (inclusive)')
    parser.add_argument('--inspect', action='store_true',
                        help='Inspect the first downloaded file')
    
    args = parser.parse_args()
    
    # Initialize loader
    loader = BridgeDatasetLoader(local_dir=args.local_dir)
    
    # Get dataset info
    dataset_info = loader.get_dataset_info()
    print("\nDataset information:")
    print(f"Features: {list(dataset_info.keys())}")
    
    # Download files
    downloaded_files = loader.download_dataset(
        split=args.split,
        start_index=args.start_index,
        end_index=args.end_index
    )
    
    print(f"\nDownloaded {len(downloaded_files)} files")
    
    # Inspect first file if requested
    if args.inspect and downloaded_files:
        print("\nInspecting first downloaded file...")
        loader.inspect_file(str(downloaded_files[0]))

if __name__ == "__main__":
    main() 