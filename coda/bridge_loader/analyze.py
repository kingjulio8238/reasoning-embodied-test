import argparse
from loader import BridgeDatasetLoader
from rich.console import Console
from typing import Optional

def analyze_dataset(loader: BridgeDatasetLoader, split: str = "train", 
                   start_index: int = 0, end_index: Optional[int] = None):
    """
    Analyze the dataset files.
    
    Args:
        loader: BridgeDatasetLoader instance
        split: Dataset split to analyze
        start_index: Starting shard index
        end_index: Ending shard index (inclusive)
    """
    console = Console()
    
    # Get dataset info
    dataset_info = loader.get_dataset_info()
    console.print("\n[bold]Dataset Information:[/bold]")
    console.print(f"Features: {list(dataset_info.keys())}")
    
    # Download files if needed
    files = loader.download_dataset(split=split, start_index=start_index, end_index=end_index)
    
    # Analyze each file
    console.print(f"\n[bold]Analyzing {len(files)} files:[/bold]")
    
    total_examples = 0
    total_size = 0
    
    for file_path in files:
        # Get file stats
        stats = loader.get_dataset_stats(file_path)
        
        # Add to totals
        total_examples += stats['num_examples']
        total_size += file_path.stat().st_size
        
        # Print file stats
        console.print(f"\n[bold]{file_path.name}:[/bold]")
        console.print(f"  Examples: {stats['num_examples']}")
        console.print(f"  Size: {file_path.stat().st_size / (1024*1024):.2f} MB")
        
        # Print feature structure
        console.print("\n  Feature Structure:")
        for feature_name, feature in stats['feature_description'].items():
            console.print(f"    {feature_name}: {feature.dtype}")
    
    # Print summary
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"Total Examples: {total_examples}")
    console.print(f"Total Size: {total_size / (1024*1024*1024):.2f} GB")
    console.print(f"Average Examples per File: {total_examples/len(files):.2f}")
    console.print(f"Average File Size: {total_size/len(files)/(1024*1024):.2f} MB")

def main():
    parser = argparse.ArgumentParser(description='Analyze Bridge Dataset files')
    parser.add_argument('--local-dir', type=str, default='test_data',
                        help='Local directory containing downloaded files')
    parser.add_argument('--split', type=str, choices=['train', 'val'], default='train',
                        help='Dataset split to analyze')
    parser.add_argument('--start-index', type=int, default=0,
                        help='Starting shard index')
    parser.add_argument('--end-index', type=int, default=None,
                        help='Ending shard index (inclusive)')
    
    args = parser.parse_args()
    
    # Initialize loader
    loader = BridgeDatasetLoader(local_dir=args.local_dir)
    
    # Run analysis
    analyze_dataset(loader, args.split, args.start_index, args.end_index)

if __name__ == "__main__":
    main() 