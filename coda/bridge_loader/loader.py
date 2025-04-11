import tensorflow as tf
from typing import List, Optional, Dict
from pathlib import Path
from tqdm import tqdm
import requests
import json
import os

class BridgeDatasetLoader:
    """Loader for the Bridge v2 dataset."""
    
    def __init__(self, base_url: str = "https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0/",
                 local_dir: str = "~/reasoning-embodied-test/data/bridge", chunk_size: int = 8192, max_retries: int = 3):
        """
        Initialize the dataset loader.
        
        Args:
            base_url: Base URL for the dataset files
            local_dir: Local directory to store downloaded files (default: ~/reasoning-embodied-test/data/bridge)
            chunk_size: Size of chunks for downloading files
            max_retries: Maximum number of retries for failed downloads
        """
        self.base_url = base_url
        # Expand the home directory path
        self.local_dir = Path(os.path.expanduser(local_dir))
        self.chunk_size = chunk_size
        self.max_retries = max_retries
        
        # Create local directory structure
        self.local_dir.mkdir(parents=True, exist_ok=True)
        
        # Split configuration
        self.split_config = {
            "train": {
                "prefix": "bridge_dataset-train.tfrecord",
                "num_shards": 1024,
                "padding": 5
            },
            "val": {
                "prefix": "bridge_dataset-val.tfrecord",
                "num_shards": 128,
                "padding": 5
            }
        }
    
    def _download_file(self, url: str, local_path: Path) -> bool:
        """
        Download a single file with progress tracking.
        
        Args:
            url: URL of the file to download
            local_path: Local path to save the file
            
        Returns:
            bool: True if download was successful, False otherwise
        """
        if local_path.exists():
            print(f"File already exists: {local_path}")
            return True
            
        try:
            print(f"Attempting to download from URL: {url}")
            response = requests.get(url, stream=True)
            # print(f"Response status code: {response.status_code}")
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            # print(f"Total file size: {total_size} bytes")
            
            with open(local_path, 'wb') as f, tqdm(
                desc=local_path.name,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=self.chunk_size):
                    size = f.write(data)
                    pbar.update(size)
                    
            return True
            
        except Exception as e:
            print(f"Error downloading {url}: {str(e)}")
            if local_path.exists():
                local_path.unlink()
            return False
    
    def download_dataset(self, split: str = "train", start_index: int = 0, 
                        end_index: Optional[int] = None) -> List[Path]:
        """
        Download dataset files for a specific split.
        
        Args:
            split: Dataset split to download ("train" or "val")
            start_index: Starting shard index
            end_index: Ending shard index (inclusive)
            
        Returns:
            List of paths to downloaded files
        """
        if split not in self.split_config:
            raise ValueError(f"Invalid split: {split}. Must be one of {list(self.split_config.keys())}")
            
        config = self.split_config[split]
        if end_index is None:
            end_index = config["num_shards"] - 1
            
        # Create split directory
        split_dir = self.local_dir / split
        split_dir.mkdir(exist_ok=True)
        
        print(f"\nDownloading {split} split files {start_index} to {end_index}...")
        downloaded_files = []
        for i in range(start_index, end_index + 1):
            shard_str = str(i).zfill(config["padding"])
            total_shards = str(config["num_shards"]).zfill(config["padding"])
            filename = f"{config['prefix']}-{shard_str}-of-{total_shards}"
            url = f"{self.base_url}{filename}"
            local_path = split_dir / filename
            
            print(f"Downloading file {i+1}/{end_index+1}: {filename}")
            if self._download_file(url, local_path):
                downloaded_files.append(local_path)
                
        return downloaded_files
    
    def load_tfrecord(self, file_path: Path) -> tf.data.TFRecordDataset:
        """
        Load a TFRecord file into a TensorFlow dataset.
        
        Args:
            file_path: Path to the TFRecord file
            
        Returns:
            TensorFlow dataset
        """
        if not file_path.exists():
            raise FileNotFoundError(f"TFRecord file not found: {file_path}")
            
        return tf.data.TFRecordDataset([str(file_path)])
        
    def get_dataset_info(self) -> Dict:
        """
        Get dataset information from features.json.
        
        Returns:
            Dictionary containing dataset information
        """
        url = f"{self.base_url}features.json"
        local_path = self.local_dir / "features.json"
        
        print("\nDownloading dataset information (features.json)...")
        if not local_path.exists():
            self._download_file(url, local_path)
            
        with open(local_path, 'r') as f:
            return json.load(f)
            
    def verify_download(self, file_path: Path) -> bool:
        """
        Verify that a downloaded file is valid.
        
        Args:
            file_path: Path to the file to verify
            
        Returns:
            bool: True if file is valid, False otherwise
        """
        if not file_path.exists():
            return False
            
        try:
            # Try to read the first record
            dataset = self.load_tfrecord(file_path)
            next(iter(dataset))
            return True
        except Exception as e:
            print(f"Error verifying {file_path}: {str(e)}")
            return False
        
    def get_dataset_stats(self, file_path: Path) -> Dict:
        """
        Get statistics for a TFRecord file.
        
        Args:
            file_path: Path to the TFRecord file
            
        Returns:
            Dictionary containing file statistics
        """
        # Get dataset info
        dataset_info = self.get_dataset_info()
        
        # Create feature description from dataset info
        feature_description = {}
        for feature_name, feature_info in dataset_info['featuresDict'].items():
            # Handle nested features
            if isinstance(feature_info, dict) and 'feature' in feature_info:
                nested_feature = feature_info['feature']
                if nested_feature['type'] == 'BytesList':
                    feature_description[feature_name] = tf.io.FixedLenFeature([], tf.string)
                elif nested_feature['type'] == 'FloatList':
                    feature_description[feature_name] = tf.io.FixedLenFeature([], tf.float32)
                elif nested_feature['type'] == 'Int64List':
                    feature_description[feature_name] = tf.io.FixedLenFeature([], tf.int64)
                else:
                    print(f"Warning: Unsupported feature type {nested_feature['type']} for feature {feature_name}")
                    continue
        
        # Count examples
        dataset = tf.data.TFRecordDataset([str(file_path)])
        num_examples = sum(1 for _ in dataset)
        
        return {
            'num_examples': num_examples,
            'feature_description': feature_description
        }

    def inspect_file(self, file_path: str) -> None:
        """
        Inspect the contents of a TFRecord file.
        
        Args:
            file_path: Path to the TFRecord file to inspect
        """
        if not Path(file_path).exists():
            print(f"Error: File not found: {file_path}")
            return
            
        print(f"\nInspecting TFRecord file: {file_path}")
        
        # Create a TFRecordDataset
        dataset = tf.data.TFRecordDataset([file_path])
        
        # Get the first example
        try:
            for raw_record in dataset.take(1):
                example = tf.train.Example()
                example.ParseFromString(raw_record.numpy())
                
                print("\nFirst record features:")
                for feature_name, feature in example.features.feature.items():
                    print(f"\n{feature_name}:")
                    if feature.HasField('bytes_list'):
                        if feature_name == 'action':
                            # Action is a string
                            decoded = [val.decode('utf-8') for val in feature.bytes_list.value]
                            print(f"  Type: string")
                            print(f"  Value: {decoded}")
                        elif feature_name == 'language_embedding':
                            # Language embedding is a float array
                            print(f"  Type: float array")
                            print(f"  Shape: {len(feature.bytes_list.value)}")
                            print(f"  First few values: {[float(x) for x in feature.bytes_list.value[:5]]}")
                        elif feature_name == 'observation':
                            # Observation is an image
                            print(f"  Type: image")
                            print(f"  Size: {len(feature.bytes_list.value[0])} bytes")
                    elif feature.HasField('float_list'):
                        print(f"  Type: float")
                        print(f"  Value: {feature.float_list.value}")
                    elif feature.HasField('int64_list'):
                        print(f"  Type: int")
                        print(f"  Value: {feature.int64_list.value}")
                        
        except Exception as e:
            print(f"Error inspecting file: {str(e)}") 