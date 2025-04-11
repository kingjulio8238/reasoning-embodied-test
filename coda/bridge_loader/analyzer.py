import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import tensorflow as tf
from tqdm import tqdm


class BridgeDatasetAnalyzer:
    def __init__(self, data_dir: str):
        """
        Initialize the Bridge v2 dataset analyzer.
        
        Args:
            data_dir: Directory containing TFRecord files
        """
        self.data_dir = Path(data_dir)
        
    def load_tfrecord(self, file_path: Path) -> tf.data.TFRecordDataset:
        """Load a TFRecord file."""
        return tf.data.TFRecordDataset([str(file_path)])
        
    def parse_episode(self, episode_proto) -> Dict:
        """Parse a single episode from TFRecord."""
        feature_description = {
            'steps': tf.io.RaggedFeature(
                tf.io.FixedLenFeature([], tf.string)
            ),
            'episode_metadata': tf.io.FixedLenFeature([], tf.string)
        }
        
        parsed = tf.io.parse_single_example(episode_proto, feature_description)
        
        # Parse steps
        steps = tf.io.parse_tensor(parsed['steps'], tf.string)
        
        # Parse episode metadata
        metadata = tf.io.parse_tensor(parsed['episode_metadata'], tf.string)
        
        return {
            'steps': steps,
            'metadata': metadata
        }
        
    def analyze_episode(self, episode: Dict) -> Dict:
        """Analyze a single episode."""
        stats = {
            'num_steps': len(episode['steps']),
            'has_language': episode['metadata']['has_language'],
            'available_images': {
                'image_0': episode['metadata']['has_image_0'],
                'image_1': episode['metadata']['has_image_1'],
                'image_2': episode['metadata']['has_image_2'],
                'image_3': episode['metadata']['has_image_3']
            },
            'episode_id': episode['metadata']['episode_id'],
            'file_path': episode['metadata']['file_path']
        }
        
        # Analyze steps if available
        if episode['steps']:
            first_step = episode['steps'][0]
            last_step = episode['steps'][-1]
            
            stats.update({
                'first_step': {
                    'is_first': first_step['is_first'],
                    'has_language_instruction': bool(first_step['language_instruction']),
                    'has_language_embedding': bool(first_step['language_embedding']),
                    'state_shape': first_step['observation']['state'].shape,
                    'action_shape': first_step['action'].shape
                },
                'last_step': {
                    'is_last': last_step['is_last'],
                    'is_terminal': last_step['is_terminal'],
                    'reward': last_step['reward'],
                    'discount': last_step['discount']
                }
            })
            
        return stats
        
    def analyze_file(self, file_path: Path) -> Dict:
        """Analyze a single TFRecord file."""
        dataset = self.load_tfrecord(file_path)
        
        file_stats = {
            'file_path': str(file_path),
            'total_episodes': 0,
            'episodes': [],
            'summary': {
                'total_steps': 0,
                'episodes_with_language': 0,
                'episodes_with_images': {
                    'image_0': 0,
                    'image_1': 0,
                    'image_2': 0,
                    'image_3': 0
                },
                'min_steps': float('inf'),
                'max_steps': 0,
                'avg_steps': 0
            }
        }
        
        for episode_proto in tqdm(dataset, desc=f"Analyzing {file_path.name}"):
            try:
                episode = self.parse_episode(episode_proto)
                episode_stats = self.analyze_episode(episode)
                
                file_stats['episodes'].append(episode_stats)
                file_stats['total_episodes'] += 1
                
                # Update summary statistics
                file_stats['summary']['total_steps'] += episode_stats['num_steps']
                file_stats['summary']['min_steps'] = min(
                    file_stats['summary']['min_steps'],
                    episode_stats['num_steps']
                )
                file_stats['summary']['max_steps'] = max(
                    file_stats['summary']['max_steps'],
                    episode_stats['num_steps']
                )
                
                if episode_stats['has_language']:
                    file_stats['summary']['episodes_with_language'] += 1
                    
                for img in ['image_0', 'image_1', 'image_2', 'image_3']:
                    if episode_stats['available_images'][img]:
                        file_stats['summary']['episodes_with_images'][img] += 1
                        
            except Exception as e:
                print(f"Error analyzing episode in {file_path}: {e}")
                
        # Calculate average steps
        if file_stats['total_episodes'] > 0:
            file_stats['summary']['avg_steps'] = (
                file_stats['summary']['total_steps'] / file_stats['total_episodes']
            )
            
        return file_stats
        
    def analyze_dataset(
        self,
        file_pattern: str = "*.tfrecord",
        output_file: Optional[str] = None
    ) -> Dict:
        """
        Analyze the entire dataset.
        
        Args:
            file_pattern: Pattern to match TFRecord files
            output_file: Optional path to save analysis results
            
        Returns:
            Dictionary containing analysis results
        """
        files = list(self.data_dir.glob(file_pattern))
        if not files:
            raise ValueError(f"No files found matching pattern: {file_pattern}")
            
        dataset_stats = {
            'total_files': len(files),
            'files': [],
            'summary': {
                'total_episodes': 0,
                'total_steps': 0,
                'episodes_with_language': 0,
                'episodes_with_images': {
                    'image_0': 0,
                    'image_1': 0,
                    'image_2': 0,
                    'image_3': 0
                },
                'min_steps': float('inf'),
                'max_steps': 0,
                'avg_steps': 0
            }
        }
        
        for file_path in files:
            file_stats = self.analyze_file(file_path)
            dataset_stats['files'].append(file_stats)
            
            # Update summary statistics
            dataset_stats['summary']['total_episodes'] += file_stats['total_episodes']
            dataset_stats['summary']['total_steps'] += file_stats['summary']['total_steps']
            dataset_stats['summary']['min_steps'] = min(
                dataset_stats['summary']['min_steps'],
                file_stats['summary']['min_steps']
            )
            dataset_stats['summary']['max_steps'] = max(
                dataset_stats['summary']['max_steps'],
                file_stats['summary']['max_steps']
            )
            dataset_stats['summary']['episodes_with_language'] += (
                file_stats['summary']['episodes_with_language']
            )
            
            for img in ['image_0', 'image_1', 'image_2', 'image_3']:
                dataset_stats['summary']['episodes_with_images'][img] += (
                    file_stats['summary']['episodes_with_images'][img]
                )
                
        # Calculate average steps
        if dataset_stats['summary']['total_episodes'] > 0:
            dataset_stats['summary']['avg_steps'] = (
                dataset_stats['summary']['total_steps'] /
                dataset_stats['summary']['total_episodes']
            )
            
        # Save results if output file is specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(dataset_stats, f, indent=2)
                
        return dataset_stats 