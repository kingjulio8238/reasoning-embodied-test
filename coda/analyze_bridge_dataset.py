import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import datetime

# Add parent directory to path to import primitive_movements
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.generate_embodied_data.primitive_movements import classify_movement, describe_move

def load_dataset(tfrecord_path):
    """Load the TFRecord dataset."""
    raw_dataset = tf.data.TFRecordDataset([tfrecord_path])
    return raw_dataset

def parse_example(example_proto):
    """Parse a single example from the TFRecord."""
    try:
        # We'll try to parse the example and return its features
        # The exact feature specification will be determined after examining the data
        example = tf.train.Example.FromString(example_proto.numpy())
        return example
    except Exception as e:
        print(f"Error parsing example: {e}")
        return None

def explore_dataset_structure(tfrecord_path):
    """Explore the structure of the dataset by examining a few examples."""
    print(f"Analyzing dataset: {tfrecord_path}")
    
    dataset = load_dataset(tfrecord_path)
    
    # Look at the first example to understand the structure
    for i, example_proto in enumerate(dataset.take(1)):
        example = parse_example(example_proto)
        if example:
            print("\nExample Features:")
            features = example.features.feature
            for key in features:
                feature = features[key]
                if feature.HasField('bytes_list'):
                    print(f"  {key}: bytes[{len(feature.bytes_list.value)}]")
                elif feature.HasField('float_list'):
                    print(f"  {key}: float[{len(feature.float_list.value)}]")
                elif feature.HasField('int64_list'):
                    print(f"  {key}: int64[{len(feature.int64_list.value)}]")
    
    # Count the total number of examples
    count = 0
    for _ in dataset:
        count += 1
    print(f"\nTotal examples in dataset: {count}")

def decode_example(example_proto, feature_description):
    """Decode an example using the feature description."""
    return tf.io.parse_single_example(example_proto, feature_description)

def extract_state_and_actions(dataset, feature_description):
    """Extract state and action data from the dataset."""
    states = []
    actions = []
    
    for example_proto in dataset:
        parsed_example = decode_example(example_proto, feature_description)
        
        # Extract state and action based on the feature description
        # This will be updated after exploring the dataset structure
        # For example:
        # state = parsed_example['observation/state'].numpy()
        # action = parsed_example['action'].numpy()
        
        # states.append(state)
        # actions.append(action)
    
    return np.array(states), np.array(actions)

#TODO: just highlight/count total number of steps
def generate_movement_primitives(states, actions):
    """Generate movement primitives from state and action data."""
    movement_primitives = []
    
    # Create trajectories (windows of 4 consecutive states)
    move_trajs = [states[i:i+4] for i in range(len(states) - 3)]
    
    # Classify each trajectory into a primitive movement
    for traj in move_trajs:
        description, move_vec = classify_movement(traj)
        movement_primitives.append((description, move_vec))
    
    return movement_primitives

def visualize_episode(states, actions, episode_idx, movement_primitives=None, output_dir="episode_visualizations"):
    """Visualize a single episode's trajectory and movements."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(15, 6))
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Extract position data (assuming first 3 dimensions are x, y, z)
    x = states[:, 0]
    y = states[:, 1]
    z = states[:, 2]
    
    # Plot trajectory
    ax1.plot(x, y, z, 'b-', alpha=0.5, label='Trajectory')
    ax1.scatter(x, y, z, c=range(len(x)), cmap='viridis', s=20)
    
    # Add start and end points
    ax1.scatter(x[0], y[0], z[0], c='green', s=100, label='Start')
    ax1.scatter(x[-1], y[-1], z[-1], c='red', s=100, label='End')
    
    # Add labels and title
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'Episode {episode_idx} Trajectory')
    ax1.legend()
    
    # Movement primitive plot
    if movement_primitives:
        ax2 = fig.add_subplot(122)
        primitive_counts = {}
        for desc, _ in movement_primitives:
            if desc in primitive_counts:
                primitive_counts[desc] += 1
            else:
                primitive_counts[desc] = 1
        
        ax2.bar(primitive_counts.keys(), primitive_counts.values())
        ax2.set_xlabel('Movement Primitive')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Episode {episode_idx} Movement Primitives')
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    # Save with episode number padded with zeros for proper sorting
    plt.savefig(os.path.join(output_dir, f'episode_{episode_idx:03d}_analysis.png'))
    plt.close()

def get_dataset_path():
    #Note that this is just one specific example 
    """Get the path to the TFRecord dataset."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, "../data/bridge/train/bridge_dataset-train.tfrecord-00005-of-01024")

def get_feature_description():
    """Define the feature description for the dataset."""
    return {
        'steps/observation/state': tf.io.FixedLenSequenceFeature([7], tf.float32, allow_missing=True),
        'steps/action': tf.io.FixedLenSequenceFeature([7], tf.float32, allow_missing=True),
        'steps/observation/image_0': tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
        'steps/observation/image_1': tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
        'steps/observation/image_2': tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
        'steps/observation/image_3': tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
        'steps/language_instruction': tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
        'steps/language_embedding': tf.io.FixedLenSequenceFeature([512], tf.float32, allow_missing=True),
        'steps/is_first': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'steps/is_last': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'steps/is_terminal': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'steps/reward': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'steps/discount': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'episode_metadata/has_image_0': tf.io.FixedLenFeature([], tf.int64),
        'episode_metadata/has_image_1': tf.io.FixedLenFeature([], tf.int64),
        'episode_metadata/has_image_2': tf.io.FixedLenFeature([], tf.int64),
        'episode_metadata/has_image_3': tf.io.FixedLenFeature([], tf.int64),
        'episode_metadata/has_language': tf.io.FixedLenFeature([], tf.int64),
        'episode_metadata/episode_id': tf.io.FixedLenFeature([], tf.int64),
        'episode_metadata/file_path': tf.io.FixedLenFeature([], tf.string)
    }

def process_dataset(dataset, feature_description, num_examples=100):
    """Process the dataset and extract state and action data."""
    parsed_dataset = dataset.map(lambda x: decode_example(x, feature_description))
    
    all_states = []
    all_actions = []
    episode_states = []
    episode_actions = []
    
    for example in parsed_dataset.take(num_examples):
        state_sequence = example['steps/observation/state'].numpy()
        action_sequence = example['steps/action'].numpy()
        
        # Store complete sequences for each episode
        episode_states.append(state_sequence)
        episode_actions.append(action_sequence)
        
        # Also store flattened for overall analysis
        all_states.extend(state_sequence)
        all_actions.extend(action_sequence)
    
    return np.array(all_states), np.array(all_actions), episode_states, episode_actions

def analyze_and_visualize(states, actions, episode_states, episode_actions, tfrecord_path):
    """Analyze the data and generate visualizations."""
    print(f"\nProcessed {len(episode_states)} episodes")
    print(f"Total steps across all episodes: {len(states)}")
    print(f"State shape: {states.shape}")
    print(f"Action shape: {actions.shape}")
    
    # Create directory based on TFRecord file name
    file_name = os.path.basename(tfrecord_path)
    output_dir = f"visualizations_{file_name.replace('.tfrecord', '')}"
    print(f"\nSaving visualizations to directory: {output_dir}")
    
    # Generate movement primitives for each episode
    for i, (ep_states, ep_actions) in enumerate(zip(episode_states, episode_actions)):
        print(f"\nAnalyzing Episode {i+1}...")
        # print(f"Steps in episode: {len(ep_states)}")
        
        # Generate primitives for this episode
        movement_primitives = generate_movement_primitives(ep_states, ep_actions)
        
        # Visualize this episode
        visualize_episode(ep_states, ep_actions, i+1, movement_primitives, output_dir)
        
        # Print primitive summary
        print(f"Movement primitives in episode {i+1}:")
        primitive_counts = {}
        for desc, _ in movement_primitives:
            if desc in primitive_counts:
                primitive_counts[desc] += 1
            else:
                primitive_counts[desc] = 1
        for primitive, count in primitive_counts.items():
            print(f"  {primitive}: {count}")
    
    print(f"\nVisualizations saved in: {output_dir}")
    print("Files are named as: episode_XXX_analysis.png where XXX is the episode number")

def main():
    # Get dataset path
    tfrecord_path = get_dataset_path()
    print(f"Looking for dataset at: {tfrecord_path}")
    
    # Check if file exists
    if not os.path.exists(tfrecord_path):
        print(f"Error: File not found at {tfrecord_path}")
        return
    
    # Explore dataset structure
    explore_dataset_structure(tfrecord_path)
    
    # Get feature description
    feature_description = get_feature_description()
    
    # Load and process dataset
    dataset = load_dataset(tfrecord_path)
    states, actions, episode_states, episode_actions = process_dataset(dataset, feature_description)
    
    # Analyze and visualize results
    analyze_and_visualize(states, actions, episode_states, episode_actions, tfrecord_path)

if __name__ == "__main__":
    main() 