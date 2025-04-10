import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys

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

def visualize_movements(states, movement_primitives):
    """Visualize the movement primitives."""
    # Plot the trajectory in 3D space
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract position data (assuming first 3 dimensions are x, y, z)
    x = states[:, 0]
    y = states[:, 1]
    z = states[:, 2]
    
    ax.plot(x, y, z, 'b-', alpha=0.5)
    ax.scatter(x, y, z, c=range(len(x)), cmap='viridis')
    
    # Add labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Trajectory with Movement Primitives')
    
    plt.savefig('trajectory_with_primitives.png')
    plt.close()
    
    # Plot movement primitive frequency
    primitive_counts = {}
    for desc, _ in movement_primitives:
        if desc in primitive_counts:
            primitive_counts[desc] += 1
        else:
            primitive_counts[desc] = 1
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(primitive_counts.keys(), primitive_counts.values())
    ax.set_xlabel('Movement Primitive')
    ax.set_ylabel('Frequency')
    ax.set_title('Movement Primitive Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('primitive_distribution.png')

def main():
    # Update the path to use the absolute path
    # Option 1: Use absolute path from project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tfrecord_path = os.path.join(project_root, "data/bridge/bridge_dataset-train.tfrecord-00002-of-01024")
    
    # Option 2: Make sure the path is correct based on your current working directory
    # If the script is run from the project root directory:
    # tfrecord_path = "data/bridge/bridge_dataset-train.tfrecord-00002-of-01024"
    
    # Print the path for debugging
    print(f"Looking for dataset at: {tfrecord_path}")
    
    # Check if the file exists
    if not os.path.exists(tfrecord_path):
        print(f"Error: File not found at {tfrecord_path}")
        print("Please check the path and make sure the file exists.")
        
        # Option 3: Try to find the file in the directory structure
        for root, dirs, files in os.walk(project_root):
            for file in files:
                if "bridge_dataset-train.tfrecord" in file:
                    potential_path = os.path.join(root, file)
                    print(f"Found a potential match at: {potential_path}")
        
        return
    
    # Step 1: Explore the dataset structure
    explore_dataset_structure(tfrecord_path)
    
    # After exploring, uncomment and modify the following code based on the actual structure
    
    # # Step 2: Define the feature description based on exploration
    # feature_description = {
    #     'observation/state': tf.io.FixedLenFeature([STATE_DIM], tf.float32),
    #     'action': tf.io.FixedLenFeature([ACTION_DIM], tf.float32),
    #     # Add other features as needed
    # }
    
    # # Step 3: Extract state and action data
    # dataset = load_dataset(tfrecord_path)
    # states, actions = extract_state_and_actions(dataset, feature_description)
    
    # # Step 4: Generate movement primitives
    # movement_primitives = generate_movement_primitives(states, actions)
    
    # # Step 5: Analyze and visualize the results
    # print(f"Total movement primitives generated: {len(movement_primitives)}")
    # visualize_movements(states, movement_primitives)
    
    # # Print the first few primitives
    # print("\nSample movement primitives:")
    # for i, (desc, _) in enumerate(movement_primitives[:10]):
    #     print(f"{i+1}. {desc}")

if __name__ == "__main__":
    main() 