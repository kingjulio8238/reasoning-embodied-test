import os
import tensorflow as tf
import numpy as np
import sys
import pickle

# Add parent directory to path to import primitive_movements
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.generate_embodied_data.primitive_movements import classify_movement, describe_move

def load_dataset(tfrecord_path):
    """Load the TFRecord dataset."""
    raw_dataset = tf.data.TFRecordDataset([tfrecord_path])
    return raw_dataset

def parse_sequence_features(example_proto):
    """Parse the sequence features from the example."""
    feature_description = {
        'steps/observation/state': tf.io.VarLenFeature(tf.float32),
        'steps/action': tf.io.VarLenFeature(tf.float32),
        'steps/is_first': tf.io.VarLenFeature(tf.int64),
        'steps/is_last': tf.io.VarLenFeature(tf.int64),
        'episode_metadata/episode_id': tf.io.FixedLenFeature([1], tf.int64),
    }
    
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    
    # Convert sparse tensors to dense
    state_seq = tf.sparse.to_dense(parsed_features['steps/observation/state'])
    action_seq = tf.sparse.to_dense(parsed_features['steps/action'])
    is_first = tf.sparse.to_dense(parsed_features['steps/is_first'])
    is_last = tf.sparse.to_dense(parsed_features['steps/is_last'])
    episode_id = parsed_features['episode_metadata/episode_id'][0]
    
    return {
        'state_seq': state_seq,
        'action_seq': action_seq,
        'is_first': is_first,
        'is_last': is_last,
        'episode_id': episode_id
    }

def restructure_episode_data(parsed_data):
    """Restructure the flat sequence data into steps with state and action."""
    state_seq = parsed_data['state_seq'].numpy()
    action_seq = parsed_data['action_seq'].numpy()
    is_first = parsed_data['is_first'].numpy()
    
    # Determine the dimensionality of state and action
    num_steps = len(is_first)
    state_dim = len(state_seq) // num_steps
    action_dim = len(action_seq) // num_steps
    
    # Reshape the sequences
    states = state_seq.reshape(num_steps, state_dim)
    actions = action_seq.reshape(num_steps, action_dim)
    
    return {
        'states': states,
        'actions': actions[:, :7],  # Focus on the first 7 dimensions for movement actions
        'episode_id': parsed_data['episode_id'].numpy()
    }

def get_move_primitives_episode(episode_data):
    """Get movement primitives for an episode, similar to the original function."""
    states = episode_data['states']
    actions = episode_data['actions']
    
    move_trajs = [states[i:i+4] for i in range(len(states) - 3)]
    primitives = [classify_movement(move) for move in move_trajs]
    
    # Append the last primitive to match the number of actions
    primitives.append(primitives[-1] if primitives else (None, None))
    
    move_actions = {}
    
    # Match primitives with corresponding actions
    for (move, _), action in zip(primitives, actions):
        if move in move_actions:
            move_actions[move].append(action)
        else:
            move_actions[move] = [action]
    
    return primitives, move_actions

def process_dataset(tfrecord_path, output_dir):
    """Process the entire dataset and generate movement primitives."""
    dataset = load_dataset(tfrecord_path)
    
    # Dictionary to store all movement actions across episodes
    all_move_actions = {}
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing {tfrecord_path}...")
    
    # Process each example (episode)
    for i, example_proto in enumerate(dataset):
        try:
            # Parse and restructure the episode data
            parsed_data = parse_sequence_features(example_proto)
            episode_data = restructure_episode_data(parsed_data)
            
            # Get movement primitives in the original format
            primitives, move_actions = get_move_primitives_episode(episode_data)
            
            # Add to the overall dictionary
            for move, actions in move_actions.items():
                if move in all_move_actions:
                    all_move_actions[move].extend(actions)
                else:
                    all_move_actions[move] = actions.copy()
            
        except Exception as e:
            print(f"Error processing example {i}: {e}")
    
    # Save the move_actions dictionary
    with open(os.path.join(output_dir, 'move_actions.pkl'), 'wb') as f:
        pickle.dump(all_move_actions, f)
    
    # Print summary
    print("\nMovement primitives summary:")
    for move, actions in sorted(all_move_actions.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"{move}: {len(actions)} instances")
    
    return all_move_actions

def main():
    # Path to the TFRecord file
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tfrecord_path = os.path.join(project_root, "data/bridge/bridge_dataset-train.tfrecord-00002-of-01024")
    
    # Output directory for primitives
    output_dir = os.path.join(project_root, "coda/primitive_movements_output")
    
    # Process the dataset
    move_actions = process_dataset(tfrecord_path, output_dir)
    
    # Print total primitives
    total_actions = sum(len(actions) for actions in move_actions.values())
    print(f"\nTotal primitive movements: {total_actions}")
    print(f"Unique primitive types: {len(move_actions)}")
    print(f"\nOutput saved to {output_dir}/move_actions.pkl")

if __name__ == "__main__":
    main() 