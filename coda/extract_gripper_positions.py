# No SAM (memory efficient ; worse accuracy) 
# python coda/extract_gripper_positions.py --gpu 0 --output-dir coda/gripper_positions --data-path /home/ubuntu/reasoning-embodied-test/data/bridge/bridge_dataset-train.tfrecord-00002-of-01024 
# TODO: add max-epsiodes arg 

import os
import json
import cv2
import numpy as np
import torch
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from transformers import pipeline
from sklearn.linear_model import RANSACRegressor
import gc

# Set constants
IMAGE_DIMS = (256, 256)
GRIPPER_PROMPT = "robotic gripper, robot hand, gripper"
BATCH_SIZE = 1  # Process this many images at once

def load_dataset(tfrecord_path):
    """Load the TFRecord dataset."""
    raw_dataset = tf.data.TFRecordDataset([tfrecord_path])
    return raw_dataset

def parse_sequence_features(example_proto):
    """Parse the sequence features from the example."""
    feature_description = {
        'steps/observation/image_0': tf.io.VarLenFeature(tf.string),
        'steps/observation/state': tf.io.VarLenFeature(tf.float32),
        'episode_metadata/episode_id': tf.io.FixedLenFeature([1], tf.int64),
    }
    
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    
    # Convert sparse tensors to dense
    images = tf.sparse.to_dense(parsed_features['steps/observation/image_0'])
    states = tf.sparse.to_dense(parsed_features['steps/observation/state'])
    episode_id = parsed_features['episode_metadata/episode_id'][0]
    
    return {
        'images': images,
        'states': states,
        'episode_id': episode_id
    }

def load_owlvit_model(device, verbose=True):
    """Load OWLv2 model."""
    if verbose:
        print("Loading OWLv2 model...")
    
    # Load OWLv2 for zero-shot object detection
    owlv2_checkpoint = "google/owlvit-base-patch16"
    detector = pipeline(model=owlv2_checkpoint, task="zero-shot-object-detection", device=device)
    
    if verbose:
        print("OWLv2 model loaded successfully!")
    
    return detector

def get_bounding_boxes(img, detector, prompt=GRIPPER_PROMPT, threshold=0.01):
    """Get bounding boxes for the gripper using OWLv2."""
    # Ensure the image is in PIL format
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    
    predictions = detector(img, candidate_labels=[prompt], threshold=threshold)
    return predictions

def get_center_from_bbox(bbox):
    """Get the center point of a bounding box."""
    x_center = (bbox["xmin"] + bbox["xmax"]) / 2
    y_center = (bbox["ymin"] + bbox["ymax"]) / 2
    return (int(x_center), int(y_center))

def get_gripper_pos_raw(img, detector):
    """Get the gripper position from a raw image."""
    # Get bounding boxes for gripper
    predictions = get_bounding_boxes(img, detector)

    if len(predictions) > 0:
        # Sort predictions by score and take the highest
        predictions.sort(key=lambda x: x["score"], reverse=True)
        pos = get_center_from_bbox(predictions[0]["box"])
        return pos, predictions[0]
    else:
        return (-1, -1), None

def process_trajectory(images, states, detector):
    """Process an entire trajectory to get gripper positions."""
    # Get raw trajectory data
    print("Processing trajectory images...")
    raw_positions = []
    raw_predictions = []
    
    # Process images individually to manage memory
    for i, img in enumerate(tqdm(images)):
        pos, pred = get_gripper_pos_raw(img, detector)
        raw_positions.append(pos)
        raw_predictions.append(pred)
        
        # Clean up memory
        if i % 5 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    # Handle missing detections by filling with nearest valid positions
    valid_indices = [i for i, pred in enumerate(raw_predictions) if pred is not None]
    
    if not valid_indices:
        print("Warning: Gripper never detected in this trajectory")
        return None
    
    # Fill missing positions
    filled_positions = []
    for i in range(len(raw_positions)):
        if raw_predictions[i] is not None:
            # If we have a valid detection, use it
            filled_positions.append(raw_positions[i])
        else:
            # Find nearest valid detection
            distances = [abs(i - valid_idx) for valid_idx in valid_indices]
            nearest_valid_idx = valid_indices[np.argmin(distances)]
            filled_positions.append(raw_positions[nearest_valid_idx])
    
    # Create the final trajectory with states
    trajectory = [(pos, state) for pos, state in zip(filled_positions, states)]
    return trajectory

def apply_ransac(trajectory, debug=False):
    """Apply RANSAC to get corrected 2D positions from 3D states."""
    if trajectory is None or len(trajectory) < 10:
        print("Not enough valid points for RANSAC")
        return None
    
    # Extract 2D positions and 3D states
    pos_2d = np.array([traj[0] for traj in trajectory], dtype=np.float32)
    pos_3d = np.array([traj[1][:3].numpy() if isinstance(traj[1], tf.Tensor) else traj[1][:3] for traj in trajectory])
    
    # Add homogeneous coordinates
    pos_3d_h = np.concatenate([pos_3d, np.ones_like(pos_3d[:, :1])], axis=-1)
    pos_2d_h = np.concatenate([pos_2d, np.ones_like(pos_2d[:, :1])], axis=-1)
    
    try:
        # Fit RANSAC regressor
        reg = RANSACRegressor(random_state=0).fit(pos_3d_h, pos_2d_h)
        
        # Predict corrected 2D positions
        corrected_pos = reg.predict(pos_3d_h)[:, :-1].astype(int)
        
        if debug:
            print(f"RANSAC score: {reg.score(pos_3d_h, pos_2d_h)}")
        
        return corrected_pos
    except Exception as e:
        print(f"RANSAC failed: {e}")
        return None

def visualize_trajectory(images, positions, output_path):
    """Create a visualization of the gripper positions on the images."""
    # Create a directory for visualizations
    viz_dir = os.path.dirname(output_path)
    os.makedirs(viz_dir, exist_ok=True)
    
    # Create video frames with positions marked
    frames = []
    for img, pos in zip(images, positions):
        # Ensure the image is a numpy array
        if isinstance(img, tf.Tensor):
            img = img.numpy()
        
        # Draw position on image
        frame = cv2.circle(
            img.copy(), 
            (int(pos[0]), int(pos[1])), 
            radius=5, 
            color=(255, 0, 0), 
            thickness=-1
        )
        frames.append(frame)
    
    # Save the frames as a video
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, 10, (width, height))
    
    for frame in frames:
        video.write(frame)
    
    video.release()
    
    return frames

def process_episode(episode_data, detector, output_dir, episode_id, visualize=True):
    """Process a single episode to extract gripper positions."""
    # Extract images and states
    images_data = episode_data['images'].numpy()
    states_data = episode_data['states'].numpy()
    
    # Reshape states if needed
    num_steps = len(images_data)
    state_dim = len(states_data) // num_steps
    states = states_data.reshape(num_steps, state_dim)
    
    # Decode images
    images = []
    for img_data in images_data:
        try:
            img = tf.image.decode_jpeg(img_data).numpy()
            images.append(img)
        except:
            print(f"Error decoding image, using blank image instead")
            images.append(np.zeros((256, 256, 3), dtype=np.uint8))
    
    # Process trajectory
    trajectory = process_trajectory(images, states, detector)
    
    if trajectory is None:
        print(f"Failed to process trajectory for episode {episode_id}")
        return None
    
    # Apply RANSAC to get corrected positions
    corrected_positions = apply_ransac(trajectory)
    
    if corrected_positions is None:
        print(f"RANSAC failed for episode {episode_id}")
        # Fall back to original positions
        corrected_positions = np.array([traj[0] for traj in trajectory], dtype=np.int32)
    
    # Create visualization if requested
    if visualize:
        video_path = os.path.join(output_dir, f"gripper_trajectory_{episode_id}.mp4")
        visualize_trajectory(images, corrected_positions, video_path)
    
    # Prepare results
    results = {
        "episode_id": int(episode_id),
        "num_steps": len(images),
        "gripper_positions": corrected_positions.tolist()
    }
    
    # Save to JSON
    json_path = os.path.join(output_dir, f"gripper_positions_{episode_id}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract gripper positions from bridge dataset')
    parser.add_argument('--data-path', type=str, 
                        default='/home/ubuntu/embodied-CoT/data/bridge/bridge_dataset-train.tfrecord-00002-of-01024',
                        help='Path to the TFRecord dataset')
    parser.add_argument('--output-dir', type=str, default='./gripper_positions_light',
                        help='Directory to save outputs')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization videos')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    detector = load_owlvit_model(device)
    
    # Load dataset
    print(f"Loading dataset from {args.data_path}...")
    dataset = load_dataset(args.data_path)
    
    # Process all episodes
    all_results = {}
    for ep_idx, example_proto in enumerate(dataset):
        try:
            # Parse episode data
            episode_data = parse_sequence_features(example_proto)
            episode_id = int(episode_data['episode_id'].numpy())
            
            print(f"\nProcessing episode {episode_id} ({ep_idx+1})...")
            
            # Process the episode
            results = process_episode(
                episode_data, 
                detector,
                args.output_dir, 
                episode_id,
                args.visualize
            )
            
            if results:
                all_results[str(episode_id)] = results
            
            # Save aggregated results periodically
            if (ep_idx + 1) % 5 == 0 or ep_idx == 0:
                with open(os.path.join(args.output_dir, 'all_gripper_positions.json'), 'w') as f:
                    json.dump(all_results, f, indent=2)
                
                print(f"Saved results for {len(all_results)} episodes")
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"Error processing episode {ep_idx}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save final aggregated results
    with open(os.path.join(args.output_dir, 'all_gripper_positions.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nComplete! Processed {len(all_results)} episodes successfully.")
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()  