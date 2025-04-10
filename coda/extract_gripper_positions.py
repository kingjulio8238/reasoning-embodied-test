#SAM Included - may not be needed for this task (test diff)

import os
import json
import cv2
import numpy as np
import torch
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from transformers import SamModel, SamProcessor, pipeline
from sklearn.linear_model import RANSACRegressor
import gc

# Set constants
IMAGE_DIMS = (256, 256)
GRIPPER_PROMPT = "robotic gripper, robot hand, gripper"
BATCH_SIZE = 5  # Process this many images at once to manage memory

def load_dataset(tfrecord_path):
    """Load the TFRecord dataset."""
    raw_dataset = tf.data.TFRecordDataset([tfrecord_path])
    return raw_dataset

def parse_sequence_features(example_proto):
    """Parse the sequence features from the example."""
    feature_description = {
        'steps/observation/image_0': tf.io.VarLenFeature(tf.string),
        'steps/observation/state': tf.io.VarLenFeature(tf.float32),
        'steps/language_instruction': tf.io.VarLenFeature(tf.string),
        'episode_metadata/episode_id': tf.io.FixedLenFeature([1], tf.int64),
    }
    
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    
    # Convert sparse tensors to dense
    images = tf.sparse.to_dense(parsed_features['steps/observation/image_0'])
    states = tf.sparse.to_dense(parsed_features['steps/observation/state'])
    instructions = tf.sparse.to_dense(parsed_features['steps/language_instruction'])
    episode_id = parsed_features['episode_metadata/episode_id'][0]
    
    return {
        'images': images,
        'states': states,
        'instructions': instructions,
        'episode_id': episode_id
    }

def load_models(device, verbose=True):
    """Load OWLv2 and SAM models."""
    if verbose:
        print("Loading OWLv2 and SAM models...")
    
    # Load OWLv2 for zero-shot object detection
    owlv2_checkpoint = "google/owlvit-base-patch16"
    detector = pipeline(model=owlv2_checkpoint, task="zero-shot-object-detection", device=device)
    
    # Load SAM model for segmentation
    sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    
    if verbose:
        print("Models loaded successfully!")
    
    return detector, sam_model, sam_processor

def get_bounding_boxes(img, detector, prompt=GRIPPER_PROMPT, threshold=0.01):
    """Get bounding boxes for the gripper using OWLv2."""
    predictions = detector(img, candidate_labels=[prompt], threshold=threshold)
    return predictions

def get_gripper_mask(img, pred, sam_model, sam_processor, device):
    """Generate a mask for the gripper using SAM."""
    box = [
        round(pred["box"]["xmin"], 2),
        round(pred["box"]["ymin"], 2),
        round(pred["box"]["xmax"], 2),
        round(pred["box"]["ymax"], 2),
    ]

    inputs = sam_processor(img, input_boxes=[[[box]]], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = sam_model(**inputs)

    mask = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
    )[0][0][0].cpu().numpy()

    # Clean up to save memory
    del inputs, outputs
    torch.cuda.empty_cache()
    gc.collect()

    return mask

def sq(w, h):
    """Create a grid of positions."""
    return np.concatenate(
        [(np.arange(w * h).reshape(h, w) % w)[:, :, None], 
         (np.arange(w * h).reshape(h, w) // w)[:, :, None]], 
        axis=-1
    )

def mask_to_pos_naive(mask):
    """Convert mask to position using a naive approach."""
    pos = sq(*IMAGE_DIMS)
    weight = pos[:, :, 0] + pos[:, :, 1]
    min_pos = np.argmax((weight * mask).flatten())

    return min_pos % IMAGE_DIMS[0] - (IMAGE_DIMS[0] / 16), min_pos // IMAGE_DIMS[0] - (IMAGE_DIMS[0] / 24)

def get_gripper_pos_raw(img, detector, sam_model, sam_processor, device):
    """Get the gripper position from a raw image."""
    # Convert to PIL Image if it's a numpy array
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    
    # Get bounding boxes for gripper
    predictions = get_bounding_boxes(img, detector)

    if len(predictions) > 0:
        # Sort predictions by score and take the highest
        predictions.sort(key=lambda x: x["score"], reverse=True)
        mask = get_gripper_mask(img, predictions[0], sam_model, sam_processor, device)
        pos = mask_to_pos_naive(mask)
    else:
        mask = np.zeros(IMAGE_DIMS)
        pos = (-1, -1)
        predictions = [None]

    return (int(pos[0]), int(pos[1])), mask, predictions[0]

def process_trajectory(images, states, detector, sam_model, sam_processor, device):
    """Process an entire trajectory to get gripper positions."""
    # Get raw trajectory data
    print("Processing trajectory images...")
    raw_trajectory = []
    
    # Process in batches to manage memory
    for i in tqdm(range(0, len(images), BATCH_SIZE)):
        batch_images = images[i:i+BATCH_SIZE]
        batch_states = states[i:i+BATCH_SIZE]
        
        for img, state in zip(batch_images, batch_states):
            pos, mask, pred = get_gripper_pos_raw(img, detector, sam_model, sam_processor, device)
            raw_trajectory.append((pos, mask, pred, state))
        
        # Clean up memory
        torch.cuda.empty_cache()
        gc.collect()
    
    # Handle missing detections by filling with nearest valid positions
    prev_found = list(range(len(raw_trajectory)))
    next_found = list(range(len(raw_trajectory)))

    prev_found[0] = 0  # First point references itself if no gripper is found
    next_found[-1] = len(raw_trajectory) - 1  # Last point references itself if no gripper is found

    # Build forward index of valid detections
    for i in range(1, len(raw_trajectory)):
        if raw_trajectory[i][2] is None:
            prev_found[i] = prev_found[i - 1]
        else:
            prev_found[i] = i

    # Build backward index of valid detections
    for i in reversed(range(0, len(raw_trajectory) - 1)):
        if raw_trajectory[i][2] is None:
            next_found[i] = next_found[i + 1]
        else:
            next_found[i] = i

    # If gripper was never found, return None
    valid_detections = [i for i, traj in enumerate(raw_trajectory) if traj[2] is not None]
    if not valid_detections:
        print("Warning: Gripper never detected in this trajectory")
        return None

    # Replace the not found positions with the closest neighbor
    filled_trajectory = []
    for i in range(0, len(raw_trajectory)):
        # Choose closest valid detection
        if raw_trajectory[i][2] is None:
            dist_to_prev = i - prev_found[i] if raw_trajectory[prev_found[i]][2] is not None else float('inf')
            dist_to_next = next_found[i] - i if raw_trajectory[next_found[i]][2] is not None else float('inf')
            
            if dist_to_prev <= dist_to_next and dist_to_prev != float('inf'):
                filled_trajectory.append(raw_trajectory[prev_found[i]])
            elif dist_to_next != float('inf'):
                filled_trajectory.append(raw_trajectory[next_found[i]])
            else:
                # If somehow there are no valid detections (shouldn't happen due to our check above)
                filled_trajectory.append(((-1, -1), np.zeros(IMAGE_DIMS), None, raw_trajectory[i][3]))
        else:
            filled_trajectory.append(raw_trajectory[i])

    return filled_trajectory

def apply_ransac(trajectory, debug=False):
    """Apply RANSAC to get corrected 2D positions from 3D states."""
    if trajectory is None or len(trajectory) < 10:
        print("Not enough valid points for RANSAC")
        return None
    
    # Extract 2D positions and 3D states
    pos_2d = np.array([traj[0] for traj in trajectory], dtype=np.float32)
    pos_3d = np.array([traj[3][:3].numpy() if isinstance(traj[3], tf.Tensor) else traj[3][:3] for traj in trajectory])
    
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
            print(f"Original positions shape: {pos_2d.shape}")
            print(f"Corrected positions shape: {corrected_pos.shape}")
        
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
        
        # Convert grayscale to RGB if needed
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[-1] == 1:
            img = np.concatenate([img, img, img], axis=-1)
        
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

def process_episode(episode_data, detector, sam_model, sam_processor, device, output_dir, episode_id, visualize=True):
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
    trajectory = process_trajectory(images, states, detector, sam_model, sam_processor, device)
    
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
    parser.add_argument('--output-dir', type=str, default='./gripper_positions',
                        help='Directory to save outputs')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--batch-size', type=int, default=5, help='Batch size for processing')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization videos')
    
    args = parser.parse_args()
    
    # Update global batch size
    global BATCH_SIZE
    BATCH_SIZE = args.batch_size
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load models
    detector, sam_model, sam_processor = load_models(device)
    
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
                sam_model, 
                sam_processor, 
                device, 
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