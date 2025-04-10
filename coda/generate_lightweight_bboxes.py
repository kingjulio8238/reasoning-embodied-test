import argparse
import json
import os
import time
import warnings
import tensorflow as tf
import numpy as np
import torch
import gc
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

warnings.filterwarnings("ignore")

class NumpyFloatValuesEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy numeric values."""
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64, np.int64, np.int32, np.uint8)):
            return float(obj) if isinstance(obj, (np.float32, np.float64)) else int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def create_detection_prompt(instruction):
    """Create a better detection prompt based on the instruction."""
    # Common objects in robotics manipulation tasks
    robot_objects = [
        "robot arm", "gripper", "robotic hand", 
        "block", "cube", "sphere", "ball", "cylinder", 
        "container", "bowl", "basket", "box", 
        "table", "surface", "platform", "shelf",
        "cloth", "towel", "blanket", 
        "pot", "pan", "utensil", "fork", "spoon", "knife",
        "food", "cheese", "vegetable", "fruit",
        "drawer", "cabinet", "door", "handle",
        "microwave", "oven", "appliance", "laundry machine",
        "toy", "figure", "object"
    ]
    
    # Always include these general object categories
    prompt = "robot arm, gripper, objects on table, containers"
    
    # Add specific objects from instruction if present
    for obj in robot_objects:
        if obj.lower() in instruction.lower():
            prompt += f", {obj}"
    
    return prompt

def save_debug_image(image, boxes, labels, scores, save_path):
    """Save debug image with bounding boxes for visualization."""
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    ax = plt.gca()
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, linewidth=2, edgecolor='r')
        ax.add_patch(rect)
        ax.text(x1, y1, f"{label}: {score:.2f}", color='white', backgroundcolor='red', fontsize=8)
    
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()

def load_dataset(tfrecord_path):
    """Load the TFRecord dataset."""
    raw_dataset = tf.data.TFRecordDataset([tfrecord_path])
    return raw_dataset

def parse_sequence_features(example_proto):
    """Parse the sequence features from the example."""
    feature_description = {
        'steps/observation/image_0': tf.io.VarLenFeature(tf.string),
        'steps/language_instruction': tf.io.VarLenFeature(tf.string),
        'episode_metadata/episode_id': tf.io.FixedLenFeature([1], tf.int64),
    }
    
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    
    # Convert sparse tensors to dense
    images = tf.sparse.to_dense(parsed_features['steps/observation/image_0'])
    instructions = tf.sparse.to_dense(parsed_features['steps/language_instruction'])
    episode_id = parsed_features['episode_metadata/episode_id'][0]
    
    return {
        'images': images,
        'instructions': instructions,
        'episode_id': episode_id
    }

def process_batch(model, processor, batch_images, prompt, device, box_threshold, text_threshold, debug=False, debug_dir=None, episode_id=None, start_idx=0):
    """Process a batch of images to detect objects."""
    batch_results = []
    
    # Process each image in the batch
    for i, img_data in enumerate(batch_images):
        step_idx = start_idx + i
        try:
            # Clear GPU cache periodically
            if i % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            
            # Decode the image
            image = Image.fromarray(tf.image.decode_jpeg(img_data).numpy())
            
            # Process with Grounding DINO
            inputs = processor(
                images=image,
                text=prompt,
                return_tensors="pt",
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            dino_results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[image.size[::-1]],
            )[0]
            
            # Extract results
            logits = dino_results["scores"].cpu().numpy()
            phrases = dino_results["labels"]
            boxes = dino_results["boxes"].cpu().numpy()
            
            # Delete tensors to free memory
            del inputs, outputs, dino_results
            
            # Store bounding box results
            bboxes = []
            for lg, p, b in zip(logits, phrases, boxes):
                # Convert numpy types to Python native types
                score = float(lg)
                box = [int(val) for val in b.astype(int)]
                bboxes.append({"score": score, "label": p, "box": box})
            
            # Save debug image if requested
            if debug and debug_dir and episode_id is not None and i % 10 == 0:
                boxes_list = [bbox["box"] for bbox in bboxes]
                labels_list = [bbox["label"] for bbox in bboxes]
                scores_list = [bbox["score"] for bbox in bboxes]
                
                debug_path = os.path.join(debug_dir, f"ep{episode_id}_step{step_idx}.jpg")
                save_debug_image(np.array(image), boxes_list, labels_list, scores_list, debug_path)
            
            batch_results.append({"bboxes": bboxes})
            
        except Exception as e:
            print(f"  Error processing step {step_idx}: {e}")
            batch_results.append({"bboxes": []})
    
    return batch_results

def convert_to_json_serializable(data):
    """Convert numpy types to Python native types to ensure JSON serialization."""
    if isinstance(data, dict):
        return {k: convert_to_json_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_json_serializable(item) for item in data]
    elif isinstance(data, (np.int64, np.int32)):
        return int(data)
    elif isinstance(data, (np.float64, np.float32)):
        return float(data)
    elif isinstance(data, np.ndarray):
        return convert_to_json_serializable(data.tolist())
    else:
        return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--result-path", default="./lightweight_bridge_bboxes", help="Path to save results")
    parser.add_argument("--data-path", type=str, default="/home/ubuntu/embodied-CoT/data/bridge/bridge_dataset-train.tfrecord-00002-of-01024", 
                       help="Path to the TFRecord dataset")
    parser.add_argument("--debug", action="store_true", help="Save debug images with bounding boxes")
    parser.add_argument("--box-threshold", type=float, default=0.25, help="Box confidence threshold")
    parser.add_argument("--text-threshold", type=float, default=0.15, help="Text confidence threshold")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size for processing images")
    
    args = parser.parse_args()
    
    # Create result directory if it doesn't exist
    os.makedirs(args.result_path, exist_ok=True)
    result_json_path = os.path.join(args.result_path, "lightweight_bridge_bboxes.json")
    
    # Create debug directory if needed
    if args.debug:
        debug_dir = os.path.join(args.result_path, "debug_images")
        os.makedirs(debug_dir, exist_ok=True)
    else:
        debug_dir = None
    
    # Load the dataset
    print("Loading data...")
    dataset = load_dataset(args.data_path)
    print("Done.")
    
    # Set device
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load Grounding DINO model
    print(f"Loading Grounding DINO...")
    try:
        # Try to load a more memory-efficient model
        dino_model_id = "IDEA-Research/grounding-dino-tiny"
        processor = AutoProcessor.from_pretrained(dino_model_id, size={"shortest_edge": 224, "longest_edge": 224})
        model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_id).to(device)
    except:
        # Fall back to the base model if tiny is not available
        print("Tiny model not found, using base model instead.")
        dino_model_id = "IDEA-Research/grounding-dino-base"
        processor = AutoProcessor.from_pretrained(dino_model_id, size={"shortest_edge": 224, "longest_edge": 224})
        model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_id).to(device)
    
    print("Model loaded.")
    
    # Set detection thresholds
    BOX_THRESHOLD = args.box_threshold
    TEXT_THRESHOLD = args.text_threshold
    BATCH_SIZE = args.batch_size
    
    # Results storage
    results_json = {}
    
    # Process each episode
    num_episodes = 0
    for ep_idx, example_proto in enumerate(dataset):
        try:
            # Parse the episode data
            parsed_data = parse_sequence_features(example_proto)
            episode_id = int(parsed_data['episode_id'].numpy())
            
            print(f"Starting episode: {episode_id} ({ep_idx+1})")
            
            # Extract language instruction
            instructions = parsed_data['instructions'].numpy()
            if len(instructions) > 0:
                try:
                    language_instruction = instructions[0].decode('utf-8')
                except:
                    language_instruction = ""
            else:
                language_instruction = ""
            
            # Create detection prompt
            detection_prompt = create_detection_prompt(language_instruction)
            print(f"  Using prompt: '{detection_prompt}'")
            
            # Get all images for this episode
            images_data = parsed_data['images'].numpy()
            num_steps = len(images_data)
            
            # Process images in batches to save memory
            step_results = []
            
            for batch_start in range(0, num_steps, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, num_steps)
                print(f"  Processing batch {batch_start//BATCH_SIZE + 1}/{(num_steps + BATCH_SIZE - 1)//BATCH_SIZE}")
                
                # Extract batch
                batch_images = images_data[batch_start:batch_end]
                
                # Process batch
                batch_results = process_batch(
                    model=model,
                    processor=processor,
                    batch_images=batch_images,
                    prompt=detection_prompt,
                    device=device,
                    box_threshold=BOX_THRESHOLD,
                    text_threshold=TEXT_THRESHOLD,
                    debug=args.debug,
                    debug_dir=debug_dir,
                    episode_id=episode_id,
                    start_idx=batch_start
                )
                
                # Add batch results to step results
                step_results.extend(batch_results)
                
                # Free memory
                torch.cuda.empty_cache()
                gc.collect()
            
            # Make sure all values are JSON serializable
            episode_data = {
                "episode_id": int(episode_id),  # Convert from np.int64 to Python int
                "steps": step_results,
                "instruction": language_instruction,
                "num_steps": len(step_results),
                "prompt_used": detection_prompt
            }
            
            # Store episode results
            results_json[str(episode_id)] = episode_data
            
            # Save results after each episode
            try:
                # First try with our custom encoder
                with open(result_json_path, "w") as f:
                    json.dump(results_json, f, cls=NumpyFloatValuesEncoder, indent=2)
            except TypeError as e:
                print(f"  Warning: Custom encoder failed: {e}. Using manual conversion.")
                # If that fails, try with manual conversion
                serializable_data = convert_to_json_serializable(results_json)
                with open(result_json_path, "w") as f:
                    json.dump(serializable_data, f, indent=2)
            
            num_episodes += 1
            print(f"  Finished episode {episode_id}. Processed {len(step_results)} steps.")
            
        except Exception as e:
            print(f"Error processing episode {ep_idx}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"All episodes processed ({num_episodes} total). Results saved to {result_json_path}")

if __name__ == "__main__":
    main() 