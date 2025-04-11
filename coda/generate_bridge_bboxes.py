# Without SAM (acts like the lightweight version):
# python generate_bridge_bboxes.py --gpu 0 --result-path ./bridge_bboxes --data-path /home/ubuntu/reasoning-embodied-test/data/bridge/bridge_dataset-train.tfrecord-00002-of-01024

# With SAM (full functionality):
# python generate_bridge_bboxes.py --gpu 0 --result-path ./bridge_bboxes --data-path /home/ubuntu/reasoning-embodied-test/data/bridge/bridge_dataset-train.tfrecord-00002-of-01024 --use-sam

# With debugging and limited episode processing:
# python generate_bridge_bboxes.py --gpu 0 --result-path ./bridge_bboxes --data-path /home/ubuntu/reasoning-embodied-test/data/bridge/bridge_dataset-train.tfrecord-00002-of-01024 --use-sam --debug --max-episodes 2 --batch-size 3

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
import logging
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bridge_bboxes_debug.log')
    ]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

# Global variable to store dataset-specific object terms
DATASET_SPECIFIC_OBJECTS = []

class NumpyFloatValuesEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy numeric values."""
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64, np.int64, np.int32, np.uint8)):
            return float(obj) if isinstance(obj, (np.float32, np.float64)) else int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def create_detection_prompt(instruction):
    """Create a more focused detection prompt with individual object categories."""
    global DATASET_SPECIFIC_OBJECTS
    
    # Define clear individual object categories - keeping them singular
    base_objects = [
        "robot arm", "gripper", "table", "surface", "block",
        "cube", "container", "tool", "object"
    ]
    
    # Define colors and attributes as separate categories
    colors = ["red", "blue", "green", "yellow", "orange", "white", "black", "brown", "purple", "pink"]
    shapes = ["round", "square", "rectangular", "triangular", "cylindrical", "flat", "long", "short"]
    
    # Start with basic categories
    prompt_objects = base_objects.copy()
    
    # Add dataset-specific objects - keeping them separate
    for obj in DATASET_SPECIFIC_OBJECTS:
        if obj not in prompt_objects and len(obj) > 2:
            if obj not in colors and obj not in shapes:  # Don't add colors/shapes as separate objects
                prompt_objects.append(obj)
    
    # Extract mentioned objects from instruction
    if instruction:
        words = instruction.lower().replace(".", " ").replace(",", " ").split()
        
        # Look for specific objects mentioned in this instruction
        for word in words:
            if len(word) > 2 and word not in ["the", "and", "to", "of", "a", "an", "on", "in"]:
                if word not in prompt_objects and word not in colors and word not in shapes:
                    prompt_objects.append(word)
    
    # Join all objects with separator that encourages them to be detected individually
    prompt = " . ".join(prompt_objects)  # Using period as separator to encourage distinct detections
    
    logger.debug(f"Detection prompt: {prompt}")
    return prompt

def save_debug_image(image, boxes, labels, scores, masks=None, save_path=None):
    """Save debug image with bounding boxes and masks for visualization."""
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    ax = plt.gca()
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        x1, y1, x2, y2 = box
        
        # Create a more visible bounding box
        rect = plt.Rectangle(
            (x1, y1), x2-x1, y2-y1, 
            fill=False, 
            linewidth=2, 
            edgecolor='r'  # Keep the red color for the box
        )
        ax.add_patch(rect)
        
        # Improve label visibility with black background and white text
        ax.text(
            x1, y1-10 if y1 > 20 else y1+5,  # Position text above box if there's room
            f"{label}: {score:.2f}", 
            color='white',                    # White text
            backgroundcolor='black',          # Black background
            fontsize=8,
            fontweight='bold',                # Bold text for better readability
            bbox=dict(
                facecolor='black',            # Black background
                alpha=0.8,                    # Slightly transparent
                pad=3,                        # Padding around text
                edgecolor='red',              # Red edge to match the bounding box
                boxstyle='round,pad=0.3'      # Rounded corners
            )
        )
        
        # Plot mask if available
        if masks is not None and i < len(masks):
            mask = masks[i]
            if mask is not None:
                # Convert mask to boolean if it's not already
                if not isinstance(mask, np.ndarray):
                    mask = np.array(mask)
                if mask.dtype != bool:
                    mask = mask > 0.5
                
                # Create a colored overlay for the mask
                colored_mask = np.zeros((mask.shape[0], mask.shape[1], 4))
                # Use a more vibrant red with slightly higher alpha for better visibility
                colored_mask[mask] = [1, 0, 0, 0.4]  # Red with alpha=0.4
                plt.imshow(colored_mask)
    
    plt.axis('off')
    
    # Add a title with episode and step information if available
    if save_path:
        title = os.path.basename(save_path).replace('.jpg', '')
        plt.title(title, fontsize=14, color='white', backgroundcolor='black', pad=10)
        
        # Save the image with higher DPI for better quality
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
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

def process_batch(dino_model, dino_processor, sam_model, sam_processor, batch_images, 
                 prompt, device, box_threshold, text_threshold, use_sam=False,
                 debug=False, debug_dir=None, episode_id=None, start_idx=0):
    """Process a batch of images to detect objects and generate masks if requested."""
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
                inputs = dino_processor(
                    images=image,
                text=prompt,
                    return_tensors="pt",
                ).to(device)
                
                with torch.no_grad():
                    outputs = dino_model(**inputs)
                
                dino_results = dino_processor.post_process_grounded_object_detection(
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
            
            # Delete DINO tensors to free memory
            del inputs, outputs, dino_results
            
            # Clean and simplify labels - keep them focused on single objects
            bboxes = []
            for lg, p, b in zip(logits, phrases, boxes):
                # Convert numpy types to Python native types
                score = float(lg)
                box = [int(val) for val in b.astype(int)]
                
                # Simplify the label: if it's a compound, take only the main object term
                label_parts = p.split()
                if len(label_parts) > 2:
                    # If it's a compound label, try to extract the main object term
                    main_term = label_parts[0]  # Default to first term
                    
                    # Look for known object terms in the label
                    object_indicators = ["arm", "gripper", "table", "block", "cube", "tool", "object", "container"]
                    for indicator in object_indicators:
                        if indicator in label_parts:
                            main_term = indicator
                            break
                    
                    p = main_term  # Use the simplified label
                
                bboxes.append({"score": score, "label": p, "box": box})
            
            # Process with SAM if needed and if there are bounding boxes detected
            masks = []
            all_sam_masks = []
                    
            if use_sam and len(bboxes) > 0:
                for bbox in bboxes:
                    try:
                        # Convert bbox to input points for SAM
                        box = bbox["box"]
                        
                        # Clear cache before each mask generation
                        torch.cuda.empty_cache()
                        
                        # Use a smaller input size for SAM to reduce memory usage
                        sam_inputs = sam_processor(
                            image,
                            input_boxes=[[box]],
                            return_tensors="pt",
                        ).to(device)
                        
                        with torch.no_grad():
                            sam_outputs = sam_model(**sam_inputs)
                        
                        masks_tensor = sam_processor.image_processor.post_process_masks(
                            sam_outputs.pred_masks.cpu(),
                            sam_inputs["original_sizes"].cpu(),
                            sam_inputs["reshaped_input_sizes"].cpu()
                        )[0]
                        
                        # Get the mask with highest IoU with the box
                        best_mask = masks_tensor[0][0].numpy()
                        
                        # Only store downsampled masks to save memory in JSON
                        # Downsample mask to 1/4 resolution to reduce JSON size
                        h, w = best_mask.shape
                        downsampled_mask = best_mask[::4, ::4].tolist()
                        
                        masks.append({"box": box, "mask": downsampled_mask, "downsampled": True})
                        all_sam_masks.append(best_mask)  # Keep full resolution for visualization
                        
                        # Delete tensors immediately
                        del sam_inputs, sam_outputs, masks_tensor, best_mask
                        
                    except Exception as e:
                        logger.error(f"Error generating mask for box {box}: {e}")
                        masks.append({"box": box, "mask": None})
                        all_sam_masks.append(None)
            
            # Save debug image if requested
            if debug and debug_dir and episode_id is not None and i % 5 == 0:
                boxes_list = [bbox["box"] for bbox in bboxes]
                labels_list = [bbox["label"] for bbox in bboxes]
                scores_list = [bbox["score"] for bbox in bboxes]
                
                debug_path = os.path.join(debug_dir, f"ep{episode_id}_step{step_idx}.jpg")
                save_debug_image(np.array(image), boxes_list, labels_list, scores_list, 
                                 masks=all_sam_masks if use_sam else None, save_path=debug_path)
            
            # Create step result with or without masks
            step_result = {"bboxes": bboxes}
            if use_sam:
                step_result["masks"] = masks
                
            batch_results.append(step_result)
            
        except Exception as e:
            logger.error(f"Error processing step {step_idx}: {e}")
            import traceback
            traceback.print_exc()
            batch_results.append({"bboxes": [], "masks": [] if use_sam else None})
    
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

def extract_dataset_object_terms(dataset, max_samples=50):
    """Extract potential object terms from a sample of dataset instructions."""
    common_words = set()
    object_indicators = ["pick", "place", "move", "put", "stack", "grab", "hold", "lift", "push", "pull"]
    
    # Process a limited number of examples to extract terms
    sample_count = 0
    for example_proto in dataset:
        if sample_count >= max_samples:
            break
            
        try:
            parsed_data = parse_sequence_features(example_proto)
            instructions = parsed_data['instructions'].numpy()
            
            if len(instructions) > 0:
                try:
                    instruction = instructions[0].decode('utf-8').lower()
                    words = instruction.replace('.', ' ').replace(',', ' ').split()
                    
                    # Look for nouns that appear after action verbs
                    for i, word in enumerate(words):
                        if word in object_indicators and i < len(words) - 1:
                            # The word after an action verb is often an object
                            potential_object = words[i+1]
                            if len(potential_object) > 2 and potential_object not in ["the", "a", "an", "to", "up", "down"]:
                                common_words.add(potential_object)
                            
                            # Check for two-word objects (often with "the" in between)
                            if i < len(words) - 3 and words[i+1] in ["the", "a", "an"]:
                                potential_object = words[i+2]
                                if len(potential_object) > 2 and potential_object not in ["to", "up", "down"]:
                                    common_words.add(potential_object)
                except:
                    pass
                    
            sample_count += 1
        except:
            continue
    
    return list(common_words)

def save_results_to_json(results_json, output_path):
    """Save results to a JSON file with error handling and proper flushing."""
    try:
        with open(output_path, "w") as f:
            json.dump(results_json, f, cls=NumpyFloatValuesEncoder, indent=2)
            f.flush()
            os.fsync(f.fileno())  # Ensure data is flushed to disk
        logger.info(f"Results saved to {output_path}")
        return True
    except TypeError as e:
        logger.warning(f"Custom encoder failed: {e}. Using manual conversion.")
        try:
            # Fall back to manual conversion
            serializable_data = convert_to_json_serializable(results_json)
            with open(output_path, "w") as f:
                json.dump(serializable_data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            logger.info(f"Results saved to {output_path} after manual conversion")
            return True
        except Exception as e:
            logger.error(f"Failed to save results to {output_path}: {e}")
            return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--result-path", default="./bridge_bboxes", help="Path to save results")
    parser.add_argument("--data-path", type=str, default="/home/ubuntu/reasoning-embodied-test/data/bridge/bridge_dataset-train.tfrecord-00002-of-01024", 
                       help="Path to the TFRecord dataset")
    parser.add_argument("--use-sam", action="store_true", help="Whether to use SAM for segmentation")
    parser.add_argument("--debug", action="store_true", help="Save debug images with bounding boxes")
    parser.add_argument("--box-threshold", type=float, default=0.25, help="Box confidence threshold")
    parser.add_argument("--text-threshold", type=float, default=0.25, help="Text confidence threshold")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size for processing images")
    parser.add_argument("--max-episodes", type=int, default=None, help="Maximum number of episodes to process")
    parser.add_argument("--episode-ids", type=str, default=None, help="Comma-separated list of specific episode IDs to process")
    parser.add_argument("--sam-model", type=str, default="facebook/sam-vit-base", 
                       choices=["facebook/sam-vit-base", "facebook/sam-vit-small", "facebook/sam-vit-huge"],
                       help="SAM model size to use (smaller is more memory efficient for A100 40GB SXM4)")
    
    args = parser.parse_args()
    
    # Log all arguments for debugging
    logger.info(f"Starting with args: {vars(args)}")
    logger.info(f"CUDA_VISIBLE_DEVICES env var: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    
    # Print explicit message about max episodes limit
    if args.max_episodes is not None:
        print(f"\n======================================")
        print(f"LIMITING PROCESSING TO {args.max_episodes} EPISODES")
        print(f"======================================\n")
    
    # Check GPU availability
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Create result directory if it doesn't exist
    os.makedirs(args.result_path, exist_ok=True)
    result_json_path = os.path.join(args.result_path, "bridge_bboxes.json")
    
    # Log memory status before starting
    if torch.cuda.is_available():
        logger.info(f"Initial GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        logger.info(f"Initial GPU memory reserved: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
    
    # Create debug directory if needed
    if args.debug:
        debug_dir = os.path.join(args.result_path, "debug_images")
        os.makedirs(debug_dir, exist_ok=True)
    else:
        debug_dir = None
    
    # Load the dataset
    logger.info("Loading data...")
    try:
        dataset = load_dataset(args.data_path)
        logger.info("Dataset loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    
    # After loading the dataset
    logger.info("Analyzing dataset instructions to extract relevant object terms...")
    try:
        dataset_objects = extract_dataset_object_terms(dataset)
        logger.info(f"Extracted {len(dataset_objects)} potential object terms: {', '.join(dataset_objects)}")
        # Reset the dataset iterator 
        dataset = load_dataset(args.data_path)
    except Exception as e:
        logger.warning(f"Could not extract dataset-specific terms: {e}")
        dataset_objects = []
    
    # Make dataset objects available globally for the prompt creation
    global DATASET_SPECIFIC_OBJECTS
    DATASET_SPECIFIC_OBJECTS = dataset_objects
    
    # Set device
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device_str}")
    
    # Load Grounding DINO model
    logger.info("Loading Grounding DINO...")
    try:
        start_time = time.time()
        # Try to load a more memory-efficient model
        dino_model_id = "IDEA-Research/grounding-dino-tiny"
        dino_processor = AutoProcessor.from_pretrained(dino_model_id, size={"shortest_edge": 224, "longest_edge": 224})
        dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_id).to(device_str)
        logger.info(f"DINO model loaded in {time.time() - start_time:.2f} seconds.")
        if torch.cuda.is_available():
            logger.info(f"After DINO model load - GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    except Exception as e:
        logger.warning(f"Error loading tiny model: {e}")
        try:
            # Fall back to the base model if tiny is not available
            logger.info("Tiny model not found, using base model instead.")
            dino_model_id = "IDEA-Research/grounding-dino-base"
            dino_processor = AutoProcessor.from_pretrained(dino_model_id, size={"shortest_edge": 224, "longest_edge": 224})
            dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_id).to(device_str)
            if torch.cuda.is_available():
                logger.info(f"After base DINO model load - GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        except Exception as e:
            logger.error(f"Failed to load DINO models: {e}")
            return
    
    # Load SAM model if needed
    sam_model = None
    sam_processor = None
    if args.use_sam:
        logger.info(f"Loading SAM model ({args.sam_model})...")
        try:
            start_time = time.time()
            sam_model_id = args.sam_model
            sam_processor = SamProcessor.from_pretrained(sam_model_id)
            sam_model = SamModel.from_pretrained(sam_model_id).to(device_str)
            logger.info(f"SAM model loaded in {time.time() - start_time:.2f} seconds.")
            if torch.cuda.is_available():
                logger.info(f"After SAM model load - GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        except Exception as e:
            logger.error(f"Failed to load SAM model: {e}")
            args.use_sam = False
            logger.warning("Disabling SAM segmentation due to model loading error.")
    
    # Set detection thresholds
    BOX_THRESHOLD = args.box_threshold
    TEXT_THRESHOLD = args.text_threshold
    BATCH_SIZE = args.batch_size
    logger.info(f"Batch size: {BATCH_SIZE}, Box threshold: {BOX_THRESHOLD}, Text threshold: {TEXT_THRESHOLD}")
    
    # Parse specific episode IDs if provided
    target_episode_ids = None
    if args.episode_ids:
        try:
            target_episode_ids = [int(ep_id.strip()) for ep_id in args.episode_ids.split(",")]
            print(f"Will only process specific episodes: {target_episode_ids}")
        except:
            logger.warning(f"Could not parse episode IDs from '{args.episode_ids}', will process sequentially")
    
    # After setting up everything but before processing, create an empty set to track processed episode IDs
    processed_episode_ids = set()
    num_episodes_processed = 0
    
    # Results storage
    results_json = {}
    
    # Reset dataset for processing
    dataset = load_dataset(args.data_path)
    
    # Process each example in the dataset
    for ep_idx, example_proto in enumerate(dataset):
        # STRICT ENFORCEMENT: Check if we've already processed the maximum number of episodes
        if args.max_episodes is not None and num_episodes_processed >= args.max_episodes:
            logger.info(f"Reached maximum number of episodes ({args.max_episodes}), stopping.")
            print(f"\n======================================")
            print(f"COMPLETED {args.max_episodes} EPISODES - STOPPING")
            print(f"======================================\n")
            break
        
        try:
            # Parse the episode data
            parsed_data = parse_sequence_features(example_proto)
            episode_id = int(parsed_data['episode_id'].numpy())
            
            # Skip if we're targeting specific episodes and this isn't one of them
            if target_episode_ids is not None and episode_id not in target_episode_ids:
                continue
            
            # Skip this episode if we've already processed maximum number of episodes
            if args.max_episodes is not None and num_episodes_processed >= args.max_episodes:
                continue
                
            # Double-check that we haven't already processed this episode (could be duplicated in the dataset)
            if episode_id in processed_episode_ids:
                logger.info(f"Skipping duplicate episode ID: {episode_id}")
                continue
                
            # Add this episode ID to our tracking set
            processed_episode_ids.add(episode_id)
            
            # Now process the episode
            logger.info(f"Starting episode {ep_idx}, memory: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
            print(f"Starting episode: {episode_id} ({ep_idx+1}) [{num_episodes_processed+1}/{args.max_episodes if args.max_episodes else 'all'}]")
            
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
                    dino_model=dino_model,
                    dino_processor=dino_processor,
                    sam_model=sam_model,
                    sam_processor=sam_processor,
                    batch_images=batch_images,
                    prompt=detection_prompt,
                    device=device_str,
                    box_threshold=BOX_THRESHOLD,
                    text_threshold=TEXT_THRESHOLD,
                    use_sam=args.use_sam,
                    debug=args.debug,
                    debug_dir=debug_dir,
                    episode_id=episode_id,
                    start_idx=batch_start
                )
                
                # Add batch results to step results
                step_results.extend(batch_results)
                
                # Free memory - log before and after
                logger.info(f"  Before cleanup - Memory: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
                torch.cuda.empty_cache()
                gc.collect()
                logger.info(f"  After cleanup - Memory: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
            
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
            save_success = save_results_to_json(results_json, result_json_path)
            if not save_success:
                print(f"  WARNING: Could not save results after episode {episode_id}")
            
            # Increment counter AFTER successfully processing the episode
            num_episodes_processed += 1
            print(f"  Finished episode {episode_id}. Processed {len(step_results)} steps.")
            print(f"  Episodes processed: {num_episodes_processed}/{args.max_episodes if args.max_episodes else 'all'}")
            
            # ADDITIONAL SAFETY CHECK: Break if we've reached the maximum
            if args.max_episodes is not None and num_episodes_processed >= args.max_episodes:
                logger.info(f"Processed maximum number of episodes ({args.max_episodes}), stopping.")
                break
            
        except Exception as e:
            print(f"Error processing episode {ep_idx}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"All episodes processed ({num_episodes_processed} total). Results saved to {result_json_path}")

if __name__ == "__main__":
    main() 