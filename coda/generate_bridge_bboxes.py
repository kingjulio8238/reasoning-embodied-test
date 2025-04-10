#SAM Included 

import argparse
import json
import os
import time
import warnings
import tensorflow as tf
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor
import sys

warnings.filterwarnings("ignore")

class NumpyFloatValuesEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy float values."""
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def post_process_caption(description, instruction):
    """Process the description and instruction for better object detection prompts."""
    # Combine description and instruction, focus on objects
    objects = []
    
    # Extract potential objects from description and instruction
    combined_text = description + " " + instruction
    
    # Simple object extraction - you might want to improve this
    common_objects = [
        "cube", "block", "box", "cylinder", "sphere", "ball", 
        "table", "surface", "bowl", "container", "target", "gripper",
        "arm", "robot", "hand", "platform", "bridge", "stack"
    ]
    
    for obj in common_objects:
        if obj in combined_text.lower():
            objects.append(obj)
    
    # If no objects found, return a generic prompt
    if not objects:
        return "objects"
    
    return ", ".join(objects)

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--result-path", default="./bridge_bboxes", help="Path to save results")
    parser.add_argument("--data-path", type=str, default="/home/ubuntu/embodied-CoT/data/bridge/bridge_dataset-train.tfrecord-00002-of-01024", 
                       help="Path to the TFRecord dataset")
    parser.add_argument("--use-sam", action="store_true", help="Whether to use SAM for segmentation")
    
    args = parser.parse_args()
    
    # Create result directory if it doesn't exist
    os.makedirs(args.result_path, exist_ok=True)
    result_json_path = os.path.join(args.result_path, "bridge_bboxes.json")
    
    # Load the dataset
    print("Loading data...")
    dataset = load_dataset(args.data_path)
    print("Done.")
    
    # Load Grounding DINO model
    print(f"Loading Grounding DINO to device cuda:{args.gpu}...")
    dino_model_id = "IDEA-Research/grounding-dino-base"
    device = f"cuda:{args.gpu}"
    
    dino_processor = AutoProcessor.from_pretrained(dino_model_id, size={"shortest_edge": 256, "longest_edge": 256})
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_id).to(device)
    print("Done.")
    
    # Load SAM model if needed
    if args.use_sam:
        print("Loading SAM model...")
        sam_model_id = "facebook/sam-vit-base"
        sam_processor = SamProcessor.from_pretrained(sam_model_id)
        sam_model = SamModel.from_pretrained(sam_model_id).to(device)
        print("Done.")
    
    # Detection thresholds
    BOX_THRESHOLD = 0.3
    TEXT_THRESHOLD = 0.2
    
    # Results storage
    results_json = {}
    
    # Process each episode
    for ep_idx, example_proto in enumerate(dataset):
        try:
            # Parse the episode data
            parsed_data = parse_sequence_features(example_proto)
            episode_id = int(parsed_data['episode_id'].numpy())
            
            print(f"Starting episode: {episode_id}")
            
            # Initialize episode results
            if str(episode_id) not in results_json:
                results_json[str(episode_id)] = {}
            
            # Extract language instruction
            instructions = parsed_data['instructions'].numpy()
            if len(instructions) > 0:
                language_instruction = instructions[0].decode('utf-8')
            else:
                language_instruction = ""
            
            # Process each image in the episode
            start = time.time()
            images_data = parsed_data['images'].numpy()
            step_results = []
            
            for step_idx, img_data in enumerate(images_data):
                # Decode the image
                try:
                    image = Image.fromarray(tf.image.decode_jpeg(img_data).numpy())
                except:
                    print(f"Skipping step {step_idx} due to image decoding error")
                    step_results.append({"bboxes": [], "masks": [] if args.use_sam else None})
                    continue
                
                # Create a description for object detection
                description = language_instruction
                
                # Process with Grounding DINO
                inputs = dino_processor(
                    images=image,
                    text=post_process_caption(description, language_instruction),
                    return_tensors="pt",
                ).to(device)
                
                with torch.no_grad():
                    outputs = dino_model(**inputs)
                
                dino_results = dino_processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=BOX_THRESHOLD,
                    text_threshold=TEXT_THRESHOLD,
                    target_sizes=[image.size[::-1]],
                )[0]
                
                logits, phrases, boxes = (
                    dino_results["scores"].cpu().numpy(),
                    dino_results["labels"],
                    dino_results["boxes"].cpu().numpy(),
                )
                
                # Store bounding box results
                bboxes = []
                for lg, p, b in zip(logits, phrases, boxes):
                    b = list(b.astype(int))
                    lg = float(lg)
                    bboxes.append({"score": lg, "label": p, "box": b})
                
                step_result = {"bboxes": bboxes}
                
                # Process with SAM if needed
                if args.use_sam and len(bboxes) > 0:
                    masks = []
                    
                    for bbox in bboxes:
                        # Convert bbox to input points for SAM
                        box = bbox["box"]
                        
                        sam_inputs = sam_processor(
                            image,
                            input_boxes=[[box]],
                            return_tensors="pt"
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
                        
                        masks.append({"box": box, "mask": best_mask.tolist()})
                    
                    step_result["masks"] = masks
                
                step_results.append(step_result)
                
                # Save intermediate results every 10 steps
                if step_idx % 10 == 0:
                    print(f"  Processed step {step_idx}")
            
            end = time.time()
            
            # Store episode results
            results_json[str(episode_id)] = {
                "episode_id": episode_id,
                "steps": step_results,
                "instruction": language_instruction,
                "num_steps": len(step_results)
            }
            
            # Save results after each episode
            with open(result_json_path, "w") as f:
                json.dump(results_json, f, cls=NumpyFloatValuesEncoder)
            
            print(f"Finished episode {episode_id}. Elapsed time: {round(end - start, 2)} seconds")
            
        except Exception as e:
            print(f"Error processing episode {ep_idx}: {e}")
    
    print(f"All episodes processed. Results saved to {result_json_path}")

if __name__ == "__main__":
    main() 