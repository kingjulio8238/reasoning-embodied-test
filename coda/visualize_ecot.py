import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import textwrap
from PIL import Image, ImageDraw, ImageFont
import cv2
import argparse

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
    episode_id = parsed_features['episode_metadata/episode_id'][0]
    
    return {
        'images': images,
        'episode_id': episode_id
    }

def extract_images_for_episode(tfrecord_path, episode_id):
    """Extract images for a specific episode from the TFRecord file."""
    # Load the dataset
    dataset = load_dataset(tfrecord_path)
    
    # Find the specific episode
    episode_data = None
    for i, example_proto in enumerate(dataset):
        try:
            parsed_data = parse_sequence_features(example_proto)
            if int(parsed_data['episode_id'].numpy()) == episode_id:
                episode_data = parsed_data
                print(f"Found episode {episode_id} at index {i}")
                break
        except Exception as e:
            print(f"Error parsing example {i}: {e}")
    
    if episode_data is None:
        print(f"Error: Episode {episode_id} not found in the dataset")
        return None
    
    # Extract images
    images_data = episode_data['images'].numpy()
    images = []
    
    for img_data in images_data:
        # Decode the image
        try:
            img = tf.image.decode_jpeg(img_data).numpy()
            images.append(img)
        except Exception as e:
            print(f"Error decoding image: {e}")
            # Add a blank image as fallback
            images.append(np.ones((256, 256, 3), dtype=np.uint8) * 240)
    
    return images

def name_to_random_color(name):
    """Generate consistent random color for an object name."""
    return [(hash(name) // (256**i)) % 256 for i in range(3)]

def resize_pos(pos, img_size=(256, 256)):
    """Resize position to match image dimensions."""
    return [int(x) for x in pos]

def draw_gripper(img, pos_list, img_size=(256, 256)):
    """Draw gripper position on image."""
    for i, pos in enumerate(reversed(pos_list)):
        if len(pos) != 2:
            continue
        pos = resize_pos(pos, img_size)
        scale = 255 - int(255 * i / max(1, len(pos_list)))
        cv2.circle(img, pos, 6, (0, 0, 0), -1)
        cv2.circle(img, pos, 5, (scale, scale, 255), -1)

def draw_bboxes(img, bboxes, img_size=(256, 256)):
    """Draw bounding boxes on image."""
    for name, bbox in bboxes.items():
        if len(bbox) != 4:
            continue
            
        # Ensure coordinates are within image bounds
        x1, y1, x2, y2 = [max(0, min(x, size-1)) for x, size in zip(bbox[:2] + bbox[2:], img_size*2)]
        
        # Draw rectangle
        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            name_to_random_color(name),
            2
        )
        
        # Add label
        font_scale = 0.5
        font_thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(name, font, font_scale, font_thickness)[0]
        
        # Draw text background
        cv2.rectangle(
            img,
            (x1, y1 - text_size[1] - 5),
            (x1 + text_size[0], y1),
            name_to_random_color(name),
            -1
        )
        
        # Draw text
        cv2.putText(
            img,
            name,
            (x1, y1 - 5),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA
        )

def extract_reasoning_sections(raw_text):
    """Extract main sections from the raw output text."""
    sections = {}
    
    # Try to extract overview section
    if "Overview of the Task" in raw_text:
        sections["Overview"] = raw_text.split("Overview of the Task")[1].split("Reasoning for Each Trajectory Step")[0].strip()
    
    # Try to extract reasoning dictionary
    if "reasoning = {" in raw_text:
        sections["Reasoning"] = raw_text.split("reasoning = {")[1].split("}", 1)[0].strip()
    
    return sections

def create_text_image(raw_text, metadata, img_size=(480, 640)):
    """Create image with formatted text."""
    height, width = img_size
    
    # Create blank white image
    text_img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Extract instruction and episode ID
    title = f"Episode {metadata['episode_id']}: {metadata['language_instruction']}"
    
    # Extract sections from the text
    sections = extract_reasoning_sections(raw_text)
    
    # Format sections for display
    formatted_text = title + "\n\n"
    
    if "Overview" in sections:
        # Extract first few paragraphs of overview
        overview = sections["Overview"].split("\n\n")[0]
        formatted_text += "Overview:\n" + overview + "\n\n"
    
    if "Reasoning" in sections:
        # Extract reasoning for first few steps
        reasoning_lines = sections["Reasoning"].split("\n")[:5]
        formatted_text += "Step-by-step reasoning (first steps):\n" + "\n".join(reasoning_lines)
    
    # Convert to PIL Image for text drawing
    pil_img = Image.fromarray(text_img)
    draw = ImageDraw.Draw(pil_img)
    
    # Use default font
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    # Wrap and draw text
    wrapper = textwrap.TextWrapper(width=80)
    wrapped_text = []
    for line in formatted_text.split('\n'):
        if line.strip():
            wrapped_text.extend(wrapper.wrap(line))
        else:
            wrapped_text.append('')
    
    y_position = 20
    for line in wrapped_text:
        draw.text((20, y_position), line, fill=(0, 0, 0), font=font)
        y_position += 20
    
    return np.array(pil_img)

def visualize_single_step(image, gripper_pos, objects, step_idx, move_primitive):
    """Visualize a single step with annotations."""
    img = image.copy()
    
    # Add step number and movement type
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        img,
        f"Step {step_idx}: {move_primitive}",
        (10, 30),
        font,
        0.7,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )
    
    # Draw gripper position
    draw_gripper(img, [gripper_pos])
    
    # Draw bounding boxes
    if objects:
        bboxes = {obj['label']: obj['box'] for obj in objects}
        draw_bboxes(img, bboxes)
    
    return img

def visualize_ecot_annotation(json_path, tfrecord_path, output_dir, step_idx=0):
    """
    Create visualization of ECoT annotation data.
    
    Args:
        json_path: Path to ECoT annotation JSON file
        tfrecord_path: Path to TFRecord dataset
        output_dir: Directory to save visualizations
        step_idx: Specific step to visualize (default: 0)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract relevant data
    features = data['features']
    metadata = data['metadata']
    raw_text = data.get('raw_output', '')
    episode_id = metadata['episode_id']
    
    # Extract images from TFRecord
    print(f"Extracting images for episode {episode_id}...")
    images = extract_images_for_episode(tfrecord_path, episode_id)
    
    if not images:
        print(f"No images found for episode {episode_id}, creating blank canvas")
        # Create blank canvas as fallback
        images = [np.ones((256, 256, 3), dtype=np.uint8) * 240] * len(features['gripper_position'])
    
    # Ensure step_idx is valid
    if step_idx >= len(images) or step_idx >= len(features['gripper_position']):
        print(f"Warning: Requested step {step_idx} out of range. Using step 0 instead.")
        step_idx = 0
    
    # Get specific step data
    image = images[step_idx]
    gripper_pos = features['gripper_position'][step_idx]
    
    # Get objects if available for this step
    objects = features['objects'][step_idx] if step_idx < len(features['objects']) else []
    
    # Get move primitive
    move_primitive = features['move_primitive'][step_idx] if step_idx < len(features['move_primitive']) else "unknown"
    
    # Visualize single step
    vis_img = visualize_single_step(image, gripper_pos, objects, step_idx, move_primitive)
    
    # Create text image
    text_img = create_text_image(raw_text, metadata)
    
    # Combine images horizontally
    # Ensure both images have the same height
    h1, w1 = vis_img.shape[:2]
    h2, w2 = text_img.shape[:2]
    
    if h1 != h2:
        # Resize images to match heights
        scale = h2 / h1
        vis_img = cv2.resize(vis_img, (int(w1 * scale), h2))
    
    combined_img = np.concatenate([vis_img, text_img], axis=1)
    
    # Save combined image
    base_filename = os.path.basename(json_path).replace('.json', '')
    combined_path = os.path.join(output_dir, f"{base_filename}_step{step_idx}.jpg")
    cv2.imwrite(combined_path, combined_img)
    print(f"Saved visualization to {combined_path}")
    
    # Create a multi-step visualization
    num_steps = min(len(images), len(features['gripper_position']))
    step_indices = list(range(0, num_steps, max(1, num_steps // 10)))  # Sample ~10 steps
    
    multi_step_imgs = []
    for i in step_indices:
        if i < len(images):
            step_img = visualize_single_step(
                images[i], 
                features['gripper_position'][i], 
                features['objects'][i] if i < len(features['objects']) else [],
                i,
                features['move_primitive'][i] if i < len(features['move_primitive']) else "unknown"
            )
            multi_step_imgs.append(step_img)
    
    # Create grid of images (arrange in rows of 3)
    if multi_step_imgs:
        grid_rows = [multi_step_imgs[i:i+3] for i in range(0, len(multi_step_imgs), 3)]
        grid_img = None
        
        for row in grid_rows:
            # Ensure all images in the row have the same height
            max_h = max(img.shape[0] for img in row)
            resized_row = []
            for img in row:
                h, w = img.shape[:2]
                if h != max_h:
                    scale = max_h / h
                    img = cv2.resize(img, (int(w * scale), max_h))
                resized_row.append(img)
                
            row_img = np.concatenate(resized_row, axis=1)
            if grid_img is None:
                grid_img = row_img
            else:
                grid_img = np.concatenate([grid_img, row_img], axis=0)
        
        # Save multi-step grid
        grid_path = os.path.join(output_dir, f"{base_filename}_multi_step.jpg")
        cv2.imwrite(grid_path, grid_img)
        print(f"Saved multi-step visualization to {grid_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize ECoT annotations')
    parser.add_argument('--json', type=str, required=True, help='Path to ECoT annotation JSON file')
    parser.add_argument('--tfrecord', type=str, 
                      default='/home/ubuntu/embodied-CoT/data/bridge/bridge_dataset-train.tfrecord-00002-of-01024', 
                      help='Path to TFRecord dataset')
    parser.add_argument('--output', type=str, default='./visualizations', help='Output directory')
    parser.add_argument('--step', type=int, default=0, help='Specific step to visualize')
    
    args = parser.parse_args()
    
    visualize_ecot_annotation(args.json, args.tfrecord, args.output, args.step)

if __name__ == "__main__":
    main()  