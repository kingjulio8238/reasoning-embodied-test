import json
import os
import re
import time
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import argparse
import openai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

class OpenAIAssistant:
    def __init__(self, api_key=None, model="o3-mini-2025-01-31"):
        """Initialize OpenAI Assistant with the specified model."""
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Please provide it as an argument or set the OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        print(f"Initialized OpenAI Assistant with model: {model}")

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=60))
    def generate(self, prompt):
        """Generate text using OpenAI's API with retry logic for rate limits and errors."""
        try:
            # Different parameter handling based on model
            if self.model.startswith('o3-') or self.model.startswith('o2-') or self.model.startswith('o1-'):
                # For Claude-style models
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert in robotics and embodied reasoning, providing detailed chain-of-thought annotations for robotic trajectories."},
                        {"role": "user", "content": prompt}
                    ],
                    max_completion_tokens=4000
                )
            else:
                # For GPT models
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert in robotics and embodied reasoning, providing detailed chain-of-thought annotations for robotic trajectories."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=4000
                )
            
            return response.choices[0].message.content
            
        except openai.RateLimitError:
            print("Rate limit reached. Retrying after exponential backoff...")
            raise  # Let tenacity handle the retry
        except Exception as e:
            print(f"Error during OpenAI API call: {e}")
            raise

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
    states = tf.sparse.to_dense(parsed_features['steps/observation/state'])
    instructions = tf.sparse.to_dense(parsed_features['steps/language_instruction'])
    episode_id = parsed_features['episode_metadata/episode_id'][0]
    
    return {
        'states': states,
        'instructions': instructions,
        'episode_id': episode_id
    }

def load_bboxes(bbox_path, episode_id):
    """Load bounding box data for a specific episode."""
    with open(bbox_path, 'r') as f:
        bbox_data = json.load(f)
    
    if str(episode_id) in bbox_data:
        return bbox_data[str(episode_id)]
    else:
        print(f"Warning: No bounding box data found for episode {episode_id}")
        return None

def load_gripper_positions(gripper_path, episode_id):
    """Load gripper position data for a specific episode."""
    filename = f"gripper_positions_{episode_id}.json"
    full_path = os.path.join(gripper_path, filename)
    
    if os.path.exists(full_path):
        with open(full_path, 'r') as f:
            return json.load(f)
    else:
        # Try loading from the combined file
        combined_path = os.path.join(gripper_path, "all_gripper_positions.json")
        if os.path.exists(combined_path):
            with open(combined_path, 'r') as f:
                all_data = json.load(f)
                if str(episode_id) in all_data:
                    return all_data[str(episode_id)]
        
        print(f"Warning: No gripper position data found for episode {episode_id}")
        return None

def load_movement_primitives(primitives_path, episode_id):
    """Load movement primitives for a specific episode."""
    # First, try loading the episode-specific JSON file
    filename = f"episode_{episode_id}_primitives.json"
    full_path = os.path.join(primitives_path, filename)
    
    if os.path.exists(full_path):
        try:
            with open(full_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading primitives JSON for episode {episode_id}: {e}")
    
    # If that fails, try loading from the pickle file as fallback
    pkl_path = os.path.join(primitives_path, "move_actions.pkl")
    if os.path.exists(pkl_path):
        try:
            import pickle
            with open(pkl_path, 'rb') as f:
                all_move_data = pickle.load(f)
                
                # Create a simulated primitives structure with all possible movements
                primitives = []
                for key in all_move_data.keys():
                    primitives.append({"description": key})
                
                # Save this as a JSON file for future use
                output_json = {"primitives": primitives}
                with open(full_path, 'w') as f:
                    json.dump(output_json, f, indent=2)
                
                return output_json
        except Exception as e:
            print(f"Error loading primitives pickle for episode {episode_id}: {e}")
    
    # If we still don't have primitives, create a default set
    print(f"Warning: No movement primitives found for episode {episode_id}")
    # Create some basic movement primitives as a fallback
    default_moves = ["move_left", "move_right", "move_up", "move_down", 
                    "move_forward", "move_backward", "grasp", "release", "stop"]
    return {"primitives": [{"description": move} for move in default_moves]}

def build_prompt(features, language_instruction, caption=None, list_only_moves=False):
    """Build the prompt for generating chain-of-thought annotations."""
    structured_features = "{\n"

    keys = list(features.keys())

    for i in range(len(features[keys[0]])):
        if list_only_moves:
            structured_features = structured_features + f'    {i}: "{features["move_primitive"][i]}"\n'
        else:
            structured_features = structured_features + f'    {i}: {"{"}\n'

            for key in keys:
                feature_value = features[key][i]
                if isinstance(feature_value, str):
                    feature_value = f'"{feature_value}"'

                structured_features = structured_features + f'        "{key}": {feature_value},\n'

            structured_features = structured_features + "    },\n"

    structured_features = structured_features + "}"

    if list_only_moves:
        features_desc = (
            "Each entry in that dictionary corresponds to a single step on the "
            "trajectory and describes the move that is about to be executed."
        )
    else:
        features_desc = (
            "Each entry in that dictionary corresponds to a single step on "
            "the trajectory. The provided features are the following:\n"
            "\n"
            '- "state_3d" are the current 3d coordinates of the robotic arm end effector; '
            "moving forward increases the first coordinate; moving left increases the second "
            "coordinate; moving up increases the third coordinate,\n"
            '- "move_primitive" describes the move that is about to be executed,\n'
            '- "gripper_position" denotes the location of the gripper in the 256x256 image observation,\n'
            '- "objects" lists the detected objects in the scene with their bounding boxes'
        )

    if caption is None:
        caption = ""
    else:
        caption = f"""## Scene description

The robot is operating in the following environment. {caption}

"""

    break_line = ""  # for line formatting

    return f"""# Annotate the training trajectory with reasoning

## Specification of the experimental setup

You're an expert reinforcement learning researcher. You've trained an optimal policy for controlling a robotic arm. The
robot successfully completed a task specified by the instruction: "{language_instruction}". For that purpose, the
robotic arm executed a sequence of actions. Consecutive moves that were executed are the following:


```python
trajectory_features = {structured_features}
```

{features_desc}

{caption}## Your objective

I want you to annotate the given trajectory with reasoning. That is, for each step, I need to know not only {
break_line}which action should be chosen, but importantly what reasoning justifies that action choice. I want you to {
break_line}be descriptive and include all the relevant information available. The reasoning should include the task {
break_line}to complete, the remaining high-level steps, the high-level movements that should be executed and why they {
break_line}are required, the premises that allow inferring the direction of each move, including the locations of {
break_line}relevant objects, possible obstacles or difficulties to avoid, and any other relevant justification.

### Begin by describing the task

Start by giving an overview of the task. Make it more comprehensive than the simple instruction. Include the activity, {
break_line}the objects the robotic arm interacts with, and their relative locations in the environment. Then, describe {
break_line}the high-level movements that were most likely executed, based on the task that was completed and the {
break_line}primitive movements that were executed. Then, for each high-level movement write the interval of steps that {
break_line}movement consists of. Also, for each high-level movement write a justification for why it should be {
break_line}executed. Write an answer for this part using markdown and natural language. Be descriptive and highlight {
break_line}all the relevant details, but ensure that your description is consistent with the trajectory that was {
break_line}executed, specified by the features listed above in the `trajectory_features` dictionary.

### List the reasonings for each step

Finally, for each step describe the reasoning that allows to determine the correct action. For each step describe the {
break_line}remaining part of the objective, the current progress, the objects that are still relevant for determining {
break_line}the plan, and the plan for the next steps, based on the available features. Start the reasoning from a high {
break_line}level and gradually add finer features. I need you to be descriptive and very precise. Ensure that the {
break_line}reasoning is consistent with the task and the executed trajectory. Write the answer for this part as a {
break_line}Python-executable dictionary. For every step in the initial trajectory there should be exactly one separate {
break_line}item of the form <step id>:<reasoning>. Do not group the answers. The final dictionary should have exactly {
break_line}the same set of integer keys as the dictionary of features provided in the `trajectory_features` dictionary {
break_line}above. The reasoning should be a single string that describes the reasoning in natural language and {
break_line}includes all the required features.

Each reasoning string should have the following form:
- Describe the full task that remains to be completed (but only describe what remains), and place it inside a {
break_line}tag <task>.
- Describe the complete high-level plan for completing the remaining task (the list of remaining high-level steps), {
break_line}and place it inside a tag <plan>.
- Describe the high-level step that should be executed now (chosen from the list of high-level steps), and place it {
break_line}inside a tag <subtask>.
- Describe why the chosen high-level step should be executed now, which features of the current environment influence {
break_line}that decision, and how it should be done. Place it within a tag <subtask_reason>.
- Describe the current primitive movement of the arm that needs to be executed, and place it inside a tag <move>.
- Describe why the chosen movement should be executed now and which features of the current environment influence that {
break_line}decision. Place it inside a tag <move_reason>.

## Task summary

Here is a breakdown of what needs to be done:

- Describe the task.
- Describe the high-level movements that were executed, based on the completed task and the listed features.
- Describe the plan for the solution that allowed the robot to complete the task successfully.
- For each step on the trajectory, describe the reasoning that leads to determining the correct action. The reasoning {
break_line}should be descriptive and precise. You should provide exactly one reasoning string for each step on the {
break_line}trajectory specified by `trajectory_features`.
- At the very end of the response, write a single label FINISHED to indicate that the answer is complete."""

def find_task_occurrences(input_string, tags):
    """Find occurrences of tagged content in the reasoning output."""
    pattern = r"(\d+):"
    for tag in tags:
        pattern = pattern + r"\s*<" + tag + r">([^<]*)<\/" + tag + ">"

    matches = re.findall(pattern, input_string)
    return matches

def extract_reasoning_dict(reasoning_output, tags=("task", "plan", "subtask", "subtask_reason", "move", "move_reason")):
    """Extract structured reasoning from the LLM output."""
    if reasoning_output is None:
        return dict()

    trajectory = dict()

    matches = find_task_occurrences(reasoning_output, tags)

    for match in matches:
        trajectory[int(match[0])] = dict(zip(tags, match[1:]))

    return trajectory

def get_reasoning_dict(features, metadata, llm):
    """Get reasoning dictionary from the LLM."""
    language_instruction = metadata["language_instruction"]
    caption = metadata.get("caption", None)

    prompt = build_prompt(features, language_instruction, caption=caption, list_only_moves=False)
    print(f"\nGenerating reasoning for episode {metadata['episode_id']} with instruction: {language_instruction}")
    
    reasoning_output = llm.generate(prompt)
    
    # Check if FINISHED is in the output
    if "FINISHED" not in reasoning_output:
        print("Warning: LLM response may be incomplete. 'FINISHED' marker not found.")
    
    reasoning_dict = extract_reasoning_dict(reasoning_output)
    print(f"Extracted reasoning for {len(reasoning_dict)} steps")
    
    return reasoning_dict, reasoning_output

def build_features_dict(episode_id, tfrecord_path, bboxes_path, primitives_path, gripper_path):
    """Build features dictionary for an episode."""
    # Load the dataset
    dataset = load_dataset(tfrecord_path)
    
    # Find the specific episode
    episode_data = None
    for example_proto in dataset:
        parsed_data = parse_sequence_features(example_proto)
        if int(parsed_data['episode_id'].numpy()) == episode_id:
            episode_data = parsed_data
            break
    
    if episode_data is None:
        print(f"Error: Episode {episode_id} not found in the dataset")
        return None, None
    
    # Extract states
    states_data = episode_data['states'].numpy()
    
    # Determine the state dimension dynamically
    total_elements = states_data.shape[0]
    
    # First, try to determine how many steps we have
    # We can use the length of the instruction tensor as a guide
    instructions = episode_data['instructions'].numpy()
    if len(instructions) > 0:
        num_steps = len(instructions)
    else:
        # If that doesn't work, we need to estimate from the states
        # Try common dimensions
        for dim in [189, 266, 133]:
            if total_elements % dim == 0:
                state_dim = dim
                num_steps = total_elements // state_dim
                print(f"Detected {num_steps} steps with state dimension {state_dim}")
                break
        else:
            # If no common dimension works, just use the first 3 dimensions
            # and estimate the number of steps
            state_dim = 3  # Just use XYZ coordinates
            num_steps = total_elements // state_dim
            print(f"Using estimated {num_steps} steps with state dimension {state_dim}")
    
    # Reshape states based on detected dimensions
    try:
        states_3d = states_data.reshape(-1, 3)[:num_steps].tolist()
    except ValueError:
        # If reshaping fails, just take the first 3 values of each step
        states_3d = []
        for i in range(0, min(num_steps * 3, total_elements), 3):
            if i + 2 < total_elements:
                states_3d.append(states_data[i:i+3].tolist())
            else:
                # Pad with zeros if we don't have enough data
                states_3d.append(states_data[i:].tolist() + [0] * (3 - (total_elements - i)))
    
    # Extract language instruction
    if len(instructions) > 0:
        language_instruction = instructions[0].decode('utf-8')
    else:
        language_instruction = ""
    
    # Load movement primitives
    movement_data = load_movement_primitives(primitives_path, episode_id)
    
    # Load gripper positions
    gripper_data = load_gripper_positions(gripper_path, episode_id)
    
    # Load bounding boxes
    bbox_data = load_bboxes(bboxes_path, episode_id)
    
    # Build features dictionary
    features = {}
    
    # Add 3D state
    features["state_3d"] = states_3d
    
    # Add movement primitives
    if movement_data:
        primitives = [p["description"] for p in movement_data["primitives"]]
        # Ensure we have a primitive for each step
        if len(primitives) < num_steps:
            primitives.extend(["stop"] * (num_steps - len(primitives)))
        features["move_primitive"] = primitives[:num_steps]
    else:
        features["move_primitive"] = ["unknown"] * num_steps
    
    # Add gripper positions
    if gripper_data:
        positions = gripper_data["gripper_positions"]
        features["gripper_position"] = positions
    else:
        features["gripper_position"] = [[0, 0]] * num_steps
    
    # Add object bounding boxes
    if bbox_data:
        objects = []
        for step_data in bbox_data["steps"]:
            step_objects = []
            for bbox in step_data["bboxes"]:
                obj = {
                    "label": bbox["label"],
                    "box": bbox["box"],
                    "score": bbox["score"]
                }
                step_objects.append(obj)
            objects.append(step_objects)
        features["objects"] = objects
    else:
        features["objects"] = [[]] * num_steps
    
    # Build metadata
    metadata = {
        "episode_id": episode_id,
        "language_instruction": language_instruction,
        "n_steps": num_steps
    }
    
    # Ensure all feature lists have the same length
    min_length = min(len(features[key]) for key in features)
    for key in features:
        features[key] = features[key][:min_length]
    
    return features, metadata

def generate_reasoning(episode_id, tfrecord_path, bboxes_path, primitives_path, gripper_path, llm, output_path):
    """Generate reasoning for a single episode."""
    features, metadata = build_features_dict(
        episode_id, tfrecord_path, bboxes_path, primitives_path, gripper_path
    )
    
    if not features or not metadata:
        print(f"Error: Could not build features for episode {episode_id}")
        return None
    
    try:
        reasoning_dict, raw_output = get_reasoning_dict(features, metadata, llm)
        
        entry = {
            "reasoning": reasoning_dict,
            "features": features,
            "metadata": metadata,
            "raw_output": raw_output
        }
        
        # Save individual result
        episode_path = os.path.join(output_path, f"ecot_{episode_id}.json")
        with open(episode_path, 'w') as f:
            json.dump(entry, f, indent=2)
        
        return entry
    
    except Exception as e:
        print(f"Error generating reasoning for episode {episode_id}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description='Generate Embodied Chain-of-Thought annotations')
    parser.add_argument('--data-path', type=str, 
                        default='/home/ubuntu/embodied-CoT/data/bridge/bridge_dataset-train.tfrecord-00002-of-01024',
                        help='Path to the TFRecord dataset')
    parser.add_argument('--bbox-path', type=str, 
                        default='/home/ubuntu/embodied-CoT/lightweight_bridge_bboxes/lightweight_bridge_bboxes.json',
                        help='Path to the bounding box data')
    parser.add_argument('--primitives-path', type=str, 
                        default='/home/ubuntu/embodied-CoT/coda/primitive_movements_output',
                        help='Path to the movement primitives data')
    parser.add_argument('--gripper-path', type=str, 
                        default='/home/ubuntu/embodied-CoT/gripper_positions_light',
                        help='Path to the gripper positions data')
    parser.add_argument('--output-path', type=str, 
                        default='/home/ubuntu/embodied-CoT/coda/ecot_annotations',
                        help='Path to save the annotations')
    parser.add_argument('--model', type=str, default='o3-mini-2025-01-31',
                        help='OpenAI model to use for reasoning')
    parser.add_argument('--api-key', type=str, default=None,
                        help='OpenAI API key (or set OPENAI_API_KEY environment variable)')
    parser.add_argument('--episodes', type=str, default='all',
                        help='Episodes to process, comma-separated or "all"')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Initialize OpenAI client
    try:
        llm = OpenAIAssistant(api_key=args.api_key, model=args.model)
    except ValueError as e:
        print(f"Error initializing OpenAI client: {e}")
        print("Please provide an OpenAI API key using --api-key or by setting the OPENAI_API_KEY environment variable.")
        return
    
    # Get list of episodes to process
    if args.episodes == 'all':
        # Get all episode IDs from the gripper positions directory
        episode_ids = []
        for filename in os.listdir(args.gripper_path):
            if filename.startswith('gripper_positions_') and filename.endswith('.json'):
                try:
                    ep_id = int(filename.replace('gripper_positions_', '').replace('.json', ''))
                    episode_ids.append(ep_id)
                except:
                    continue
    else:
        episode_ids = [int(ep) for ep in args.episodes.split(',')]
    
    print(f"Processing {len(episode_ids)} episodes")
    
    # Process each episode
    all_results = {}
    for episode_id in tqdm(episode_ids):
        print(f"\nProcessing episode {episode_id}")
        
        # Check if this episode has already been processed
        output_file = os.path.join(args.output_path, f"ecot_{episode_id}.json")
        if os.path.exists(output_file):
            print(f"Episode {episode_id} already processed, skipping")
            with open(output_file, 'r') as f:
                entry = json.load(f)
            all_results[str(episode_id)] = entry
            continue
        
        # Generate reasoning
        entry = generate_reasoning(
            episode_id, 
            args.data_path, 
            args.bbox_path, 
            args.primitives_path, 
            args.gripper_path, 
            llm, 
            args.output_path
        )
        
        if entry:
            all_results[str(episode_id)] = entry
        
        # Save aggregated results periodically
        if len(all_results) % 5 == 0:
            with open(os.path.join(args.output_path, 'all_ecot_annotations.json'), 'w') as f:
                json.dump(all_results, f)
    
    # Save final aggregated results
    with open(os.path.join(args.output_path, 'all_ecot_annotations.json'), 'w') as f:
        json.dump(all_results, f)
    
    print(f"\nComplete! Generated ECoT annotations for {len(all_results)} episodes.")
    print(f"Results saved to {args.output_path}")

if __name__ == "__main__":
    main() 