# Have to run this script for a step specified 
# python coda/parse_reasoning.py --input /home/ubuntu/reasoning-embodied-test/coda/ecot_annotations/ecot_13.json --output coda/reasoning_13.txt --step 5


"""
Parse reasoning from ECoT annotations using OpenAI's GPT models.
This script processes the raw LLM output and formats it in a consistent structure.
"""

import json
import argparse
import os
import sys
from typing import Dict, Any, Optional
from openai import OpenAI

def get_openai_client(api_key=None):
    """Initialize OpenAI client with API key."""
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        print("Error: OpenAI API key is required. Please provide it as an argument or set the OPENAI_API_KEY environment variable.")
        sys.exit(1)
    return OpenAI(api_key=key)

def parse_with_gpt(client, raw_output: str, step: Optional[int] = None) -> str:
    """
    Parse the raw reasoning output using GPT-4o-mini.
    
    Args:
        client: OpenAI client
        raw_output: Raw reasoning text from the JSON file
        step: Specific step to format (if None, formats all steps)
        
    Returns:
        Formatted reasoning text
    """
    if not raw_output or raw_output.strip() == "":
        return "No raw output found to parse."

    # Construct the prompt based on whether a specific step is requested
    if step is not None:
        prompt = f"""
        Extract and format the reasoning for step {step} from the following chain-of-thought reasoning output.
        
        IMPORTANT GUIDELINES:
        1. Never use phrases like "the robot" - use neutral or task-focused terms like "the device," "the gripper," "it," or "the system"
        2. Match exact wording and capitalization from the source text
        3. Ensure object descriptions are precise and identical to the source
        4. Match coordinates and formatting in "VISIBLE OBJECTS" and "GRIPPER POSITION" exactly
        5. Do not add additional details or omit required information
        6. Keep all formatting consistent with the source

        Format the output using EXACTLY these section headers:
        TASK: [task description]
        PLAN: [plan description]
        SUBTASK REASONING: [subtask reasoning]
        SUBTASK: [subtask description]
        MOVE REASONING: [move reasoning]
        MOVE: [move description]
        VISIBLE OBJECTS: [visible objects with coordinates]
        GRIPPER POSITION: [gripper coordinates]

        For example, the formatting should look EXACTLY like:
        TASK: Place the watermelon on the towel.

        PLAN: Move to the watermelon, grasp it, move to the towel, release the watermelon.

        SUBTASK REASONING: The watermelon is grasped and needs to be moved to the towel.

        SUBTASK: Move to the towel.

        MOVE REASONING: The towel is to the right of the watermelon, so the gripper needs to move right to reach it.

        MOVE: Move right.

        VISIBLE OBJECTS: a pink watermelon [65, 84, 135, 110], a pink watermelon [66, 83, 135, 109], the scene [4, 0, 243, 249], a yellow spoon the spoon [14, 120, 97, 150], the towel [156, 99, 230, 158], a red mushroom and the mushroom is on the table near the [107, 150, 153, 203]

        GRIPPER POSITION: [113, 91]

        Extract the exact information without any modifications, additions, or omissions, and format it exactly as shown above.

        Here's the raw reasoning output:
        
        {raw_output}
        """
    else:
        prompt = f"""
        Extract and format the main task and plan from the following chain-of-thought reasoning output.
        
        IMPORTANT GUIDELINES:
        1. Never use phrases like "the robot" - use neutral or task-focused terms like "the device," "the gripper," "it," or "the system"
        2. Match exact wording and capitalization from the source text
        3. Do not add additional details or omit required information
        4. Keep all formatting consistent with the source
        
        Format as shown below with each section clearly labeled:
        
        TASK: [task description]
        PLAN: [plan description]
        
        Extract the exact information without any modifications, additions, or omissions, and format it exactly as shown above.
        
        Here's the raw reasoning output:
        
        {raw_output}
        """

    # Call the OpenAI API
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": "You are an expert at precisely extracting and formatting chain-of-thought reasoning from robotic task instructions. Extract the relevant sections exactly as they appear in the source, without any modifications."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,  # Use deterministic output for exact matching
            max_tokens=1000
        )
        
        formatted_result = response.choices[0].message.content.strip()
        
        # Additional verification pass to ensure "the robot" is not present
        if "the robot" in formatted_result.lower():
            # Make a second call to clean up any remaining references
            cleanup_prompt = f"""
            The following formatted text still contains references to "the robot", which must be replaced with neutral terms like "the device," "the gripper," "it," or "the system".
            
            Please correct these references while maintaining exact formatting and all other content:
            
            {formatted_result}
            """
            
            cleanup_response = client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {"role": "system", "content": "You are an expert at precisely correcting and formatting text. Fix references to 'the robot' while maintaining everything else exactly as provided."},
                    {"role": "user", "content": cleanup_prompt}
                ],
                temperature=0.0,
                max_tokens=1000
            )
            
            return cleanup_response.choices[0].message.content.strip()
            
        return formatted_result
    
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return f"Error parsing output: {e}"

def parse_ecot_file(file_path: str, client, step: Optional[int] = None) -> Dict[str, Any]:
    """
    Parse an ECoT JSON file using GPT.
    
    Args:
        file_path: Path to the ECoT JSON file
        client: OpenAI client
        step: Specific step to format (if None, formats overview)
        
    Returns:
        Dictionary with parsed content and formatted result
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract the raw output text
        raw_output = data.get('raw_output', '')
        
        if not raw_output:
            print(f"Warning: No raw_output found in {file_path}")
            print("Available keys:", data.keys())
            return {"error": "No raw output found in file"}
        
        # Use GPT to parse the raw output
        formatted_result = parse_with_gpt(client, raw_output, step)
        
        # Return both the formatted result and metadata
        return {
            "metadata": data.get('metadata', {}),
            "formatted_result": formatted_result
        }
    
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description='Parse ECoT reasoning annotations using GPT.')
    parser.add_argument('--input', '-i', type=str, required=True, 
                        help='Path to ECoT annotation JSON file or directory')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path to save parsed output (if not specified, prints to console)')
    parser.add_argument('--step', '-s', type=int, default=None,
                        help='Specific step to extract reasoning for')
    parser.add_argument('--api-key', '-k', type=str, default=None,
                        help='OpenAI API key (or set OPENAI_API_KEY environment variable)')
    
    args = parser.parse_args()
    
    # Initialize OpenAI client
    client = get_openai_client(args.api_key)
    
    # Process a single file or a directory
    if os.path.isdir(args.input):
        # Process all JSON files in the directory
        files = [os.path.join(args.input, f) for f in os.listdir(args.input) 
                 if f.endswith('.json') and f.startswith('ecot_')]
        
        all_results = {}
        for file_path in files:
            episode_id = os.path.basename(file_path).replace('ecot_', '').replace('.json', '')
            parsed = parse_ecot_file(file_path, client, args.step)
            all_results[episode_id] = parsed
        
        # Save or print results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(all_results, f, indent=2)
        else:
            # Print text representation of the first file or a specific episode
            if all_results:
                first_key = next(iter(all_results))
                print(all_results[first_key].get("formatted_result", "No formatted result available"))
    
    else:
        # Process a single file
        parsed = parse_ecot_file(args.input, client, args.step)
        
        if args.output:
            with open(args.output, 'w') as f:
                if isinstance(parsed.get("formatted_result"), str):
                    f.write(parsed["formatted_result"])
                else:
                    json.dump(parsed, f, indent=2)
        else:
            print(parsed.get("formatted_result", "No formatted result available"))

if __name__ == "__main__":
    main() 