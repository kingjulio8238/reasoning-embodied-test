#!/usr/bin/env python3
"""
Parse reasoning from ECoT annotations to extract structured data from tags.
This script processes the raw LLM output and extracts tagged content.
"""

import re
import json
import argparse
import os
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any


class CotTag(Enum):
    """Tags used in chain-of-thought reasoning."""
    TASK = "task"
    PLAN = "plan"
    SUBTASK = "subtask"
    SUBTASK_REASONING = "subtask_reason"
    MOVE = "move"
    MOVE_REASONING = "move_reason"
    VISIBLE_OBJECTS = "visible_objects"
    GRIPPER_POSITION = "gripper_position"
    ACTION = "action"


def get_cot_tags_list():
    """Get list of all CoT tags."""
    return [tag.value for tag in CotTag]


def extract_tags_from_text(text: str, tag_name: str) -> List[Tuple[int, str]]:
    """
    Extract all instances of a specific tag from text.
    
    Args:
        text: Raw text containing tags
        tag_name: Name of the tag to extract
        
    Returns:
        List of tuples containing (step_number, tag_content)
    """
    # Look for special Python dictionary format like: 
    # reasoning_dict = { 0: "<task>text</task> <plan>text</plan> ...", 1: "..." }
    dict_regex = r'(?:reasoning|trajectory_reasoning|reasoning_dict)\s*=\s*\{(.*?)\}'
    dict_match = re.search(dict_regex, text, re.DOTALL | re.IGNORECASE)
    
    if dict_match:
        dict_content = dict_match.group(1)
        
        # Get all entries in the dictionary (step: "content")
        entries_regex = r'(\d+)\s*:\s*"(.*?)",?\s*(?=\d+\s*:|$)'
        entries = re.findall(entries_regex, dict_content, re.DOTALL)
        
        results = []
        for step, content in entries:
            # Extract the specific tag from each entry
            tag_regex = rf'<\s*{tag_name}\s*>(.*?)<\s*/\s*{tag_name}\s*>'
            tag_match = re.search(tag_regex, content, re.DOTALL)
            
            if tag_match:
                results.append((int(step), tag_match.group(1).strip()))
        
        return results
    
    # Fallback to other extraction methods if dictionary format isn't found
    # Direct pattern matching for tags associated with step numbers
    pattern = rf'(\d+)\s*:(?:(?!\d+\s*:).)*?<\s*{tag_name}\s*>(.*?)<\s*/\s*{tag_name}\s*>'
    
    try:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        return [(int(step), content.strip()) for step, content in matches]
    except Exception as e:
        print(f"Error extracting tag '{tag_name}': {e}")
        return []


def extract_all_tags_from_text(text: str) -> Dict[str, Dict[int, str]]:
    """
    Extract all tagged content from the text.
    
    Args:
        text: Raw text containing tags
        
    Returns:
        Dictionary mapping tags to dictionaries of {step_number: content}
    """
    result = {}
    
    # Process each tag
    for tag in get_cot_tags_list():
        tag_matches = extract_tags_from_text(text, tag)
        
        # Create nested dictionary for this tag
        if tag_matches:
            result[tag] = {step: content for step, content in tag_matches}
    
    return result


def extract_high_level_overview(text: str) -> Dict[str, str]:
    """
    Extract high-level overview sections from the reasoning output.
    
    Args:
        text: Raw LLM output text
        
    Returns:
        Dictionary with overview sections
    """
    overview = {}
    
    # First, try to divide the text into sections
    # Look for the part before "reasoning for each step" or similar sections
    overview_section = text
    reasoning_section_markers = [
        "reasoning for each step", 
        "list the reasonings for each step",
        "reasoning for each trajectory step",
        "step-by-step reasoning",
        "reasoning = {"
    ]
    
    # Try to separate overview from step-by-step reasoning
    for marker in reasoning_section_markers:
        if marker.lower() in text.lower():
            parts = re.split(re.escape(marker), text, flags=re.IGNORECASE, maxsplit=1)
            if len(parts) > 1:
                overview_section = parts[0]
                break
    
    # Extract task overview
    task_patterns = [
        r"(?:^|\n)#+\s*(?:Task|Overview|Description)[^\n]*\n+(.*?)(?=\n#+|\n\d+:|$)",
        r"(?:^|\n)The task is to(.*?)(?=\n#+|\n\d+:|$)",
        r"(?:^|\n)In this task,(.*?)(?=\n#+|\n\d+:|$)",
        r"(?:^|\n)This task requires(.*?)(?=\n#+|\n\d+:|$)",
        r"(?:^|\n)The robot needs to(.*?)(?=\n#+|\n\d+:|$)"
    ]
    
    for pattern in task_patterns:
        match = re.search(pattern, overview_section, re.DOTALL | re.IGNORECASE)
        if match and match.group(1).strip():
            overview['task_overview'] = match.group(1).strip()
            break
    
    # Extract high-level movements
    movement_patterns = [
        r"(?:^|\n)#+\s*(?:High-Level|Movements|Steps)[^\n]*\n+(.*?)(?=\n#+|\n\d+:|$)",
        r"(?:^|\n)(?:The )?high-level movements(?: executed)? (?:are|include|consist)[^\n]*\n+(.*?)(?=\n#+|\n\d+:|$)",
        r"(?:^|\n)To accomplish this task, the robot(.*?)(?=\n#+|\n\d+:|$)",
        r"(?:^|\n)The robot executes the following movements:(.*?)(?=\n#+|\n\d+:|$)"
    ]
    
    for pattern in movement_patterns:
        match = re.search(pattern, overview_section, re.DOTALL | re.IGNORECASE)
        if match and match.group(1).strip():
            overview['high_level_movements'] = match.group(1).strip()
            break
    
    # Extract plan
    plan_patterns = [
        r"(?:^|\n)#+\s*(?:Plan|Solution)[^\n]*\n+(.*?)(?=\n#+|\n\d+:|$)",
        r"(?:^|\n)The plan for the solution(.*?)(?=\n#+|\n\d+:|$)",
        r"(?:^|\n)The solution plan(.*?)(?=\n#+|\n\d+:|$)",
        r"(?:^|\n)To solve this task, the plan is(.*?)(?=\n#+|\n\d+:|$)"
    ]
    
    for pattern in plan_patterns:
        match = re.search(pattern, overview_section, re.DOTALL | re.IGNORECASE)
        if match and match.group(1).strip():
            overview['plan'] = match.group(1).strip()
            break
    
    # If we couldn't extract a task overview, use the first paragraph
    if 'task_overview' not in overview and overview_section.strip():
        paragraphs = overview_section.split('\n\n')
        for paragraph in paragraphs:
            if paragraph.strip() and not paragraph.startswith('#'):
                overview['task_overview'] = paragraph.strip()
                break
    
    return overview


def parse_ecot_file(file_path: str) -> Dict[str, Any]:
    """
    Parse an ECoT JSON file to extract structured reasoning.
    
    Args:
        file_path: Path to the ECoT JSON file
        
    Returns:
        Dictionary with parsed content
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract the raw output text
        raw_output = data.get('raw_output', '')
        
        if not raw_output:
            print(f"Warning: No raw_output found in {file_path}")
            print("Available keys:", data.keys())
            return {}
        
        # Print the first 200 characters to debug
        print(f"Parsing file: {file_path}")
        print(f"Raw output starts with: {raw_output[:100]}...")
        print(f"Raw output length: {len(raw_output)} characters")
        
        # Parse the results
        result = {
            'metadata': data.get('metadata', {}),
            'high_level': extract_high_level_overview(raw_output),
            'step_reasoning': extract_all_tags_from_text(raw_output),
            'features': data.get('features', {})
        }
        
        # Show what was extracted
        print(f"Extracted high-level sections: {list(result['high_level'].keys())}")
        if result['step_reasoning']:
            tags_extracted = list(result['step_reasoning'].keys())
            print(f"Extracted tags: {tags_extracted}")
            steps_count = len(next(iter(result['step_reasoning'].values())))
            print(f"Extracted reasoning for {steps_count} steps")
        else:
            print("No step reasoning extracted")
        
        return result
    
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return {}


def format_parsed_reasoning(parsed_data: Dict[str, Any], step: Optional[int] = None) -> str:
    """
    Format the parsed reasoning into a readable string.
    
    Args:
        parsed_data: Parsed reasoning data
        step: Specific step to format (if None, formats overview)
        
    Returns:
        Formatted string representation
    """
    result = []
    
    # Add metadata
    metadata = parsed_data.get('metadata', {})
    episode_id = metadata.get('episode_id', 'Unknown')
    instruction = metadata.get('language_instruction', 'No instruction provided')
    
    result.append(f"Episode {episode_id}: {instruction}")
    result.append("=" * 50)
    
    # If no specific step is requested, show the overview
    if step is None:
        # Add high-level overview
        high_level = parsed_data.get('high_level', {})
        if 'task_overview' in high_level:
            result.append("\n🎯 TASK OVERVIEW")
            result.append("-" * 30)
            result.append(high_level['task_overview'])
            result.append("")
        
        if 'high_level_movements' in high_level:
            result.append("\n🔄 HIGH-LEVEL MOVEMENTS")
            result.append("-" * 30)
            result.append(high_level['high_level_movements'])
            result.append("")
        
        if 'plan' in high_level:
            result.append("\n📋 SOLUTION PLAN")
            result.append("-" * 30)
            result.append(high_level['plan'])
            result.append("")
    
    # If a specific step is requested, show that step's reasoning
    else:
        step_reasoning = parsed_data.get('step_reasoning', {})
        
        result.append(f"\n⚙️ STEP {step} REASONING")
        result.append("-" * 30)
        
        # Display each tag's content for this step
        for tag in get_cot_tags_list():
            if tag in step_reasoning and step in step_reasoning[tag]:
                content = step_reasoning[tag][step]
                tag_label = tag.upper().replace('_', ' ')
                result.append(f"\n📌 {tag_label}:")
                result.append(content)
        
    return "\n".join(result)


def main():
    parser = argparse.ArgumentParser(description='Parse ECoT reasoning annotations.')
    parser.add_argument('--input', '-i', type=str, required=True, 
                        help='Path to ECoT annotation JSON file or directory')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path to save parsed output (if not specified, prints to console)')
    parser.add_argument('--step', '-s', type=int, default=None,
                        help='Specific step to extract reasoning for')
    parser.add_argument('--format', '-f', choices=['text', 'json'], default='text',
                        help='Output format (text or json)')
    
    args = parser.parse_args()
    
    # Process a single file or a directory
    if os.path.isdir(args.input):
        # Process all JSON files in the directory
        files = [os.path.join(args.input, f) for f in os.listdir(args.input) 
                 if f.endswith('.json') and f.startswith('ecot_')]
        
        all_results = {}
        for file_path in files:
            episode_id = os.path.basename(file_path).replace('ecot_', '').replace('.json', '')
            parsed = parse_ecot_file(file_path)
            all_results[episode_id] = parsed
        
        # Save or print results
        if args.format == 'json':
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(all_results, f, indent=2)
            else:
                print(json.dumps(all_results, indent=2))
        else:
            # Print text representation of the first file or a specific episode
            if all_results:
                first_key = next(iter(all_results))
                formatted = format_parsed_reasoning(all_results[first_key], args.step)
                
                if args.output:
                    with open(args.output, 'w') as f:
                        f.write(formatted)
                else:
                    print(formatted)
    
    else:
        # Process a single file
        parsed = parse_ecot_file(args.input)
        
        if args.format == 'json':
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(parsed, f, indent=2)
            else:
                print(json.dumps(parsed, indent=2))
        else:
            formatted = format_parsed_reasoning(parsed, args.step)
            
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(formatted)
            else:
                print(formatted)


if __name__ == "__main__":
    main() 