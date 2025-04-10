#!/usr/bin/env python3
# When running the script, after including the path to the json file in the command include false false for not including overview and headers
# Example: python extract_reasoning.py ecot_39.json false false


"""
Script to extract complete reasoning from an ECoT annotation file and save it to a text file.
This script handles the specific format used in the ECoT annotations.
"""

import json
import re
import sys
import os
from typing import Dict, Any, List

def extract_dictionary_content(raw_output: str) -> Dict[int, Dict[str, str]]:
    """
    Extract the Python dictionary content from the raw output, which contains
    the step-by-step reasoning.
    
    Args:
        raw_output: The full raw output text from the ECoT file
        
    Returns:
        Dictionary mapping step numbers to dictionaries of tag content
    """
    print("Raw output length:", len(raw_output))
    
    # Try to find Python code blocks with dictionaries
    python_blocks = re.findall(r'```python\s*(.*?)```', raw_output, re.DOTALL)
    
    if python_blocks:
        print(f"Found {len(python_blocks)} Python code blocks")
        for block in python_blocks:
            # Look for dictionary assignment
            dict_match = re.search(r'(\w+)\s*=\s*\{', block)
            if dict_match:
                dict_name = dict_match.group(1)
                print(f"Found dictionary: {dict_name}")
                dict_content = block
                break
    else:
        print("No Python code blocks found. Trying direct dictionary search...")
        # Try to find a dictionary directly
        dict_regex = r'(?:reasoning_dict|reasoning|trajectory_reasoning|reasonings)\s*=\s*\{(.*?)(?:\}\s*\n|\Z)'
        dict_match = re.search(dict_regex, raw_output, re.DOTALL)
        
        if dict_match:
            dict_content = dict_match.group(1)
            print(f"Found dictionary content starting with: {dict_content[:50]}...")
        else:
            # Last resort: try to find any step-based pattern
            print("No standard dictionary found. Trying to extract steps directly...")
            result = {}
            step_pattern = r'(\d+):\s*"<task>(.*?)</task>\s*<plan>(.*?)</plan>\s*<subtask>(.*?)</subtask>\s*<subtask_reason>(.*?)</subtask_reason>\s*<move>(.*?)</move>\s*<move_reason>(.*?)</move_reason>"'
            steps = re.findall(step_pattern, raw_output, re.DOTALL)
            
            if steps:
                print(f"Found {len(steps)} steps with direct pattern")
                for step_match in steps:
                    step = int(step_match[0])
                    result[step] = {
                        "task": step_match[1].strip(),
                        "plan": step_match[2].strip(),
                        "subtask": step_match[3].strip(),
                        "subtask_reason": step_match[4].strip(),
                        "move": step_match[5].strip(),
                        "move_reason": step_match[6].strip()
                    }
                return result
            else:
                print("No steps found with direct pattern")
                return {}
    
    # Get individual step entries
    result = {}
    
    # Look for entries like: 0: "<task>...</task> <plan>...</plan> ...",
    entries_regex = r'(\d+)\s*:\s*"(.*?)"(?:,|\s*\})'
    entries = re.findall(entries_regex, dict_content, re.DOTALL)
    
    if entries:
        print(f"Found {len(entries)} step entries")
        for step_str, content in entries:
            step = int(step_str)
            result[step] = {}
            
            # Extract each tag content
            tag_names = ["task", "plan", "subtask", "subtask_reason", "move", "move_reason"]
            for tag in tag_names:
                tag_regex = rf'<{tag}>(.*?)</{tag}>'
                tag_match = re.search(tag_regex, content, re.DOTALL)
                if tag_match:
                    result[step][tag] = tag_match.group(1).strip()
    else:
        print("No step entries found in dictionary")
    
    return result

def extract_high_level_sections(raw_output: str) -> Dict[str, str]:
    """
    Extract high-level sections from the raw output text.
    
    Args:
        raw_output: The full raw output text
        
    Returns:
        Dictionary of section names to content
    """
    sections = {}
    
    # Split by separator lines (e.g., "----------")
    # Look for sections demarcated by separator lines like "-------"
    separator_pattern = r'[-]{5,}'
    parts = re.split(separator_pattern, raw_output)
    
    if len(parts) > 1:
        print(f"Split text into {len(parts)} parts using separator lines")
        
        # Process each part to identify overview, movements, plan sections
        for part in parts:
            part = part.strip()
            if not part:
                continue
                
            # Look for section headers within the part
            if re.search(r'(?:Overview|Task\s+Overview|Overview\s+of\s+the\s+Task)', part, re.IGNORECASE):
                sections["Overview"] = part
            elif re.search(r'(?:High.Level\s+Movements|Movements\s+and\s+their\s+Intervals)', part, re.IGNORECASE):
                sections["Movements"] = part
            elif re.search(r'(?:Detailed\s+Step.by.Step\s+Reasoning)', part, re.IGNORECASE):
                sections["Step Reasoning"] = part
    else:
        # If no separator lines, try directly matching sections
        # Find Overview/Task description section
        overview_regex = r'(?:Overview\s+of\s+the\s+Task|Task\s+Description|Task\s+Overview)[^\n]*\n+(.*?)(?=\n+(?:High|Detailed|Step|\d+:|$))'
        overview_match = re.search(overview_regex, raw_output, re.DOTALL | re.IGNORECASE)
        if overview_match:
            sections["Overview"] = overview_match.group(1).strip()
        
        # Find High-Level Movements section
        movements_regex = r'(?:High(?:\-|\s)Level\s+Movements|Movements\s+and\s+their\s+Intervals)[^\n]*\n+(.*?)(?=\n+(?:Detailed|Step|\d+:|$))'
        movements_match = re.search(movements_regex, raw_output, re.DOTALL | re.IGNORECASE)
        if movements_match:
            sections["Movements"] = movements_match.group(1).strip()
    
    return sections

def format_reasoning(steps_dict: Dict[int, Dict[str, str]], high_level: Dict[str, str], metadata: Dict[str, Any], include_overview=False, include_headers=False) -> str:
    """
    Format the extracted reasoning into a readable text format.
    
    Args:
        steps_dict: Dictionary of step reasoning
        high_level: Dictionary of high-level sections
        metadata: Metadata from the original file
        include_overview: Whether to include the high-level overview sections
        include_headers: Whether to include section headers and episode info
        
    Returns:
        Formatted string with complete reasoning
    """
    lines = []
    
    # Add metadata only if headers are requested
    if include_headers:
        episode_id = metadata.get("episode_id", "Unknown")
        instruction = metadata.get("language_instruction", "")
        lines.append(f"Episode {episode_id}: {instruction}")
        lines.append("=" * 50)
        lines.append("")
    
    # Add high-level sections if requested
    if include_overview and high_level and include_headers:
        lines.append("HIGH-LEVEL OVERVIEW")
        lines.append("-" * 30)
        for section, content in high_level.items():
            lines.append(f"\n## {section}")
            lines.append(content)
            lines.append("")
    
    # Add step-by-step reasoning (always included)
    if steps_dict:
        # Only add the section header if headers are requested
        if include_headers:
            lines.append("STEP-BY-STEP REASONING")
            lines.append("-" * 30)
            lines.append("")
        
        # Sort steps by number
        steps = sorted(steps_dict.keys())
        
        for step in steps:
            step_data = steps_dict[step]
            # No newline before first step
            if step != steps[0] or include_headers:
                lines.append("")
            lines.append(f"Step {step}:")
            
            for tag in ["task", "plan", "subtask", "subtask_reason", "move", "move_reason"]:
                if tag in step_data:
                    tag_title = tag.replace('_', ' ').upper()
                    lines.append(f"  {tag_title}: {step_data[tag]}")
    
    return "\n".join(lines)

def main():
    # Check if input file is provided
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                 "pipeline_output/ecot_annotations/ecot_39.json")
    
    # Determine output file name
    base_name = os.path.basename(input_file).replace('.json', '')
    output_file = f"reasoning_{base_name.replace('ecot_', '')}.txt"
    
    # Add command line arguments for output customization
    include_overview = False  # Default to not include overview
    include_headers = False   # Default to not include headers
    
    if len(sys.argv) > 2:
        if sys.argv[2].lower() in ('yes', 'true', 't', 'y', '1'):
            include_overview = True
    
    if len(sys.argv) > 3:
        if sys.argv[3].lower() in ('yes', 'true', 't', 'y', '1'):
            include_headers = True
    
    print(f"Extracting reasoning from {input_file}")
    print(f"Including overview sections: {include_overview}")
    print(f"Including headers: {include_headers}")
    print(f"Saving results to {output_file}")
    
    try:
        # Load JSON file
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Extract raw output
        raw_output = data.get('raw_output', '')
        if not raw_output:
            print("Error: No raw_output found in file")
            return
        
        # Extract high-level sections
        high_level = extract_high_level_sections(raw_output)
        print(f"Extracted {len(high_level)} high-level sections")
        
        # Extract step-by-step reasoning dictionary
        steps_dict = extract_dictionary_content(raw_output)
        print(f"Extracted reasoning for {len(steps_dict)} steps")
        
        # Format the reasoning without headers
        formatted = format_reasoning(
            steps_dict, 
            high_level, 
            data.get('metadata', {}), 
            include_overview=include_overview,
            include_headers=include_headers
        )
        
        # Save to file
        with open(output_file, 'w') as f:
            f.write(formatted)
        
        print(f"Successfully saved reasoning to {output_file}")
        
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 