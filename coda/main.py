#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import subprocess
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments for the pipeline."""
    parser = argparse.ArgumentParser(description='Execute the complete bridge dataset processing pipeline')
    
    # Data paths
    parser.add_argument('--data-path', type=str, 
                      default='/home/ubuntu/embodied-CoT/data/bridge/bridge_dataset-train.tfrecord-00002-of-01024',
                      help='Path to the TFRecord dataset')
    
    # Output paths
    parser.add_argument('--output-dir', type=str, default='./pipeline_output',
                       help='Base directory for all outputs')
    
    # GPU selection
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID to use for processing')
    
    # Pipeline control
    parser.add_argument('--steps', type=str, default='all',
                       help='Comma-separated list of steps to run (all,analyze,primitives,bboxes,gripper,ecot,visualize)')
    parser.add_argument('--force', action='store_true',
                       help='Force rerunning steps even if output exists')
    parser.add_argument('--debug', action='store_true',
                       help='Generate additional debug visualizations')
    
    # LLM parameters
    parser.add_argument('--model', type=str, default='o3-mini-2025-01-31',
                       help='OpenAI model to use for ECoT annotations')
    parser.add_argument('--api-key', type=str, default=None,
                       help='OpenAI API key (or set OPENAI_API_KEY environment variable)')
    parser.add_argument('--episodes', type=str, default='all',
                       help='Episodes to process for ECoT annotations, comma-separated or "all"')
    
    return parser.parse_args()

def run_script(script_name, args, output_dir):
    """Run a script and handle errors."""
    start_time = time.time()
    script_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        script_name
    )
    
    # Create command with proper arguments
    cmd = [sys.executable, script_path]
    
    for key, value in args.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.append(f"--{key}")
            cmd.append(str(value))
    
    # Log the command
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        # Run the script
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        # Log output
        logger.info(f"Output from {script_name}:")
        for line in process.stdout.split('\n'):
            if line.strip():
                logger.info(f"  {line}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Successfully completed {script_name} in {elapsed_time:.2f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script_name}: {e}")
        logger.error(f"STDERR: {e.stderr}")
        return False

def main():
    """Main pipeline execution function."""
    args = parse_arguments()
    
    # Create output directory structure
    base_output_dir = args.output_dir
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Set up specific output directories
    output_dirs = {
        'analyze': os.path.join(base_output_dir, "analysis"),
        'primitives': os.path.join(base_output_dir, "primitive_movements"),
        'lightweight_bboxes': os.path.join(base_output_dir, "lightweight_bboxes"),
        'gripper': os.path.join(base_output_dir, "gripper_positions"),
        'ecot': os.path.join(base_output_dir, "ecot_annotations"),
        'visualize': os.path.join(base_output_dir, "visualizations")
    }
    
    # Create all output directories
    for directory in output_dirs.values():
        os.makedirs(directory, exist_ok=True)
    
    # Determine which steps to run
    steps_to_run = args.steps.lower().split(',')
    run_all = 'all' in steps_to_run
    
    # Start pipeline
    logger.info("=" * 80)
    logger.info(f"Starting Bridge Dataset Processing Pipeline at {datetime.now()}")
    logger.info(f"Output directory: {base_output_dir}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Steps to run: {args.steps}")
    logger.info("=" * 80)
    
    pipeline_success = True
    
    # Step 1: Analyze bridge dataset
    if run_all or 'analyze' in steps_to_run:
        logger.info("Step 1: Analyzing Bridge Dataset")
        script_args = {
            'data_path': args.data_path,
            'output_dir': output_dirs['analyze']
        }
        success = run_script('analyze_bridge_dataset.py', script_args, output_dirs['analyze'])
        pipeline_success = pipeline_success and success
    
    # Step 2: Generate primitive movements
    if (run_all or 'primitives' in steps_to_run) and pipeline_success:
        logger.info("Step 2: Generating Primitive Movements")
        script_args = {
            'data_path': args.data_path,
            'output_dir': output_dirs['primitives']
        }
        success = run_script('primitive_movements_bridge.py', script_args, output_dirs['primitives'])
        pipeline_success = pipeline_success and success
    
    # Step 3: Generate lightweight bounding boxes
    if (run_all or 'bboxes' in steps_to_run) and pipeline_success:
        logger.info("Step 3: Generating Lightweight Bounding Boxes")
        script_args = {
            'data_path': args.data_path,
            'result_path': output_dirs['lightweight_bboxes'],
            'gpu': args.gpu,
            'debug': args.debug
        }
        success = run_script('generate_lightweight_bboxes.py', script_args, output_dirs['lightweight_bboxes'])
        pipeline_success = pipeline_success and success
    
    # Step 4: Extract gripper positions
    if (run_all or 'gripper' in steps_to_run) and pipeline_success:
        logger.info("Step 4: Extracting Gripper Positions")
        script_args = {
            'data_path': args.data_path,
            'output_dir': output_dirs['gripper'],
            'gpu': args.gpu,
            'visualize': args.debug
        }
        success = run_script('extract_gripper_lightweight.py', script_args, output_dirs['gripper'])
        pipeline_success = pipeline_success and success
    
    # Step 5: Generate ECoT annotations
    ecot_files = []
    if (run_all or 'ecot' in steps_to_run) and pipeline_success:
        logger.info("Step 5: Generating Embodied Chain-of-Thought Annotations")
        
        # Check if required files exist
        gripper_path = os.path.join(output_dirs['gripper'], 'all_gripper_positions.json')
        bboxes_path = os.path.join(output_dirs['lightweight_bboxes'], 'lightweight_bridge_bboxes.json')
        
        if not os.path.exists(gripper_path) or not os.path.exists(bboxes_path):
            logger.warning("ECoT generation requires gripper and bbox outputs, which are missing.")
            logger.warning("Skipping ECoT annotation step.")
        else:
            # Run the ECoT annotation script
            script_args = {
                'data_path': args.data_path,
                'bbox_path': bboxes_path,
                'primitives_path': output_dirs['primitives'],
                'gripper_path': output_dirs['gripper'],
                'output_path': output_dirs['ecot'],
                'model': args.model,
                'api_key': args.api_key,
                'episodes': args.episodes
            }
            
            success = run_script('generate_ecot_annotations.py', script_args, output_dirs['ecot'])
            pipeline_success = pipeline_success and success
            
            # Find generated ECoT files for visualization
            if success and os.path.exists(output_dirs['ecot']):
                ecot_files = [f for f in os.listdir(output_dirs['ecot']) 
                             if f.startswith('ecot_') and f.endswith('.json')]
                logger.info(f"Found {len(ecot_files)} ECoT annotation files for visualization")
    
    # Step 6: Visualize ECoT annotations
    if (run_all or 'visualize' in steps_to_run) and pipeline_success:
        logger.info("Step 6: Visualizing ECoT Annotations")
        
        # If we have ECoT files, visualize them
        if ecot_files:
            for ecot_file in ecot_files[:5]:  # Limit to first 5 for demonstration
                ecot_path = os.path.join(output_dirs['ecot'], ecot_file)
                
                logger.info(f"Visualizing {ecot_file}")
                script_args = {
                    'json': ecot_path,
                    'tfrecord': args.data_path,
                    'output': output_dirs['visualize']
                }
                success = run_script('visualize_ecot.py', script_args, output_dirs['visualize'])
                if not success:
                    logger.warning(f"Failed to visualize {ecot_file}")
        else:
            logger.warning("No ECoT annotation files found for visualization.")
            logger.warning("Skipping visualization step.")
    
    # Summarize pipeline results
    logger.info("=" * 80)
    if pipeline_success:
        logger.info("Pipeline completed successfully.")
        logger.info(f"All outputs available at: {base_output_dir}")
    else:
        logger.error("Pipeline completed with errors. See logs for details.")
    logger.info("=" * 80)
    
    return 0 if pipeline_success else 1

if __name__ == "__main__":
    sys.exit(main()) 