#!/usr/bin/env python3
#python main.py --steps all --data-path /home/ubuntu/reasoning-embodied-test/data/bridge/... --output-dir ./pipeline_output --episodes all 

import os
import sys
import argparse
import logging
import subprocess
import time
from datetime import datetime
import gc
import json

# Add torch import for GPU memory management
try:
    import torch
except ImportError:
    torch = None

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
                      default='/home/ubuntu/reasoning-embodied-test/data/bridge/bridge_dataset-train.tfrecord-00002-of-01024',
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
    parser.add_argument('--episodes', type=str, default='5,18',
                       help='Episodes to process for ECoT annotations, comma-separated or "all"')
    
    # Visualization parameters
    parser.add_argument('--visualization-step', type=int, default=0,
                       help='Which step to visualize in each episode (default: 0)')
    parser.add_argument('--visualization-episode', type=str, default=None,
                       help='Specific episode ID to visualize (e.g., 39)')
    
    return parser.parse_args()

def run_script(script_name, args, output_dir, timeout=None):
    """Run a script and handle errors with timeout support."""
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
    logger.info(f"Running command: {script_name} with {len(args)} arguments")
    
    try:
        # Pass the current environment and set CUDA_VISIBLE_DEVICES explicitly
        env = os.environ.copy()
        if 'gpu' in args:
            env['CUDA_VISIBLE_DEVICES'] = str(args['gpu'])
        
        # Set PYTORCH_CUDA_ALLOC_CONF to limit memory usage
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        # Run the process with real-time output handling
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            env=env
        )
        
        # Function to handle real-time output
        def read_output(pipe, prefix):
            for line in iter(pipe.readline, ''):
                if line and line.strip():
                    logger.info(f"{prefix}: {line.strip()}")
            pipe.close()
        
        # Create threads to read output in real-time
        import threading
        stdout_thread = threading.Thread(
            target=read_output, 
            args=(process.stdout, "OUTPUT")
        )
        stderr_thread = threading.Thread(
            target=read_output, 
            args=(process.stderr, "ERROR")
        )
        
        # Start the threads
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()
        
        # Wait for the process to complete with timeout
        try:
            if timeout:
                return_code = process.wait(timeout=timeout)
            else:
                return_code = process.wait()
            
            # Ensure threads are done
            stdout_thread.join(2)
            stderr_thread.join(2)
            
            # Check the return code
            if return_code != 0:
                logger.error(f"Process completed with non-zero return code: {return_code}")
                return False
            
            elapsed_time = time.time() - start_time
            logger.info(f"Successfully completed {script_name} in {elapsed_time:.2f} seconds")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error(f"Process timed out after {timeout} seconds")
            process.kill()
            return False
        
    except Exception as e:
        logger.error(f"Error running {script_name}: {e}")
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
        
        # Clean up resources before running
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Reduce TensorFlow logging
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        # Important: Match the EXACT parameters used in generate_lightweight_bboxes.py
        # Using default values from generate_lightweight_bboxes.py
        script_args = {
            'data-path': args.data_path,
            'result-path': output_dirs['lightweight_bboxes'],
            'gpu': args.gpu,
            'box-threshold': 0.25,  # Changed to match default in generate_lightweight_bboxes.py
            'text-threshold': 0.15,  # Changed to match default in generate_lightweight_bboxes.py
            'batch-size': 5,  # Changed to match default in generate_lightweight_bboxes.py
            'debug': args.debug
        }
        
        logger.info("Starting bounding box generation with exact parameters from generate_lightweight_bboxes.py...")
        
        # Run the script with the parameters exactly as in generate_lightweight_bboxes.py
        success = run_script(
            'generate_lightweight_bboxes.py', 
            script_args, 
            output_dirs['lightweight_bboxes'],
            timeout=10800  # 3 hours timeout
        )
        
        # Clean up resources after running
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        pipeline_success = pipeline_success and success
    
    # Step 4: Extract gripper positions
    if (run_all or 'gripper' in steps_to_run) and pipeline_success:
        logger.info("Step 4: Extracting Gripper Positions")
        
        # Fix: Use hyphens in argument names to match the script's expectations
        script_args = {
            'data-path': args.data_path,  # Changed from data_path to data-path
            'output-dir': output_dirs['gripper'],  # Changed from output_dir to output-dir
            'gpu': args.gpu,
            'visualize': args.debug  # Pass debug flag as visualize
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
            logger.warning("Required files for ECoT annotations not found:")
            if not os.path.exists(gripper_path):
                logger.warning(f"Missing: {gripper_path}")
            if not os.path.exists(bboxes_path):
                logger.warning(f"Missing: {bboxes_path}")
            logger.warning("Skipping ECoT annotation step.")
        else:
            # Use hyphenated argument names to match what generate_ecot_annotations.py expects
            script_args = {
                'data-path': args.data_path,
                'bbox-path': bboxes_path,
                'primitives-path': output_dirs['primitives'],
                'gripper-path': output_dirs['gripper'],
                'output-path': output_dirs['ecot'],
                'model': args.model,
                'api-key': args.api_key,
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
        
        # Check if a specific episode was requested for visualization
        if args.visualization_episode:
            ecot_file = f"ecot_{args.visualization_episode}.json"
            ecot_path = os.path.join(output_dirs['ecot'], ecot_file)
            
            if os.path.exists(ecot_path):
                logger.info(f"Visualizing specific episode {args.visualization_episode}, Step {args.visualization_step}")
                script_args = {
                    'json': ecot_path,
                    'tfrecord': args.data_path,
                    'output': output_dirs['visualize'],
                    'step': args.visualization_step
                }
                success = run_script('visualize_ecot.py', script_args, output_dirs['visualize'])
                if not success:
                    logger.warning(f"Failed to visualize {ecot_file}")
            else:
                logger.warning(f"Requested ECoT file {ecot_file} not found in {output_dirs['ecot']}")
        else:
            # Find all available ECoT files
            if os.path.exists(output_dirs['ecot']):
                ecot_files = [f for f in os.listdir(output_dirs['ecot']) 
                             if f.startswith('ecot_') and f.endswith('.json')]
                
                if ecot_files:
                    logger.info(f"Found {len(ecot_files)} ECoT annotation files")
                    for ecot_file in ecot_files[:5]:  # Limit to first 5 for demonstration
                        ecot_path = os.path.join(output_dirs['ecot'], ecot_file)
                        episode_id = ecot_file.replace('ecot_', '').replace('.json', '')
                        
                        logger.info(f"Visualizing {ecot_file} (Episode {episode_id}, Step {args.visualization_step})")
                        script_args = {
                            'json': ecot_path,
                            'tfrecord': args.data_path,
                            'output': output_dirs['visualize'],
                            'step': args.visualization_step
                        }
                        success = run_script('visualize_ecot.py', script_args, output_dirs['visualize'])
                        if not success:
                            logger.warning(f"Failed to visualize {ecot_file}")
                else:
                    logger.warning("No ECoT annotation files found for visualization.")
                    logger.warning("Skipping visualization step.")
            else:
                logger.warning(f"ECoT directory {output_dirs['ecot']} does not exist.")
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