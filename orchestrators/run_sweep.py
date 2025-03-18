'''
run_train.py

Hyperparameter optimization orchestrator for RiskFormer training.
Uses Optuna for hyperparameter search and Weights & Biases for tracking.
Author: landeros10
'''
import os
import sys
import logging
import argparse
import subprocess
import json
from datetime import datetime
import uuid
import shutil

import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
import wandb
import boto3

from riskformer.utils.logger_config import logger_setup, log_event
from riskformer.utils.training_utils import load_train_config, validate_config
from riskformer.utils.aws_utils import (
    initialize_s3_client,
    upload_large_files_to_bucket,
    upload_model_to_s3,
    build_model_s3_key,
    get_s3_client_with_fallback
)

logger = logging.getLogger(__name__)

def extract_metric_from_results(results, metric_name, default_value=None, direction='minimize'):
    """Extract a metric from different possible locations in the results
    
    Args:
        results (dict): Results dictionary
        metric_name (str): Name of the metric to extract
        default_value (any): Default value if metric not found
        direction (str): Direction of optimization ('minimize' or 'maximize')
        
    Returns:
        any: Extracted metric value or default
    """
    # Check direct key
    if metric_name in results:
        return results[metric_name]
    
    # Check in test_results
    if 'test_results' in results and metric_name in results['test_results']:
        return results['test_results'][metric_name]
    
    # Check with test_ prefix
    if f'test_{metric_name}' in results:
        return results[f'test_{metric_name}']
    
    # Use default value if specified
    if default_value is not None:
        return default_value
        
    # Generate worst case default based on direction
    if direction == 'minimize':
        return float('inf')
    else:
        return -float('inf')

def parse_args():
    """Parse command line arguments for the hyperparameter optimization orchestrator"""
    parser = argparse.ArgumentParser(description="RiskFormer Training Hyperparameter Optimization")

    # Base configuration
    parser.add_argument("--base_config", type=str, required=True,
                        help="Path to base configuration file")
    parser.add_argument("--sweep_config", type=str, required=True,
                        help="Path to hyperparameter sweep configuration file")
    parser.add_argument("--study_name", type=str, default=f"riskformer-optuna-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                        help="Name for the Optuna study")
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Number of trials for hyperparameter search")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Timeout for hyperparameter search in seconds")
    
    # Output and storage
    parser.add_argument("--output_dir", type=str, default="./optuna_results",
                        help="Directory to save Optuna study results")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage URL (e.g., sqlite:///optuna.db)")
    
    # Wandb configuration
    parser.add_argument("--wandb_project", type=str, default="riskformer-hpo",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Weights & Biases entity name")
    
    # Execution parameters
    parser.add_argument("--run_mode", type=str, choices=["sequential", "parallel"], default="sequential",
                        help="Run trials sequentially or in parallel")
    parser.add_argument("--max_parallel_jobs", type=int, default=1,
                        help="Maximum number of parallel jobs (if run_mode is parallel)")
    
    # S3 Configuration for model uploads
    parser.add_argument("--s3_bucket", type=str, default=None,
                        help="S3 bucket for uploading model checkpoints")
    parser.add_argument("--s3_prefix", type=str, default="models",
                        help="S3 prefix for model checkpoints")
    parser.add_argument("--aws_profile", type=str, default=None,
                        help="AWS profile name for S3 operations")
    parser.add_argument("--aws_region", type=str, default=None,
                        help="AWS region for S3 operations")
    
    # Model upload criteria
    parser.add_argument("--upload_best_only", action="store_true",
                        help="Upload only the best models to S3")
    parser.add_argument("--upload_threshold", type=float, default=None,
                        help="Only upload models with metric better than this threshold")
    
    # Debug mode
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode (fewer trials, more logging)")

    return parser.parse_args()


def load_sweep_config(config_path):
    """Load hyperparameter sweep configuration from YAML file
    
    Args:
        config_path (str): Path to the hyperparameter sweep configuration file
        
    Returns:
        dict: Loaded hyperparameter sweep configuration
    """
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        log_event("info", "sweep_config_loaded", "Hyperparameter sweep configuration loaded", config_path=config_path)
        return config
    except Exception as e:
        log_event("error", "sweep_config_load_error", "Failed to load hyperparameter sweep configuration", 
                 config_path=config_path, error_message=str(e))
        raise ValueError(f"Failed to load sweep config from {config_path}: {str(e)}")


def define_search_space(trial, sweep_config):
    """Define the hyperparameter search space for Optuna based on sweep configuration.
    
    Args:
        trial: Optuna trial object
        sweep_config: Hyperparameter sweep configuration dictionary
        
    Returns:
        dict: Dictionary containing hyperparameters for this trial
    """
    params = {}
    
    # Process each parameter in the sweep configuration
    for param_name, param_config in sweep_config.items():
        param_type = param_config.get('type')
        
        if param_type == 'categorical':
            values = param_config.get('values', [])
            params[param_name] = trial.suggest_categorical(param_name, values)
            
        elif param_type == 'float':
            low = param_config.get('min')
            high = param_config.get('max')
            log = param_config.get('log', False)
            params[param_name] = trial.suggest_float(param_name, low, high, log=log)
            
        elif param_type == 'int':
            low = param_config.get('min')
            high = param_config.get('max')
            step = param_config.get('step', 1)
            params[param_name] = trial.suggest_int(param_name, low, high, step=step)
            
        elif param_type == 'conditional':
            # Handle conditional parameters based on the value of another parameter
            condition_param = param_config.get('condition_on')
            condition_value = param_config.get('condition_value')
            
            # Check if we have already set the condition parameter
            if condition_param in params and params[condition_param] == condition_value:
                inner_param_config = param_config.get('param_config', {})
                inner_param_type = inner_param_config.get('type')
                
                if inner_param_type == 'categorical':
                    values = inner_param_config.get('values', [])
                    params[param_name] = trial.suggest_categorical(param_name, values)
                elif inner_param_type == 'float':
                    low = inner_param_config.get('min')
                    high = inner_param_config.get('max')
                    log = inner_param_config.get('log', False)
                    params[param_name] = trial.suggest_float(param_name, low, high, log=log)
                elif inner_param_type == 'int':
                    low = inner_param_config.get('min')
                    high = inner_param_config.get('max')
                    step = inner_param_config.get('step', 1)
                    params[param_name] = trial.suggest_int(param_name, low, high, step=step)
    
    return params


def save_and_manage_results(results, experiment_name, optimization_metric, results_dir, sweep_config, s3_config=None, trial_number=None):
    """Save results to file and handle S3 uploads if configured.
    
    Args:
        results (dict): Results dictionary from the training job
        experiment_name (str): Name of the experiment
        optimization_metric (str): Name of the optimization metric
        results_dir (str): Directory to save results
        sweep_config (dict): Sweep configuration
        s3_config (dict): S3 configuration for model upload
        trial_number (int/str): Trial number or identifier (None for final model)
        
    Returns:
        tuple: (results_file_path, results) - Updated path and results
    """
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Extract metric value for filename
    metric_value = extract_metric_from_results(
        results, 
        optimization_metric, 
        direction=sweep_config.get('direction', 'minimize')
    )
    
    # Create descriptive filename
    descriptive_filename = f"{experiment_name}_{timestamp}"
    
    # Add trial number if provided
    if trial_number is not None:
        descriptive_filename = f"{descriptive_filename}_trial_{trial_number}"
    
    # Add metric if available
    if metric_value is not None:
        if isinstance(metric_value, float):
            metric_str = f"{optimization_metric}={metric_value:.4f}"
        else:
            metric_str = f"{optimization_metric}={metric_value}"
        descriptive_filename += f"_{metric_str}"
    
    # Clean filename
    descriptive_filename = descriptive_filename.replace(" ", "_").replace("/", "_")
    descriptive_filename += ".json"
    
    # Create full path
    results_file_path = os.path.join(results_dir, descriptive_filename)
    
    # Save results to file
    with open(results_file_path, "w") as f:
        json.dump(results, f, indent=4)
    
    log_event("info", "results_saved", f"Results saved to {results_file_path}")
    
    # Upload model to S3 if configured
    if s3_config and s3_config.get('s3_bucket'):
        # Check if we have a best checkpoint path
        best_checkpoint_path = results.get('best_checkpoint_path')
        if best_checkpoint_path and os.path.exists(best_checkpoint_path):
            # Determine if we should upload based on threshold and direction
            should_upload = True
            threshold = s3_config.get('upload_threshold')
            direction = sweep_config.get('direction', 'minimize')
            
            if threshold is not None:
                if direction == 'minimize':
                    should_upload = metric_value <= threshold
                else:  # maximize
                    should_upload = metric_value >= threshold
            
            if should_upload:
                # Get S3 client
                s3_client = s3_config.get('s3_client')
                if not s3_client:
                    s3_client = get_s3_client_with_fallback(
                        s3_config.get('aws_profile'),
                        s3_config.get('aws_region'),
                        logger
                    )
                    s3_config['s3_client'] = s3_client
                
                if s3_client:
                    # Build S3 key
                    s3_key = build_model_s3_key(
                        experiment_name=s3_config.get('study_name', 'riskformer-hpo'),
                        trial_identifier=trial_number if trial_number is not None else 'final',
                        model_file=best_checkpoint_path,
                        is_best=True,
                        prefix=s3_config.get('s3_prefix', 'models'),
                        is_final_model=(trial_number is None)  # If trial_number is None, it's the final model
                    )
                    
                    # Upload model to S3
                    s3_path = upload_model_to_s3(
                        best_checkpoint_path,
                        s3_config['s3_bucket'],
                        s3_key,
                        s3_client,
                        logger
                    )
                    
                    if s3_path:
                        # Store S3 path in results
                        results['s3_model_path'] = s3_path
                        
                        # Update results file with S3 path
                        with open(results_file_path, "w") as f:
                            json.dump(results, f, indent=4)
                        
                        # Log to W&B if enabled - only log the path as text, not as an artifact
                        if 'wandb' in sys.modules:
                            try:
                                if trial_number is not None:
                                    # Just log the S3 path as a string, not linking to the actual model
                                    wandb.log({
                                        f"trial_{trial_number}/s3_model_path_text": str(s3_path),
                                        f"trial_{trial_number}/{optimization_metric}": metric_value
                                    })
                                else:
                                    # Just log the S3 path as a string, not linking to the actual model
                                    wandb.log({
                                        "final_model/s3_model_path_text": str(s3_path),
                                        f"final_model/{optimization_metric}": metric_value
                                    })
                            except Exception as e:
                                log_event("warning", "wandb_log_failed", "Failed to log to W&B", error=str(e))
                else:
                    log_event("warning", "s3_upload_skipped", "Skipping S3 upload due to client initialization failure")
    
    return results_file_path, results


def run_training_job(params, base_config_path, trial_number, s3_config=None):
    """Run a training job with the given hyperparameters
    
    Args:
        params (dict): Hyperparameters for this trial
        base_config_path (str): Path to base configuration file
        trial_number (int): Current trial number
        s3_config (dict): S3 configuration for model upload
        
    Returns:
        dict: Results from the training job including validation metrics
    """
    # Create a unique experiment name for this trial
    experiment_name = f"trial_{trial_number}"
    
    # Generate a temporary config file for this specific trial
    config = load_train_config(base_config_path)
    
    # Update the base config with the trial hyperparameters
    for key, value in params.items():
        config[key] = value
    
    # Set experiment name and wandb config
    config["experiment_name"] = experiment_name
    config["use_wandb"] = True
    
    # Configure wandb to not save artifacts
    config["wandb_log_artifacts"] = False
    config["wandb_log_model"] = False
    
    # Set environment variable to disable code uploads
    os.environ['WANDB_DISABLE_CODE'] = 'true'
    
    # Write the updated config to a temporary file
    import yaml
    temp_config_path = f"temp_config_trial_{trial_number}.yaml"
    with open(temp_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Create a results directory for this trial
    results_dir = os.path.join(args.output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Define the initial results file path (where train.py will save results)
    # Use a simple name initially since we'll rename it later with more details
    initial_results_file_path = os.path.join(results_dir, f"{experiment_name}.json")
    
    # Build command to run training script
    cmd = [
        "python", "entrypoints/train.py",
        "--config", temp_config_path,
        "--use_wandb",
        "--no_wandb_artifacts",  # Add flag to disable wandb artifacts
        "--results_file_path", initial_results_file_path
    ]
    
    log_event("info", "trial_started", f"Starting trial {trial_number}", params=params)
    
    # Run the training process
    try:
        process = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            check=True
        )
        
        # Log stdout and stderr for debugging
        log_event("debug", "trial_stdout", f"Trial {trial_number} stdout", stdout=process.stdout)
        if process.stderr:
            log_event("debug", "trial_stderr", f"Trial {trial_number} stderr", stderr=process.stderr)
            
        log_event("info", "trial_completed", f"Trial {trial_number} completed successfully")
        
    except subprocess.CalledProcessError as e:
        log_event("error", "trial_failed", f"Trial {trial_number} failed", 
                 error=str(e), stderr=e.stderr)
        # Return a very poor result to signal failure to Optuna
        return {"val_loss": float('inf')}
    
    # Parse results from the training job
    try:
        if os.path.exists(initial_results_file_path):
            with open(initial_results_file_path, "r") as f:
                results = json.load(f)
                log_event("info", "results_loaded", f"Loaded results from {initial_results_file_path}")
        else:
            log_event("error", "results_not_found", "Could not find results file, returning default failure metrics")
            return {"val_loss": float('inf')}
        
        # Save results with more descriptive filename and handle S3 upload if configured
        optimization_metric = sweep_config.get('optimization_metric', 'val_loss')
        
        results_file_path, results = save_and_manage_results(
            results=results,
            experiment_name=experiment_name,
            optimization_metric=optimization_metric,
            results_dir=results_dir,
            sweep_config=sweep_config,
            s3_config=s3_config,
            trial_number=trial_number
        )
        
        # Clean up temporary files
        try:
            os.remove(temp_config_path)
            log_event("debug", "temp_config_removed", f"Removed temporary config file {temp_config_path}")
            
            # Remove the initial results file if it exists and is different from the descriptive file
            if os.path.exists(initial_results_file_path) and initial_results_file_path != results_file_path:
                os.remove(initial_results_file_path)
                log_event("debug", "initial_results_removed", f"Removed initial results file {initial_results_file_path}")
        except Exception as e:
            log_event("warning", "temp_file_cleanup_failed", "Failed to clean up temporary files", error=str(e))
        
        return results
    except Exception as e:
        log_event("error", "parsing_results_failed", f"Failed to parse results for trial {trial_number}", 
                 error=str(e), exc_info=True)
        return {"val_loss": float('inf')}


def objective(trial):
    """Optuna objective function that defines the hyperparameter space and runs training
    
    Args:
        trial: Optuna trial object
        
    Returns:
        float: Primary metric to optimize (e.g., validation loss)
    """
    global args, sweep_config, s3_config
    
    # Generate hyperparameters for this trial
    params = define_search_space(trial, sweep_config)
    
    # Execute the training with these hyperparameters
    trial_number = trial.number
    results = run_training_job(params, args.base_config, trial_number, s3_config)
    
    # Extract the optimization metric
    optimization_metric = sweep_config.get('optimization_metric', 'val_loss')
    
    # Extract the primary metric using our helper function
    primary_metric = extract_metric_from_results(
        results,
        optimization_metric,
        direction=sweep_config.get('direction', 'minimize')
    )
    
    # Log the primary metric and trial parameters
    log_event("info", "trial_metric", f"Trial {trial_number} {optimization_metric}", 
             metric_value=primary_metric, params=params)
    
    # Log additional metrics to Optuna
    for key, value in results.items():
        if (key != optimization_metric and 
            isinstance(value, (int, float)) and
            not key.startswith('config') and
            key not in ['run_id', 'timestamp']):
            trial.set_user_attr(key, value)
    
    # Also look for metrics in test_results if it exists
    if 'test_results' in results and isinstance(results['test_results'], dict):
        for key, value in results['test_results'].items():
            if isinstance(value, (int, float)):
                trial.set_user_attr(f"test_{key}", value)
    
    return primary_metric


def setup_wandb_callback():
    """Set up the Weights & Biases callback for Optuna
    
    Returns:
        WeightsAndBiasesCallback: Configured W&B callback for Optuna
    """
    # Disable artifacts and model logging in wandb to save storage
    os.environ['WANDB_DISABLE_CODE'] = 'true'
    
    wandb_kwargs = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "name": args.study_name,
        "config": vars(args),
        # Explicitly disable artifact storage and model saving
        "settings": wandb.Settings(
            _disable_stats=True,
            _disable_artifacts=True,
            _disable_model_checkpoints=True
        )
    }
    
    return WeightsAndBiasesCallback(
        metric_name="val_loss",
        wandb_kwargs=wandb_kwargs
    )


def generate_train_command(best_params, base_config_path, output_path=None):
    """Generate a train.py command to run with the best parameters
    
    Args:
        best_params (dict): Best parameters found by the hyperparameter search
        base_config_path (str): Path to the base configuration file
        output_path (str, optional): Path to save the best parameters config
        
    Returns:
        str: Command to train with the best parameters
    """
    # Create config file path for the best parameters
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = f"best_params_{timestamp}.yaml"
    
    # Generate parameter string for command line
    param_str = " ".join([f"--{k}={v}" for k, v in best_params.items() 
                         if k not in ['experiment_name']])
    
    # Create the command
    cmd = f"python entrypoints/train.py --config {base_config_path} {param_str} --experiment_name best_model"
    
    # Suggest writing parameters to a config file for reproducibility
    import yaml
    try:
        # Load base config
        with open(base_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update with best parameters
        for k, v in best_params.items():
            config[k] = v
        
        # Set a descriptive experiment name
        config['experiment_name'] = 'best_model'
        
        # Write to file
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Also provide a command to use this config
        config_cmd = f"python entrypoints/train.py --config {output_path}"
        return f"Option 1 - Use parameter file (recommended):\n{config_cmd}\n\nOption 2 - Use command line parameters:\n{cmd}"
    except Exception as e:
        log_event("warning", "train_command_generation", 
                 f"Failed to generate config file for best parameters: {str(e)}")
        return cmd


def main():
    """Main function to run the hyperparameter optimization"""
    global args, sweep_config, s3_config
    args = parse_args()
    
    # Check if AWS region is specified when needed
    if args.s3_bucket and not args.aws_region:
        print("Warning: S3 bucket specified but AWS region is not. Will try to use default region.")
        # Try to get region from environment
        args.aws_region = os.environ.get('AWS_DEFAULT_REGION')
        if not args.aws_region:
            print("No AWS region found in environment. CloudWatch logging and S3 uploads may fail.")
    
    # Set up logger with CloudWatch integration
    logger_setup(
        log_group="riskformer-hpo",
        debug=args.debug,
        use_cloudwatch=True,  # Enable CloudWatch logging
        region_name=args.aws_region,  # Use the same region as specified for S3
        profile_name=args.aws_profile  # Use the same AWS profile as specified for S3
    )
    
    # Set environment variables to disable wandb artifacts
    os.environ['WANDB_DISABLE_CODE'] = 'true'
    
    log_event("info", "run_sweep_started", "Started hyperparameter sweep", 
              base_config=args.base_config, 
              sweep_config=args.sweep_config,
              n_trials=args.n_trials,
              aws_region=args.aws_region)
    
    # Validate base configuration
    try:
        validate_config(args.base_config)
        log_event("info", "base_config_validated", "Base configuration file validated", config_path=args.base_config)
    except ValueError as e:
        log_event("error", "invalid_base_config", "Invalid base configuration file", config_path=args.base_config, error_message=str(e))
        return 1
    
    # Load hyperparameter sweep configuration
    try:
        sweep_config = load_sweep_config(args.sweep_config)
    except ValueError as e:
        log_event("error", "invalid_sweep_config", "Invalid sweep configuration file", error_message=str(e))
        return 1
    
    # Set up S3 configuration if bucket is provided
    s3_config = None
    if args.s3_bucket:
        s3_config = {
            's3_bucket': args.s3_bucket,
            's3_prefix': args.s3_prefix,
            'aws_profile': args.aws_profile,
            'aws_region': args.aws_region,
            'upload_threshold': args.upload_threshold,
            'upload_best_only': args.upload_best_only,
            'study_name': args.study_name,
        }
        
        # Initialize S3 client once to reuse
        s3_client = get_s3_client_with_fallback(args.aws_profile, args.aws_region, logger)
        if s3_client:
            s3_config['s3_client'] = s3_client
            log_event("info", "s3_upload_enabled", "S3 model upload enabled", 
                     bucket=args.s3_bucket, prefix=args.s3_prefix)
        else:
            log_event("warning", "s3_upload_disabled", "S3 model upload disabled due to client initialization failure")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up W&B callback for Optuna
    wandb_callback = setup_wandb_callback()
    
    # Adjust number of trials for debug mode
    if args.debug:
        args.n_trials = min(args.n_trials, 3)
        log_event("debug", "debug_mode", "Running in debug mode with reduced trials", n_trials=args.n_trials)
    
    # Create or load Optuna study
    study_name = args.study_name
    storage = args.storage
    direction = sweep_config.get('direction', 'minimize')
    
    log_event("info", "study_creation", "Creating Optuna study", 
             study_name=study_name, storage=storage, direction=direction)
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        load_if_exists=True,
    )
    
    # Run optimization
    log_event("info", "optimization_started", "Starting hyperparameter optimization", 
             n_trials=args.n_trials, timeout=args.timeout)
    
    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        callbacks=[wandb_callback],
    )
    
    # Get the best trial and parameters
    best_trial = study.best_trial
    best_params = best_trial.params
    best_value = best_trial.value
    
    log_event("info", "optimization_completed", "Hyperparameter optimization completed", 
             best_value=best_value, best_params=best_params)
    
    # Save study results
    results_file = os.path.join(args.output_dir, f"{study_name}_results.json")
    
    # Extract all trial information for the results
    trial_data = []
    for trial in study.trials:
        if trial.state.is_finished():
            trial_info = {
                'number': trial.number,
                'params': trial.params,
                'value': trial.value,
                'state': str(trial.state),
                'user_attrs': trial.user_attrs,
            }
            trial_data.append(trial_info)
    
    # Save comprehensive study results
    with open(results_file, "w") as f:
        json.dump({
            "best_params": best_params,
            "best_value": best_value,
            "best_trial_number": best_trial.number,
            "n_trials": args.n_trials,
            "n_completed_trials": len([t for t in study.trials if t.state.is_finished()]),
            "study_name": study_name,
            "direction": direction,
            "datetime": datetime.now().isoformat(),
            "optimization_metric": sweep_config.get('optimization_metric', 'val_loss'),
            "trials": trial_data,
        }, f, indent=4)
    
    log_event("info", "results_saved", "Study results saved", file_path=results_file)
    
    # Print information about the best parameters
    print("\n\n" + "="*80)
    print(f"Best trial: {best_trial.number}")
    print(f"Best value: {best_value}")
    print("Best parameters:")
    for param_name, param_value in best_params.items():
        print(f"  {param_name}: {param_value}")
    print("="*80 + "\n")
    
    print(f"Results saved to: {results_file}")
    
    # Save best parameters to a config file and generate training command
    best_params_file = os.path.join(args.output_dir, f"{study_name}_best_params.yaml")
    train_cmd = generate_train_command(best_params, args.base_config, best_params_file)
    
    print("\nTo train a final model with the best parameters, use:")
    print(train_cmd)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())