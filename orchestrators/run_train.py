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

import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
import wandb
import boto3
from botocore.exceptions import ClientError

from riskformer.utils.logger_config import logger_setup, log_event
from riskformer.utils.training_utils import load_train_config, validate_config
from riskformer.utils.aws_utils import (
    is_s3_path,
    initialize_boto3_session,
    initialize_s3_client,
    upload_large_files_to_bucket,
    generate_s3_key
)

logger = logging.getLogger(__name__)

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
    
    # Write the updated config to a temporary file
    import yaml
    temp_config_path = f"temp_config_trial_{trial_number}.yaml"
    with open(temp_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Build command to run training script
    cmd = [
        "python", "entrypoints/train.py",
        "--config", temp_config_path,
        "--use_wandb",
        "--save_results"
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
        log_event("info", "trial_completed", f"Trial {trial_number} completed successfully")
    except subprocess.CalledProcessError as e:
        log_event("error", "trial_failed", f"Trial {trial_number} failed", 
                 error=str(e), stderr=e.stderr)
        # Return a very poor result to signal failure to Optuna
        return {"val_loss": float('inf')}
    
    # Parse results from the training job
    try:
        # Example: Read results from a file that train.py creates
        results_file = f"results_{experiment_name}.json"
        with open(results_file, "r") as f:
            results = json.load(f)
            
        # Upload model to S3 if configured and the model meets criteria
        if s3_config and s3_config.get('s3_bucket'):
            # Check if we have a best checkpoint path
            best_checkpoint_path = results.get('best_checkpoint_path')
            if best_checkpoint_path and os.path.exists(best_checkpoint_path):
                # Check if the model meets the performance threshold
                optimization_metric = sweep_config.get('optimization_metric', 'val_loss')
                metric_value = results.get(optimization_metric, float('inf'))
                direction = sweep_config.get('direction', 'minimize')
                
                # Determine if we should upload based on threshold and direction
                should_upload = True
                threshold = s3_config.get('upload_threshold')
                if threshold is not None:
                    if direction == 'minimize':
                        should_upload = metric_value <= threshold
                    else:  # maximize
                        should_upload = metric_value >= threshold
                
                if should_upload:
                    # Initialize S3 client if not already done
                    s3_client = s3_config.get('s3_client')
                    if not s3_client:
                        s3_client = get_s3_client(
                            s3_config.get('aws_profile'),
                            s3_config.get('aws_region')
                        )
                        s3_config['s3_client'] = s3_client
                    
                    if s3_client:
                        # Build S3 key
                        s3_key = build_s3_key(
                            experiment_name=s3_config.get('study_name', 'riskformer-hpo'),
                            trial_number=trial_number,
                            model_file=best_checkpoint_path,
                            is_best=True,
                            prefix=s3_config.get('s3_prefix', 'models')
                        )
                        
                        # Upload model to S3
                        s3_path = upload_model_to_s3(
                            best_checkpoint_path,
                            s3_config['s3_bucket'],
                            s3_key,
                            s3_client
                        )
                        
                        if s3_path:
                            # Store S3 path in results
                            results['s3_model_path'] = s3_path
                            
                            # Update results file with S3 path
                            with open(results_file, "w") as f:
                                json.dump(results, f, indent=4)
                            
                            # Log to W&B if enabled
                            if 'wandb' in sys.modules:
                                try:
                                    wandb.log({
                                        f"trial_{trial_number}/s3_model_path": s3_path,
                                        f"trial_{trial_number}/{optimization_metric}": metric_value
                                    })
                                except Exception as e:
                                    log_event("warning", "wandb_log_failed", "Failed to log to W&B", error=str(e))
                    else:
                        log_event("warning", "s3_upload_skipped", "Skipping S3 upload due to client initialization failure")
        
        # Clean up temporary files
        try:
            os.remove(temp_config_path)
        except Exception as e:
            log_event("warning", "temp_config_cleanup_failed", "Failed to clean up temporary config file", error=str(e))
        
        return results
    except Exception as e:
        log_event("error", "parsing_results_failed", f"Failed to parse results for trial {trial_number}", 
                 error=str(e))
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
    
    # Extract the primary metric for optimization
    primary_metric = results.get(sweep_config.get('optimization_metric', 'val_loss'), float('inf'))
    
    # Log additional metrics to Optuna
    for key, value in results.items():
        if key != sweep_config.get('optimization_metric', 'val_loss') and isinstance(value, (int, float)):
            trial.set_user_attr(key, value)
    
    return primary_metric


def setup_wandb_callback():
    """Set up the Weights & Biases callback for Optuna
    
    Returns:
        WeightsAndBiasesCallback: Configured W&B callback for Optuna
    """
    wandb_kwargs = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "name": args.study_name,
        "config": vars(args),
    }
    
    return WeightsAndBiasesCallback(
        metric_name="val_loss",
        wandb_kwargs=wandb_kwargs
    )


def get_s3_client(aws_profile=None, aws_region=None):
    """Initialize an S3 client
    
    Args:
        aws_profile (str): AWS profile name
        aws_region (str): AWS region
        
    Returns:
        boto3.client: S3 client or None if initialization fails
    """
    try:
        # Try to initialize with the provided profile and region
        s3_client = initialize_s3_client(aws_profile, aws_region)
        
        # If that fails, try with default credentials
        if not s3_client:
            log_event("warning", "s3_client_init_failed", 
                     "Failed to initialize S3 client with provided profile, falling back to default credentials")
            s3_client = boto3.client('s3')
        
        # Test the connection with a simple operation
        s3_client.list_buckets()
        
        log_event("info", "s3_client_initialized", "S3 client initialized successfully",
                 profile=aws_profile, region=aws_region)
        return s3_client
    except Exception as e:
        log_event("error", "s3_client_failed", "Failed to initialize S3 client", 
                 error=str(e), profile=aws_profile, region=aws_region)
        return None


def upload_model_to_s3(local_path, s3_bucket, s3_key, s3_client=None):
    """Upload a model checkpoint to S3
    
    Args:
        local_path (str): Path to local model file
        s3_bucket (str): S3 bucket name
        s3_key (str): S3 key for the file
        s3_client (boto3.client): S3 client to use for upload
        
    Returns:
        bool: True if upload successful, False otherwise
    """
    try:
        # Check if the file exists
        if not os.path.exists(local_path):
            log_event("error", "model_file_not_found", "Model file not found", local_path=local_path)
            return False
            
        # Create a client if one wasn't provided
        if s3_client is None:
            s3_client = boto3.client('s3')

        # Use upload_large_files_to_bucket for larger files
        if os.path.getsize(local_path) > 100 * 1024 * 1024:  # 100MB
            # This function has better handling for large files with multipart uploads
            prefix = '/'.join(s3_key.split('/')[:-1])
            filename = s3_key.split('/')[-1]
            upload_large_files_to_bucket(
                s3_client, 
                s3_bucket,
                [local_path],
                file_names=[filename], 
                prefix=prefix,
                reupload=True
            )
        else:
            # Use direct upload for smaller files
            s3_client.upload_file(local_path, s3_bucket, s3_key)
            
        log_event("info", "s3_upload_success", "Successfully uploaded model to S3", 
                 local_path=local_path, s3_bucket=s3_bucket, s3_key=s3_key)
        
        # Return the full S3 path
        return f"s3://{s3_bucket}/{s3_key}"
    except Exception as e:
        log_event("error", "s3_upload_failed", "Failed to upload model to S3", 
                 local_path=local_path, s3_bucket=s3_bucket, s3_key=s3_key, error_message=str(e))
        return False


def build_s3_key(experiment_name, trial_number, model_file, is_best=False, prefix="models"):
    """Build a standardized S3 key for model checkpoints
    
    Args:
        experiment_name (str): Name of the experiment
        trial_number (int/str): Trial number or identifier
        model_file (str): Name or path of the model file
        is_best (bool): Whether this is the best model
        prefix (str): S3 prefix
        
    Returns:
        str: S3 key for the model checkpoint
    """
    # Extract just the filename from the model path
    model_name = os.path.basename(model_file)
    
    # Generate a timestamp
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    
    # Create a unique run ID
    run_id = str(uuid.uuid4())[:8]
    
    if is_best:
        s3_key = f"{prefix}/{experiment_name}/best/trial_{trial_number}_{timestamp}_{run_id}_{model_name}"
    else:
        s3_key = f"{prefix}/{experiment_name}/checkpoints/trial_{trial_number}_{timestamp}_{run_id}_{model_name}"
    
    return s3_key


def main():
    """Main function to run the hyperparameter optimization"""
    global args, sweep_config, s3_config
    args = parse_args()
    
    # Set up logger
    logger_setup(
        log_group="riskformer-hpo",
        debug=args.debug,
    )
    
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
        s3_client = get_s3_client(args.aws_profile, args.aws_region)
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
    with open(results_file, "w") as f:
        json.dump({
            "best_params": best_params,
            "best_value": best_value,
            "n_trials": args.n_trials,
            "study_name": study_name,
            "datetime": datetime.now().isoformat(),
        }, f, indent=4)
    
    log_event("info", "results_saved", "Study results saved", file_path=results_file)
    
    # Train final model with best parameters (optional)
    final_training = input("Do you want to train a final model with the best parameters? (y/n): ")
    if final_training.lower() == "y":
        log_event("info", "final_training_started", "Starting final model training with best parameters")
        final_results = run_training_job(best_params, args.base_config, "final", s3_config)
        log_event("info", "final_training_completed", "Final model training completed", metrics=final_results)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())