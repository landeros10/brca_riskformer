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

import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
import wandb

from riskformer.utils.logger_config import logger_setup, log_event
from riskformer.utils.training_utils import load_train_config, validate_config

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


def run_training_job(params, base_config_path, trial_number):
    """Run a training job with the given hyperparameters
    
    Args:
        params (dict): Hyperparameters for this trial
        base_config_path (str): Path to base configuration file
        trial_number (int): Current trial number
        
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
    # This assumes your train.py script outputs a JSON with metrics to stdout
    # or saves metrics to a file that we can read
    try:
        # Example: Read results from a file that train.py creates
        results_file = f"results_{experiment_name}.json"
        with open(results_file, "r") as f:
            results = json.load(f)
        
        # Clean up temporary files
        os.remove(temp_config_path)
        os.remove(results_file)
        
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
    # TODO: should this be val or test loss?
    
    # Generate hyperparameters for this trial
    params = define_search_space(trial, sweep_config)
    
    # Execute the training with these hyperparameters
    trial_number = trial.number
    results = run_training_job(params, args.base_config, trial_number)
    
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


def main():
    """Main function to run the hyperparameter optimization"""
    global args, sweep_config
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
        final_results = run_training_job(best_params, args.base_config, "final")
        log_event("info", "final_training_completed", "Final model training completed", metrics=final_results)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())