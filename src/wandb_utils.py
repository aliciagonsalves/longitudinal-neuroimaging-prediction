# src/wandb_utils.py (or src/experiment_utils.py)

import wandb
import torch
from pathlib import Path
import json # For type hinting, not direct use in this function
import time # For the initialize_wandb_run function

# Import your model classes using relative imports
try:
    from .models import BaselineLSTMRegressor, ModularLateFusionLSTM
except ImportError:
    from models import BaselineLSTMRegressor, ModularLateFusionLSTM

def initialize_wandb_run(base_project_config: dict, 
                         job_group: str, # E.g., "DataProcessing", "Training", "Analysis"
                         job_specific_type: str, # E.g., "BaselineLSTM-OASIS2", "HybridCNN-OASIS2"
                         run_specific_config: dict, # The HP dict or config dict for this run
                         custom_run_name_elements: list = None, # List of key elements for name, e.g., [HP['lr'], HP['batch_size']]
                         notes: str = ""):
    """
    Initializes and returns a new W&B run with standardized naming and job types.

    Args:
        base_project_config (dict): Loaded main config.json (must contain 'wandb':'entity' and 'wandb':'project_name').
        job_group (str): Broad category of the job (e.g., "Training", "Analysis").
        job_specific_type (str): Specific type of job (e.g., "BaselineLSTM-OASIS2").
        run_specific_config (dict): Configuration dictionary (e.g., HP) for this run.
        custom_run_name_elements (list, optional): List of strings/values to include in the run name.
        notes (str, optional): Notes for the W&B run.

    Returns:
        wandb.sdk.wandb_run.Run or None: The initialized W&B run object, or None if init fails.
    """
    try:
        entity = base_project_config['wandb']['entity']
        project = base_project_config['wandb']['project_name']
    except KeyError as e:
        print(f"Error: Missing 'wandb':'entity' or 'wandb':'project_name' in base_project_config: {e}")
        return None
    
    timestamp = time.strftime('%Y%m%d-%H%M%S') # Added seconds for more uniqueness
    
    name_parts = []
    if custom_run_name_elements: # Use these as the core of the name
        name_parts.extend([str(el) for el in custom_run_name_elements])
    else: # Fallback if no custom elements
        name_parts.append(job_specific_type)
    name_parts.append(timestamp) # Always add timestamp
    run_name = "-".join(name_parts)

    full_job_type = f"{job_group}-{job_specific_type}" 

    # Add job group and type to the config being logged for traceability
    if 'job_group' not in run_specific_config: run_specific_config['job_group'] = job_group
    if 'job_specific_type' not in run_specific_config: run_specific_config['job_specific_type'] = job_specific_type
    if 'run_name_generated' not in run_specific_config: run_specific_config['run_name_generated'] = run_name

    try:
        run = wandb.init(
            project=project,
            entity=entity,
            job_type=full_job_type,
            name=run_name,
            config=run_specific_config,
            notes=notes
        )
        print(f"W&B run '{run.name}' (Job: '{run.job_type}') initialized. View at: {run.url}")
        return run
    except Exception as e_init:
        print(f"Error during wandb.init: {e_init}")
        return None

def load_model_from_wandb_artifact(run_path: str, 
                                   base_config_dict: dict, # Main config.json content for fallbacks
                                   artifact_type_to_load: str = 'model', 
                                   artifact_alias_to_load: str = 'best', 
                                   device_to_load: str = 'cpu'):
    """
    Loads a PyTorch model from a W&B artifact associated with a specific training run.
    It infers model type and parameters primarily from the run's logged W&B config,
    using base_config_dict for critical fallbacks like CNN input shapes if not found.

    Args:
        run_path (str): Full W&B run path "entity/project/run_id".
        base_config_dict (dict): Loaded main project config.json. Used for fallbacks
                                 (e.g., cnn_model_params if not in run's W&B config).
        artifact_type_to_load (str): The type of the W&B artifact (e.g., 'model').
        artifact_alias_to_load (str): The alias of the W&B artifact (e.g., 'best').
        device_to_load (str or torch.device): Device to load the model onto.

    Returns:
        tuple: (loaded_model, original_run_config_dict, is_hybrid_flag) or (None, None, None) on failure.
               original_run_config_dict is the W&B config of the training run.
    """
    loaded_model = None
    original_run_config_dict = None
    is_hybrid_model_flag = None

    try:
        print(f"Attempting to load model artifact from W&B run: {run_path}, type: '{artifact_type_to_load}', alias: '{artifact_alias_to_load}'")
        wandb_api = wandb.Api(timeout=29) 
        
        training_run_obj = wandb_api.run(run_path)
        if not training_run_obj:
            print(f"Error: Could not fetch W&B run object for path {run_path}")
            return None, None, None # Return 3 Nones
            
        print(f"  Successfully fetched W&B run object: {training_run_obj.name} (ID: {training_run_obj.id})")
        original_run_config_dict = dict(training_run_obj.config) # Convert to a standard dict

        found_artifact_obj = None
        print(f"  Searching for artifact with type='{artifact_type_to_load}' and alias='{artifact_alias_to_load}'...")
        for art_obj in training_run_obj.logged_artifacts():
            if art_obj.type == artifact_type_to_load and artifact_alias_to_load in art_obj.aliases:
                found_artifact_obj = art_obj
                print(f"    Found artifact: {found_artifact_obj.name} (Type: {found_artifact_obj.type}, Aliases: {found_artifact_obj.aliases}, Version: {found_artifact_obj.version})")
                break 
        
        if not found_artifact_obj:
            print(f"  ERROR: No artifact with type '{artifact_type_to_load}' and alias '{artifact_alias_to_load}' found for run {run_path}.")
            print("  Available logged artifacts for this run:")
            for art_debug in training_run_obj.logged_artifacts():
                print(f"    - Name: {art_debug.name}, Type: {art_debug.type}, Aliases: {art_debug.aliases}, Version: {art_debug.version}")
            return None, None, None # Return 3 Nones
            
        model_artifact_dir = found_artifact_obj.download()
        checkpoint_files = list(Path(model_artifact_dir).glob("*.pth"))
        
        if not checkpoint_files:
            print(f"  ERROR: No .pth file found in downloaded artifact directory: {model_artifact_dir}")
            return None, None, None # Return 3 Nones
        
        checkpoint_file_path = checkpoint_files[0] 
        if len(checkpoint_files) > 1:
            print(f"  Warning: Multiple .pth files found in artifact. Using first one: {checkpoint_file_path.name}")
        print(f"  Model artifact downloaded. Checkpoint file: {checkpoint_file_path}")

        # Infer model type (job_type is more reliable if set consistently)
        job_type_str = training_run_obj.job_type.lower() if training_run_obj.job_type else ""
        run_name_str = training_run_obj.name.lower() if training_run_obj.name else ""
        is_hybrid_model_flag = "hybrid" in job_type_str or "hybrid" in run_name_str
        
        # Define default hyperparameter values
        default_lstm_hidden_size = 128
        default_num_layers = 2
        default_lstm_dropout_prob = 0.0 # A safe default like 0.0 or 0.3
        default_tabular_input_size = 0 # Should be an error if not found or inferred

        if is_hybrid_model_flag:
            print("  Instantiating ModularLateFusionLSTM...")
            # Get CNN params: Prefer snapshot from run config, then direct from run config, then from base_config_dict
            cnn_params_run = original_run_config_dict.get('project_cnn_model_params_snapshot', 
                                                           original_run_config_dict.get('cnn_model_params', {}))
            cnn_params_to_use = cnn_params_run if cnn_params_run else base_config_dict.get('cnn_model_params', {})

            cnn_input_channels = cnn_params_to_use.get('input_shape', [1,91,109,91])[0]
            cnn_output_features = cnn_params_to_use.get('output_features', 128)
            
            tabular_input_size = original_run_config_dict.get('input_size')
            if tabular_input_size is None:
                features_setting = original_run_config_dict.get('features_setting_for_dataset', 
                                                              original_run_config_dict.get('features', {}))
                time_varying = features_setting.get('time_varying', [])
                static = features_setting.get('static', [])
                tabular_input_size = len(time_varying) + len(static)
                if tabular_input_size == 0 and not (time_varying or static): # If keys were missing or lists empty
                     print(f"Warning: Could not determine tabular_input_size for hybrid model from W&B config of run {run_path}. Using default {default_tabular_input_size}.")
                     tabular_input_size = default_tabular_input_size


            mri_lstm_hs = original_run_config_dict.get('mri_lstm_hidden_size', 
                                         original_run_config_dict.get('lstm_hidden_size', default_lstm_hidden_size))
            tabular_lstm_hs = original_run_config_dict.get('tabular_lstm_hidden_size', 
                                             original_run_config_dict.get('lstm_hidden_size', default_lstm_hidden_size))
            
            num_layers_val = original_run_config_dict.get('num_lstm_layers')
            final_num_layers = num_layers_val if num_layers_val is not None else default_num_layers
            
            lstm_dropout_val = original_run_config_dict.get('lstm_dropout_prob')
            final_lstm_dropout = lstm_dropout_val if lstm_dropout_val is not None else default_lstm_dropout_prob

            loaded_model = ModularLateFusionLSTM(
                cnn_input_channels=cnn_input_channels,
                cnn_output_features=cnn_output_features,
                tabular_input_size=tabular_input_size,
                mri_lstm_hidden_size=mri_lstm_hs,
                tabular_lstm_hidden_size=tabular_lstm_hs,
                num_lstm_layers=final_num_layers,
                lstm_dropout_prob=final_lstm_dropout,
                modality_dropout_rate=0.0 # Always 0.0 for deterministic evaluation loading
            )
        else: # Baseline
            print("  Instantiating BaselineLSTMRegressor...")
            tabular_input_size_baseline = original_run_config_dict.get('input_size')
            if tabular_input_size_baseline is None:
                features_setting = original_run_config_dict.get('features_setting_for_dataset', 
                                                              original_run_config_dict.get('features', {}))
                time_varying = features_setting.get('time_varying', [])
                static = features_setting.get('static', [])
                tabular_input_size_baseline = len(time_varying) + len(static)
                if tabular_input_size_baseline == 0 and not (time_varying or static):
                     print(f"Warning: Could not determine tabular_input_size for baseline model from W&B config of run {run_path}. Using default {default_tabular_input_size}.")
                     tabular_input_size_baseline = default_tabular_input_size
            
            retrieved_hidden_size = original_run_config_dict.get('lstm_hidden_size', default_lstm_hidden_size)
            retrieved_num_layers_val = original_run_config_dict.get('num_lstm_layers')
            final_num_layers = retrieved_num_layers_val if retrieved_num_layers_val is not None else default_num_layers
            retrieved_dropout_prob_val = original_run_config_dict.get('lstm_dropout_prob')
            final_dropout_prob = retrieved_dropout_prob_val if retrieved_dropout_prob_val is not None else default_lstm_dropout_prob

            loaded_model = BaselineLSTMRegressor(
                input_size=tabular_input_size_baseline,
                hidden_size=retrieved_hidden_size,
                num_layers=final_num_layers,
                dropout_prob=final_dropout_prob
            )
        
        checkpoint_data = torch.load(checkpoint_file_path, map_location=device_to_load)
        if isinstance(checkpoint_data, dict) and 'model_state_dict' in checkpoint_data:
            loaded_model.load_state_dict(checkpoint_data['model_state_dict'])
            print("  Loaded model_state_dict from checkpoint dictionary.")
        else: 
            loaded_model.load_state_dict(checkpoint_data)
            print("  Loaded state_dict directly from checkpoint file.")

        loaded_model.to(device_to_load)
        loaded_model.eval() 
        print(f"  Model loaded successfully from {checkpoint_file_path} and set to eval mode.")
        
        return loaded_model, original_run_config_dict, is_hybrid_model_flag

    except Exception as e:
        print(f"Error in load_model_from_wandb_artifact for run '{run_path}': {e}")
        import traceback
        traceback.print_exc() 
        return None, None, None # Return 3 Nones