# src/paths_utils.py
"""
Utility functions for resolving and managing file/directory paths for the project
based on a central configuration.
"""
from pathlib import Path
import time # For timestamped fallback output dirs in get_notebook_run_output_dir

def get_dataset_paths(project_root: Path, 
                            app_config: dict, 
                            dataset_identifier: str, 
                            stage: str):
    """
    Resolves and returns a dictionary of key data and preprocessor paths 
    for a specific dataset and pipeline stage, using details from app_config.

    This function expects app_config (loaded from main config.json) to contain:
    - app_config['data']['output_dir_base']: Base for outputs like split data, preprocessors.
    - app_config['paths']['preprocessed_mri_dir']: Directory for preprocessed MRI scans.
    - app_config['preprocessing_config']['scaling_strategy'] and ['imputation_strategy'].
    - A section keyed by f"pipeline_artefact_locators_{dataset_identifier}" (e.g., 
      "pipeline_artefact_locators_oasis2") which itself contains keys like:
        - "feature_eng_subdir": Subdirectory for feature-engineered data.
        - "preprocessors_subdir": Subdirectory for saved preprocessor objects.
        - "train_data_fname", "val_data_fname", "test_data_fname": Filenames for data splits.
        - "scaler_fname_pattern", "imputer_fname_pattern": Filename patterns for preprocessors.

    Args:
        project_root (Path): The absolute root directory of the project.
        app_config (dict): The loaded main config.json content.
        dataset_identifier (str): Identifier for the dataset (e.g., "oasis2", "adni").
        stage (str): The pipeline stage. Expected values: "training", "testing", "analysis".
                     Determines which data splits (train/val vs. test) are returned.

    Returns:
        dict: A dictionary where keys are descriptive (e.g., 'train_data_parquet', 'scaler_path')
              and values are resolved Path objects.

    Raises:
        KeyError: If essential keys are missing from app_config.
        ValueError: If an unsupported stage is provided.
    """
    paths = {}
    try:
        # --- Base Output Directory (common for NB03, NB04 outputs) ---
        # Example: PROJECT_ROOT / "notebooks/outputs/"
        output_dir_base = project_root / app_config['data']['output_dir_base']
        
        # --- Dataset-Specific Locators ---
        # This key points to a dictionary in config.json defining subdirs and filenames for a dataset
        locators_key = f"pipeline_artefact_locators_{dataset_identifier}"
        locators = app_config.get(locators_key)
        if locators is None:
            raise KeyError(f"Missing section '{locators_key}' in config.json. This section should define "
                           "subdirectories like 'feature_eng_subdir', 'preprocessors_subdir', "
                           "and filenames/patterns for data splits and preprocessors.")

        # --- General Paths (e.g., for MRI data not tied to a specific notebook output stage) ---
        general_paths_config = app_config.get("paths", {})
        
        # --- Preprocessing Config (for filename patterns) ---
        preprocessing_cfg = app_config.get('preprocessing_config', {})

        # --- Construct specific output directories from Notebook 03 and 04 ---
        # Example: PROJECT_ROOT / "notebooks/outputs/03_Feature_Engineering_Splitting"
        feat_eng_dir = output_dir_base / locators.get("feature_eng_subdir", 
                                                      f"03_Feature_Engineering_Splitting_{dataset_identifier}_default")
        # Example: PROJECT_ROOT / "notebooks/outputs/04_Fit_Preprocessors"
        preproc_files_dir = output_dir_base / locators.get("preprocessors_subdir", 
                                                         f"04_Fit_Preprocessors_{dataset_identifier}_default")

        # --- Resolve Stage-Dependent Data Files ---
        if stage.lower() == "training":
            paths['train_data_parquet'] = feat_eng_dir / locators.get("train_data_fname", "cohort_train.parquet")
            paths['val_data_parquet'] = feat_eng_dir / locators.get("val_data_fname", "cohort_val.parquet")
        elif stage.lower() in ["testing", "analysis"]:
            # Assuming 'analysis' stage uses the 'test' data split and train for SHAP analysis
            paths['test_data_parquet'] = feat_eng_dir / locators.get("test_data_fname", "cohort_test.parquet")
            paths['train_data_parquet'] = feat_eng_dir / locators.get("train_data_fname", "cohort_train.parquet")
        else:
            raise ValueError(f"Unsupported 'stage': {stage}. Must be one of 'training', 'testing', 'analysis'.")
            
        # --- Resolve Preprocessor Paths ---
        scaler_fname = locators.get("scaler_fname_pattern", "{scaling_strategy}.joblib").format(
            scaling_strategy=preprocessing_cfg.get('scaling_strategy', 'standard_scaler').lower()
        )
        paths['scaler_path'] = preproc_files_dir / scaler_fname

        imputer_fname = locators.get("imputer_fname_pattern", "simple_imputer_{imputation_strategy}_{dataset_identifier}.joblib").format(
            imputation_strategy=preprocessing_cfg.get('imputation_strategy', 'median'),
            dataset_identifier=dataset_identifier
        )
        paths['imputer_path'] = preproc_files_dir / imputer_fname
        
        # --- Resolve MRI Data Directory ---
        # Example: PROJECT_ROOT / "data/preprocessed_mni"
        paths['mri_data_dir'] = project_root / general_paths_config.get("preprocessed_mri_dir", 
                                                                      f"data/preprocessed_mri_{dataset_identifier}_default")
        
        # --- Basic Existence Checks for Resolved INPUT Paths ---
        # These checks are for paths that should already exist when this function is called.
        # The calling notebook/script might perform more specific checks or handle FileNotFoundError.
        # For example, if stage is "training", train_data_parquet should exist (output of NB03).
        # If stage is "analysis", test_data_parquet, scaler_path, imputer_path should exist.
        # This is a simple check for all resolved paths; refine as needed.
        for key, path_obj in paths.items():
            if not path_obj.exists():
            # MRI data dir is a directory, others are files
                if path_obj.is_dir() and key == 'mri_data_dir': continue 
                print(f"Warning from paths_utils: Path for '{key}' does not exist: {path_obj}")
        
    except KeyError as e:
        print(f"Error resolving paths: Missing a critical key '{e}' in the application config (config.json) "
              f"or in the '{locators_key}' section.")
        raise # Re-raise to indicate failure to the caller
    except Exception as e_general:
        print(f"An unexpected error occurred in get_dataset_stage_paths: {e_general}")
        raise
        
    return paths


def get_notebook_run_output_dir(project_root: Path, 
                                app_config: dict, 
                                notebook_locator_key: str, # Key in locators dict for this notebook's base output, e.g., "analysis_subdir_nb08"
                                wandb_run_obj=None,      # Active W&B run object
                                dataset_identifier: str = "oasis2"):
    """
    Creates and returns a run-specific output directory for a notebook's artifacts (plots, tables).
    The directory structure will be: <output_dir_base>/<notebook_base_folder>/<wandb_run_name_or_timestamp>/.
    Not used when the output dir is static.

    Args:
        project_root (Path): The absolute root directory of the project.
        app_config (dict): The loaded main config.json content.
        notebook_locator_key (str): The key within app_config's dataset-specific locators
                                    that specifies the base folder name for this notebook's outputs.
                                    Example: "analysis_subdir_nb08" (defined in config.json).
        wandb_run_obj (wandb.Run, optional): The active W&B run object. If provided, its name
                                             is used for the run-specific subdirectory.
        dataset_identifier (str): Identifier for the dataset (e.g., "oasis2", "adni").

    Returns:
        Path: The Path object for the created run-specific output directory.
    """
    final_run_output_dir = None
    try:
        output_dir_base = project_root / app_config['data']['output_dir_base']
        
        locators_key_full = f"pipeline_artefact_locators_{dataset_identifier}"
        locators = app_config.get(locators_key_full)
        if locators is None:
            raise KeyError(f"Missing section '{locators_key_full}' in config.json for dataset '{dataset_identifier}'.")

        notebook_base_folder_name = locators.get(notebook_locator_key)
        if notebook_base_folder_name is None:
            # Fallback if specific key for notebook output folder isn't in config locators
            notebook_base_folder_name = f"{notebook_locator_key}_outputs_default" 
            print(f"Warning: Using default notebook output folder name: {notebook_base_folder_name}")
            
        notebook_level_output_dir = output_dir_base / notebook_base_folder_name
        
        run_specific_subdir_name = ""
        if wandb_run_obj and hasattr(wandb_run_obj, 'name') and wandb_run_obj.name:
            run_specific_subdir_name = wandb_run_obj.name
        else:
            run_specific_subdir_name = f"local_run_{time.strftime('%Y%m%d-%H%M%S')}"
            
        final_run_output_dir = notebook_level_output_dir / run_specific_subdir_name
        final_run_output_dir.mkdir(parents=True, exist_ok=True)
        
    except KeyError as e:
        print(f"Error creating notebook run output directory: Missing key {e} in app_config.")
        # Fallback to a simpler structure to avoid outright failure if possible
        fallback_dir = project_root / "outputs_fallback" / f"{notebook_locator_key}_{time.strftime('%Y%m%d-%H%M%S')}"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        print(f"Using fallback output directory: {fallback_dir}")
        return fallback_dir
    except Exception as e_general:
        print(f"Unexpected error creating notebook run output directory: {e_general}")
        # Fallback
        fallback_dir = project_root / "outputs_fallback" / f"{notebook_locator_key}_{time.strftime('%Y%m%d-%H%M%S')}"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        print(f"Using fallback output directory: {fallback_dir}")
        return fallback_dir
        
    return final_run_output_dir