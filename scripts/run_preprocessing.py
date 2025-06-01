# scripts/run_preprocessing.py
"""
Executes the MRI preprocessing pipeline for a cohort of subjects.

This script manages the following workflow:
1.  Loads main project configuration and necessary file paths.
2.  Fetches a standard MNI template using TemplateFlow if registration is enabled.
3.  Loads a list of scans to be processed, based on prior verification (from Notebook 01)
    and final cohort definition (from Notebook 02).
4.  Implements a resume capability by checking for already existing final preprocessed outputs.
5.  Calls the 'preprocess_scan.py' script in parallel (using joblib) for each selected
    T1w MRI scan that needs processing.
6.  Summarizes the results of the preprocessing operations.

This script is intended to be run from the command line, typically from the project root.
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
from joblib import Parallel, delayed
import subprocess
import time
import os 

# Attempt to import TemplateFlow and set availability flag
try:
    from templateflow import api as tf
    TEMPLATEFLOW_AVAILABLE = True
    print("Successfully imported 'templateflow.api'. Template downloading is available.")
except ImportError:
    print("Warning: templateflow-client not installed (e.g., `pip install templateflow-client`). "
          "Template downloading will fail if registration to a TemplateFlow template is configured.")
    TEMPLATEFLOW_AVAILABLE = False

def run_single_scan_preprocessing(scan_info: dict, 
                                  script_path: Path, 
                                  final_output_dir_for_scan: Path, 
                                  intermediate_dir_base_for_scan: Path, 
                                  config_path_for_scan: Path, 
                                  template_file_path_str: str | None) -> tuple[str, bool, str]:
    """
    Worker function to execute 'preprocess_scan.py' for a single MRI scan via subprocess.

    Args:
        scan_info (dict): Dictionary containing information about the scan. 
                          Must include 'subject_id', 'scan_id' (MRI ID), and 'input_img_path'.
        script_path (Path): Absolute path to the 'preprocess_scan.py' script.
        final_output_dir_for_scan (Path): Base directory where the final processed scan will be saved
                                          by 'preprocess_scan.py'.
        intermediate_dir_base_for_scan (Path): Base directory where 'preprocess_scan.py' will create
                                               a subject-specific subdirectory for intermediate files.
        config_path_for_scan (Path): Absolute path to the main 'config.json' file.
        template_file_path_str (str or None): Absolute path (as a string) to the MNI template file.
                                              Passed to 'preprocess_scan.py' if registration is enabled.
                                              Will be None if registration is disabled in the config.

    Returns:
        tuple: Contains (scan_id, success_status (bool), message_or_error (str)).
    """
    subject_id = scan_info['subject_id']
    scan_id = scan_info['scan_id'] # This is the MRI ID from the clinical data sheet
    input_img_path = Path(scan_info['input_img_path']) # Ensure it's a Path object

    # Construct the command to run the preprocessing script
    # using the same Python interpreter that is running this orchestrator script.
    command_to_run = [
        sys.executable, 
        str(script_path),
        '--input-img', str(input_img_path),
        '--output-dir', str(final_output_dir_for_scan),
        '--intermediate-dir', str(intermediate_dir_base_for_scan),
        '--config-path', str(config_path_for_scan),
        '--subject-id', str(subject_id),
        '--scan-id', str(scan_id)
    ]

    # Add template path argument to the command ONLY if a template path is provided
    if template_file_path_str:
        command_to_run.extend(['--template-path', str(template_file_path_str)])

    print(f"\n---> Starting preprocessing task for: Subject {subject_id}, Scan {scan_id}")
    print(f"     Input: {input_img_path.name}")
    # Log the full command for reproducibility and easier debugging if needed
    # print(f"     Executing Command: {' '.join(command_to_run)}") 

    try:
        # Execute the subprocess.
        # 'check=True' will raise CalledProcessError if preprocess_scan.py returns a non-zero exit code.
        # 'capture_output=True' captures stdout and stderr from the subprocess.
        # 'text=True' decodes stdout/stderr as text (requires encoding).
        process_result = subprocess.run(
            command_to_run, 
            check=True, 
            capture_output=True, 
            text=True, 
            encoding='utf-8' # Specify encoding for robust text decoding
        )
        # If check=True and an error occurs, CalledProcessError is raised.
        # If successful, preprocess_scan.py exits with 0.
        print(f"<--- Finished preprocessing for: Subject {subject_id}, Scan {scan_id} --- SUCCESS")
        # Optional: print stdout from preprocess_scan.py if needed for detailed logging,
        # but it can be verbose. preprocess_scan.py already prints its progress.
        # if process_result.stdout:
        #     print(f"     Stdout from preprocess_scan.py:\n{process_result.stdout.strip()}")
        return (scan_id, True, "Preprocessing successful.")
    
    except subprocess.CalledProcessError as e_called_process:
        # This block executes if preprocess_scan.py returns a non-zero exit code (indicating failure)
        print(f"<--- Finished preprocessing for: Subject {subject_id}, Scan {scan_id} --- FAILED (Script Error)")
        print(f"     Error Code: {e_called_process.returncode}")
        # Log a snippet of stdout and stderr for debugging the failed script
        stdout_snippet = e_called_process.stdout.strip()[:500] if e_called_process.stdout else "None"
        stderr_snippet = e_called_process.stderr.strip()[:1000] if e_called_process.stderr else "None"
        print(f"     Stdout (first 500 chars): {stdout_snippet}")
        print(f"     Stderr (first 1000 chars): {stderr_snippet}")
        return (scan_id, False, f"Script failed (exit code {e_called_process.returncode}). Stderr snippet: {stderr_snippet[:200]}")
    
    except Exception as e_general_subprocess:
        # Catch any other unexpected Python errors during subprocess setup or execution
        print(f"<--- Finished preprocessing for: Subject {subject_id}, Scan {scan_id} --- FAILED (Unexpected Python Error)")
        print(f"     Error during subprocess call: {e_general_subprocess}")
        return (scan_id, False, f"Unexpected Python error: {e_general_subprocess}")

def main(cli_args: argparse.Namespace):
    """
    Main function to orchestrate the MRI preprocessing pipeline for a cohort.

    It performs the following steps:
    1. Loads the main project configuration (`config.json`).
    2. Defines and creates necessary input/output directories.
    3. If registration is enabled in the config, fetches the MNI template using TemplateFlow.
    4. Loads the list of scans to be processed from the verification CSV (output of NB01)
       and filters it against the final analysis cohort (output of NB02).
    5. Selects the specific MPRAGE scan for each session based on config.
    6. Implements a resume capability by checking for already existing final preprocessed output files.
    7. Calls the `preprocess_scan.py` script in parallel for each scan that needs processing,
       using `joblib.Parallel`.
    8. Summarizes the overall preprocessing results (success/failure counts).
    
    Args:
        cli_args (argparse.Namespace): Command-line arguments, including 'config_path' and 'n_jobs'.
    """
    overall_pipeline_start_time = time.time()
    print("--- Starting MRI Preprocessing Pipeline Orchestrator ---")

    # --- 1. Load Configuration and Define Project Root ---
    config_file_path: Path = Path(cli_args.config_path).resolve()
    if not config_file_path.is_file():
        print(f"Error: Main configuration file not found at {config_file_path}")
        sys.exit(1)

    PROJECT_ROOT: Path = config_file_path.parent # Assuming config.json is at project root
    app_config: dict
    try:
        with open(config_file_path, 'r', encoding='utf-8') as f:
            app_config = json.load(f)
        print(f"Loaded main configuration from: {config_file_path}")
    except Exception as e_cfg:
        print(f"Error reading or parsing main config file '{config_file_path}': {e_cfg}")
        sys.exit(1)

    # --- 2. Resolve and Set Up All Necessary Paths ---
    print("\n--- Resolving and Setting Up File Paths ---")
    paths_config: dict = app_config.get('paths', {})
    preprocessing_pipeline_config: dict = app_config.get('preprocessing_config', {}) # Renamed from prep_config for clarity
    
    # Focusing on a specific dataset-c"oasis2".
    # In future, make an argument or part of app_config if the script needs to be more generic.
    dataset_identifier: str = "oasis2" 
    locators_key: str = f"pipeline_artefact_locators_{dataset_identifier}"
    dataset_locators: dict = app_config.get(locators_key, {})

    if not paths_config:
        print("Warning: 'paths' section missing in config.json. Some path resolutions might use defaults or fail.")
    if not preprocessing_pipeline_config:
        print("Warning: 'preprocessing_config' section missing in config.json. Preprocessing steps might use defaults or fail.")
    if not dataset_locators:
        print(f"Warning: '{locators_key}' section missing in config.json. "
              "Resolution of dataset-specific file/subdir names will use defaults.")

    try:
        # Base directory for outputs of previous notebooks (NB01, NB02, NB03, NB04)
        # This comes from config.json: data.output_dir_base
        notebook_outputs_base_dir: Path = PROJECT_ROOT / app_config['data']['output_dir_base']

        # Inputs from previous notebooks, using locators for subdirs and filenames
        nb01_out_subdir = notebook_outputs_base_dir / dataset_locators.get("exploration_subdir", f"01_Data_Exploration_{dataset_identifier}_default")
        verification_csv_path: Path = nb01_out_subdir / dataset_locators.get("verification_csv_fname", "verification_details.csv")

        nb02_out_subdir = notebook_outputs_base_dir / dataset_locators.get("cohort_def_subdir", f"02_Cohort_Definition_{dataset_identifier}_default")
        final_cohort_definition_path: Path = nb02_out_subdir / dataset_locators.get("final_cohort_fname", "final_analysis_cohort.csv")
        
        # Output directories for this preprocessing pipeline
        final_processed_output_dir: Path = PROJECT_ROOT / paths_config['preprocessed_mri_dir']
        base_intermediate_files_dir: Path = PROJECT_ROOT / paths_config['intermediate_mri_dir']
        
        # TemplateFlow cache directory
        templateflow_home_dir: Path = PROJECT_ROOT / paths_config.get('template_cache_dir', f'data/templates_tf_{dataset_identifier}_default')

        # Path to the worker script (preprocess_scan.py)
        mri_processing_script_path: Path = PROJECT_ROOT / 'scripts' / 'preprocess_scan.py'

        # Create directories if they don't exist
        final_processed_output_dir.mkdir(parents=True, exist_ok=True)
        base_intermediate_files_dir.mkdir(parents=True, exist_ok=True)
        templateflow_home_dir.mkdir(parents=True, exist_ok=True)
        
        # Set TEMPLATEFLOW_HOME environment variable for TemplateFlow client
        os.environ['TEMPLATEFLOW_HOME'] = str(templateflow_home_dir.resolve())
        
        # Verify existence of critical input files and the worker script
        if not verification_csv_path.is_file():
            raise FileNotFoundError(f"Input verification CSV not found: {verification_csv_path}. Please run Notebook 01.")
        if not final_cohort_definition_path.is_file():
            raise FileNotFoundError(f"Input final cohort definition CSV not found: {final_cohort_definition_path}. Please run Notebook 02.")
        if not mri_processing_script_path.is_file(): 
            raise FileNotFoundError(f"MRI processing worker script not found: {mri_processing_script_path}")

        # Print resolved paths for user confirmation
        print(f"Project Root: {PROJECT_ROOT}")
        print(f"Main Config File: {config_file_path}")
        print(f"Verification List CSV (Input from NB01): {verification_csv_path}")
        print(f"Final Cohort Definition CSV (Input from NB02): {final_cohort_definition_path}")
        print(f"Final Output Directory (for processed scans): {final_processed_output_dir}")
        print(f"Base Directory for Intermediate Files: {base_intermediate_files_dir}")
        print(f"TemplateFlow Cache (TEMPLATEFLOW_HOME): {os.environ['TEMPLATEFLOW_HOME']}")
        print(f"Worker Script (preprocess_scan.py): {mri_processing_script_path}")

    except KeyError as e_key: 
        print(f"Error: Missing a critical key in config.json needed for path setup: {e_key}")
        sys.exit(1)
    except FileNotFoundError as e_file: 
        print(f"Error: A critical input file or script was not found: {e_file}")
        sys.exit(1)
    except Exception as e_path_setup:
        print(f"An unexpected error occurred during path setup: {e_path_setup}")
        sys.exit(1)

    # --- 3. Handle TemplateFlow Download (if registration is enabled) ---
    registration_template_path_str = None 
    if preprocessing_pipeline_config.get('run_register', False):
        print("\n--- Checking/Fetching MNI Template via TemplateFlow ---")
        if not TEMPLATEFLOW_AVAILABLE:
            print("Error: Registration is enabled in config, but TemplateFlow library is not installed. "
                  "Please install with `pip install templateflow` or `conda install -c conda-forge templateflow`.")
            sys.exit(1)
        try:
            templateflow_query_details = preprocessing_pipeline_config.get('templateflow_id', {})
            if not templateflow_query_details:
                print("Error: 'templateflow_id' details (template name, resolution, etc.) are missing "
                      "in 'preprocessing_config' section of config.json, but registration is enabled.")
                sys.exit(1)

            print(f"Requesting template using details: {templateflow_query_details}")
            
            # Ensure all arguments for tf.get are strings or ints as expected by TemplateFlow
            tf_get_args = {
                key: val for key, val in templateflow_query_details.items() 
                if key in ['template', 'atlas', 'cohort', 'desc', 'extension', 'hemi', 
                           'label', 'resolution', 'space', 'suffix', 'datatype']
            }
            # Ensure resolution is int if present
            if 'resolution' in tf_get_args and tf_get_args['resolution'] is not None:
                tf_get_args['resolution'] = int(tf_get_args['resolution'])

            template_paths_from_tf = tf.get(**tf_get_args) # Use dictionary unpacking

            resolved_template_path = None
            if isinstance(template_paths_from_tf, list):
                if template_paths_from_tf: # List is not empty
                    first_path_obj = Path(template_paths_from_tf[0])
                    if first_path_obj.exists(): resolved_template_path = first_path_obj
                    else: print(f"Warning: TemplateFlow returned list but first path doesn't exist: {first_path_obj}")
                else: print("Warning: TemplateFlow query returned an empty list (no template found).")
            elif isinstance(template_paths_from_tf, (str, Path)):
                 single_path_obj = Path(template_paths_from_tf)
                 if single_path_obj.exists(): resolved_template_path = single_path_obj
                 else: print(f"Warning: TemplateFlow returned a single path that doesn't exist: {single_path_obj}")
            elif template_paths_from_tf is None:
                 print("Warning: TemplateFlow query returned None (no template found).")

            if resolved_template_path:
                registration_template_path_str = str(resolved_template_path.resolve()) # Store absolute path as string
                print(f"Using TemplateFlow template found at: {registration_template_path_str}")
            else:
                raise FileNotFoundError(f"TemplateFlow could not find a valid, existing template file "
                                        f"matching criteria: {templateflow_query_details}")
        except Exception as e_tf:
            print(f"Error during TemplateFlow operation: {e_tf}")
            print("  - Check TemplateFlow installation and network connection.")
            print("  - Verify 'templateflow_id' in config.json matches available TemplateFlow templates.")
            print(f"  - Ensure TEMPLATEFLOW_HOME ('{os.environ['TEMPLATEFLOW_HOME']}') is writable.")
            sys.exit(1)

    # --- 4. Load and Filter Scan List for Processing ---
    print(f"\n--- Loading and Filtering Scan List ---")
    print(f"Using verification data from: {verification_csv_path.name}")
    print(f"Using final cohort definition from: {final_cohort_definition_path.name}")
    try:
        verification_df = pd.read_csv(verification_csv_path)
        final_cohort_df = pd.read_csv(final_cohort_definition_path)
        final_cohort_mri_ids_set = set(final_cohort_df['MRI ID'].unique()) # For efficient lookup

        print(f"Loaded verification data: {len(verification_df)} entries.")
        print(f"Loaded final cohort: {len(final_cohort_mri_ids_set)} unique MRI IDs from "
              f"{final_cohort_df['Subject ID'].nunique()} subjects.")

        # Filter 1: Keep only scans where the MRI folder was verified to exist
        scans_with_folders_df = verification_df[verification_df['mri_folder_exists'] == True].copy()
        
        # Filter 2: Keep only scans belonging to the subjects in the final defined cohort
        # This assumes 'mri_id' in verification_df corresponds to 'MRI ID' in final_cohort_df
        scans_in_final_cohort_df = scans_with_folders_df[
            scans_with_folders_df['mri_id'].isin(final_cohort_mri_ids_set)
        ].copy()
        print(f"Filtered to {len(scans_in_final_cohort_df)} verified scans belonging to the final cohort.")

        # Filter 3: Select specific MPRAGE scan (e.g., "mpr-1") and construct full input path
        mpr_scan_to_select = preprocessing_pipeline_config.get('select_mpr_scan', 'mpr-1')
        if 'mri_base_path_used' not in scans_in_final_cohort_df.columns:
             raise KeyError("'mri_base_path_used' column is missing in the verification CSV. "
                            "This column is needed to construct input image paths.")

        scan_processing_jobs = []
        # Use a set to track MRI IDs already added to jobs, as verification_df might have multiple entries per MRI ID (e.g. from different visits)
        # but we typically preprocess each unique MRI scan (identified by MRI ID) only once.
        mri_ids_added_to_jobs = set() 

        for _, row_info in scans_in_final_cohort_df.iterrows():
            mri_id = row_info['mri_id']
            if mri_id in mri_ids_added_to_jobs: # Process each unique MRI scan only once
                 continue

            subject_id = row_info['subject_id']
            # Construct absolute path to the base directory where this MRI ID's folder resides
            mri_session_base_path = Path(row_info['mri_base_path_used'])
            if not mri_session_base_path.is_absolute():
                 mri_session_base_path = PROJECT_ROOT / mri_session_base_path
            
            # Path to the specific .img file (as expected by preprocess_scan.py's original logic)
            # Assumes structure: <mri_session_base_path>/<mri_id>/RAW/<scan_to_select>.nifti.img
            selected_input_img_file = mri_session_base_path / mri_id / 'RAW' / f"{mpr_scan_to_select}.nifti.img"

            if selected_input_img_file.is_file():
                scan_processing_jobs.append({
                    'subject_id': subject_id,
                    'scan_id': mri_id, # This is the unique MRI session identifier
                    'input_img_path': selected_input_img_file 
                })
                mri_ids_added_to_jobs.add(mri_id)
            else:
                print(f"  Skipping Scan: Selected input file {selected_input_img_file} not found for MRI ID {mri_id}, Subject {subject_id}.")
        
        print(f"Identified {len(scan_processing_jobs)} unique scans with existing specified input files for potential processing.")

        # --- Resume Logic: Filter out scans if their final output already exists ---
        final_output_filename_suffix = preprocessing_pipeline_config.get('final_output_suffix', '_preprocessed_mni.nii.gz')
        scans_actually_to_run = []
        num_existing_outputs_skipped = 0
        print("\nChecking for already processed final output files (resume capability)...")

        for job_details in scan_processing_jobs:
            subject_id = job_details['subject_id']
            scan_id = job_details['scan_id'] # MRI ID
            filename_base = f"{subject_id}_{scan_id}"
            expected_final_output_file = final_processed_output_dir / f"{filename_base}{final_output_filename_suffix}"

            if expected_final_output_file.exists():
                num_existing_outputs_skipped += 1
                continue # Skip this job as output already exists
            scans_actually_to_run.append(job_details)

        if num_existing_outputs_skipped > 0:
            print(f"Resuming: Found and skipped {num_existing_outputs_skipped} scans whose final outputs already exist.")
        
        if not scans_actually_to_run:
            print("All identified scans seem to be already processed based on existing final outputs. Exiting.")
            sys.exit(0)
        
        print(f"Number of scans queued for preprocessing: {len(scans_actually_to_run)}")

    except FileNotFoundError as e_file_filter:
        print(f"Error: A required CSV file for scan list filtering not found: {e_file_filter}")
        sys.exit(1)
    except KeyError as e_key_filter:
        print(f"Error: A required column key is missing in a CSV for scan list filtering: {e_key_filter}")
        sys.exit(1)
    except Exception as e_filter: 
        print(f"Error during scan list preparation: {e_filter}")
        sys.exit(1)

    # --- 5. Run Preprocessing in Parallel ---
    num_parallel_jobs = cli_args.n_jobs # Value from command line arg
    # joblib uses number of CPUs if n_jobs is -1
    cores_to_use_str = f"{num_parallel_jobs if num_parallel_jobs != -1 else os.cpu_count() or 1}"
    print(f"\n--- Starting Parallel Preprocessing for {len(scans_actually_to_run)} scans using ~{cores_to_use_str} cores ---")

    parallel_processing_start_time = time.time()
    processing_results = []
    
    # Using joblib.Parallel for parallel execution of the worker function
    # verbose=10 gives some progress, verbose=50 gives more (as in your original)
    with Parallel(n_jobs=num_parallel_jobs, backend='loky', verbose=10) as parallel_executor:
        # Pass all necessary arguments to the worker function
        # template_path_str is registration_template_path_str defined in TemplateFlow section
        processing_results = parallel_executor(
            delayed(run_single_scan_preprocessing)(
                scan_info_item, 
                mri_processing_script_path, 
                final_processed_output_dir, 
                base_intermediate_files_dir, 
                config_file_path, # Pass the main config.json path
                registration_template_path_str 
            ) for scan_info_item in scans_actually_to_run
        )
    
    parallel_processing_end_time = time.time()
    print(f"\n--- Parallel Processing Block Finished in {parallel_processing_end_time - parallel_processing_start_time:.2f} seconds ---")

    # --- 6. Summarize Results ---
    num_successful = sum(1 for res_tuple in processing_results if res_tuple[1] is True)
    num_failed = len(processing_results) - num_successful
    
    print(f"\n--- Preprocessing Pipeline Summary ---")
    print(f"Total scans attempted in this run: {len(processing_results)}")
    print(f"  Successfully processed: {num_successful}")
    print(f"  Failed to process: {num_failed}")

    if num_failed > 0:
        print("\nDetails of failed scans:")
        for res_tuple in processing_results:
            if res_tuple[1] is False: # If success status is False
                scan_id_failed = res_tuple[0]
                failure_reason = res_tuple[2]
                print(f"  - Scan ID (MRI ID): {scan_id_failed}, Reason: {failure_reason}")

    overall_pipeline_end_time = time.time()
    print(f"\n--- Entire Preprocessing Pipeline Finished. Total time: {overall_pipeline_end_time - overall_pipeline_start_time:.2f} seconds ---")

# --- Command Line Argument Parser ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Orchestrates the OASIS MRI preprocessing pipeline by calling preprocess_scan.py for multiple scans.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Config path is now resolved relative to this script's location if a relative path is given at CLI
    # This makes it easier to run from anywhere if config.json is, e.g., in project root.
    parser.add_argument(
        "--config-path",
        default="../config.json", # Default assumes script is in 'scripts/' and config.json in parent
        type=str,
        help="Path to the project's main config.json file."
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1, 
        help="Number of parallel processes for joblib (-1 uses all available cores, 1 for sequential)."
    )
    cli_args_parsed = parser.parse_args()

    # Resolve config_path to an absolute path before passing to main
    # If the provided path is already absolute, this does nothing.
    # If it's relative, it's relative to where the script is run from.
    # To make it relative to script location for the default:
    if cli_args_parsed.config_path == "../config.json": # If default is used
         script_dir = Path(__file__).parent.resolve()
         cli_args_parsed.config_path = (script_dir / cli_args_parsed.config_path).resolve()
    else: # User provided a path, resolve it
         cli_args_parsed.config_path = Path(cli_args_parsed.config_path).resolve()

    main(cli_args_parsed)