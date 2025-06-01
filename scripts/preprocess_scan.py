# scripts/preprocess_scan.py
"""
Performs a series of preprocessing steps on a single input T1w MRI scan.
Steps include format conversion (if needed), reorientation to canonical space,
N4 bias field correction, skull stripping (using FSL BET or ANTsPyNet),
and registration to a specified template.

The execution of these steps is controlled by a JSON configuration file.
Intermediate files are typically stored in a subject-specific subdirectory within
a base intermediate directory, and the final processed scan is saved to a
specified output directory.
"""

import argparse
import json
import sys
from pathlib import Path
import nibabel as nib
import time
import shutil
import os
import subprocess # Kept for FSL BET if used

# --- Imports for ANTs/ANTsPyNet (with conditional availability checks) ---
try:
    import ants # Official import name for antspyx
    ANTSPYX_AVAILABLE = True
    # Informative print, not a debug print, good for user awareness of environment
    print("Successfully imported 'ants' (from antspyx library). ANTs-based N4 and Registration are available.")
except ImportError:
    print("Warning: antspyx library (imported as 'ants') not found. "
          "ANTs-based N4 bias field correction and registration steps cannot run.")
    ANTSPYX_AVAILABLE = False

try:
    from antspynet.utilities import brain_extraction as antspynet_brain_extraction
    ANTSPYNET_AVAILABLE = True
    print("Successfully imported 'brain_extraction' from 'antspynet.utilities'. ANTsPyNet skull stripping is available.")
except ImportError:
    print("Warning: antspynet library not found. "
          "ANTsPyNet-based skull stripping cannot run.")
    ANTSPYNET_AVAILABLE = False

# --- Define Functions for Each Preprocessing Step ---

def convert_to_nii_gz(input_img_path: Path, output_nii_gz_path: Path) -> bool:
    """Loads an Analyze 7.5 .img/.hdr file pair and saves it as a .nii.gz file.

    Args:
        input_img_path (Path): Path to the input .img file.
        output_nii_gz_path (Path): Path where the output .nii.gz file will be saved.

    Returns:
        bool: True if conversion was successful, False otherwise.
    """
    try:
        print(f"  Converting {input_img_path.name} -> {output_nii_gz_path.name}...")
        img = nib.load(str(input_img_path)) # nibabel often prefers string paths
        output_nii_gz_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(img, str(output_nii_gz_path))
        print(f"  Successfully converted to {output_nii_gz_path.name}.")
        return True
    except FileNotFoundError:
        print(f"  Error: Input file not found for conversion: {input_img_path}")
        return False
    except Exception as e:
        print(f"  Error during conversion to .nii.gz: {e}")
        return False

def reorient_to_canonical(input_nii_path: Path, output_nii_path: Path) -> bool:
    """Reorients a NIfTI image to canonical (RAS+) orientation.

    Args:
        input_nii_path (Path): Path to the input NIfTI file.
        output_nii_path (Path): Path to save the reoriented NIfTI file.

    Returns:
        bool: True if reorientation was successful, False otherwise.
    """
    try:
        print(f"  Reorienting {input_nii_path.name} to canonical -> {output_nii_path.name}...")
        img = nib.load(str(input_nii_path))
        canonical_img = nib.as_closest_canonical(img)
        # Log the determined canonical orientation
        canonical_orientation_code = ''.join(nib.aff2axcodes(canonical_img.affine))
        print(f"  Image reoriented to canonical ({canonical_orientation_code}).")
        output_nii_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(canonical_img, str(output_nii_path))
        print(f"  Successfully reoriented and saved.")
        return True
    except Exception as e:
        print(f"  Error during reorientation: {e}")
        return False

def run_n4_bias_correction(input_nii_path: Path, output_nii_path: Path) -> bool:
    """Performs N4 Bias Field Correction using the antspyx library.

    Args:
        input_nii_path (Path): Path to the input NIfTI file.
        output_nii_path (Path): Path to save the N4 corrected NIfTI file.

    Returns:
        bool: True if N4 correction was successful or skipped (if antspyx not available), 
              False if an error occurred during processing.
    """
    if not ANTSPYX_AVAILABLE:
        print("  Skipping N4 Bias Field Correction: antspyx library not available.")
        return True # Gracefully skip if library is not present

    try:
        print(f"  Running N4 Bias Field Correction: {input_nii_path.name} -> {output_nii_path.name}...")
        ants_image_input = ants.image_read(str(input_nii_path), pixeltype='float')
        ants_image_n4_corrected = ants.n4_bias_field_correction(ants_image_input)
        output_nii_path.parent.mkdir(parents=True, exist_ok=True)
        # ants.image_write(ants_image_n4_corrected, str(output_nii_path)) # Alternative save
        ants_image_n4_corrected.to_file(str(output_nii_path)) # As per your original
        print(f"  N4 Bias Field Correction completed successfully.")
        return True
    except Exception as e:
        print(f"  Error during N4 Bias Field Correction: {e}")
        return False

def run_skull_strip(input_nii_path: Path, output_nii_path: Path, method: str = "fsl_bet") -> bool:
    """Performs skull stripping using the specified method ('fsl_bet' or 'antspynet').

    Args:
        input_nii_path (Path): Path to the input NIfTI file (ideally N4 corrected).
        output_nii_path (Path): Path to save the skull-stripped brain NIfTI file.
        method (str): Skull stripping method to use ("fsl_bet" or "antspynet").

    Returns:
        bool: True if skull stripping was successful, False otherwise.
    """
    print(f"  Skull Stripping (Method: {method}): {input_nii_path.name} -> {output_nii_path.name}")
    output_nii_path.parent.mkdir(parents=True, exist_ok=True)

    if method.lower() == "fsl_bet":
        try:
            if shutil.which('bet') is None: # Check if FSL BET command is available
                print("    Error: FSL 'bet' command not found in system PATH. Cannot use 'fsl_bet' method.")
                return False
            # Standard FSL BET command arguments for robust brain extraction
            bet_command_args = [ 'bet', str(input_nii_path), str(output_nii_path), '-R', '-f', '0.4', '-m']
            print(f"    Running FSL BET command: {' '.join(bet_command_args)}")
            # Using subprocess.run for better control and error capturing
            result = subprocess.run(bet_command_args, check=True, capture_output=True, text=True, encoding='utf-8')
            if result.stderr: # Log any standard error output from BET
                print(f"    FSL BET stderr:\n{result.stderr.strip()}")
            if not output_nii_path.exists(): # Verify output file was created
                print(f"    Error: FSL BET output file not created at {output_nii_path}!")
                return False
            print(f"    FSL BET completed successfully.")
            return True
        except subprocess.CalledProcessError as e_subproc: # Specific error for subprocess failure
            print(f"    Error during FSL BET execution (CalledProcessError): {e_subproc}")
            print(f"    BET Stderr: {e_subproc.stderr}")
            return False
        except Exception as e:
            print(f"    An unexpected error occurred during FSL BET: {e}")
            return False

    elif method.lower() == "antspynet":
        if not ANTSPYNET_AVAILABLE:
            print("    Error: ANTsPyNet library is not available. Cannot use 'antspynet' method.")
            return False
        if not ANTSPYX_AVAILABLE: # ANTsPyNet often relies on antspyx
            print("    Error: 'ants' (antspyx) library needed by ANTsPyNet is not available.")
            return False
        try:
            print(f"    Running ANTsPyNet Brain Extraction (modality='t1')...")
            step_start_time_antspynet = time.time()
            ants_image_input = ants.image_read(str(input_nii_path), pixeltype='float')
            # ANTsPyNet brain_extraction often returns a probability mask
            brain_probability_mask = antspynet_brain_extraction(ants_image_input, modality="t1", verbose=False)
            # Threshold the probability mask to get a binary mask
            binary_brain_mask = ants.threshold_image(brain_probability_mask, 0.5, 1.0, 1, 0)
            # Apply the binary mask to the original image
            skull_stripped_image = ants_image_input * binary_brain_mask # Element-wise multiplication
            skull_stripped_image.to_file(str(output_nii_path)) # Save the result
            step_end_time_antspynet = time.time()
            print(f"    ANTsPyNet Brain Extraction completed in {step_end_time_antspynet - step_start_time_antspynet:.2f}s.")
            return True
        except Exception as e:
            print(f"    Error during ANTsPyNet Brain Extraction: {e}")
            return False
    else:
        print(f"    Error: Unknown skull stripping method specified: '{method}'. Supported: 'fsl_bet', 'antspynet'.")
        return False

def run_registration(input_nii_path: Path, output_nii_path: Path, template_path: Path) -> bool:
    """Performs registration to a template (e.g., MNI) using antspyx SyN algorithm.

    Args:
        input_nii_path (Path): Path to the input NIfTI file (ideally skull-stripped).
        output_nii_path (Path): Path to save the registered NIfTI file.
        template_path (Path): Path to the template NIfTI file to register to.

    Returns:
        bool: True if registration was successful or skipped, False otherwise.
    """
    if not ANTSPYX_AVAILABLE:
        print("  Skipping Registration: antspyx library not available.")
        return True # Gracefully skip

    if template_path is None or not template_path.exists(): # Check template path validity
        print(f"  Error: Template path for registration is invalid or not provided: {template_path}")
        return False
    try:
        print(f"  Running ANTs Registration (Transform: SyN): {input_nii_path.name} -> Template: {template_path.name}")
        step_start_time_reg = time.time()
        
        print("    Loading moving and fixed images for registration...")
        moving_image = ants.image_read(str(input_nii_path), pixeltype='float')
        fixed_template_image = ants.image_read(str(template_path), pixeltype='float')
        
        print("    Starting antspyx.registration (type_of_transform='SyN'). This may take time...")
        # SyN is a common choice for accurate, deformable registration
        registration_output = ants.registration(
            fixed=fixed_template_image, 
            moving=moving_image, 
            type_of_transform='SyN', 
            verbose=False # Set to True for more detailed ANTs output if debugging
        )
        print("    antspyx.registration finished.")
        
        warped_moving_image = registration_output['warpedmovout'] # The registered moving image
        
        output_nii_path.parent.mkdir(parents=True, exist_ok=True)
        # ants.image_write(warped_moving_image, str(output_nii_path))
        warped_moving_image.to_file(str(output_nii_path)) # As per your original
        step_end_time_reg = time.time()
        print(f"  Registration completed successfully in {step_end_time_reg - step_start_time_reg:.2f}s. Output: {output_nii_path.name}")
        return True
    except Exception as e:
        print(f"  Error during ANTs Registration: {e}")
        return False


# --- Main Processing Orchestration Function ---
def preprocess_scan(input_img_file_str: str, 
                    output_dir_base_str: str, 
                    intermediate_dir_base_str: str, 
                    config_file_path_str: str, 
                    subject_id: str, 
                    scan_id: str, 
                    template_file_path_str: str = None) -> bool:
    """
    Main function to preprocess a single MRI scan based on steps defined in a configuration file.
    Intermediate files are saved in a subject-specific subdirectory within intermediate_dir_base.
    The final processed file is saved directly in output_dir_base.

    Args:
        input_img_file_str (str): Path string to the input .img or .nii(.gz) file.
        output_dir_base_str (str): Path string to the base directory for FINAL preprocessed files.
        intermediate_dir_base_str (str): Path string to the base directory for intermediate files.
        config_file_path_str (str): Path string to the project's config.json file.
        subject_id (str): Subject ID for naming intermediate and final files.
        scan_id (str): Scan ID (e.g., MRI ID) for naming files.
        template_file_path_str (str, optional): Path string to the registration template NIfTI file.

    Returns:
        bool: True if all configured steps completed successfully, False otherwise.
    """
    # Convert string paths to Path objects for robust handling
    input_img_file = Path(input_img_file_str)
    output_dir_base = Path(output_dir_base_str) 
    intermediate_dir_base = Path(intermediate_dir_base_str)
    config_path = Path(config_file_path_str)
    template_path = Path(template_file_path_str) if template_file_path_str else None
    
    overall_start_time = time.time()
    print(f"\n--- Starting Preprocessing for Scan ---")
    print(f"Input Image File : {input_img_file}")
    print(f"Subject ID: {subject_id}, Scan ID: {scan_id}")
    print(f"Configuration File: {config_path}")

    # Load Configuration for preprocessing steps
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            loaded_config = json.load(f)
        prep_config = loaded_config.get('preprocessing_config', {})
        if not prep_config:
            print(f"Warning: 'preprocessing_config' section not found in {config_path}. Using default behaviors (most steps off).")
    except Exception as e_cfg:
        print(f"Error loading or parsing config file {config_path}: {e_cfg}")
        return False

    # Define subject-specific directory for intermediate files
    # Example: intermediate_dir_base / "OASIS_Subject123"
    subject_intermediate_dir = intermediate_dir_base / subject_id
    subject_intermediate_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure final output base directory exists
    output_dir_base.mkdir(parents=True, exist_ok=True)

    # Consistent base name for files related to this scan
    file_base_name = f"{subject_id}_{scan_id}" 
    
    current_processed_file = input_img_file # Start with the original input
    step_overall_success = True
    intermediate_files_created_this_run = [] # Track files created by this specific run for cleanup

    # --- Step 1: Convert to NIfTI (.nii.gz) if necessary ---
    # Output goes to: subject_intermediate_dir / file_base_name_T1w.nii.gz
    if step_overall_success and prep_config.get('run_convert_to_nii', True) and input_img_file.suffix.lower() == '.img':
        # Standardized intermediate name
        converted_nii_gz_file = subject_intermediate_dir / f"{file_base_name}_T1w.nii.gz"
        step_overall_success = convert_to_nii_gz(current_processed_file, converted_nii_gz_file)
        if step_overall_success:
            current_processed_file = converted_nii_gz_file
            intermediate_files_created_this_run.append(current_processed_file)
    elif step_overall_success and (input_img_file.suffix.lower() == '.nii' or ".nii.gz" in input_img_file.name.lower()):
        print(f"  Input {input_img_file.name} is already in NIfTI format.")
        # Copy to intermediate directory with standardized name for consistent pipeline flow
        standardized_nii_file_in_interm = subject_intermediate_dir / f"{file_base_name}_T1w.nii.gz"
        if standardized_nii_file_in_interm.resolve() != current_processed_file.resolve():
            try:
                shutil.copyfile(str(current_processed_file), str(standardized_nii_file_in_interm))
                print(f"  Copied/using NIfTI input in intermediate dir: {standardized_nii_file_in_interm.name}")
                current_processed_file = standardized_nii_file_in_interm
                intermediate_files_created_this_run.append(current_processed_file)
            except Exception as e_copy:
                print(f"  Error copying NIfTI to intermediate directory: {e_copy}")
                step_overall_success = False
        else: # Input was already the target standardized file in intermediate dir
             intermediate_files_created_this_run.append(current_processed_file) # Still track it if it's an intermediate
    elif step_overall_success : # Not .img and not .nii or .nii.gz - unsupported format
        print(f"  Error: Unsupported input file format: {input_img_file.name}. Expected .img or .nii(.gz).")
        step_overall_success = False


    # --- Step 2: Reorient to Canonical ---
    # Output: subject_intermediate_dir / file_base_name_reorient.nii.gz
    if step_overall_success and prep_config.get('run_reorient', True): # Default to True if key missing
        reoriented_file = subject_intermediate_dir / f"{file_base_name}_reorient.nii.gz"
        step_overall_success = reorient_to_canonical(current_processed_file, reoriented_file)
        if step_overall_success:
            current_processed_file = reoriented_file
            intermediate_files_created_this_run.append(current_processed_file)

    # --- Step 3: N4 Bias Field Correction ---
    # Output: subject_intermediate_dir / file_base_name_n4.nii.gz
    if step_overall_success and prep_config.get('run_n4', True): # Default to True
        n4_corrected_file = subject_intermediate_dir / f"{file_base_name}_n4.nii.gz"
        step_overall_success = run_n4_bias_correction(current_processed_file, n4_corrected_file)
        if step_overall_success:
            current_processed_file = n4_corrected_file
            intermediate_files_created_this_run.append(current_processed_file)

    # --- Step 4: Skull Stripping ---
    # Output: subject_intermediate_dir / file_base_name_brain.nii.gz
    if step_overall_success and prep_config.get('run_skullstrip', True): # Default to True
        skull_stripped_file = subject_intermediate_dir / f"{file_base_name}_brain.nii.gz"
        skullstrip_method = prep_config.get('skullstrip_method', 'fsl_bet') # Default method
        step_overall_success = run_skull_strip(current_processed_file, skull_stripped_file, method=skullstrip_method)
        if step_overall_success:
            current_processed_file = skull_stripped_file
            intermediate_files_created_this_run.append(current_processed_file)

    # --- Step 5: Registration to Template ---
    # Final output location: output_dir_base / file_base_name<final_output_suffix>.nii.gz
    final_output_filename_suffix = prep_config.get('final_output_suffix', '_preprocessed_mni.nii.gz')
    final_registered_output_file = output_dir_base / f"{file_base_name}{final_output_filename_suffix}"
    
    if step_overall_success and prep_config.get('run_register', True): # Default to True
        if template_path is None or not template_path.exists():
             print(f"  Error: Cannot run registration. Template path invalid or not provided: {template_path}")
             step_overall_success = False # Critical failure if registration is configured but template missing
        else:
             step_overall_success = run_registration(current_processed_file, final_registered_output_file, template_path)
             if step_overall_success:
                 current_processed_file = final_registered_output_file # This is now the final output
             # The final output (registered_file) is NOT added to intermediate_files_created_this_run
    elif step_overall_success: # Registration is configured to be skipped, but previous steps were successful
         print(f"  Registration configured to be skipped. Copying last processed file '{current_processed_file.name}' to final output location with suffix: {final_registered_output_file.name}")
         try:
            shutil.copyfile(str(current_processed_file), str(final_registered_output_file))
            current_processed_file = final_registered_output_file # This is now the final output
            print(f"  Successfully copied to final output: {current_processed_file}")
         except Exception as e_copy_final:
            print(f"  Error copying to final output location: {e_copy_final}")
            step_overall_success = False
    # If step_overall_success is False at this point, current_processed_file is the last successfully created intermediate.

    # --- Step 6: Cleanup Intermediate Files ---
    # Only cleanup if all configured steps were successful AND keep_intermediates is False
    if step_overall_success and not prep_config.get('keep_intermediates', False):
        if subject_intermediate_dir.exists(): # Check if intermediate dir itself exists
            print(f"  Cleaning up intermediate files from {subject_intermediate_dir} for this run...")
            deleted_files_count = 0
            for f_path_to_delete in intermediate_files_created_this_run:
                # Do not delete the final output file if it happens to be in the list by mistake
                # (e.g., if no registration and last intermediate was copied to final output name but still tracked)
                # However, current_processed_file points to the final file, and it's not added to intermediate_files_created_this_run
                # if it's the result of registration or the final copy.
                if f_path_to_delete.exists() and f_path_to_delete.is_file(): 
                    try:
                        f_path_to_delete.unlink()
                        deleted_files_count += 1
                    except OSError as e_del:
                        print(f"  Warning: Could not delete intermediate file {f_path_to_delete}. Error: {e_del}")
            print(f"  Deleted {deleted_files_count} intermediate file(s) created during this run.")
            
            # Optionally, try to remove the subject_intermediate_dir if it's empty
            # This is safer after deleting specific files, as other runs might use the same subject_intermediate_dir
            try:
                # Check if directory is empty (handles potential hidden files like .DS_Store)
                if not any(subject_intermediate_dir.iterdir()):
                    subject_intermediate_dir.rmdir()
                    print(f"  Removed empty intermediate subject directory: {subject_intermediate_dir}")
            except OSError: 
                # Directory might not be empty due to other files or permissions
                print(f"  Note: Intermediate subject directory {subject_intermediate_dir} not removed (may not be empty or other reason).")
    elif not step_overall_success:
        print(f"  Skipping cleanup of intermediate files due to processing failure. Files kept in {subject_intermediate_dir}")
    else: # keep_intermediates is True
        print(f"  Keeping intermediate files in {subject_intermediate_dir} as configured.")


    # --- Final Status Reporting ---
    overall_end_time = time.time()
    final_status_message = "SUCCESS" if step_overall_success else "FAILED"
    print(f"--- Finished Preprocessing Scan for {file_base_name} ---")
    print(f"Overall Status: {final_status_message}")
    print(f"Total processing time: {overall_end_time - overall_start_time:.2f} seconds")
    if step_overall_success:
        print(f"Final preprocessed output: {current_processed_file}") # This should be final_registered_output_file
    else:
        print(f"Processing failed. Last successfully processed file (if any): {current_processed_file if current_processed_file != input_img_file else 'None beyond input'}")
    
    return step_overall_success


# --- Argument Parser and Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocesses a single T1w MRI scan using a defined pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help
    )
    parser.add_argument("--input-img", required=True, type=str, 
                        help="Path to the input .img or .nii(.gz) file.")
    parser.add_argument("--output-dir", required=True, type=str, 
                        help="Base directory to save the FINAL preprocessed NIfTI file.")
    parser.add_argument("--intermediate-dir", required=True, type=str, 
                        help="Base directory to save intermediate files. A subject-specific subdirectory will be created here.")
    parser.add_argument("--config-path", required=True, type=str, 
                        help="Path to the project's main config.json file.")
    parser.add_argument("--subject-id", required=True, type=str, 
                        help="Subject ID (e.g., OASIS ID) for naming outputs and intermediate subdirectory.")
    parser.add_argument("--scan-id", required=True, type=str, 
                        help="Scan ID (e.g., MRI ID from clinical data sheet) for naming outputs.")
    parser.add_argument("--template-path", type=str, default=None, 
                        help="Path to the NIfTI template file for registration (e.g., MNI template). Required if registration is enabled in config.")

    cli_args = parser.parse_args()

    # Call the main preprocessing function with arguments from the command line
    overall_success = preprocess_scan(
        input_img_file_str=cli_args.input_img,
        output_dir_base_str=cli_args.output_dir,
        intermediate_dir_base_str=cli_args.intermediate_dir,
        config_file_path_str=cli_args.config_path,
        subject_id=cli_args.subject_id,
        scan_id=cli_args.scan_id,
        template_file_path_str=cli_args.template_path
    )

    # Exit with appropriate status code
    sys.exit(0 if overall_success else 1)