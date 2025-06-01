# src/datasets.py
"""
PyTorch Dataset and collate function for loading and preparing longitudinal
OASIS-2 data, with optional inclusion of MRI scan data.
"""
import torch
from torch.utils.data import Dataset # DataLoader is typically imported in notebooks/scripts that use the Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import nibabel as nib

class OASISDataset(Dataset):
    """
    PyTorch Dataset for loading longitudinal data (e.g., OASIS-2), 
    applying pre-fitted preprocessors, and preparing sequences of tabular features,
    with an option to include corresponding MRI scan data.

    Expects a configuration dictionary detailing feature names, preprocessing info,
    and MRI parameters.
    """
    def __init__(self, 
                 data_parquet_path: Path | str, 
                 scaler_path: Path | str, 
                 imputer_path: Path | str, 
                 config: dict, 
                 mri_data_dir: Path | str = None, 
                 include_mri: bool = True):
        """
        Args:
            data_parquet_path (Path or str): Path to the Parquet file 
                                             (e.g., train, validation, or test split).
            scaler_path (Path or str): Path to the saved fitted StandardScaler object.
            imputer_path (Path or str): Path to the saved fitted SimpleImputer object.
            config (dict): Configuration dictionary. Expected to contain nested keys for:
                           'features': {'time_varying': [...], 'static': [...]},
                           'preprocess': {'scaling_cols': [...], 'imputation_cols': [...]},
                           'preprocessing_config': {'final_output_suffix': ... (for MRI)},
                           'cnn_model_params': {'input_shape': ... (for MRI)}.
                           This config is typically sourced from W&B run configs or a main JSON config.
            mri_data_dir (Path or str, optional): Path to the directory containing 
                                                  preprocessed MRI .nii.gz files. 
                                                  Required if include_mri is True. Defaults to None.
            include_mri (bool): Flag to determine if MRI data should be loaded and included. 
                                Defaults to True.
        """
        super().__init__()

        print(f"Initializing OASISDataset with data: {data_parquet_path}")
        self.data_path = Path(data_parquet_path)
        self.scaler_path = Path(scaler_path)
        self.imputer_path = Path(imputer_path)
        self.config = config 
        self.include_mri = include_mri

        # Initialize MRI-related attributes and counters
        self.mri_data_dir = None
        self.final_output_suffix = ""
        self.expected_mri_shape_cdhw = (0,0,0,0) # Default for no MRI

        if self.include_mri:
            if mri_data_dir is None:
                raise ValueError("If 'include_mri' is True, 'mri_data_dir' must be provided.")
            self.mri_data_dir = Path(mri_data_dir)
            
            mri_prep_config = self.config.get('preprocessing_config', {})
            self.final_output_suffix = mri_prep_config.get('final_output_suffix', '_preprocessed_mni.nii.gz')
            
            cnn_params = self.config.get('cnn_model_params', {})
            self.expected_mri_shape_cdhw = tuple(cnn_params.get('input_shape', (1, 91, 109, 91))) # Default example shape
            
            # MRI integrity counters (reset per dataset instance)
            self.mri_files_found_count = 0
            self.mri_files_not_found_count = 0
            self.mri_shape_mismatch_count = 0
            self.mri_load_error_count = 0
            self.total_mri_expected = 0 # Incremented in __getitem__ for sequences processed

        # Load and preprocess tabular data
        try:
            self.data_df = pd.read_parquet(self.data_path)
        except Exception as e_load:
            print(f"ERROR: Failed to load Parquet file at {self.data_path}: {e_load}")
            raise # Re-raise critical error
        
        available_cols = self.data_df.columns.tolist()
        
        # Load preprocessors
        self.scaler = None
        try:
            if self.scaler_path.exists(): # Check existence before loading
                self.scaler = joblib.load(self.scaler_path)
            else:
                print(f"Warning: Scaler file not found at {self.scaler_path}. Proceeding without scaler.")
        except Exception as e:
            print(f"Warning: Error loading scaler object from {self.scaler_path}: {e}. Proceeding without scaler.")

        self.imputer = None
        try:
            if self.imputer_path.exists(): # Check existence
                self.imputer = joblib.load(self.imputer_path)
            else:
                print(f"Warning: Imputer file not found at {self.imputer_path}. Proceeding without imputer.")
        except Exception as e:
            print(f"Warning: Error loading imputer object from {self.imputer_path}: {e}. Proceeding without imputer.")

        # Get feature lists from config
        features_config_dict = self.config.get('features', {})
        preprocess_config_dict = self.config.get('preprocess', {})

        self.time_varying_feats_from_config = features_config_dict.get('time_varying', [])
        self.static_feats_from_config = features_config_dict.get('static', [])
        imputation_cols_from_config = preprocess_config_dict.get('imputation_cols', [])
        scaling_cols_from_config = preprocess_config_dict.get('scaling_cols', [])

        # Filter feature lists based on columns available in the loaded DataFrame
        self.actual_time_varying_feats = [f for f in self.time_varying_feats_from_config if f in available_cols]
        self.actual_static_feats = [f for f in self.static_feats_from_config if f in available_cols]
        self.actual_imputation_cols = [f for f in imputation_cols_from_config if f in available_cols]
        self.actual_scaling_cols = [f for f in scaling_cols_from_config if f in available_cols]
        
        # --- Apply Tabular Preprocessing ---
        # 1. Imputation
        if self.imputer and self.actual_imputation_cols:
            print(f"Applying imputer to columns: {self.actual_imputation_cols}")
            self.data_df[self.actual_imputation_cols] = self.imputer.transform(self.data_df[self.actual_imputation_cols])
        elif self.actual_imputation_cols: # Warn if imputation was expected but imputer is not available
             print(f"Warning: Imputation specified for {self.actual_imputation_cols} but imputer object not loaded/available.")

        # 2. Encode Categorical Features (Example: M/F)
        # M/F_encoded is expected to be handled if 'M/F_encoded' is in static_feats_from_config
        # and 'M/F' is an available column.
        created_mf_encoded_col = False
        if 'M/F_encoded' in self.static_feats_from_config and 'M/F' in available_cols:
            print("Encoding 'M/F' feature to 'M/F_encoded' (F=0, M=1).")
            self.data_df['M/F_encoded'] = self.data_df['M/F'].apply(lambda x: 1 if x == 'M' else 0).astype(np.float32)
            created_mf_encoded_col = True
        elif 'M/F_encoded' in self.static_feats_from_config and 'M/F_encoded' in available_cols:
             print("'M/F_encoded' column already exists in data. Using existing.")
             created_mf_encoded_col = True # To ensure M/F is dropped if M/F_encoded is used
        # No warning needed if M/F_encoded not in config, or M/F not in data.

        # Re-evaluate available columns after potential new column creation (M/F_encoded)
        current_df_columns = self.data_df.columns.tolist()
        self.actual_scaling_cols = [f for f in scaling_cols_from_config if f in current_df_columns] # Re-filter based on current columns

        # 3. Scaling
        if self.scaler and self.actual_scaling_cols:
            print(f"Applying scaler to columns: {self.actual_scaling_cols}")
            self.data_df[self.actual_scaling_cols] = self.scaler.transform(self.data_df[self.actual_scaling_cols])
        elif self.actual_scaling_cols: # Warn if scaling was expected but scaler is not available
            print(f"Warning: Scaling specified for {self.actual_scaling_cols} but scaler object not loaded/available.")

        # --- Define Final Feature Lists for Model Input (after all tabular processing) ---
        final_df_columns = self.data_df.columns.tolist()
        
        # Use the original configured feature lists and filter them based on what's actually in the DataFrame NOW
        final_time_varying_feats = [f for f in self.time_varying_feats_from_config if f in final_df_columns]
        final_static_feats = [f for f in self.static_feats_from_config if f in final_df_columns]

        # If M/F_encoded was the target and created, ensure it's used if present.
        # If M/F_encoded was expected in config but not created/present, it will be naturally excluded.
        # If M/F_encoded was created, remove original M/F if it was in static_feats_from_config
        if created_mf_encoded_col and 'M/F' in final_static_feats: 
            # This case means 'M/F' was in static_feats_from_config and also in final_df_columns,
            # but M/F_encoded was also created (or already existed and was targeted).
            # We should prefer M/F_encoded if it's in final_static_feats.
            if 'M/F_encoded' in final_static_feats:
                 final_static_feats.remove('M/F')
        
        self.model_input_features = sorted(list(set(final_time_varying_feats + final_static_feats)))
        # Final sanity check: ensure all model_input_features are indeed columns in the processed data_df
        self.model_input_features = [f for f in self.model_input_features if f in self.data_df.columns]

        # Standard column names
        self.target_col = 'CDR_next_visit'
        self.id_col = 'Subject ID'
        self.visit_col = 'Visit'
        if self.target_col not in self.data_df.columns:
             print(f"Warning: Target column '{self.target_col}' not found in final DataFrame columns!")

        # Group data by subject
        self.data_df = self.data_df.sort_values(by=[self.id_col, self.visit_col]) # Ensure sorted for grouping
        self.grouped_data = self.data_df.groupby(self.id_col)
        self.subject_ids = list(self.grouped_data.groups.keys())
        
        # Final summary print
        print(f"\n--- OASISDataset Initialized ---")
        print(f"Loaded data from: {self.data_path}, Shape: {self.data_df.shape}")
        if self.scaler: print(f"Scaler loaded from: {self.scaler_path}")
        if self.imputer: print(f"Imputer loaded from: {self.imputer_path}")
        print(f"Final Tabular Model Input Features ({len(self.model_input_features)}): {self.model_input_features}")
        if self.include_mri:
            print(f"MRI data included. Source: {self.mri_data_dir}")
            print(f"  Expected MRI shape (C,D,H,W): {self.expected_mri_shape_cdhw}, Suffix: {self.final_output_suffix}")
        print(f"Number of subjects (sequences): {len(self.subject_ids)}")
        print(f"Target column: {self.target_col}")
        print("--- End of OASISDataset Initialization ---\n")


    def __len__(self):
        """Returns the number of subjects (sequences) in the dataset."""
        return len(self.subject_ids)

    def __getitem__(self, idx: int) -> tuple:
        """
        Retrieves the processed sequence data (tabular, optionally MRI) and 
        target for a single subject.
        """
        subject_id = self.subject_ids[idx]
        subject_data_df = self.grouped_data.get_group(subject_id).copy() # Use .copy()

        tabular_features_for_sequence = []
        mri_scans_for_sequence = [] if self.include_mri else None # Initialize only if needed

        if self.include_mri:
            # Increment total expected MRIs for this subject's sequence.
            # This helps in assessing data integrity if an analysis function iterates through the dataset.
            self.total_mri_expected += len(subject_data_df)

        for _, visit_row in subject_data_df.iterrows():
            # Extract tabular features for the current visit using the finalized list
            current_tabular_visit_features = visit_row[self.model_input_features].values.astype(np.float32)
            tabular_features_for_sequence.append(current_tabular_visit_features)

            if self.include_mri:
                mri_id_for_visit = visit_row['MRI ID']
                mri_filename = f"{subject_id}_{mri_id_for_visit}{self.final_output_suffix}"
                mri_filepath = self.mri_data_dir / mri_filename # type: ignore
                
                # Initialize with a zero tensor of the correct expected shape
                mri_tensor_for_visit = torch.zeros(self.expected_mri_shape_cdhw, dtype=torch.float32)

                if mri_filepath.is_file():
                    try:
                        mri_img = nib.load(str(mri_filepath)) # Ensure path is string for nibabel
                        mri_data_array = mri_img.get_fdata(dtype=np.float32)
                        
                        # Basic Min-Max Normalization per scan to [0,1]
                        min_val, max_val = mri_data_array.min(), mri_data_array.max()
                        if max_val > min_val: # Avoid division by zero
                            mri_data_array = (mri_data_array - min_val) / (max_val - min_val)
                        else: # If all values are the same (e.g., all zeros)
                            mri_data_array = np.zeros_like(mri_data_array, dtype=np.float32)
                        
                        temp_mri_tensor = torch.from_numpy(mri_data_array)
                        
                        # Add channel dimension if loaded as 3D and expected is 4D (C=1)
                        if temp_mri_tensor.ndim == 3 and self.expected_mri_shape_cdhw[0] == 1 and \
                           tuple(temp_mri_tensor.shape) == self.expected_mri_shape_cdhw[1:]:
                            temp_mri_tensor = temp_mri_tensor.unsqueeze(0)

                        if temp_mri_tensor.shape == self.expected_mri_shape_cdhw:
                            mri_tensor_for_visit = temp_mri_tensor
                            self.mri_files_found_count += 1 
                        else:
                            print(f"Warning: MRI shape mismatch for {mri_filepath}. "
                                  f"Expected {self.expected_mri_shape_cdhw}, got {temp_mri_tensor.shape}. Using zero tensor.")
                            self.mri_shape_mismatch_count += 1
                    except Exception as e_mri_load:
                        print(f"Error loading/processing MRI {mri_filepath}: {e_mri_load}. Using zero tensor.")
                        self.mri_load_error_count += 1
                else:
                    # This warning can be verbose if many files are missing. Consider logging frequency.
                    # print(f"Warning: MRI file not found: {mri_filepath}. Using zero tensor.")
                    self.mri_files_not_found_count += 1
                
                if mri_scans_for_sequence is not None: # Should always be true if self.include_mri
                    mri_scans_for_sequence.append(mri_tensor_for_visit)

        # Convert list of tabular feature arrays to a single tensor for the sequence
        sequence_tabular_features = torch.tensor(np.array(tabular_features_for_sequence), dtype=torch.float32)
        
        # Extract target from the last visit in the subject_data_df
        target_val = np.nan # Default if target column missing or all NaNs
        if self.target_col in subject_data_df.columns:
            # Ensure there's at least one non-NaN target if possible, otherwise take last
            valid_targets = subject_data_df[self.target_col].dropna()
            if not valid_targets.empty:
                 target_val = valid_targets.iloc[-1] # Use last valid target in sequence
            elif not subject_data_df[self.target_col].empty: # If all are NaN, take last NaN
                 target_val = subject_data_df[self.target_col].iloc[-1]

        target_tensor = torch.tensor([target_val], dtype=torch.float32) # Ensure target is float

        if self.include_mri:
            return sequence_tabular_features, mri_scans_for_sequence, target_tensor
        else:
            return sequence_tabular_features, target_tensor


def pad_collate_fn(batch: list) -> tuple:
    """
    Collate function for DataLoader. Handles batches of sequences with varying lengths,
    supporting both tabular-only and tabular+MRI data structures from OASISDataset.

    Args:
        batch (list): A list of tuples. Each tuple is the output of OASISDataset.__getitem__.
                      It can be (sequence_tabular, target_tensor) if MRI is not included,
                      or (sequence_tabular, mri_scans_list, target_tensor) if MRI is included.

    Returns:
        tuple: Depending on MRI inclusion:
               If MRI included: (seq_tab_padded, seq_mri_padded, lengths, targets_stacked, masks)
               If MRI not included: (seq_tab_padded, lengths, targets_stacked, masks)
    """
    if not batch: # Handle empty batch list
        # Determine expected output structure based on a hypothetical item, or return fixed empty structure
        # This is tricky if batch can sometimes have MRI and sometimes not (though unlikely for a single DataLoader)
        # For now, assume if it's empty, it could have been an MRI batch
        print("Warning: pad_collate_fn received an empty batch.")
        return torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)
    
    # Determine if MRI data is present in this batch based on the structure of the first item
    # (assuming all items in a batch have the same structure)
    item_example = batch[0]
    batch_includes_mri = len(item_example) == 3 # (tabular, mri_list, target)

    if batch_includes_mri:
        sequences_tabular = [item[0] for item in batch]
        sequences_mri_lists = [item[1] for item in batch] # List of lists of MRI Tensors
        targets = [item[2] for item in batch]
    else: # Only (tabular, target)
        sequences_tabular = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        sequences_mri_lists = None # Explicitly None

    # --- 1. Process Tabular Sequences, Lengths, and Masks (Common to both cases) ---
    lengths = torch.tensor([len(seq) for seq in sequences_tabular], dtype=torch.long)
    sequences_tabular_padded = pad_sequence(sequences_tabular, batch_first=True, padding_value=0.0)
    
    batch_size_actual = 0
    max_seq_len = 0
    if sequences_tabular_padded.numel() > 0: # If tensor is not empty
        batch_size_actual, max_seq_len, _ = sequences_tabular_padded.shape
    
    # Handle case where batch_size_actual might be 0 (e.g., if all input sequences were empty, though unlikely)
    if batch_size_actual == 0:
        masks = torch.empty((0,0), dtype=torch.bool)
        # Try to get target feature dimension from first target if available, else assume 1
        target_dim = targets[0].shape[0] if targets and isinstance(targets[0], torch.Tensor) else 1
        targets_stacked = torch.empty((0, target_dim), dtype=torch.float32) 
        
        if batch_includes_mri:
            # Need a placeholder for MRI shape for empty MRI tensor
            # This is difficult without access to dataset.expected_mri_shape_cdhw
            # For now, create a truly empty tensor with placeholder other dims.
            # A better approach might be to pass expected_mri_shape_cdhw to collate_fn.
            empty_mri_shape_other_dims = (0, 1, 91, 109, 91) # (N, C, D, H, W) with N=0
            sequences_mri_padded = torch.empty(empty_mri_shape_other_dims, dtype=torch.float32)
            return sequences_tabular_padded, sequences_mri_padded, lengths, targets_stacked, masks
        else:
            return sequences_tabular_padded, lengths, targets_stacked, masks

    masks = torch.arange(max_seq_len).expand(batch_size_actual, max_seq_len) < lengths.unsqueeze(1)
    targets_stacked = torch.stack(targets) # Shape becomes (batch_size, num_target_dims) e.g. (B,1)

    # --- 2. Process MRI Scan Sequences (Only if MRI is included in batch) ---
    if batch_includes_mri and sequences_mri_lists is not None:
        # Determine MRI shape (C,D,H,W) from the first valid tensor in the batch.
        # This relies on OASISDataset.__getitem__ ensuring all MRI tensors (including placeholders)
        # have a consistent, expected shape (self.expected_mri_shape_cdhw).
        mri_c, mri_d, mri_h, mri_w = 1, 91, 109, 91 # Default/fallback shape
        found_shape = False
        for subj_mri_list in sequences_mri_lists: # Iterate through subjects in batch
            if subj_mri_list: # If the list of MRI tensors for this subject is not empty
                for mri_tensor_sample in subj_mri_list:
                    if mri_tensor_sample is not None and mri_tensor_sample.ndim == 4 and mri_tensor_sample.numel() > 0:
                        mri_c, mri_d, mri_h, mri_w = mri_tensor_sample.shape # Shape is (C,D,H,W)
                        found_shape = True
                        break
            if found_shape:
                break
        if not found_shape:
            print(f"Warning (pad_collate_fn): Could not infer MRI C,D,H,W from batch data. "
                  f"Using default shape: {(mri_c, mri_d, mri_h, mri_w)}. "
                  "Ensure OASISDataset provides correctly shaped placeholder MRI tensors.")

        padded_mri_sequences_for_batch = []
        for i in range(batch_size_actual): # Iterate through subjects in the batch
            subject_mri_list = sequences_mri_lists[i] # This is a list of (C,D,H,W) tensors for one subject
            subject_actual_len = lengths[i].item()    # Original sequence length for this subject
            
            current_subject_padded_mri_sequence = []
            expected_mri_shape_this_collate = (mri_c, mri_d, mri_h, mri_w)

            for t_step in range(max_seq_len): # Iterate up to the max_seq_len of the batch
                if t_step < subject_actual_len:
                    mri_tensor_for_visit = subject_mri_list[t_step]
                    # Ensure it's the target shape; OASISDataset should have handled this, but double check.
                    if mri_tensor_for_visit.shape != expected_mri_shape_this_collate:
                        print(f"Warning (pad_collate_fn): MRI tensor shape {mri_tensor_for_visit.shape} for subject {i}, visit {t_step} "
                              f"does not match inferred batch MRI shape {expected_mri_shape_this_collate}. Using zero tensor.")
                        mri_tensor_for_visit = torch.zeros(expected_mri_shape_this_collate, dtype=torch.float32)
                else:
                    # This time step is padding; add a zero tensor for the MRI scan
                    mri_tensor_for_visit = torch.zeros(expected_mri_shape_this_collate, dtype=torch.float32)
                current_subject_padded_mri_sequence.append(mri_tensor_for_visit)
            
            if current_subject_padded_mri_sequence: # If list is not empty (should be if max_seq_len > 0)
                padded_mri_sequences_for_batch.append(torch.stack(current_subject_padded_mri_sequence, dim=0))
            else: # Should only happen if max_seq_len is 0
                padded_mri_sequences_for_batch.append(torch.empty((0, mri_c, mri_d, mri_h, mri_w), dtype=torch.float32))
        
        # Stack all subjects' padded MRI sequences: (batch_size, max_seq_len, C, D, H, W)
        if padded_mri_sequences_for_batch:
            sequences_mri_padded = torch.stack(padded_mri_sequences_for_batch, dim=0)
        else: # Fallback for entirely empty batch scenario
             sequences_mri_padded = torch.empty((0, max_seq_len, mri_c, mri_d, mri_h, mri_w), dtype=torch.float32)

        return sequences_tabular_padded, sequences_mri_padded, lengths, targets_stacked, masks
    else:
        # MRI not included in this batch, return only tabular-related data
        return sequences_tabular_padded, lengths, targets_stacked, masks