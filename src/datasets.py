# src/datasets.py

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

class OASISDataset(Dataset):
    """
    PyTorch Dataset for loading OASIS longitudinal data, applying
    pre-fitted preprocessors, and preparing sequences.
    Expects config with nested keys, e.g., config['features']['time_varying'].
    """
    def __init__(self, data_parquet_path, scaler_path, imputer_path, config):
        """
        Args:
            data_parquet_path (str or Path): Path to the parquet file (train, val, or test split).
            scaler_path (str or Path): Path to the saved fitted StandardScaler object.
            imputer_path (str or Path): Path to the saved fitted SimpleImputer object.
            config (dict): Dictionary containing configuration, expected to have nested keys like
                           config['features']['time_varying'], config['preprocess']['scaling_cols'], etc.
                           (Should be loaded from the relevant W&B run.config in the calling notebook).
        """
        super().__init__()

        print(f"Initializing OASISDataset with data: {data_parquet_path}")
        self.data_path = Path(data_parquet_path)
        self.scaler_path = Path(scaler_path)
        self.imputer_path = Path(imputer_path)
        self.config = config # Store the loaded config

        # Load data
        self.data_df = pd.read_parquet(self.data_path)
        print(f"Loaded data shape: {self.data_df.shape}")
        available_cols = self.data_df.columns.tolist() # 

        # Load fitted preprocessors
        try:
            self.scaler = joblib.load(self.scaler_path)
            print(f"Loaded scaler from {self.scaler_path}")
        except FileNotFoundError:
             print(f"Warning: Scaler file not found at {self.scaler_path}. Proceeding without scaler.")
             self.scaler = None
        except Exception as e:
            print(f"Warning: Error loading scaler object: {e}. Proceeding without scaler.")
            self.scaler = None

        try:
            self.imputer = joblib.load(self.imputer_path)
            print(f"Loaded imputer from {self.imputer_path}")
        except FileNotFoundError:
            print(f"Warning: Imputer file not found at {self.imputer_path}. Proceeding without imputer.")
            self.imputer = None
        except Exception as e:
            print(f"Warning: Error loading imputer object: {e}. Proceeding without imputer.")
            self.imputer = None


        # Get feature lists and preprocess lists from the nested config structure using .get() for safety
        features_dict = self.config.get('features', {})
        preprocess_dict = self.config.get('preprocess', {})

        # Get lists from sub-dictionaries, default to empty list if key is missing
        self.time_varying_feats_config = features_dict.get('time_varying', [])
        self.static_feats_config = features_dict.get('static', [])
        self.imputation_cols_config = preprocess_dict.get('imputation_cols', [])
        self.scaling_cols_config = preprocess_dict.get('scaling_cols', [])

        # Filter lists based on columns in the loaded dataframe
        self.time_varying_feats = [f for f in self.time_varying_feats_config if f in available_cols]
        self.static_feats = [f for f in self.static_feats_config if f in available_cols]
        self.imputation_cols = [f for f in self.imputation_cols_config if f in available_cols]
        self.scaling_cols = [f for f in self.scaling_cols_config if f in available_cols]

        print(f"Using Time-Varying Features: {self.time_varying_feats}")
        print(f"Using Static Features (initial): {self.static_feats}")
        print(f"Using Imputation Columns: {self.imputation_cols}")
        print(f"Using Scaling Columns: {self.scaling_cols}")

        self.target_col = 'CDR_next_visit' # Target assumed created in NB 03
        if self.target_col not in available_cols:
             print(f"Warning: Target column '{self.target_col}' not found in loaded data!")
             # Consider raising an error depending on requirements
             # raise ValueError(f"Target column '{self.target_col}' not found.")
        self.id_col = 'Subject ID'
        self.visit_col = 'Visit'

        # --- Pre-processing ---
        # 1. Imputation
        if self.imputer and self.imputation_cols:
            print(f"Applying imputer to columns: {self.imputation_cols}")
            # Ensure only existing columns are passed to transform
            cols_to_impute_present = [col for col in self.imputation_cols if col in self.data_df.columns]
            if cols_to_impute_present:
                 self.data_df[cols_to_impute_present] = self.imputer.transform(self.data_df[cols_to_impute_present])
            else:
                 print("Warning: None of the specified imputation columns found in DataFrame.")
        elif self.imputation_cols:
             print(f"Warning: Imputation columns specified ({self.imputation_cols}) but imputer object not loaded/available.")

        # 2. Encode Categorical Static Features (Example: M/F)
        # Check if 'M/F' exists and is in the static list *before* trying to encode
        # This avoids errors if 'M/F' was already removed or renamed, or if M/F_encoded is already there
        create_mf_encoded = False
        if 'M/F_encoded' in self.static_feats_config and 'M/F' in available_cols:
            print("Encoding 'M/F' feature (F=0, M=1) as 'M/F_encoded' is expected.")
            self.data_df['M/F_encoded'] = self.data_df['M/F'].apply(lambda x: 1 if x == 'M' else 0).astype(np.float32)
            create_mf_encoded = True
        elif 'M/F_encoded' in self.static_feats_config and 'M/F_encoded' in available_cols:
             print("'M/F_encoded' column already exists in data.")
             create_mf_encoded = True # Assume it's okay to use existing one
        else:
             print("'M/F' not found for encoding OR 'M/F_encoded' not in static features config. Skipping encoding.")


        # 3. Scaling
        available_cols_post_encoding = self.data_df.columns.tolist()
        self.scaling_cols = [f for f in self.scaling_cols_config if f in available_cols_post_encoding]
        if self.scaler and self.scaling_cols:
            print(f"Applying scaler to columns: {self.scaling_cols}")
            self.data_df[self.scaling_cols] = self.scaler.transform(self.data_df[self.scaling_cols])
        elif self.scaling_cols:
            print(f"Warning: Scaling specified for {self.scaling_cols} but scaler not loaded.")

        # --- *** Define FINAL Feature Lists After Processing *** ---
        final_available_cols = self.data_df.columns.tolist()

        # Filter configured lists based on columns ACTUALLY present in the processed dataframe
        self.time_varying_feats = [f for f in self.time_varying_feats_config if f in final_available_cols]
        self.static_feats = [f for f in self.static_feats_config if f in final_available_cols]

        # Sanity check: If we expected M/F_encoded but didn't create/find it, remove it
        if 'M/F_encoded' in self.static_feats_config and 'M/F_encoded' not in final_available_cols:
             print("Warning: 'M/F_encoded' was expected in config but is not in final dataframe columns. Removing from static feature list.")
             if 'M/F_encoded' in self.static_feats: # Should have been filtered already, but double check
                 self.static_feats.remove('M/F_encoded')

        # Ensure M/F is NOT in the final list if M/F_encoded was created/used
        if create_mf_encoded and 'M/F' in self.static_feats:
             self.static_feats.remove('M/F')

        self.static_feats = sorted(list(set(self.static_feats))) # Ensure unique & sorted

        # --- Set other attributes ---
        self.target_col = 'CDR_next_visit'
        if self.target_col not in final_available_cols:
             print(f"Warning: Target column '{self.target_col}' not found in final data columns!")
        self.id_col = 'Subject ID'
        self.visit_col = 'Visit'


        # --- Group data by subject ---
        print("Grouping data by subject...")
        # Ensure sorting before grouping if not guaranteed by input parquet file
        self.data_df = self.data_df.sort_values(by=[self.id_col, self.visit_col])
        self.grouped_data = self.data_df.groupby(self.id_col)
        self.subject_ids = list(self.grouped_data.groups.keys())
        print(f"Dataset initialized for {len(self.subject_ids)} subjects.")

        # Define final combined feature list for model input (based on processed columns)
        self.model_input_features = sorted(list(set(self.time_varying_feats + self.static_feats)))
        # Final check: ensure all model input features actually exist in the df after processing
        self.model_input_features = [f for f in self.model_input_features if f in self.data_df.columns]
        print(f"Final Model input features ({len(self.model_input_features)}): {self.model_input_features}")


    def __len__(self):
        """Returns the number of subjects (sequences) in the dataset."""
        return len(self.subject_ids)

    def __getitem__(self, idx):
        """
        Retrieves the processed sequence data and target for a single subject.
        """
        subject_id = self.subject_ids[idx]
        # Use .copy() to avoid potential SettingWithCopyWarning if modifying subject_data later
        subject_data = self.grouped_data.get_group(subject_id).copy()

        # Select the final features defined during init for the sequence
        # Make sure static features (already processed) are selected correctly
        sequence_features_df = subject_data[self.model_input_features]

        # --- Target Definition Check ---
        # This assumes 'CDR_next_visit' was created correctly in NB 03,
        # representing the target for the state *after* the visit in the current row.
        # Getting the target from the last row of the sequence corresponds to
        # predicting the state after the final observed visit in the input sequence.
        # Verify this aligns with your prediction task definition.
        if self.target_col in subject_data.columns:
            target = subject_data[self.target_col].iloc[-1]
        else:
            print(f"Warning: Target column {self.target_col} not found for subject {subject_id}. Returning NaN target.")
            target = np.nan # Or handle differently, maybe raise error earlier

        # Convert features to PyTorch Tensor
        # Features shape: (sequence_length, num_features)
        features_tensor = torch.tensor(sequence_features_df.values, dtype=torch.float32)

        # Convert target to PyTorch Tensor (ensure it's float for regression loss)
        target_tensor = torch.tensor([target], dtype=torch.float32) # Keep as [1] shape

        # --- Placeholder for MRI features ---
        # Future: Load MRI feature vectors based on subject_data['MRI ID'] or other identifiers
        # mri_features = self.load_mri_features(subject_data) # Function to load/lookup features
        # features_tensor = torch.cat((features_tensor, mri_features), dim=1) # Concatenate along feature dim

        return features_tensor, target_tensor


# --- pad_collate_fn remains the same as it looked correct ---
def pad_collate_fn(batch):
    """
    Collate function for DataLoader to handle batches of sequences with varying lengths.
    Pads sequences to the maximum length in the batch.

    Args:
        batch (list): A list of tuples, where each tuple is (sequence_tensor, target_tensor)
                      as returned by OASISDataset.__getitem__.

    Returns:
        tuple: A tuple containing:
            - sequences_padded (Tensor): Batch of sequences, padded to max length. Shape: (batch_size, max_seq_len, num_features)
            - lengths (Tensor): Tensor containing the original lengths of sequences in the batch. Shape: (batch_size,)
            - targets (Tensor): Batch of target values. Shape: (batch_size, 1)
            - masks (Tensor): Boolean mask indicating non-padded elements. Shape: (batch_size, max_seq_len)
    """
    # Separate sequences and targets
    sequences = [item[0] for item in batch]
    targets = [item[1] for item in batch] # Targets are already shape [1]

    # Get sequence lengths before padding
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)

    # Pad sequences - batch_first=True means output shape (batch_size, max_seq_len, num_features)
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)

    # Stack targets
    targets = torch.stack(targets) # Shape becomes (batch_size, 1)

    # Create masks (True for real data, False for padding)
    batch_size, max_len, _ = sequences_padded.size()
    masks = torch.arange(max_len).expand(batch_size, max_len) < lengths.unsqueeze(1)

    return sequences_padded, lengths, targets, masks