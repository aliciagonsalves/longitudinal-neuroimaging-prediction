# src/interpretability_utils.py
"""
Utility functions for model interpretability, including Permutation Feature Importance,
Integrated Gradients for CNNs, and SHAP explanations for LSTMs and hybrid model fusion stages.
"""
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm # For progress bars

# --- Captum Import (for Integrated Gradients) ---
try:
    from captum.attr import IntegratedGradients 
    # Saliency class is also available in captum.attr if needed for other methods
except ImportError:
    print("Warning: Captum library not found. Please install with `pip install captum`. "
          "Integrated Gradients functionality will be unavailable.")
    IntegratedGradients = None # Define as None so functions can check

# --- SHAP Import ---
try:
    import shap
except ImportError:
    print("Warning: SHAP library not found. Please install with `pip install shap`. "
          "SHAP explanation functionality will be unavailable.")
    shap = None # Define as None so functions can check


def calculate_permutation_importance(model: torch.nn.Module, 
                                     data_loader: torch.utils.data.DataLoader, 
                                     feature_names_to_permute: list[str],
                                     tabular_feature_indices: dict[str, int],
                                     metric_fn: callable, # e.g., lambda y_true, y_pred: r2_score(y_true, y_pred)
                                     baseline_score: float,
                                     device: torch.device,
                                     is_hybrid_model: bool,
                                     num_permutations: int = 10) -> dict[str, float]:
    """
    Calculates permutation importance for specified tabular features by measuring
    the drop in a given performance metric when a feature is shuffled.

    Args:
        model: The trained PyTorch model.
        data_loader: DataLoader for the evaluation dataset.
        feature_names_to_permute: List of tabular feature names to calculate importance for.
        tabular_feature_indices: Dictionary mapping feature names to their column indices
                                 in the tabular feature tensor.
        metric_fn: A function that takes (y_true, y_pred) and returns a score.
        baseline_score: The model's performance score on the metric_fn with non-permuted data.
        device: The PyTorch device to use for evaluation.
        is_hybrid_model (bool): Flag to indicate if the model is the hybrid model,
                                affecting batch unpacking and model call signature.
        num_permutations (int): Number of times to permute each feature and average the score.

    Returns:
        dict: A dictionary mapping feature names to their importance scores (drop in metric).
              Higher values indicate greater importance.
    """
    print(f"\nCalculating Permutation Importance for: {feature_names_to_permute}...")
    importances = {}
    model.eval() # Ensure model is in evaluation mode
    model.to(device)

    for feature_name in tqdm(feature_names_to_permute, desc="Permuting Features", leave=False):
        if feature_name not in tabular_feature_indices:
            print(f"Warning (PFI): Feature '{feature_name}' not in provided tabular_feature_indices. Skipping.")
            importances[feature_name] = 0.0 # Or np.nan, or skip adding it
            continue
        
        feature_idx_to_permute = tabular_feature_indices[feature_name]
        permuted_metric_scores = []

        for _ in range(num_permutations):
            all_permuted_predictions = []
            all_original_targets = []

            with torch.no_grad(): # No gradients needed for inference
                for batch in data_loader:
                    targets_in_batch: torch.Tensor
                    predictions_from_model: torch.Tensor
                    
                    # Clone the tabular sequence part of the batch for permutation
                    if is_hybrid_model:
                        seq_tab, seq_mri, lengths, targets_in_batch, _ = batch
                        seq_tab_permuted = seq_tab.clone().to(device)
                        seq_mri = seq_mri.to(device)
                    else: # Baseline model
                        seq_tab, lengths, targets_in_batch, _ = batch
                        seq_tab_permuted = seq_tab.clone().to(device)
                    
                    targets_in_batch = targets_in_batch.to(device)
                    lengths_cpu = lengths.cpu() # For pack_padded_sequence if used in model

                    # Permute the specified feature column for each sample in the batch
                    # This shuffles the feature's values across the time dimension for each sample independently.
                    for sample_idx in range(seq_tab_permuted.size(0)):
                        # Create random permutation indices for the sequence length of this sample
                        # Use actual length if available, otherwise max sequence length in batch
                        # For simplicity, using seq_tab_permuted.size(1) (max_seq_len in batch)
                        # A more precise permutation might use individual sequence lengths if features
                        # are only valid up to that length, but this is simpler.
                        perm_indices_for_time = torch.randperm(seq_tab_permuted.size(1)) 
                        seq_tab_permuted[sample_idx, :, feature_idx_to_permute] = \
                            seq_tab_permuted[sample_idx, perm_indices_for_time, feature_idx_to_permute]

                    # Get predictions with the permuted feature
                    if is_hybrid_model:
                        predictions_from_model = model(seq_tab_permuted, seq_mri, lengths_cpu)
                    else:
                        predictions_from_model = model(seq_tab_permuted, lengths_cpu)
                    
                    all_permuted_predictions.extend(predictions_from_model.detach().cpu().numpy().flatten().tolist())
                    all_original_targets.extend(targets_in_batch.detach().cpu().numpy().flatten().tolist())
            
            if all_original_targets and all_permuted_predictions:
                permuted_metric_scores.append(metric_fn(np.array(all_original_targets), np.array(all_permuted_predictions)))
        
        if permuted_metric_scores:
            importances[feature_name] = baseline_score - np.mean(permuted_metric_scores)
        else:
            print(f"Warning (PFI): No permuted scores calculated for feature '{feature_name}'. Importance set to 0.")
            importances[feature_name] = 0.0 

    return importances


def generate_integrated_gradients_cnn(cnn_model_part: torch.nn.Module, 
                                      mri_input_tensor_batch: torch.Tensor, # Expects (N, C, D, H, W)
                                      n_steps: int = 50,
                                      captum_internal_batch_size: int = None # Captum's internal batching for n_steps
                                     ) -> np.ndarray | None:
    """
    Generates Integrated Gradients attribution maps for a batch of MRI inputs
    using the provided CNN model part. Attributes to the sum of the CNN's output features.

    Args:
        cnn_model_part (torch.nn.Module): The CNN part of the model (e.g., Simple3DCNN instance).
        mri_input_tensor_batch (torch.Tensor): Batch of MRI tensors, shape (N, C, D, H, W).
        n_steps (int): Number of steps for the Integrated Gradients approximation.
        captum_internal_batch_size (int, optional): If provided, Captum will batch the n_steps
                                                    computations internally.

    Returns:
        np.ndarray or None: Attribution maps as a NumPy array of shape (N, C, D, H, W),
                            or None if Captum/IntegratedGradients is unavailable or an error occurs.
    """
    if IntegratedGradients is None: # Check if captum.attr.IntegratedGradients was successfully imported
        print("Captum's IntegratedGradients is not available (likely Captum not installed). Skipping IG calculation.")
        return None

    # Validate input tensor
    if not isinstance(mri_input_tensor_batch, torch.Tensor) or mri_input_tensor_batch.ndim != 5: 
        print(f"Error (IG): Expected 5D PyTorch tensor for mri_input_tensor_batch (N, C, D, H, W), "
              f"got {type(mri_input_tensor_batch)} with shape {mri_input_tensor_batch.shape if hasattr(mri_input_tensor_batch, 'shape') else 'N/A'}.")
        return None
    if mri_input_tensor_batch.size(0) == 0: 
        print("Error (IG): mri_input_tensor_batch is empty. Skipping IG calculation.")
        return None

    print(f"\nGenerating Integrated Gradients for {mri_input_tensor_batch.size(0)} MRI sample(s)...")
    
    cnn_model_part.eval() # Ensure CNN part is in evaluation mode for consistent attributions

    # Define the forward function that Captum's IntegratedGradients will use.
    # It takes the (potentially batched by Captum) interpolated inputs and returns a scalar
    # output per instance in that batch (by summing the CNN's feature vector).
    def model_forward_for_ig(interpolated_mri_batch_from_captum: torch.Tensor) -> torch.Tensor:
        # interpolated_mri_batch_from_captum from Captum typically has shape 
        # (current_ig_processing_batch_size, C, D, H, W)
        output_features = cnn_model_part(interpolated_mri_batch_from_captum)
        # output_features shape: (current_ig_processing_batch_size, num_cnn_output_features)
        return torch.sum(output_features, dim=1) # Sum features to get a scalar: (current_ig_processing_batch_size,)

    ig_algorithm = IntegratedGradients(model_forward_for_ig)
    
    # Define baseline (all-zeros tensor of the same shape and device as the input batch)
    baseline_mris = torch.zeros_like(mri_input_tensor_batch, device=mri_input_tensor_batch.device)
    
    try:
        # Calculate attributions for the entire input batch
        attributions_batch = ig_algorithm.attribute(
            inputs=mri_input_tensor_batch,      # Original input batch, shape (N, C, D, H, W)
            baselines=baseline_mris,            # Baseline batch, shape (N, C, D, H, W)
            n_steps=n_steps,
            internal_batch_size=captum_internal_batch_size
            # No target_index is specified, so IG attributes to the scalar output of model_forward_for_ig
        )
        # attributions_batch should have the same shape as mri_input_tensor_batch: (N, C, D, H, W)
        return attributions_batch.cpu().detach().numpy()
    except Exception as e_ig_calc:
        print(f"Error during Integrated Gradients .attribute() call: {e_ig_calc}")
        import traceback
        traceback.print_exc()
        return None


def explain_lstm_with_shap(model: torch.nn.Module, 
                           background_data_loader: torch.utils.data.DataLoader, 
                           instances_to_explain_loader: torch.utils.data.DataLoader, 
                           device: torch.device,
                           feature_names: list[str] = None) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Provides SHAP explanations for an LSTM-based model (e.g., BaselineLSTMRegressor) 
    using SHAP's DeepExplainer.

    Args:
        model: The trained PyTorch LSTM-based model.
        background_data_loader: DataLoader for a subset of training data (sequences_padded should be batch[0]).
        instances_to_explain_loader: DataLoader for instances to explain (sequences_padded should be batch[0]).
        device: PyTorch device.
        feature_names: List of feature names for SHAP plots (length must match num_features).

    Returns:
        tuple: (processed_shap_values, explained_instances_np)
               - processed_shap_values (np.ndarray, shape (N, S, F)): SHAP values.
               - explained_instances_np (np.ndarray, shape (N, S, F)): The actual feature values of explained instances.
               Returns (None, None) if SHAP fails.
    """
    if shap is None: # Check if SHAP library was successfully imported at the module level
        print("SHAP library not available. Skipping SHAP explanation.")
        return None, None

    print("\n--- Running SHAP DeepExplainer for LSTM ---")
    model.to(device)
    model.eval() # SHAP explainers typically expect the model in eval mode

    # 1. Prepare background tensor from background_data_loader
    background_sequences_list = []
    print("  SHAP: Preparing background data...")
    try:
        # Limit number of batches for background to keep it manageable and faster
        num_bg_batches_to_use = min(len(background_data_loader), 5) # Example: use up to 5 batches
        for i, batch_data in enumerate(tqdm(background_data_loader, desc="SHAP Background Data", leave=False, total=num_bg_batches_to_use)):
            if i >= num_bg_batches_to_use: break
            # Assuming batch_data[0] is sequences_padded based on pad_collate_fn for baseline
            if not isinstance(batch_data[0], torch.Tensor):
                print(f"  SHAP Background: Batch item 0 is not a tensor (type: {type(batch_data[0])}). Skipping batch.")
                continue
            background_sequences_list.append(batch_data[0].to(device)) 
        
        if not background_sequences_list:
            print("  SHAP: Background data list is empty after iterating loader. Cannot initialize SHAP.")
            return None, None
        background_tensor = torch.cat(background_sequences_list, dim=0)
        print(f"  SHAP: Using background tensor of shape: {background_tensor.shape}")
    except Exception as e_bg:
        print(f"  SHAP: Error preparing background data: {e_bg}")
        return None, None

    # 2. Prepare instances tensor from instances_to_explain_loader
    instances_sequences_list = []
    actual_instances_for_plot_list = [] # To store feature values of explained instances
    print("  SHAP: Preparing instances to explain...")
    try:
        for batch_data in tqdm(instances_to_explain_loader, desc="SHAP Instances to Explain", leave=False):
            # Assuming batch_data[0] is sequences_padded
            sequences_to_explain = batch_data[0].to(device)
            if not isinstance(sequences_to_explain, torch.Tensor):
                print(f"  SHAP Instances: Batch item 0 is not a tensor (type: {type(sequences_to_explain)}). Skipping batch.")
                continue
            instances_sequences_list.append(sequences_to_explain)
            actual_instances_for_plot_list.append(sequences_to_explain.cpu().numpy()) # Store for plotting
            
        if not instances_sequences_list:
            print("  SHAP: Instances to explain list is empty after iterating loader. Skipping SHAP.")
            return None, None
        instances_tensor = torch.cat(instances_sequences_list, dim=0)
        explained_instances_np = np.concatenate(actual_instances_for_plot_list, axis=0) # For plotting features
        print(f"  SHAP: Explaining {instances_tensor.size(0)} instances, sequence tensor shape: {instances_tensor.shape}")
    except Exception as e_inst:
        print(f"  SHAP: Error preparing instances to explain: {e_inst}")
        return None, None
        
    # 3. Create SHAP DeepExplainer
    explainer = None
    try:
        # DeepExplainer expects the model and a background data tensor.
        # Model's forward should ideally handle a single tensor input (sequences) if lengths are optional.
        explainer = shap.DeepExplainer(model, background_tensor)
        print("  SHAP: DeepExplainer initialized.")
    except Exception as e_init_explainer:
        print(f"  SHAP: Error initializing DeepExplainer: {e_init_explainer}")
        print("        Ensure model.forward can be called with just the sequence tensor, "
              "or consider a model wrapper for SHAP.")
        return None, None

    # 4. Calculate SHAP values
    shap_values_raw = None
    try:
        if explainer is None: # Should be caught by previous try-except
            print("  SHAP: Explainer is None. Cannot calculate SHAP values.")
            return None, None
            
        print("  SHAP: Calculating SHAP values (this can take time)...")
        shap_values_raw = explainer.shap_values(instances_tensor)
        # For single-output regression, shap_values_raw is often (N, S, F) or (N, S, F, 1)
        # Or a list containing one such array.
        
    except Exception as e_shap_calc:
        print(f"  SHAP: Error during explainer.shap_values() call: {e_shap_calc}")
        import traceback
        traceback.print_exc()
        return None, None # Failed to get SHAP values

    # 5. Process SHAP values to expected (N, S, F) shape
    processed_shap_values = None
    if shap_values_raw is Ellipsis:
        print("  SHAP ERROR: explainer.shap_values() returned Ellipsis, indicating internal failure.")
        return None, None
    elif isinstance(shap_values_raw, list):
        if len(shap_values_raw) == 1: 
            processed_shap_values = shap_values_raw[0]
        elif len(shap_values_raw) > 1:
            print(f"  SHAP Warning: shap_values is a list of {len(shap_values_raw)} arrays (model might be seen as multi-output by SHAP). Using the first array.")
            processed_shap_values = shap_values_raw[0]
        else: # Empty list
            print("  SHAP Warning: shap_values_raw is an empty list.")
            return None, None
    elif isinstance(shap_values_raw, np.ndarray):
        processed_shap_values = shap_values_raw
    else:
        print(f"  SHAP Error: shap_values_raw has unexpected type: {type(shap_values_raw)}")
        return None, None
    
    if processed_shap_values is None: # If it's still None after list handling
        print("  SHAP Error: Could not extract valid NumPy array from shap_values_raw.")
        return None, None

    print(f"  SHAP: Intermediate processed SHAP values shape: {processed_shap_values.shape}")
    if processed_shap_values.ndim == 4 and processed_shap_values.shape[-1] == 1:
        print("  SHAP: Squeezing the last dimension of SHAP values array.")
        processed_shap_values = processed_shap_values.squeeze(-1)
    
    if not (isinstance(processed_shap_values, np.ndarray) and processed_shap_values.ndim == 3): 
        print(f"  SHAP Error: Processed SHAP values have unexpected shape {processed_shap_values.shape if hasattr(processed_shap_values, 'shape') else 'N/A'}. Expected 3D (N, S, F).")
        return None, None
    
    print(f"  SHAP: Final SHAP values shape: {processed_shap_values.shape}")
    return processed_shap_values, explained_instances_np


def explain_hybrid_fusion_with_shap(
    hybrid_model: torch.nn.Module, 
    background_data_loader: torch.utils.data.DataLoader, 
    instances_to_explain_loader: torch.utils.data.DataLoader, 
    device: torch.device,
    num_background_samples_for_kmeans: int = 20,
    num_shap_samples: str | int = 'auto' # nsamples for KernelExplainer
) -> tuple[np.ndarray | None, pd.DataFrame | None, int, int]:
    """
    Explains the fusion part of a Hybrid (CNN+LSTM) model using SHAP KernelExplainer.
    Focuses on the importance of the outputs from the MRI LSTM stream vs. Tabular LSTM stream
    as they feed into the final prediction layers.

    Args:
        hybrid_model: The trained ModularLateFusionLSTM instance.
        background_data_loader: DataLoader yielding full hybrid inputs for background summarization.
        instances_to_explain_loader: DataLoader yielding full hybrid inputs for instances to be explained.
        device: PyTorch device.
        num_background_samples_for_kmeans (int): Number of samples to use for shap.kmeans summarization.
        num_shap_samples (str or int): 'nsamples' argument for explainer.shap_values().

    Returns:
        tuple: (shap_values_fusion, explained_df, mri_stream_feature_count, tab_stream_feature_count)
               Returns (None, None, 0, 0) on failure.
    """
    if shap is None:
        print("SHAP library not available. Skipping SHAP explanation for hybrid model.")
        return None, None, 0, 0

    print("\n--- Running SHAP KernelExplainer for Hybrid Model Fusion Stage ---")
    hybrid_model.to(device)
    hybrid_model.eval()

    # Helper function to get (detached, CPU) LSTM outputs before fusion
    def get_pre_fusion_features_from_hybrid(model_to_run, data_loader_full_input):
        all_mri_lstm_outputs_list = []
        all_tab_lstm_outputs_list = []
        print(f"  SHAP Hybrid: Extracting pre-fusion features from {len(data_loader_full_input.dataset)} unique sequences (approx)...") # Using dataset length
        
        with torch.no_grad():
            for batch_data in tqdm(data_loader_full_input, desc="Extracting Pre-Fusion Feats", leave=False):
                try:
                    # Assuming 5-item batch for hybrid model from pad_collate_fn
                    seq_tab, seq_mri, lengths, _, _ = batch_data 
                    
                    seq_tab = seq_tab.to(device)
                    seq_mri = seq_mri.to(device)
                    lengths_cpu = lengths.cpu()

                    # MRI Stream: CNN + MRI LSTM
                    batch_s, seq_l, C, D, H, W_mri = seq_mri.shape # Corrected W to W_mri
                    mri_reshaped = seq_mri.reshape(batch_s * seq_l, C, D, H, W_mri)
                    cnn_feats = model_to_run.cnn_feature_extractor(mri_reshaped)
                    mri_feat_seq = cnn_feats.view(batch_s, seq_l, -1)
                    
                    # Handle empty sequences that might result from padding if lengths are all zero for some reason
                    if mri_feat_seq.size(0) > 0 and lengths_cpu.max() > 0: # Ensure there's something to pack
                        packed_mri = torch.nn.utils.rnn.pack_padded_sequence(
                            mri_feat_seq, lengths_cpu, batch_first=True, enforce_sorted=False
                        )
                        _, (h_n_mri, _) = model_to_run.mri_lstm(packed_mri)
                        all_mri_lstm_outputs_list.append(h_n_mri[-1].cpu())
                    elif mri_feat_seq.size(0) > 0: # All lengths are zero but batch exists
                         all_mri_lstm_outputs_list.append(torch.zeros((batch_s, model_to_run.mri_lstm.hidden_size), dtype=torch.float32))
                    # Else, if mri_feat_seq.size(0) == 0, this batch was empty, skip append.

                    # Tabular Stream: Tabular LSTM
                    if seq_tab.size(0) > 0 and lengths_cpu.max() > 0:
                        packed_tab = torch.nn.utils.rnn.pack_padded_sequence(
                            seq_tab, lengths_cpu, batch_first=True, enforce_sorted=False
                        )
                        _, (h_n_tab, _) = model_to_run.tabular_lstm(packed_tab)
                        all_tab_lstm_outputs_list.append(h_n_tab[-1].cpu())
                    elif seq_tab.size(0) > 0: # All lengths are zero but batch exists
                         all_tab_lstm_outputs_list.append(torch.zeros((batch_s, model_to_run.tabular_lstm.hidden_size), dtype=torch.float32))

                except Exception as e_inner_get_feats:
                    print(f"  Error in get_pre_fusion_features_from_hybrid batch processing: {e_inner_get_feats}")
                    continue # Skip this batch

        if not all_mri_lstm_outputs_list or not all_tab_lstm_outputs_list: 
            print("  SHAP Hybrid: Pre-fusion feature lists are empty. Cannot proceed.")
            return None, None
        
        # Concatenate all collected batch outputs
        final_mri_lstm_outputs = torch.cat(all_mri_lstm_outputs_list, dim=0)
        final_tab_lstm_outputs = torch.cat(all_tab_lstm_outputs_list, dim=0)
        
        return final_mri_lstm_outputs, final_tab_lstm_outputs

    # 1. Get pre-fusion features for background and instances
    print("  SHAP Hybrid: Preparing background pre-fusion features...")
    background_mri_lstm_feats, background_tab_lstm_feats = get_pre_fusion_features_from_hybrid(hybrid_model, background_data_loader)
    if background_mri_lstm_feats is None or background_tab_lstm_feats is None: # Check both
        print("  SHAP Hybrid: Failed to get background pre-fusion features.")
        return None, None, 0, 0
    
    print("  SHAP Hybrid: Preparing instances pre-fusion features...")
    instances_mri_lstm_feats, instances_tab_lstm_feats = get_pre_fusion_features_from_hybrid(hybrid_model, instances_to_explain_loader)
    if instances_mri_lstm_feats is None or instances_tab_lstm_feats is None: # Check both
        print("  SHAP Hybrid: Failed to get instances pre-fusion features.")
        return None, None, 0, 0 # Return 0 for counts if features are None
        
    # Concatenate MRI and Tabular LSTM outputs to form the features SHAP will explain
    background_for_shap_kernel = np.concatenate((background_mri_lstm_feats.numpy(), background_tab_lstm_feats.numpy()), axis=1)
    instances_for_shap = np.concatenate((instances_mri_lstm_feats.numpy(), instances_tab_lstm_feats.numpy()), axis=1)
    
    mri_feature_count_ret = background_mri_lstm_feats.shape[1]
    tab_feature_count_ret = background_tab_lstm_feats.shape[1]

    print(f"  SHAP Hybrid: Background for KernelExplainer shape: {background_for_shap_kernel.shape}")
    print(f"  SHAP Hybrid: Instances to explain shape: {instances_for_shap.shape}")

    # 2. Define predict function for KernelExplainer (uses only the fusion part of the hybrid model)
    def fusion_predict_fn_for_kernel(concatenated_lstm_outputs_np: np.ndarray) -> np.ndarray:
        input_tensor = torch.tensor(concatenated_lstm_outputs_np, dtype=torch.float32).to(device)
        with torch.no_grad():
            # Pass through the hybrid model's layers *after* LSTM outputs are concatenated
            # These are self.dropout_after_fusion and self.final_fc in ModularLateFusionLSTM
            fused_dropped = hybrid_model.dropout_after_fusion(input_tensor)
            output = hybrid_model.final_fc(fused_dropped)
        return output.cpu().numpy()

    # 3. Create KernelExplainer using a summary of the background data
    num_actual_bg_for_kmeans = background_for_shap_kernel.shape[0]
    k_for_kmeans = min(num_background_samples_for_kmeans, num_actual_bg_for_kmeans)
    if k_for_kmeans <= 0: # k must be > 0 for shap.kmeans
        print(f"  SHAP Hybrid: Not enough background samples for K-Means (k={k_for_kmeans} from {num_actual_bg_for_kmeans} available). Summarizing with raw samples if possible or skipping.")
        if num_actual_bg_for_kmeans > 0: # Use raw samples if k is too small for kmeans but samples exist
            background_summary_for_explainer = background_for_shap_kernel
            print(f"  SHAP Hybrid: Using raw background samples (count: {num_actual_bg_for_kmeans}) for KernelExplainer.")
        else:
            print("  SHAP Hybrid: No background samples available. Skipping KernelExplainer initialization.")
            return None, None, mri_feature_count_ret, tab_feature_count_ret
    else:
        print(f"  SHAP Hybrid: Summarizing background data with K-Means (k={k_for_kmeans})...")
        try:
            background_summary_for_explainer = shap.kmeans(background_for_shap_kernel, k_for_kmeans)
        except Exception as e_kmeans: # Catch potential errors in shap.kmeans e.g. if k is too large for samples
            print(f"  SHAP Hybrid: Error during shap.kmeans (k={k_for_kmeans}, samples={num_actual_bg_for_kmeans}). Using raw background data. Error: {e_kmeans}")
            background_summary_for_explainer = background_for_shap_kernel # Fallback to using all background samples

    explainer = None
    try:
        explainer = shap.KernelExplainer(fusion_predict_fn_for_kernel, background_summary_for_explainer)
        print("  SHAP Hybrid: KernelExplainer initialized.")
    except Exception as e_init_explainer:
        print(f"  SHAP Hybrid: Error initializing KernelExplainer: {e_init_explainer}")
        return None, None, mri_feature_count_ret, tab_feature_count_ret

    # 4. Get SHAP values
    shap_values_to_return = None
    explained_df_to_return = None
    
    try:
        if explainer is None: # Should have been caught above
            print("  SHAP Hybrid: Explainer is None. Cannot calculate SHAP values.")
            return None, None, mri_feature_count_ret, tab_feature_count_ret

        if instances_for_shap is None or instances_for_shap.shape[0] == 0:
             print("  SHAP Hybrid: 'instances_for_shap' is None or empty. Cannot calculate SHAP values.")
             return None, None, mri_feature_count_ret, tab_feature_count_ret

        print("  SHAP Hybrid: Calculating SHAP values for fusion stage (this might take a while)...")
        raw_shap_values = explainer.shap_values(instances_for_shap, nsamples=num_shap_samples) 
        
        if not isinstance(raw_shap_values, np.ndarray):
            print(f"  SHAP Hybrid: KernelExplainer.shap_values did not return a NumPy array as expected. Got {type(raw_shap_values)}.")
            return None, None, mri_feature_count_ret, tab_feature_count_ret

        print(f"  SHAP Hybrid: Raw SHAP values from explainer. Shape: {raw_shap_values.shape}")

        processed_shap_values = raw_shap_values
        # KernelExplainer for single output regression usually returns (N, M) or (N, M, 1)
        # Squeeze if it's (N, M, 1)
        if processed_shap_values.ndim == 3 and processed_shap_values.shape[-1] == 1:
            print("  SHAP Hybrid: Squeezing trailing dimension of size 1 from SHAP values.")
            processed_shap_values = processed_shap_values.squeeze(-1)
            print(f"  SHAP Hybrid: Squeezed SHAP values shape: {processed_shap_values.shape}")
        
        if not (isinstance(processed_shap_values, np.ndarray) and processed_shap_values.ndim == 2):
            print(f"  SHAP Hybrid: Expected 2D NumPy array (N, M) for processed_shap_values after squeeze, "
                  f"got ndim {processed_shap_values.ndim if hasattr(processed_shap_values, 'ndim') else 'N/A'} "
                  f"and shape {processed_shap_values.shape if hasattr(processed_shap_values, 'shape') else 'N/A'}.")
            return None, None, mri_feature_count_ret, tab_feature_count_ret
            
        # These counts are now correctly derived from the actual pre-fusion features computed earlier
        # (mri_feature_count_ret, tab_feature_count_ret)
        fusion_feature_names = [f"MRI_LSTM_Feat_{i}" for i in range(mri_feature_count_ret)] + \
                               [f"Tab_LSTM_Feat_{i}" for i in range(tab_feature_count_ret)]
        
        if len(fusion_feature_names) != processed_shap_values.shape[1]:
            print(f"  SHAP Hybrid: Mismatch! Number of fusion_feature_names ({len(fusion_feature_names)}) "
                  f"does not match features in SHAP values ({processed_shap_values.shape[1]}).")
            # This indicates a critical error in feature counting or SHAP value generation.
            # Create generic names based on SHAP output as a fallback for the DataFrame.
            fusion_feature_names = [f"FusedFeat_{i}" for i in range(processed_shap_values.shape[1])]

        explained_df_to_return = pd.DataFrame(instances_for_shap, columns=fusion_feature_names)
        shap_values_to_return = processed_shap_values
        
        return shap_values_to_return, explained_df_to_return, mri_feature_count_ret, tab_feature_count_ret

    except Exception as e_shap_val_proc:
        print(f"  SHAP Hybrid: Error during SHAP value generation or processing for fusion stage: {e_shap_val_proc}")
        import traceback
        traceback.print_exc()
        return None, None, mri_feature_count_ret, tab_feature_count_ret