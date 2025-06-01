# src/evaluation_utils.py
"""
Utility functions for evaluating PyTorch models, particularly for
regression tasks on sequence data.
"""
import torch
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm.auto import tqdm # For progress bars during evaluation

# If you were to use isinstance for model type checking, you'd import them:
# from .models import ModularLateFusionLSTM, BaselineLSTMRegressor 

def evaluate_model(model: torch.nn.Module, 
                   data_loader: torch.utils.data.DataLoader, 
                   criterion: torch.nn.Module, 
                   device: torch.device, 
                   model_name_for_batch_unpack: str) -> dict:
    """
    Evaluates a trained PyTorch model on a dataset provided by a DataLoader.
    Designed to work with models expecting sequence data and a specific batch structure.

    Args:
        model (torch.nn.Module): The trained PyTorch model to evaluate.
        data_loader (torch.utils.data.DataLoader): DataLoader for the validation or test set.
        criterion (torch.nn.Module): The loss function (e.g., nn.MSELoss) to calculate loss.
        device (torch.device): The device to run evaluation on (e.g., 'cpu', 'cuda', 'mps').
        model_name_for_batch_unpack (str): A flag indicating the model type to ensure correct
                                           batch unpacking and model forward pass. 
                                           Expected values: "baseline" or "hybrid".

    Returns:
        dict: A dictionary containing key performance metrics:
              'loss': Average loss per sample (calculated using the provided criterion).
              'mse': Mean Squared Error (calculated from predictions and targets).
              'mae': Mean Absolute Error.
              'r2': R-squared score.
              'all_preds': List of all predictions made (as floats, after NaN filtering).
              'all_targets': List of all actual targets (as floats, after NaN filtering).
              Metrics are np.nan if they cannot be computed (e.g., due to no valid data).
    """
    # --- Input Validations ---
    if not hasattr(model, 'eval') or not hasattr(model, 'train'): 
        raise ValueError("The provided 'model' object does not appear to be a valid PyTorch nn.Module.")
    if not isinstance(data_loader, torch.utils.data.DataLoader):
        raise ValueError("'data_loader' is not a valid PyTorch DataLoader.")
    if not hasattr(criterion, '__call__'): 
        raise ValueError("'criterion' (loss function) must be a callable PyTorch module.")
    if model_name_for_batch_unpack.lower() not in ["baseline", "hybrid"]:
        raise ValueError(f"Invalid 'model_name_for_batch_unpack': {model_name_for_batch_unpack}. Expected 'baseline' or 'hybrid'.")

    model.to(device)
    model.eval()  # Set the model to evaluation mode

    total_loss_accumulator = 0.0
    all_targets_list: list[float] = []
    all_predictions_list: list[float] = []
    num_samples_processed = 0

    # Disable gradient calculations during evaluation for efficiency and correctness    
    with torch.no_grad(): 
        pbar_desc = f"Evaluating ({model_name_for_batch_unpack})"
        for batch_idx, batch in enumerate(tqdm(data_loader, desc=pbar_desc, leave=False)): # Added batch_idx for debug
            
            targets_in_batch: torch.Tensor
            predictions_from_model: torch.Tensor
            # Use a common variable for sequences_tabular_padded for calculating batch_size_current
            sequences_tabular_padded_ref: torch.Tensor

            try:
                if model_name_for_batch_unpack.lower() == "hybrid":
                    sequences_tabular_padded, sequences_mri_padded, lengths, targets_in_batch, _ = batch
                    sequences_tabular_padded = sequences_tabular_padded.to(device)
                    sequences_mri_padded = sequences_mri_padded.to(device)
                    # lengths for pack_padded_sequence are usually expected on CPU by the model's forward pass
                    predictions_from_model = model(sequences_tabular_padded, sequences_mri_padded, lengths.cpu())
                    sequences_tabular_padded_ref = sequences_tabular_padded
                
                elif model_name_for_batch_unpack.lower() == "baseline":
                    sequences_tabular_padded, lengths, targets_in_batch, _ = batch
                    sequences_tabular_padded = sequences_tabular_padded.to(device)
                    predictions_from_model = model(sequences_tabular_padded, lengths.cpu())
                    sequences_tabular_padded_ref = sequences_tabular_padded

                # This else should not be reached due to input validation, but as a safeguard:
                # else: continue # Should have been caught by initial validation

                targets_in_batch = targets_in_batch.to(device)
                batch_size_current = sequences_tabular_padded_ref.size(0)

                loss = criterion(predictions_from_model, targets_in_batch)
                # Accumulate total loss (sum of losses for all samples in the dataset)
                if torch.isnan(loss): # Check if loss itself is NaN
                    print(f"  CRITICAL EVAL (Batch {batch_idx}): Loss IS NaN!")
                    print(f"    Input sequences (first sample, first 5 steps, first 3 feats snippet): \n"
                          f"{sequences_tabular_padded_ref[0, :min(5, sequences_tabular_padded_ref.size(1)), :min(3, sequences_tabular_padded_ref.size(2))] if sequences_tabular_padded_ref.numel() > 0 and sequences_tabular_padded_ref.size(0)>0 else 'Input empty or too small to snippet'}")
                    print(f"    Predictions (first 5 in batch): {predictions_from_model.flatten().tolist()[:5]}")
                    print(f"    Targets (first 5 in batch): {targets_in_batch.flatten().tolist()[:5]}")
                    # To stop immediately when NaN loss occurs and inspect the specific batch:
                    # raise ValueError(f"NaN loss detected during evaluation in batch {batch_idx}. Stopping.") 
                
                # Ensure loss is a valid number before accumulating
                if not torch.isnan(loss) and not torch.isinf(loss):
                    total_loss_accumulator += loss.item() * batch_size_current 
                else:
                    print(f"Warning (Batch {batch_idx}): Invalid loss (NaN or Inf) encountered. Not accumulating loss for this batch.")
                    # Decide if you want to count these samples as processed if loss is NaN
                    # For now, we still count them based on batch_size_current below

                all_targets_list.extend(targets_in_batch.detach().cpu().numpy().flatten().tolist())
                all_predictions_list.extend(predictions_from_model.detach().cpu().numpy().flatten().tolist())
                num_samples_processed += batch_size_current

            except Exception as e_batch_eval:
                print(f"Error during evaluation of a batch (model_type '{model_name_for_batch_unpack}', batch_idx {batch_idx}): {e_batch_eval}")
                import traceback
                traceback.print_exc() # Print full traceback for the batch error
                # Continue to process other batches
                continue 
    
    # Calculate final average loss per sample
    avg_loss_per_sample = total_loss_accumulator / num_samples_processed if num_samples_processed > 0 else np.nan
    
    # Initialize metrics to NaN
    mae = np.nan
    r2 = np.nan
    mse = np.nan 

    # Filtered lists for metric calculation (after removing NaNs)
    valid_targets_for_metrics: list[float] = []
    valid_predictions_for_metrics: list[float] = []

    if num_samples_processed > 0 and \
       len(all_targets_list) == num_samples_processed and \
       len(all_predictions_list) == num_samples_processed:
        try:
            targets_np = np.array(all_targets_list, dtype=float) # Ensure float for isnan
            preds_np = np.array(all_predictions_list, dtype=float) # Ensure float for isnan
            
            # Filter out pairs where either target or prediction is NaN
            # This is important because target 'CDR_next_visit' can be NaN for last visits if not perfectly filtered
            # before OASISDataset, or if a model somehow predicts NaN (less likely if loss is finite).
            valid_indices = ~np.isnan(targets_np) & ~np.isnan(preds_np)
            
            valid_targets_np = targets_np[valid_indices]
            valid_predictions_np = preds_np[valid_indices]

            valid_targets_for_metrics = valid_targets_np.tolist()
            valid_predictions_for_metrics = valid_predictions_np.tolist()

            if len(valid_targets_np) > 0: # Ensure there's at least one valid sample
                mse = mean_squared_error(valid_targets_np, valid_predictions_np)
                mae = mean_absolute_error(valid_targets_np, valid_predictions_np)
                if len(valid_targets_np) > 1: # R2 requires at least 2 samples
                    r2 = r2_score(valid_targets_np, valid_predictions_np)
                else: # R2 is undefined for a single sample or if all targets are identical
                    print("Warning (evaluate_model): R2 score requires at least 2 valid samples with variance in targets. R2 set to NaN.")
            else:
                print("Warning (evaluate_model): No valid (non-NaN) target/prediction pairs found for MAE/R2/MSE calculation.")
        except Exception as e_metrics:
            print(f"Error calculating final MAE/R2/MSE scores in evaluate_model: {e_metrics}")
    elif num_samples_processed == 0:
        print("Warning (evaluate_model): No samples were processed during evaluation. All metrics will be NaN.")
    else: 
        print("Warning (evaluate_model): Mismatch in num_samples_processed and collected targets/predictions. Metrics will be NaN.")

    return {
        'loss': float(avg_loss_per_sample), # Average loss from criterion per sample
        'mse': float(mse),                  # Mean Squared Error from final predictions
        'mae': float(mae),                 
        'r2': float(r2),                   
        'all_preds': valid_predictions_for_metrics,  # List of valid predictions used for metrics
        'all_targets': valid_targets_for_metrics    # List of valid targets used for metrics
    }