# src/uncertainty_utils.py
"""
Utility functions for performing Monte Carlo (MC) Dropout to estimate
model prediction uncertainty.
"""
import torch
import numpy as np
from tqdm.auto import tqdm # For progress bars

# Import model classes for isinstance checks,
# ensuring this utility can adapt if models have different input needs.
try:
    from .models import ModularLateFusionLSTM, BaselineLSTMRegressor
except ImportError:
    # Fallback for cases where this script might be run in a context
    # where relative imports from .models don't work as expected.
    # This assumes models.py is in the same directory or Python path.
    from models import ModularLateFusionLSTM, BaselineLSTMRegressor

def get_mc_dropout_predictions(model: torch.nn.Module, 
                               data_loader: torch.utils.data.DataLoader, 
                               num_mc_samples: int, 
                               device: torch.device) -> tuple[list[list[float]], list[float]]:
    """
    Performs MC Dropout inference for a given model and DataLoader.

    This function activates dropout layers (by setting model.train()) and runs
    multiple forward passes to get a distribution of predictions for each input sample.

    Args:
        model (torch.nn.Module): The trained PyTorch model with dropout layers.
        data_loader (torch.utils.data.DataLoader): DataLoader for the data to be evaluated.
                                                   Batches from this loader will be processed.
        num_mc_samples (int): The number of stochastic forward passes to make for each input sample.
        device (torch.device): The device ('cpu', 'cuda', 'mps') to run inference on.

    Returns:
        tuple: A tuple containing:
            - all_predictions_mc (list of lists of floats): 
                For each sample in the data_loader, a list of `num_mc_samples` predictions.
            - all_actuals_mc (list of floats): 
                The corresponding actual target values for each sample.
    """
    if not hasattr(model, 'train') or not hasattr(model, 'eval'):
        raise ValueError("Model object does not appear to be a valid PyTorch nn.Module.")
    if not isinstance(data_loader, torch.utils.data.DataLoader):
        raise ValueError("data_loader is not a valid PyTorch DataLoader.")
    if not isinstance(num_mc_samples, int) or num_mc_samples <= 0:
        raise ValueError("num_mc_samples must be a positive integer.")

    model.to(device)
    model.train()  # Activate dropout layers for MC Dropout stochasticity

    all_predictions_mc: list[list[float]] = [] 
    all_actuals_mc: list[float] = []

    print(f"Running MC Dropout: {num_mc_samples} forward passes per batch sample...")
    for batch in tqdm(data_loader, desc="MC Dropout Batches", leave=False):
        # Determine model type for correct input unpacking and forwarding
        # This check allows the function to be generic for your defined models.
        is_hybrid = isinstance(model, ModularLateFusionLSTM)
        is_baseline = isinstance(model, BaselineLSTMRegressor)

        if is_hybrid:
            # Expected batch: sequences_tabular, sequences_mri, lengths, targets, masks
            sequences_tabular_padded, sequences_mri_padded, lengths, targets, _ = batch
            sequences_tabular_padded = sequences_tabular_padded.to(device)
            sequences_mri_padded = sequences_mri_padded.to(device)
            # lengths for pack_padded_sequence are expected on CPU by model's forward pass
            model_inputs = (sequences_tabular_padded, sequences_mri_padded, lengths.cpu()) 
        elif is_baseline:
            # Expected batch: sequences_tabular, lengths, targets, masks
            sequences_tabular_padded, lengths, targets, _ = batch
            sequences_tabular_padded = sequences_tabular_padded.to(device)
            # lengths for pack_padded_sequence are expected on CPU by model's forward pass
            model_inputs = (sequences_tabular_padded, lengths.cpu())
        else:
            # If model type is unknown, cannot be sure how to unpack or call.
            # For now, this will cause an error later if model_inputs isn't set.
            # In future, develop a more generic input handling for a robust solution for unknown models
            print(f"Warning: Unknown model type {type(model)}. Attempting generic call if possible, but batch unpacking might be incorrect.")
            # Attempt a generic call if model_inputs wasn't set (this is a fallback)
            # This part is risky if the batch structure is unknown.
            # It's better if supported models are explicitly handled.
            # For now, relying on the is_hybrid/is_baseline checks.
            # If neither, model_inputs might remain undefined, caught by try-except below.
            pass


        targets = targets.to(device)

        # Store MC predictions for each sample within the current batch
        # Each element in this list will be a tensor of predictions for one MC pass: (batch_size_dl, 1)
        current_batch_mc_predictions_list: list[torch.Tensor] = []
        
        try:
            with torch.no_grad(): # Gradients are not needed for these inference passes
                for _ in range(num_mc_samples):
                    predictions = model(*model_inputs) # Unpack model_inputs tuple
                    current_batch_mc_predictions_list.append(predictions.detach().cpu())
            
            if current_batch_mc_predictions_list:
                # Stack predictions along a new dimension: (num_mc_samples, batch_size_dl, 1_output_dim)
                stacked_mc_preds_for_batch = torch.stack(current_batch_mc_predictions_list, dim=0)
                # Permute to: (batch_size_dl, num_mc_samples, 1_output_dim)
                permuted_mc_preds_for_batch = stacked_mc_preds_for_batch.permute(1, 0, 2)

                # Collect predictions for each sample in the batch
                for i in range(permuted_mc_preds_for_batch.size(0)): # Iterate over samples in this batch
                    # .squeeze() removes the last dimension if it's 1 (for single output regression)
                    # .tolist() converts the tensor of predictions for one sample to a list of floats
                    all_predictions_mc.append(permuted_mc_preds_for_batch[i].squeeze().tolist())
                    all_actuals_mc.append(targets[i].cpu().item())
            else:
                print(f"Warning: No MC predictions generated for a batch. Batch size: {targets.size(0)}")
                # Handle cases where a batch might yield no predictions (e.g., if num_mc_samples was 0)
                # by appending empty lists or appropriate placeholders if necessary for downstream consistency.
                # For now, if no predictions, these samples are effectively skipped for all_predictions_mc.

        except Exception as e_batch:
            print(f"Error during MC Dropout processing for a batch: {e_batch}")
            import traceback
            traceback.print_exc()
            # Decide how to handle: skip batch, append NaNs for actuals/predictions?
            # For now, if a batch errors, its samples are not added.
            continue


    model.eval() # CRITICAL: Set model back to evaluation mode after MC Dropout
    print(f"MC Dropout finished. Processed {len(all_actuals_mc)} samples.")
    return all_predictions_mc, all_actuals_mc


def calculate_uncertainty_metrics(mc_predictions_per_sample: list[list[float]]) -> list[dict[str, float]]:
    """
    Calculates mean, variance, and standard deviation for each sample's MC Dropout predictions.

    Args:
        mc_predictions_per_sample (list of lists of floats): 
            A list where each inner list contains the `num_mc_samples` predictions
            for a single input sample. This is the first element returned by 
            `get_mc_dropout_predictions`.

    Returns:
        list of dicts: A list where each dictionary contains 'mean', 'variance', 
                       and 'std_dev' for the corresponding input sample.
                       Returns NaNs if metrics can't be calculated for a sample.
    """
    uncertainty_results: list[dict[str, float]] = []
    if not isinstance(mc_predictions_per_sample, list):
        print("Warning: mc_predictions_per_sample is not a list. Cannot calculate uncertainty metrics.")
        return uncertainty_results

    print(f"Calculating uncertainty metrics for {len(mc_predictions_per_sample)} samples...")
    for i, preds_for_one_sample in enumerate(tqdm(mc_predictions_per_sample, desc="Calculating Uncertainty Stats", leave=False)):
        if not isinstance(preds_for_one_sample, list) or not preds_for_one_sample:
            print(f"Warning: Sample {i} has an empty or invalid list of predictions. Appending NaN metrics.")
            uncertainty_results.append({'mean': np.nan, 'variance': np.nan, 'std_dev': np.nan})
            continue
        try:
            # Ensure predictions are floats for tensor conversion
            float_preds = [float(p) for p in preds_for_one_sample]
            preds_tensor = torch.tensor(float_preds, dtype=torch.float32)
            
            mean_pred = torch.mean(preds_tensor).item()
            # Use unbiased=True for sample variance/std (ddof=1 equivalent in numpy)
            var_pred = torch.var(preds_tensor, unbiased=True).item() 
            std_pred = torch.std(preds_tensor, unbiased=True).item()
            
            uncertainty_results.append({'mean': mean_pred, 'variance': var_pred, 'std_dev': std_pred})
        except Exception as e_calc:
            print(f"Warning: Could not calculate metrics for sample {i}'s predictions. First 5 preds: {str(preds_for_one_sample[:5])}. Error: {e_calc}")
            uncertainty_results.append({'mean': np.nan, 'variance': np.nan, 'std_dev': np.nan})
            
    return uncertainty_results

# Move utility functions for plotting uncertainty here in the future