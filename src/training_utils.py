# src/training_utils.py

import torch
import time
import wandb # For logging inside the training loop
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score
# Assuming evaluate_model is in src.evaluation_utils
# We will pass evaluate_model as an argument for flexibility.

try:
    from tqdm.auto import tqdm # Auto-selects notebook or console tqdm
except ImportError:
    print("Warning: tqdm not installed. Consider `pip install tqdm` for progress bars.")
    def tqdm(iterable, **kwargs): # Dummy tqdm if not installed
        return iterable

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs,
                wandb_run, checkpoint_dir: Path, evaluate_fn, model_type_flag: str,
                hp_dict: dict, best_val_loss_init=float('inf')):
    """
    Main training and validation loop.

    Args:
        model (torch.nn.Module): The PyTorch model to train.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (torch.device): The device to train on.
        num_epochs (int): Total number of epochs to train for.
        wandb_run (wandb.sdk.wandb_run.Run): Active W&B run object for logging.
        checkpoint_dir (Path): Directory to save model checkpoints.
        evaluate_fn (callable): The function to call for evaluation 
                                (e.g., evaluate_model from src.evaluation_utils).
        model_type_flag (str): Flag for evaluate_fn (e.g., "baseline" or "hybrid").
        hp_dict (dict): Dictionary of hyperparameters, must include 'patience' for early stopping
                        and 'save_checkpoint_every_n_epochs'.
        best_val_loss_init (float): Initial best validation loss.

    Returns:
        tuple: (path_to_best_model_checkpoint, final_best_val_loss)
    """
    best_val_loss = best_val_loss_init
    epochs_no_improve = 0
    patience = hp_dict.get('patience', 10) # Default patience if not in hp_dict
    save_every_n = hp_dict.get('save_checkpoint_every_n_epochs', 0) # Default 0 (off)
    
    # Ensure checkpoint_dir exists
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path_to_best_model_checkpoint = None

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        epoch_start_time = time.time()

        # --- Training Phase ---
        model.train()
        train_loss_epoch = 0.0
        train_targets_all_epoch = []
        train_preds_all_epoch = []

        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1} Train", leave=False)
        for i, batch in train_pbar:
            try:
                # Batch unpacking needs to be flexible based on model_type_flag
                # This part assumes the model's forward pass handles the specific inputs correctly.
                # The evaluate_fn also needs this model_type_flag.
                # The batch structure itself comes from pad_collate_fn.
                
                optimizer.zero_grad()
                
                if model_type_flag.lower() == "hybrid":
                    sequences_tabular_padded, sequences_mri_padded, lengths, targets, _ = batch
                    sequences_tabular_padded = sequences_tabular_padded.to(device)
                    sequences_mri_padded = sequences_mri_padded.to(device)
                    targets = targets.to(device)
                    predictions = model(sequences_tabular_padded, sequences_mri_padded, lengths.cpu()) # lengths for pack_padded
                elif model_type_flag.lower() == "baseline":
                    sequences_tabular_padded, lengths, targets, _ = batch
                    sequences_tabular_padded = sequences_tabular_padded.to(device)
                    targets = targets.to(device)
                    predictions = model(sequences_tabular_padded, lengths.cpu()) # lengths for pack_padded
                else:
                    raise ValueError(f"Unknown model_type_flag: {model_type_flag}")

                loss = criterion(predictions, targets)
                loss.backward()
                optimizer.step()

                train_loss_epoch += loss.item() * sequences_tabular_padded.size(0) # Weighted by batch size
                train_targets_all_epoch.extend(targets.detach().cpu().numpy().flatten())
                train_preds_all_epoch.extend(predictions.detach().cpu().numpy().flatten())
                train_pbar.set_postfix({'batch_loss': loss.item()})

                if wandb_run and (i % 10 == 0): # Log batch loss less frequently
                     wandb_run.log({'train/batch_loss': loss.item(), 
                                    'train/epoch_progress': epoch + (i+1)/len(train_loader)})
            except Exception as e_train_batch:
                print(f"\nError during training batch {i} in epoch {epoch+1}: {e_train_batch}")
                import traceback
                traceback.print_exc()
                continue # Skip problematic batch

        avg_train_loss = train_loss_epoch / len(train_loader.dataset) if len(train_loader.dataset) > 0 else 0
        train_mae = mean_absolute_error(train_targets_all_epoch, train_preds_all_epoch) if train_targets_all_epoch else 0
        train_r2 = r2_score(train_targets_all_epoch, train_preds_all_epoch) if train_targets_all_epoch else 0

        # --- Validation Phase ---
        val_metrics = evaluate_fn(model, val_loader, criterion, device, model_name_for_batch_unpack=model_type_flag)
        avg_val_loss = val_metrics['loss']
        val_mae = val_metrics['mae']
        val_r2 = val_metrics['r2']
        
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{num_epochs} finished in {epoch_duration:.2f}s.")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train MAE: {train_mae:.4f} | Train R2: {train_r2:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f} | Val MAE:   {val_mae:.4f} | Val R2:   {val_r2:.4f}")

        if wandb_run:
            wandb_run.log({
                'epoch': epoch + 1,
                'train/epoch_loss': avg_train_loss, 'train/epoch_mae': train_mae, 'train/epoch_r2': train_r2,
                'val/epoch_loss': avg_val_loss, 'val/epoch_mae': val_mae, 'val/epoch_r2': val_r2,
                'epoch_duration_sec': epoch_duration
            })

        # --- Model Checkpointing (Best Model) ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            path_to_best_model_checkpoint = checkpoint_dir / f"best_model_epoch_{epoch+1}.pth"
            try:
                torch.save(model.state_dict(), path_to_best_model_checkpoint)
                print(f"  Validation loss improved to {best_val_loss:.4f}. Saving model to {path_to_best_model_checkpoint}")
                if wandb_run:
                    # Log best model artifact
                    artifact_name = f'{wandb_run.job_type}-checkpoint' # Use job_type for unique name
                    best_model_artifact = wandb.Artifact(
                        artifact_name, type='model',
                        description=f"Best model based on val_loss from job {wandb_run.job_type}",
                        metadata={'epoch': epoch+1, 'val_loss': best_val_loss, 'val_mae': val_mae, 'val_r2': val_r2}
                    )
                    best_model_artifact.add_file(str(path_to_best_model_checkpoint))
                    wandb_run.log_artifact(best_model_artifact, aliases=['best', f'epoch_{epoch+1}'])
                    print("  Logged best model checkpoint artifact to W&B.")
            except Exception as e_save_best:
                print(f"  Error saving best model checkpoint: {e_save_best}")
        else:
            epochs_no_improve += 1
            print(f"  Validation loss did not improve from {best_val_loss:.4f} ({epochs_no_improve}/{patience}).")

        # --- Periodic Checkpointing ---
        if save_every_n > 0 and (epoch + 1) % save_every_n == 0:
            periodic_checkpoint_path = checkpoint_dir / f"model_epoch_{epoch+1}_periodic.pth"
            try:
                torch.save({
                    'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 'best_val_loss': best_val_loss,
                    'avg_train_loss': avg_train_loss, 'avg_val_loss': avg_val_loss,
                }, periodic_checkpoint_path)
                print(f"  Saved periodic checkpoint to {periodic_checkpoint_path}")
                if wandb_run:
                    periodic_artifact_name = f'{wandb_run.job_type}-periodic-checkpoint'
                    periodic_model_artifact = wandb.Artifact(
                        periodic_artifact_name, type='model-checkpoint',
                        description=f"Periodic model checkpoint at Epoch {epoch+1}",
                        metadata={'epoch': epoch+1, 'val_loss': avg_val_loss}
                    )
                    periodic_model_artifact.add_file(str(periodic_checkpoint_path))
                    wandb_run.log_artifact(periodic_model_artifact, aliases=[f'epoch_{epoch+1}']) # Keep versioning simple
                    print(f"  Logged periodic checkpoint epoch {epoch+1} to W&B.")
            except Exception as e_save_periodic:
                print(f"  Error saving periodic checkpoint: {e_save_periodic}")
        
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs with no improvement on validation loss.")
            break
            
    print(f"\nTraining finished. Best validation loss achieved: {best_val_loss:.4f}")
    return path_to_best_model_checkpoint, best_val_loss