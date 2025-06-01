# src/plotting_utils.py
import matplotlib.pyplot as plt
import wandb
from pathlib import Path

def finalize_plot(fig_to_finalize: plt.Figure, 
                  plt_module: plt, 
                  wandb_run_obj, 
                  wandb_log_key: str, 
                  local_save_path: Path):
    """
    Handles saving a Matplotlib figure locally, logging it to Weights & Biases 
    (if a run is active), showing the plot, and then closing the figure resource.

    Args:
        fig_to_finalize (plt.Figure): The Matplotlib figure object to process.
        plt_module (module): The imported matplotlib.pyplot module (passed as 'plt').
        wandb_run_obj (wandb.sdk.wandb_run.Run or None): The active W&B run object. 
                                                       If None, W&B logging is skipped.
        wandb_log_key (str): The key under which to log the plot in W&B.
        local_save_path (Path): The local file path (including filename and extension, e.g., .png)
                                to save the plot. Parent directory is created if it doesn't exist.
    """
    if not isinstance(fig_to_finalize, plt.Figure):
        print(f"Warning (finalize_plot): 'fig_to_finalize' for '{wandb_log_key}' is not a valid Matplotlib Figure. Skipping.")
        return

    try:
        # Ensure directory for saving plot exists
        local_save_path.parent.mkdir(parents=True, exist_ok=True)
        # Apply tight_layout before saving for better appearance
        fig_to_finalize.tight_layout()
        fig_to_finalize.savefig(local_save_path, bbox_inches='tight')
        print(f"Plot saved locally: {local_save_path.name}")

        # Log the saved file to W&B for better reliability than logging 'plt' object
        if wandb_run_obj:
            try:
                wandb_run_obj.log({wandb_log_key: wandb.Image(str(local_save_path))})
                print(f"Plot '{wandb_log_key}' logged to W&B from file.")
            except Exception as e_wandb_log_file:
                print(f"Warning: Could not log plot file '{local_save_path}' to W&B. Error: {e_wandb_log_file}")
    
    except Exception as e_save:
        print(f"Warning: Could not save plot to '{local_save_path}'. Error: {e_save}")
            
    plt_module.show() # Show the plot inline
    plt_module.close(fig_to_finalize) # Close the figure to free memory
