# Longitudinal Prediction of Cognitive Decline using Robust Multimodal Neuroimaging and Deep Learning

## Overview

This project develops and rigorously evaluates deep learning models for predicting longitudinal cognitive changes, primarily focusing on Clinical Dementia Rating (CDR) scores using the OASIS-2 dataset. The core aim is to build a **robust, modular, and reproducible pipeline** for handling complex, real-world longitudinal clinical and neuroimaging data.

Key technical features include:
* **Modular Deep Learning Architectures:** Implementation of a baseline LSTM model (for tabular data) and a hybrid CNN+LSTM model (for combined tabular and 3D T1w MRI data) featuring late fusion.
* **MLOps Foundation:** Emphasis on reproducible research practices using:
    * **Git** for version control.
    * **Weights & Biases (W&B)** for comprehensive experiment tracking, configuration management, and versioning of datasets, preprocessors, and models as artifacts.
    * A structured **`src/` directory** with reusable Python utilities for data processing, path management, W&B interaction, model training, evaluation, uncertainty quantification, and interpretability.
    * A **sequence of Jupyter Notebooks** (01-08 for OASIS-2) for each stage of the pipeline from data exploration to model analysis.
    * **Centralized Configuration** via `config.json` to manage paths and parameters.
* **Uncertainty Quantification:** Implementation of MC Dropout to estimate prediction uncertainty.
* **Model Interpretability:** Application of techniques like Permutation Feature Importance, Integrated Gradients (for 3D CNNs), and SHAP (for LSTMs and model fusion stages) to understand model behavior.
* **Robustness Considerations:** Initial implementation of modality dropout in the hybrid model, with plans for further exploration.

## Project Status (OASIS-2 Phase)

* **Current Stage:** Initial Development & Analysis Phase for OASIS-2 Complete.
    * Data Exploration, Cohort Definition, Feature Engineering, and Preprocessor Fitting pipelines established (Notebooks 01-04).
    * DataLoader testing and verification implemented (Notebook 05).
    * Baseline LSTM and Hybrid CNN+LSTM models trained and evaluated locally (Notebooks 06 & 07), with a focus on pipeline functionality and MLOps integration.
    * Comprehensive model analysis including uncertainty quantification (MC Dropout) and interpretability (Permutation Importance, Integrated Gradients, SHAP) performed (Notebook 08).
* **Data:** Currently using the OASIS-2 longitudinal dataset.
* **Next Immediate Steps (Local):**
    Finalize and run Notebook 09 (`09_Comparison_Report_OASIS2.ipynb`) to consolidate and compare results from all OASIS-2 analyses.
* **Future Dataset:** The ADNI dataset has been acquired and is targeted for the next major phase of development to integrate more multimodal data (PET, genetics) and test model generalization and robustness to real-world missing modalities.

## Core Pipeline (OASIS-2)

The project is structured as a sequence of Jupyter notebooks and utility scripts:

1.  **Configuration (`config.json`):** Central file for paths, W&B settings, and key parameters.
2.  **Source Utilities (`src/`):**
    * `paths_utils.py`: Manages dynamic path resolution.
    * `wandb_utils.py`: Standardizes W&B run initialization and artifact interaction.
    * `datasets.py`: Defines `OASISDataset` and `pad_collate_fn` for data loading, preprocessing application, and batching, supporting conditional MRI inclusion.
    * `models.py`: Contains `BaselineLSTMRegressor`, `Simple3DCNN`, and `ModularLateFusionLSTM`.
    * `training_utils.py`: Provides a reusable `train_model` function with W&B logging, checkpointing, and early stopping.
    * `evaluation_utils.py`: Provides a reusable `evaluate_model` function for consistent metric calculation.
    * `uncertainty_utils.py`: Utilities for MC Dropout.
    * `interpretability_utils.py`: Utilities for PFI, Integrated Gradients, and SHAP.
3.  **MRI Preprocessing Scripts (`scripts/`):**
    * `preprocess_scan.py`: Performs NIfTI conversion, reorientation, N4 bias correction, skull-stripping, and registration for a single MRI scan.
    * `run_preprocessing.py`: Orchestrates `preprocess_scan.py` in parallel for a cohort, including TemplateFlow integration and resume logic.
4.  **Jupyter Notebooks (`notebooks/` - for OASIS-2):**
    * `01_Data_Exploration_OASIS2.ipynb`: Loads raw clinical data, EDA, verifies raw MRI file presence.
    * `02_Cohort_Definition_OASIS2.ipynb`: Defines the analysis cohort based on clinical criteria and MRI availability.
    * `03_Feature_Engineering_Splitting_OASIS2.ipynb`: Engineers time-based features, selects final features, creates the target variable (`CDR_next_visit`), and performs subject-level stratified train/val/test splits. Outputs are logged as W&B artifacts.
    * `04_Fit_Preprocessors_OASIS2.ipynb`: Loads the training split, fits imputers and scalers *only* on this data. Saves fitted preprocessors locally and logs them (and the definitive feature/preprocessing configuration) as W&B artifacts. This run's config becomes the source of truth for `OASISDataset`.
    * `05_Test_DataLoader_OASIS2.ipynb`: Verifies `OASISDataset` and `pad_collate_fn` by consuming preprocessor artifacts and configurations from NB04.
    * `06_Train_Baseline_Model_OASIS2.ipynb`: Trains the `BaselineLSTMRegressor`. Consumes data splits (NB03) and preprocessor setup (NB04) via W&B artifacts/lineage. Logs metrics and model artifacts.
    * `07_Train_Hybrid_Model_OASIS2.ipynb`: Trains the `ModularLateFusionLSTM`. Similar MLOps flow as NB06.
    * `08_Model_Analysis_Interpretability_Uncertainty_OASIS2.ipynb`: Loads trained models (from NB06/07 artifacts) and performs MC Dropout, PFI, Integrated Gradients, and SHAP analyses. Logs all results.
    * `09_Comparison_Report_OASIS2.ipynb` (Planned Next): Fetches results from all W&B runs to generate a comparative report.

## Usage

1.  **Environment Setup:**
    * Clone the repository.
    * Create a Python environment (e.g., using Conda) and install dependencies from `requirements.txt`.
    * Install necessary neuroimaging tools if running MRI preprocessing (FSL, ANTs/ANTsPyNet as per `scripts/preprocess_scan.py`).
2.  **Data Acquisition:**
    * Download the OASIS-2 dataset directly from [www.oasis-brains.org](https://www.oasis-brains.org/) (requires registration and DUA). This repository does **not** include the raw data.
    * Place the downloaded data in the appropriate directories.
3.  **Configuration:**
    * Copy `config.template.json` to `config.json` (if a template is provided) or directly edit `config.json`.
    * Update paths to your local data, W&B entity/project, and any other necessary parameters. Ensure `pipeline_artefact_locators_oasis2` keys match your desired output folder naming conventions for each notebook.
4.  **W&B Login:** Run `wandb login` in your terminal.
5.  **Run Notebooks:** Execute the Jupyter notebooks in the `notebooks/` directory, preferably in numerical order (01-08 for the main OASIS-2 pipeline). Pay attention to any specific instructions or W&B run IDs that need to be passed between notebooks (though this has been minimized by using W&B artifacts and API calls).

## Data Source: OASIS

This project utilizes data obtained from the Open Access Series of Imaging Studies (OASIS) project.

* **Data Used:** OASIS-2 (Longitudinal MRI Data in Nondemented and Demented Older Adults). Future phases plan to incorporate OASIS-3 data.
* **Data Use Agreement (DUA):** Access to OASIS datasets requires user registration and agreement to the terms outlined in the official OASIS Data Use Agreement. Data is **not** included in this repository and must be obtained directly from OASIS: [https://www.oasis-brains.org/](https://www.oasis-brains.org/)
* **Citation:** Users of OASIS data are expected to cite the relevant publications. The primary reference for the OASIS-2 longitudinal dataset is:
    > Marcus, DS, Fotenos, AF, Csernansky, JG, Morris, JC, Buckner, RL. (2010). Open Access Series of Imaging Studies (OASIS): Longitudinal MRI Data in Nondemented and Demented Older Adults. *Journal of Cognitive Neuroscience*, 22(12), 2677-2684. doi: 10.1162/jocn.2009.21407

## Contact

* Alicia Gonsalves - aliciag7648@gmail.com

---