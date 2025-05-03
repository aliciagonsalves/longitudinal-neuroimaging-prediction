# Longitudinal Prediction of Cognitive Decline using Robust Multimodal Neuroimaging and Deep Learning

## Overview

This project focuses on developing and evaluating deep learning models for predicting longitudinal cognitive changes in older adults. By using the OASIS longitudinal neuroimaging datasets, the primary goal is to forecast future cognitive status (e.g., based on CDR scores) over time.

A key aspect of this research is building **robust and modular model architectures** capable of handling the complexities of real-world clinical data, particularly **missing input modalities**. The project employs sequence models (like LSTMs/GRUs) trained on longitudinal data, incorporating both clinical information and features derived from structural MRI. Uncertainty quantification techniques (MC Dropout) are used to assess prediction confidence. Future work aims to integrate multimodal data (PET, genetics) from OASIS-3 and explore alternative prediction targets like disease stage conversion.


## Project Status

* **Current Stage:** Work in Progress - Initial data exploration, baseline model development
* **Data:** Currently using OASIS-2.

## Usage

The primary analysis is conducted through Jupyter notebooks located in the `notebooks/` directory. It is recommended to run them in numerical order, starting with `01_Data_Exploration_OASIS2.ipynb`.

Running the notebooks requires a Python environment with standard data science libraries installed (e.g., pandas, numpy, matplotlib, seaborn, openpyxl, wandb).

Generated outputs (plots, intermediate CSVs) will be saved locally under `notebooks/outputs/` (this directory should be configured to be ignored by Git via `.gitignore`). Experiment results, configurations, and key artifacts are tracked using Weights & Biases. Ensure you configure the paths to your local data in the `config.json` file as described in the notebook comments or related documentation.

## Data Source: OASIS

This project utilizes data obtained from the Open Access Series of Imaging Studies (OASIS) project.

* **Data Used:** OASIS-2 (Longitudinal MRI Data in Nondemented and Demented Older Adults). Future phases plan to incorporate OASIS-3 data.
* **Data Use Agreement (DUA):** Access to OASIS datasets requires user registration and agreement to the terms outlined in the official OASIS Data Use Agreement. Data is **not** included in this repository and must be obtained directly from OASIS: [https://www.oasis-brains.org/](https://www.oasis-brains.org/)
* **Citation:** Users of OASIS data are expected to cite the relevant publications. The primary reference for the OASIS-2 longitudinal dataset is:
    > Marcus, DS, Fotenos, AF, Csernansky, JG, Morris, JC, Buckner, RL. (2010). Open Access Series of Imaging Studies (OASIS): Longitudinal MRI Data in Nondemented and Demented Older Adults. *Journal of Cognitive Neuroscience*, 22(12), 2677-2684. doi: 10.1162/jocn.2009.21407

## Contact

* Alicia Gonsalves - aliciag7648@gmail.com

---