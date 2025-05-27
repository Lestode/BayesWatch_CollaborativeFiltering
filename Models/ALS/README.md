# ALS – Alternating Least Squares for Collaborative Filtering

This repository contains an implementation of the ALS algorithm for matrix factorization in the context of collaborative filtering. It was developed as part of the CIL 2025 course project at ETH Zurich.

## Overview

The ALS algorithm decomposes the user-item rating matrix into two lower-dimensional latent factor matrices, iteratively optimizing one while fixing the other. It is particularly well-suited for large, sparse datasets like the one used in this project (scientist-paper ratings).

## Getting Started

To train and evaluate the ALS model:

1. Open the `ALS.ipynb` notebook.
2. Ensure all dependencies are installed (see requirements below).
3. Run all cells sequentially. No grid search is required, as the notebook uses hyperparameters that yielded good performance during development.
4. Final model predictions are saved in a format compatible with the provided `sample_submission.csv`.

### Input Files

- Change the path of the input files
- `train_ratings.csv`: Contains the observed ratings.
- `train_tbr.csv`: Contains wishlists (to be read) used optionally for enhancement.
- `sample_submission.csv`: Format for submission.

## Model Details

- **Algorithm**: Alternating Least Squares with regularization
- **Loss Function**: Frobenius norm regularized squared error
- **Hyperparameters**:
  - Number of latent factors
  - Regularization strength
  - Number of iterations
- **Implementation Details**:
  - Sparse matrix operations using NumPy and SciPy
  - Ratings are imputed with zeros for initialization
  - Uses stochastic updates for better scalability

## Reproducibility

- The notebook is self-contained and produces final submission files.
- All random seeds are fixed to ensure reproducible results.

## Requirements

- Python 3.8+
- numpy
- pandas
- scipy
- matplotlib

To install the required packages:

```bash
pip install numpy pandas scipy matplotlib
```

## Usage

Run the notebook using Jupyter or JupyterHub (e.g. ETH student cluster):

```bash
jupyter notebook ALS.ipynb
```

## Notes

- This model is an implementation baseline for the collaborative filtering Kaggle competition hosted as part of the ETH CIL course.
- Extensions could include implicit feedback modeling, wishlist incorporation, or hybrid ensemble methods.

## Authors

This implementation was developed by BayesWatch.

## License

For academic use only, part of ETH Zurich’s CIL course.
