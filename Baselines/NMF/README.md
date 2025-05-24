# NMF – Non-negative Matrix Factorization for Collaborative Filtering

This repository provides an implementation of Non-negative Matrix Factorization (NMF) for the task of collaborative filtering, developed for the ETH Zurich CIL 2025 course project. Two variants are included: one using only explicit ratings and the other incorporating both explicit and implicit data.

## Overview

NMF factorizes the user-item matrix into non-negative user and item latent factor matrices. It is particularly effective when interpretability of factors is desired, and performs well in sparse data environments.

## Implementations

### 1. Explicit Ratings Only

- Change the input path files
- Uses `train_ratings.csv` for factorization.
- Optimizes reconstruction error on observed ratings.
- Parameters are fine-tuned using grid search and cross-validation.

### 2. Explicit + Implicit Ratings

- Combines `train_ratings.csv` with `train_tbr.csv` (wishlist entries).
- Implicit data is used to enhance learning with confidence weighting.
- Applies a heuristic to incorporate wishlist interactions as soft signals of preference.
- Also fine-tuned with a grid search, best parameters explicitly chosen.

## Getting Started

1. Open the `NMF.ipynb` notebook.
2. Install dependencies listed below.
3. Execute all cells to preprocess, train, and generate predictions.
4. Final predictions are saved in the format required for submission (`sample_submission.csv`).

### Input Files

- `train_ratings.csv`: Explicit user-item ratings (1–5 scale).
- `train_tbr.csv`: Implicit wishlist signals.
- `sample_submission.csv`: Template for final predictions.

## Model Details

- **Algorithm**: Non-negative Matrix Factorization
- **Loss Function**: Squared reconstruction error (with optional confidence weighting)
- **Regularization**: L2 on latent factors
- **Hyperparameters**:
  - Number of latent factors
  - Learning rate (if applicable)
  - Regularization strength
  - Number of iterations

## Reproducibility

- Best hyperparameters are hard-coded based on previous experiments.
- Random seeds fixed for consistent results.
- All preprocessing and training steps are included in the notebook.

## Requirements

- Python 3.8+
- numpy
- pandas
- scipy
- scikit-learn
- matplotlib

To install the required packages:

```bash
pip install numpy pandas scipy scikit-learn matplotlib
```

## Usage

To run the notebook:

```bash
jupyter notebook NMF.ipynb
```

On the ETH cluster:

- Environment path: `/cluster/courses/cil/envs/collaborative_filtering/bin`
- Required module: `cuda/12.6`

## Notes

- This implementation serves as a baseline and a stepping stone for hybrid approaches.
- Future extensions could involve adaptive confidence modeling or integration with neural network architectures.

## Authors

Developed by [Your Group Name/Number].

## License

For academic and non-commercial use only, under ETH Zurich’s CIL course guidelines.
