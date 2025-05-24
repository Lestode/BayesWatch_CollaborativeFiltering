# SVD++ – Singular Value Decomposition Plus Plus for Collaborative Filtering

This repository includes two implementations of the SVD++ algorithm, developed for the ETH Zurich CIL 2025 course project. SVD++ extends classical SVD by incorporating both explicit and implicit feedback, enabling more accurate recommendation systems.

## Overview

SVD++ is a matrix factorization technique that augments traditional collaborative filtering by accounting for users' implicit preferences (e.g., wishlist interactions). This model is suitable for sparse rating datasets where user behavior can be inferred from both ratings and engagement data.

## Implementations

### 1. Explicit Ratings Only

- Trains the model using only the `train_ratings.csv` file.
- Optimizes prediction error on observed ratings.
- Fine-tuned using grid search with best parameters set explicitly in the notebook.

### 2. Explicit + Implicit Ratings

- Combines `train_ratings.csv` with `train_tbr.csv` for enhanced user modeling.
- Wishlist entries are interpreted as implicit signals of interest.
- Incorporates an additional user influence vector to capture implicit behavior.
- Also uses fine-tuned hyperparameters for optimal performance.

## Getting Started

1. Launch `SVD_PlusPlus.ipynb` in a Jupyter environment.
2. Ensure the necessary libraries are installed.
3. Run all cells to preprocess, train, and evaluate the model.
4. Output predictions are saved in the format required by `sample_submission.csv`.

### Input Files

- `train_ratings.csv`: User-item rating data.
- `train_tbr.csv`: Implicit feedback from wishlists.
- `sample_submission.csv`: Template for submission predictions.

## Model Details

- **Algorithm**: SVD++ with optional implicit signal integration
- **Objective**: Minimize regularized mean squared error
- **Components**:
  - Latent factor matrices for users and items
  - Implicit feedback vector per user (for the second implementation)
- **Hyperparameters**:
  - Number of latent factors
  - Learning rate
  - Regularization strength
  - Number of iterations

## Reproducibility

- All data processing and training steps are included.
- Random seeds are fixed for reproducibility.
- Best hyperparameters are hard-coded following a fine-tuning procedure.

## Requirements

- Python 3.8+
- numpy
- pandas
- scipy
- matplotlib

Install dependencies via:

```bash
pip install numpy pandas scipy matplotlib
```

## Usage

To run the model locally:

```bash
jupyter notebook SVD_PlusPlus.ipynb
```

## Notes

- SVD++ is a strong baseline and can be further improved via ensemble methods or hybrid modeling approaches.
- Ideal for applications where both explicit and implicit data are available.

## Authors

Developed by BayesWatch.

## License

Intended for educational and academic use under the ETH Zurich CIL project guidelines.
