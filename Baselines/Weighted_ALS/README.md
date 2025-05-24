# Weighted ALS – Weighted Alternating Least Squares for Collaborative Filtering

This repository contains an implementation of the Weighted ALS algorithm for matrix factorization, tailored for collaborative filtering tasks where confidence in observations varies. This was developed as part of the CIL 2025 course project at ETH Zurich.

## Overview

Weighted ALS enhances traditional ALS by incorporating a confidence matrix, allowing more reliable modeling of implicit feedback data. This makes it particularly useful in scenarios where user engagement (e.g., clicks or wishlist entries) is more informative than explicit ratings.

## Getting Started

To train and evaluate the Weighted ALS model:

1. Open the `Weighted_ALS.ipynb` notebook.
2. Ensure all dependencies are installed (see requirements below).
3. Run all cells in order to preprocess data, train the model, and generate predictions.
4. Output is formatted for submission as per `sample_submission.csv`.

### Input Files

- Change the paths of the input files
- `train_ratings.csv`: User-item interactions used to construct the rating matrix.
- `train_tbr.csv`: Wishlist data used to define confidence levels.
- `sample_submission.csv`: Submission template.

## Model Details

- **Algorithm**: Weighted Alternating Least Squares with regularization
- **Confidence Matrix**: Higher confidence for interactions with stronger signals (e.g., wishlist entries)
- **Hyperparameters**:
  - Latent factor count
  - Confidence scaling factor
  - Regularization strength
  - Number of iterations
- **Implementation Notes**:
  - Confidence-weighted updates
  - Efficient sparse matrix handling
  - Initialization with small random values or heuristics

## Reproducibility

- Fixed random seeds for consistent results
- Notebook includes all necessary preprocessing and training steps

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

To run the model:

```bash
jupyter notebook Weighted_ALS.ipynb
```

## Notes

- Designed for use in the ETH CIL collaborative filtering competition.
- Extensions might include ensemble models or adaptive weighting schemes.

## Authors

Developed by BayesWatch.

## License

For educational use within the ETH Zurich CIL course.
