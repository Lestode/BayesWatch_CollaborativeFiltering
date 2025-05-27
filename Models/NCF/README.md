# Neural Collaborative Filtering

This directory contains the relevant code for the Neural Collaborative Filtering models MLP and NeuMF and their reproduction.

## Overview

The trained models and used hyperparameters are contained in the `models` directory. All model files are also saved in the `saved_models` directory in the root of this repository. 

## Requirements

- Python 3.8 or newer
- torch
- numpy
- scikit-learn
- pandas

## Creating a submission

To create a submission .csv file for Kaggle from a saved model, follow the `submission_example.py` file. Before running, 

1. Ensure all dependencies are installed
2. Fill out the model type, model dimension and the relevant directories at the top of the file
3. The model predictions are saved as a .csv file ready for submission.

## Source Code

The source code for the models, model training and hyperparameter tuning can be found in the `src` directory. Before running, fill out the necessary parameters at the top of the file.

## Remarks

The model dimension is the embedding dimension for the GMF part, and the size of the last layer for the MLP part. They represent the model capacity.

The GMF model is not a standalone neural collaborative filtering model and was not submitted as such; it is rather a simple extension of matrix factorization.
However, it is part of the NeuMF model and pretrained before training NeuMF (see paper).