import pandas as pd
import numpy as np
import os

import config
from base_model_wrappers import FunkSVDWrapper, LightGCNWrapper, NeuMFWrapper, WeightedALSWrapper, NMFWrapper

def main():
    print("Starting simple weighted averaging prediction for 5 models...")

    # 1. Load Submission Data
    print(f"Loading sample submission data from: {config.SAMPLE_SUBMISSION_PATH}")
    try:
        submission_df_orig = pd.read_csv(config.SAMPLE_SUBMISSION_PATH)
        submission_df_features = submission_df_orig.copy()
        submission_df_features[['sid', 'pid']] = submission_df_features['sid_pid'].str.split('_', expand=True).astype(int)
        X_submission = submission_df_features[['sid', 'pid']]
    except Exception as e:
        print(f"Error loading or processing sample submission data: {e}")
        return
    
    print(f"Sample submission data loaded.")

    # 2. Initialize Base Estimators and Get Predictions
    print("Initializing base models and getting predictions...")
    predictions_dict = {}
    model_names_in_order = ['FunkSVD', 'LightGCN', 'NeuMF', 'WeightedALS', 'NMF'] # must be in correct order!

    try:
        print("Initializing FunkSVD...")
        funk_svd_wrapper = FunkSVDWrapper(
            model_path=config.FUNKSVD_EXPLICIT_MODEL_PATH
        )
        funk_svd_wrapper.fit(X_submission)
        preds_funk_svd = funk_svd_wrapper.predict(X_submission)
        predictions_dict['FunkSVD'] = preds_funk_svd
        print(f"FunkSVD predictions obtained.")

        print("Initializing Seeded LightGCN...")
        light_gcn_wrapper = LightGCNWrapper(
            model_path=config.LIGHTGCN_MODEL_PATH,
            svd_seed_path=config.SVD_SEED_FOR_LIGHTGCN_PATH,
            data_dir_for_graph=config.DATA_DIR,
            model_config=config.LIGHTGCN_CONFIG
        )
        light_gcn_wrapper.fit(X_submission)
        preds_light_gcn = light_gcn_wrapper.predict(X_submission)
        predictions_dict['LightGCN'] = preds_light_gcn
        print(f"Seeded LightGCN predictions obtained.")

        print("Initializing NeuMF...")
        neumf_wrapper = NeuMFWrapper(
            model_path=config.NEUMF_WISHLIST_MODEL_PATH,
            user_map_path=config.NCF_USER_MAP_PATH,
            item_map_path=config.NCF_ITEM_MAP_PATH,
            model_config=config.NEUMF_CONFIG
        )
        neumf_wrapper.fit(X_submission)
        preds_neumf = neumf_wrapper.predict(X_submission)
        predictions_dict['NeuMF'] = preds_neumf
        print(f"NeuMF predictions obtained.")
        
        print("Initializing WeightedALS...")
        weighted_als_wrapper = WeightedALSWrapper(
            model_path=config.WEIGHTED_ALS_MODEL_PATH
        )
        weighted_als_wrapper.fit(X_submission) # Loads the model
        preds_wals = weighted_als_wrapper.predict(X_submission)
        predictions_dict['WeightedALS'] = preds_wals
        print(f"WeightedALS predictions obtained.")

        print("Initializing NMF...")
        nmf_wrapper = NMFWrapper(
            model_path=config.NMF_MODEL_PATH
        )
        nmf_wrapper.fit(X_submission)
        preds_nmf_model = nmf_wrapper.predict(X_submission)
        predictions_dict['NMF'] = preds_nmf_model
        print(f"NMF predictions obtained.")

    except Exception as e:
        print(f"Error initializing a base model wrapper or getting predictions: {e}")
        return
    
    print("Predictions heads:")
    print("LightGCN:")
    print(predictions_dict['LightGCN'][:5])
    print("FunkSVD:")
    print(predictions_dict['FunkSVD'][:5])
    print("NeuMF:")
    print(predictions_dict['NeuMF'][:5])
    print("WeightedALS:")
    print(predictions_dict['WeightedALS'][:5])
    print("NMF:")
    print(predictions_dict['NMF'][:5])

    # 3. Compute Weighted Average of Predictions
    current_model_names = [name for name in model_names_in_order if name in predictions_dict]

    weights_sum = sum(config.AVERAGE_ENSEMBLE_WEIGHTS.get(name, 0) for name in current_model_names)
    if not np.isclose(weights_sum, 1.0):
        print(f"Warning: Configured weights do not sum to 1.0 (sum: {weights_sum}). Results might be unexpected.")

    print(f"Calculating weighted average using weights: {config.AVERAGE_ENSEMBLE_WEIGHTS}")

    num_samples = len(X_submission)
    weighted_avg_predictions = np.zeros(num_samples)
    
    for model_name in current_model_names:
        if model_name in predictions_dict and model_name in config.AVERAGE_ENSEMBLE_WEIGHTS:
            weight = config.AVERAGE_ENSEMBLE_WEIGHTS[model_name]
            if weight > 0:
                 weighted_avg_predictions += predictions_dict[model_name] * weight

    print(f"Weighted average predictions calculated.")

    weighted_avg_predictions_clipped = np.clip(weighted_avg_predictions, 1, 5)

    # 4. Create Submission File
    weighted_avg_submission_path = os.path.join(config.SUBMISSIONS_DIR, 'weighted_average_submission.csv')
    print(f"Creating submission file at: {weighted_avg_submission_path}")
    submission_output_df = pd.DataFrame({
        'sid_pid': submission_df_orig['sid_pid'],
        'rating': weighted_avg_predictions_clipped
    })

    try:
        os.makedirs(config.SUBMISSIONS_DIR, exist_ok=True)
        submission_output_df.to_csv(weighted_avg_submission_path, index=False)
        print(f"Submission file created successfully at {weighted_avg_submission_path}")
    except Exception as e:
        print(f"Error saving submission file: {e}")

    print("Simple weighted averaging prediction script finished.")

if __name__ == '__main__':
    main() 