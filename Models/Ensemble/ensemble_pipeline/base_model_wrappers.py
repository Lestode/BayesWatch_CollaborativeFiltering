import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from surprise import dump
import torch
import torch.nn as nn
from torch_geometric.data import Data

import sys
import os

import config 

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_ensemble = os.path.dirname(current_script_dir)
models_root = os.path.dirname(project_root_ensemble)
seeded_lightgcn_path = os.path.join(models_root, 'Seeded LightGCN')
if seeded_lightgcn_path not in sys.path:
    sys.path.append(seeded_lightgcn_path)

ncf_src_path = os.path.join(models_root, 'NCF', 'src')
if ncf_src_path not in sys.path:
    sys.path.append(ncf_src_path)

try:
    from model import LightGCN as SeededLightGCNModel
    from model import load_svd_embeddings as lgcn_load_svd_embeddings
    from model import load_csvs as lgcn_load_csvs
    from model import build_graph as lgcn_build_graph
    print("Successfully imported Seeded LightGCN model.")
except Exception as e:
    print(f"Could not import Seeded LightGCN model: {e}")
    

try:
    from models import NeuMF as NCFNeuMFModel
    print("Successfully imported NeuMF model.")
except Exception as e:
    print(f"Could not import NeuMF model: {e}")

from scipy.sparse.linalg import svds as scipy_svds

class WeightedALSModel:
    def __init__(self,
                 rank=100,
                 num_iterations=10,
                 reg_parameter=0.1,
                 num_svd_runs=3,
                 svd_lr=0.1,
                 use_iSVD=False,
                 transpose=False,
                 bias_reg=0.01,
                 use_bias=True,
                 use_confidence=True,
                 alpha_r=40.0,
                 alpha_tbr=10.0):
        self.rank           = rank
        self.num_iters      = num_iterations
        self.lam            = reg_parameter
        self.num_svd_runs   = num_svd_runs
        self.lr             = svd_lr
        self.use_iSVD       = use_iSVD
        self.transpose      = transpose
        self.bias_reg       = bias_reg
        self.use_bias       = use_bias
        self.use_confidence = use_confidence
        self.alpha_r        = alpha_r
        self.alpha_tbr      = alpha_tbr
        self.rec_mtx     = None
        self.user_factors = None
        self.item_factors = None
        self.global_mean  = None
        self.user_bias    = None
        self.item_bias    = None

    def predict_for_submission(self, sids, pids):
        if self.rec_mtx is None:
            raise ValueError("Model not trained or loaded, rec_mtx is None.")
        preds = []
        for uu,ii in zip(sids, pids):
            if uu < self.rec_mtx.shape[0] and ii < self.rec_mtx.shape[1]:
                preds.append(self.rec_mtx[uu,ii])
            else:
                val_to_append = self.global_mean if self.global_mean is not None else 3.0
                if self.use_bias:
                    user_bias_val = 0.0
                    item_bias_val = 0.0
                    if self.user_bias is not None and uu < len(self.user_bias):
                        user_bias_val = self.user_bias[uu]
                    if self.item_bias is not None and ii < len(self.item_bias):
                        item_bias_val = self.item_bias[ii]
                    if self.user_bias is not None and uu < len(self.user_bias) and self.item_bias is not None and ii < len(self.item_bias):
                         val_to_append = self.global_mean + user_bias_val + item_bias_val
                    elif self.user_bias is not None and uu < len(self.user_bias):
                        val_to_append = self.global_mean + user_bias_val
                    elif self.item_bias is not None and ii < len(self.item_bias):
                        val_to_append = self.global_mean + item_bias_val
                preds.append(np.clip(val_to_append,1,5))
        return np.array(preds)

class BaseHFModelWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self._load_internal_model() 

    def _load_internal_model(self):
        """
        Loads the specific model from the given path.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _load_internal_model")

    def _internal_predict(self, sids: np.ndarray, pids: np.ndarray) -> np.ndarray:
        """
        Internal prediction logic for the specific model.
        Takes NumPy arrays of sids and pids.
        Returns a NumPy array of predictions.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _internal_predict")

    def fit(self, X, y=None):
        if self.model is None:
            print(f"Warning: Model for {self.__class__.__name__} was not loaded in __init__. Attempting to load now.")
            self._load_internal_model()
            if self.model is None:
                raise RuntimeError(f"Model {self.__class__.__name__} could not be loaded.")
        
        self.model_ = self.model 
        self.is_fitted_ = True
        
        return self

    def predict(self, X_df: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
             raise RuntimeError(f"This {self.__class__.__name__} instance is not fitted yet.")
        if self.model is None:
             raise RuntimeError(f"Model for {self.__class__.__name__} is not available for prediction.")

        if not isinstance(X_df, pd.DataFrame) or not all(col in X_df.columns for col in ['sid', 'pid']):
            raise ValueError("Input X_df must be a pandas DataFrame with 'sid' and 'pid' columns.")

        sids = X_df['sid'].values
        pids = X_df['pid'].values
        
        return self._internal_predict(sids, pids)

class FunkSVDWrapper(BaseHFModelWrapper):
    def __init__(self, model_path: str):
        super().__init__(model_path)

    def _load_internal_model(self):
        try:
            _predictions, algo = dump.load(self.model_path)
            self.model = algo
            print(f"FunkSVD model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"Error loading FunkSVD model from {self.model_path}: {e}")
            raise

    def _internal_predict(self, sids: np.ndarray, pids: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("FunkSVD model is not loaded.")

        predictions_output = []
        
        test_data = []
        for sid, pid in zip(sids, pids):
            test_data.append((sid, pid, 0.0)) 
        
        surprise_predictions = self.model.test(test_data)
        
        for pred in surprise_predictions:
            if pred.details.get('was_impossible', False):
                predictions_output.append(pred.est)
            else:
                predictions_output.append(pred.est)
                
        return np.array(predictions_output)

class LightGCNWrapper(BaseHFModelWrapper):
    def __init__(self, model_path: str, svd_seed_path: str, data_dir_for_graph: str, model_config: dict):
        self.svd_seed_path = svd_seed_path
        self.data_dir_for_graph = data_dir_for_graph
        self.model_config = model_config
        self.user_map = None
        self.item_map = None
        self.graph_data = None 
        self.num_total_nodes = 0
        self.h_final_embeddings = None
        self.device = config.DEVICE
        super().__init__(model_path)

    def _load_internal_model(self):
        try:
            pu, qi, self.user_map, self.item_map = lgcn_load_svd_embeddings(self.svd_seed_path)
            
            ratings_df, wishlist_df, _ = lgcn_load_csvs(self.data_dir_for_graph)

            self.graph_data = lgcn_build_graph(pu, qi, self.user_map, self.item_map, ratings_df, wishlist_df)
            
            if self.model_config.get('n_users', -1) == -1:
                self.model_config['n_users'] = self.graph_data.n_users
            if self.model_config.get('n_items', -1) == -1:
                self.model_config['n_items'] = self.graph_data.n_items

            self.model = SeededLightGCNModel(
                n_layers=self.model_config['num_layers'],
                dropout=self.model_config['dropout'],
                initial_embeds=self.graph_data.x,
                fine_tune_embed=self.model_config['fine_tune_embed']
            ).to(self.device)
            
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            print(f"LightGCN model loaded successfully from {self.model_path}")

            with torch.no_grad():
                self.h_final_embeddings = self.model.get_embeddings(self.graph_data.adj.to(self.device))
            self.num_total_nodes = self.graph_data.n_users + self.graph_data.n_items

        except Exception as e:
            print(f"Error loading LightGCN model or its components: {e}")
            raise

    def _internal_predict(self, sids: np.ndarray, pids: np.ndarray) -> np.ndarray:
        if self.model is None or self.h_final_embeddings is None:
            raise RuntimeError("LightGCN model/embeddings not loaded.")

        predictions = np.full(len(sids), 3.0, dtype=float)

        u_indices_model = []
        p_indices_model = []
        original_indices_for_preds = []

        for i, (sid, pid) in enumerate(zip(sids, pids)):
            if sid in self.user_map and pid in self.item_map:
                u_idx = self.user_map[sid]
                # Item indices in the concatenated embedding matrix are offset by n_users
                p_idx = self.item_map[pid] + self.graph_data.n_users 
                
                if u_idx < self.graph_data.n_users and self.graph_data.n_users <= p_idx < self.num_total_nodes:
                    u_indices_model.append(u_idx)
                    p_indices_model.append(p_idx)
                    original_indices_for_preds.append(i)
                else:
                    print(f"Warning: SID {sid} or PID {pid} resulted in out-of-bounds internal index. Defaulting prediction.")
            else:
                pass

        if not u_indices_model:
            return predictions

        u_tensor = torch.tensor(u_indices_model, dtype=torch.long).to(self.device)
        p_tensor = torch.tensor(p_indices_model, dtype=torch.long).to(self.device)

        model_preds_raw = []
        batch_size = 1024
        with torch.no_grad():
            for i in range(0, len(u_tensor), batch_size):
                batch_u = u_tensor[i:i+batch_size]
                batch_p = p_tensor[i:i+batch_size]
                
                user_embeds_batch = self.h_final_embeddings[batch_u]
                item_embeds_batch = self.h_final_embeddings[batch_p]
                
                scores_batch = (user_embeds_batch * item_embeds_batch).sum(dim=1)
                preds_batch = 1.0 + 4.0 * torch.sigmoid(scores_batch)
                model_preds_raw.extend(preds_batch.cpu().tolist())

        for i, model_pred_val in enumerate(model_preds_raw):
            original_idx = original_indices_for_preds[i]
            predictions[original_idx] = model_pred_val
            
        return predictions

class NeuMFWrapper(BaseHFModelWrapper):
    def __init__(self, model_path: str, user_map_path: str, item_map_path: str, model_config: dict):
        self.user_map_path = user_map_path
        self.item_map_path = item_map_path
        self.model_config = model_config
        self.user_to_idx = None
        self.item_to_idx = None
        self.device = config.DEVICE
        self.default_prediction_value = 3.0
        super().__init__(model_path)

    def _load_internal_model(self):
        import json
        try:
            with open(self.user_map_path, 'r') as f:
                self.user_to_idx = json.load(f)
            with open(self.item_map_path, 'r') as f:
                self.item_to_idx = json.load(f)
            
            n_users = len(self.user_to_idx)
            n_items = len(self.item_to_idx)

            if self.model_config.get('n_users', -1) == -1:
                self.model_config['n_users'] = n_users
            if self.model_config.get('n_items', -1) == -1:
                self.model_config['n_items'] = n_items

            self.model = NCFNeuMFModel(
                n_users=self.model_config['n_users'],
                n_items=self.model_config['n_items'],
                models_dim=self.model_config['models_dim']
            ).to(self.device)
            
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                 self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            print(f"NeuMF model loaded successfully from {self.model_path}")

        except Exception as e:
            print(f"Error loading NeuMF model or its components: {e}")
            raise

    def _internal_predict(self, sids: np.ndarray, pids: np.ndarray) -> np.ndarray:
        if self.model is None or self.user_to_idx is None or self.item_to_idx is None:
            raise RuntimeError("NeuMF model/maps not loaded.")

        predictions = np.full(len(sids), self.default_prediction_value, dtype=float)
        
        u_indices_model = []
        p_indices_model = []
        original_indices_for_preds = []

        for i, (sid, pid) in enumerate(zip(sids, pids)):
            # NCF model expects integer sids/pids that were mapped to 0-indexed integers
            str_sid, str_pid = str(sid), str(pid)
            
            if str_sid in self.user_to_idx and str_pid in self.item_to_idx:
                u_idx = self.user_to_idx[str_sid]
                p_idx = self.item_to_idx[str_pid]
                u_indices_model.append(u_idx)
                p_indices_model.append(p_idx)
                original_indices_for_preds.append(i)
            else:
                pass
        
        if not u_indices_model:
            return predictions

        u_tensor = torch.tensor(u_indices_model, dtype=torch.long).to(self.device)
        p_tensor = torch.tensor(p_indices_model, dtype=torch.long).to(self.device)

        model_preds_raw = []
        batch_size = 1024
        with torch.no_grad():
            for i in range(0, len(u_tensor), batch_size):
                batch_u = u_tensor[i:i+batch_size]
                batch_p = p_tensor[i:i+batch_size]
                preds_batch = self.model(batch_u, batch_p)
                model_preds_raw.extend(preds_batch.cpu().tolist())
        
        for i, model_pred_val in enumerate(model_preds_raw):
            original_idx = original_indices_for_preds[i]
            predictions[original_idx] = model_pred_val
            
        return predictions

class WeightedALSWrapper(BaseHFModelWrapper):
    def __init__(self, model_path: str):
        self.user_id_map = None
        self.item_id_map = None
        self.max_sid_train = 0
        self.max_pid_train = 0
        super().__init__(model_path)

    def _load_internal_model(self):
        # Need to monkeypatch WeightedALSModel to be accessible to the unpickler
        import pickle
        import sys

        original_main_wals_model = None
        main_module = sys.modules.get('__main__')
        had_original_wals_attr = False

        if main_module is not None:
            if hasattr(main_module, 'WeightedALSModel'):
                original_main_wals_model = getattr(main_module, 'WeightedALSModel')
                had_original_wals_attr = True
            setattr(main_module, 'WeightedALSModel', WeightedALSModel)
        else:
            print("__main__ module not found in sys.modules.")

        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"WeightedALS model loaded successfully from {self.model_path}")
            if not hasattr(self.model, 'rec_mtx') or self.model.rec_mtx is None:
                 print("Warning: Loaded WeightedALS model does not have 'rec_mtx' or it is None.")
            if not hasattr(self.model, 'global_mean'):
                print("Warning: Loaded WeightedALS model does not have 'global_mean'.")
                self.model.global_mean = 3.0
            if not hasattr(self.model, 'use_bias'):
                self.model.use_bias = hasattr(self.model, 'user_bias') and hasattr(self.model, 'item_bias')

            if self.model.rec_mtx is not None:
                self.max_sid_train = self.model.rec_mtx.shape[0] - 1
                self.max_pid_train = self.model.rec_mtx.shape[1] - 1
            else:
                if self.model.user_factors is not None:
                    self.max_sid_train = self.model.user_factors.shape[0] -1 
                elif self.model.user_bias is not None:
                     self.max_sid_train = len(self.model.user_bias) -1
                
                if self.model.item_factors is not None:
                    self.max_pid_train = self.model.item_factors.shape[1] -1 if len(self.model.item_factors.shape) > 1 else self.model.item_factors.shape[0] - 1
                elif self.model.item_bias is not None:
                    self.max_pid_train = len(self.model.item_bias) -1

        except Exception as e:
            print(f"Error loading WeightedALS model: {e}")
            raise
        finally:
            # Clean up monkeypatch
            if main_module is not None:
                if had_original_wals_attr:
                    setattr(main_module, 'WeightedALSModel', original_main_wals_model)
                elif hasattr(main_module, 'WeightedALSModel'):
                    delattr(main_module, 'WeightedALSModel')

    def _internal_predict(self, sids: np.ndarray, pids: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("WeightedALS model not loaded.")
        if not hasattr(self.model, 'predict_for_submission'):
            raise AttributeError("Loaded WeightedALS model object does not have 'predict_for_submission' method.")

        predictions = self.model.predict_for_submission(sids, pids)
        
        return np.clip(predictions, 1, 5)

class NMFWrapper(BaseHFModelWrapper):
    def __init__(self, model_path: str):
        super().__init__(model_path)

    def _load_internal_model(self):
        import pickle
        try:
            with open(self.model_path, 'rb') as f:
                algo = pickle.load(f)
            
            self.model = algo
            if self.model is None:
                raise ValueError("Failed to correctly assign algorithm to self.model")
            
            if not hasattr(self.model, 'test') or not callable(getattr(self.model, 'test')):
                 print("self.model does not have a callable 'test' method.")

        except Exception as e:
            print(f"Error loading NMF model from {self.model_path}: {e}")
            raise

    def _internal_predict(self, sids: np.ndarray, pids: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("NMF model is not loaded.")

        predictions_output = []
        
        test_data = []
        for sid, pid in zip(sids, pids):
            test_data.append((sid, pid, 0.0))
        
        surprise_predictions = self.model.test(test_data)
        
        for pred in surprise_predictions:
            if pred.details.get('was_impossible', False):
                predictions_output.append(pred.est) 
            else:
                predictions_output.append(pred.est)
                
        return np.array(predictions_output)