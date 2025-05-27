import torch
import torch.nn as nn


class GMF(nn.Module):
    def __init__(self, n_users: int, n_items: int, embedding_dim: int):
        """
        Generalized Matrix Factorization (GMF) model.

        Args:
            n_users (int): Number of unique users.
            n_items (int): Number of unique items.
            embedding_dim (int): Dimensionality of the embedding vectors.
        """
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, 1)
        
        self._init_weights()

    def _init_weights(self):
        # normal distribution with std=0.01
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        # xavier uniform distribution for output layer
        nn.init.xavier_uniform_(self.output_layer.weight)
        self.output_layer.bias.data.fill_(0.0)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GMF model.

        Args:
            user_ids (torch.Tensor): Tensor of user IDs.
            item_ids (torch.Tensor): Tensor of item IDs.

        Returns:
            torch.Tensor: Predicted ratings (scalar for each user-item pair).
        """
        user_embedded = self.user_embedding(user_ids)
        item_embedded = self.item_embedding(item_ids)
        
        # element-wise product (hadamard product)
        element_product = user_embedded * item_embedded
        
        prediction = self.output_layer(element_product)
        return prediction.squeeze(-1)

class MLP(nn.Module):
    def __init__(self, n_users: int, n_items: int, last_layer_size: int):
        """
        Multi-Layer Perceptron (MLP) model for collaborative filtering.

        Args:
            n_users (int): Number of unique users.
            n_items (int): Number of unique items.
            last_layer_size (int): Size of the last layer (which is a marker for model capacity)
        """
        super().__init__()
        mlp_layers = [last_layer_size * 4, last_layer_size * 2, last_layer_size]
        self.user_embedding = nn.Embedding(n_users, last_layer_size * 2)
        self.item_embedding = nn.Embedding(n_items, last_layer_size * 2)
        
        self.mlp_layers = nn.ModuleList()
        input_size = last_layer_size * 4
        for layer_size in mlp_layers:
            self.mlp_layers.append(nn.Linear(input_size, layer_size))
            self.mlp_layers.append(nn.ReLU())
            input_size = layer_size
            
        self.output_layer = nn.Linear(input_size, 1)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
        for layer in self.mlp_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0.0)
        
        nn.init.xavier_uniform_(self.output_layer.weight)
        self.output_layer.bias.data.fill_(0.0)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP model.

        Args:
            user_ids (torch.Tensor): Tensor of user IDs.
            item_ids (torch.Tensor): Tensor of item IDs.

        Returns:
            torch.Tensor: Predicted ratings (scalar for each user-item pair).
        """
        user_embedded = self.user_embedding(user_ids)
        item_embedded = self.item_embedding(item_ids)
        
        # concat user and item embeddings
        concatenated_embeddings = torch.cat([user_embedded, item_embedded], dim=-1)
        
        mlp_output = concatenated_embeddings
        for layer in self.mlp_layers:
            mlp_output = layer(mlp_output)
            
        prediction = self.output_layer(mlp_output)
        return prediction.squeeze(-1)

class NeuMF(nn.Module):
    def __init__(self, 
                 n_users: int, 
                 n_items: int,
                 models_dim: int,
                 alpha_pretrain_fusion: float = 0.5):
        """
        Neural Matrix Factorization (NeuMF) model.
        Combines GMF and MLP pathways.

        Args:
            n_users (int): Number of unique users.
            n_items (int): Number of unique items.
            models_dim (int): For GMF: embedding_dim.
                             For MLP: last_layer_size.
            alpha_pretrain_fusion (float): Weight for GMF path when initializing fusion layer from pre-trained models.
                                         MLP path gets (1-alpha_pretrain_fusion).
        """
        super().__init__()
        self.alpha_pretrain_fusion = alpha_pretrain_fusion

        # GMF path
        self.gmf_user_embedding = nn.Embedding(n_users, models_dim)
        self.gmf_item_embedding = nn.Embedding(n_items, models_dim)

        # MLP path
        mlp_hidden_layers_config = [models_dim * 4, models_dim * 2, models_dim]
        self.mlp_user_embedding = nn.Embedding(n_users, models_dim * 2)
        self.mlp_item_embedding = nn.Embedding(n_items, models_dim * 2)
        
        self.mlp_layers_neumf = nn.ModuleList()
        mlp_input_size = 4 * models_dim
        for layer_output_size in mlp_hidden_layers_config:
            self.mlp_layers_neumf.append(nn.Linear(mlp_input_size, layer_output_size))
            self.mlp_layers_neumf.append(nn.ReLU())
            mlp_input_size = layer_output_size
        
        fusion_input_dim = models_dim + mlp_hidden_layers_config[-1]
        self.fusion_layer = nn.Linear(fusion_input_dim, 1)

    def init_weights_randomly(self):
        # gmf embeddings
        nn.init.normal_(self.gmf_user_embedding.weight, std=0.01)
        nn.init.normal_(self.gmf_item_embedding.weight, std=0.01)
        
        # mlp embeddings
        nn.init.normal_(self.mlp_user_embedding.weight, std=0.01)
        nn.init.normal_(self.mlp_item_embedding.weight, std=0.01)
        
        # mlp layers
        for layer in self.mlp_layers_neumf:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0.0)
        
        # fusion layer
        nn.init.xavier_uniform_(self.fusion_layer.weight)
        self.fusion_layer.bias.data.fill_(0.0)

    def load_pretrained_weights(self, gmf_model: GMF, mlp_model: MLP, are_paths: bool = False):
        # gmf_model and mlp_model are expected to be the model instances themselves, not file paths, if are_paths is False
        gmf_output_layer_weight_pretrained = None
        mlp_output_layer_weight_pretrained = None

        if are_paths:
            gmf_state_dict = torch.load(gmf_model, map_location=lambda storage, loc: storage)
            mlp_state_dict = torch.load(mlp_model, map_location=lambda storage, loc: storage)
        else:
            gmf_state_dict = gmf_model.state_dict()
            mlp_state_dict = mlp_model.state_dict()

        # load gmf components
        try:
            self.gmf_user_embedding.weight.data.copy_(gmf_state_dict['user_embedding.weight'])
            self.gmf_item_embedding.weight.data.copy_(gmf_state_dict['item_embedding.weight'])
            gmf_output_layer_weight_pretrained = gmf_state_dict['output_layer.weight']
            print("Loaded GMF pretrained weights.")
        except Exception as e:
            print(f"Error loading GMF weights: {e}. Initializing randomly.")
            nn.init.normal_(self.gmf_user_embedding.weight, std=0.01)
            nn.init.normal_(self.gmf_item_embedding.weight, std=0.01)

        # load mlp components
        try:
            self.mlp_user_embedding.weight.data.copy_(mlp_state_dict['user_embedding.weight'])
            self.mlp_item_embedding.weight.data.copy_(mlp_state_dict['item_embedding.weight'])
            
            mlp_layer_idx_neumf = 0
            mlp_layer_idx_pretrained_mlp = 0

            while mlp_layer_idx_neumf < len(self.mlp_layers_neumf) and \
                  mlp_layer_idx_pretrained_mlp < len(mlp_model.mlp_layers):
                
                layer_neumf = self.mlp_layers_neumf[mlp_layer_idx_neumf]
                
                if isinstance(layer_neumf, nn.Linear):
                    source_linear_layer = None
                    temp_idx = mlp_layer_idx_pretrained_mlp
                    while temp_idx < len(mlp_model.mlp_layers):
                        if isinstance(mlp_model.mlp_layers[temp_idx], nn.Linear):
                            source_linear_layer = mlp_model.mlp_layers[temp_idx]
                            actual_source_module_list_idx = -1
                            for k_mlp_source, mod_mlp_source in enumerate(mlp_model.mlp_layers):
                                if mod_mlp_source == source_linear_layer:
                                    actual_source_module_list_idx = k_mlp_source
                                    break

                            pretrained_weight_key = f'mlp_layers.{actual_source_module_list_idx}.weight'
                            pretrained_bias_key = f'mlp_layers.{actual_source_module_list_idx}.bias'

                            layer_neumf.weight.data.copy_(mlp_state_dict[pretrained_weight_key])
                            layer_neumf.bias.data.copy_(mlp_state_dict[pretrained_bias_key])
                            
                            mlp_layer_idx_pretrained_mlp = temp_idx + 1
                            break 
                        temp_idx +=1
                    else:
                        break
                mlp_layer_idx_neumf +=1


            mlp_output_layer_weight_pretrained = mlp_state_dict['output_layer.weight']
            print("Loaded MLP pretrained weights.")

        except Exception as e:
            print(f"Error loading MLP weights: {e}. Initializing randomly.")
            nn.init.normal_(self.mlp_user_embedding.weight, std=0.01)
            nn.init.normal_(self.mlp_item_embedding.weight, std=0.01)
            for layer in self.mlp_layers_neumf:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    layer.bias.data.fill_(0.0)

        # initialize the fusion layer
        expected_gmf_out_dim = self.gmf_user_embedding.embedding_dim
        
        # find the last linear layer in mlp path to determine expected_mlp_out_dim
        last_neumf_mlp_linear_layer = None
        for layer in reversed(self.mlp_layers_neumf):
            if isinstance(layer, nn.Linear):
                last_neumf_mlp_linear_layer = layer
                break

        expected_mlp_out_dim = last_neumf_mlp_linear_layer.out_features

        if gmf_output_layer_weight_pretrained is not None and \
           mlp_output_layer_weight_pretrained is not None and \
           gmf_output_layer_weight_pretrained.shape[1] == expected_gmf_out_dim and \
           mlp_output_layer_weight_pretrained.shape[1] == expected_mlp_out_dim:
            
            fusion_weight = torch.cat([
                self.alpha_pretrain_fusion * gmf_output_layer_weight_pretrained,
                (1 - self.alpha_pretrain_fusion) * mlp_output_layer_weight_pretrained
            ], dim=1)
            
            if fusion_weight.shape[1] == self.fusion_layer.in_features:
                 self.fusion_layer.weight.data.copy_(fusion_weight)
                 self.fusion_layer.bias.data.fill_(0.0)
                 print("Initialized NeuMF fusion layer.")
            else:
                print("Dimension mismatch after concatenating output layer weights for fusion.")
                print("Initializing NeuMF fusion layer randomly.")
                nn.init.xavier_uniform_(self.fusion_layer.weight)
                self.fusion_layer.bias.data.fill_(0.0)
        else:
            print("Dimension mismatch for pre-trained output layer weights before concatenation.")
            print("Initializing NeuMF fusion layer randomly.")
            nn.init.xavier_uniform_(self.fusion_layer.weight)
            self.fusion_layer.bias.data.fill_(0.0)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the NeuMF model.

        Args:
            user_ids (torch.Tensor): Tensor of user IDs.
            item_ids (torch.Tensor): Tensor of item IDs.
        """
        # gmf path
        gmf_user_embedded = self.gmf_user_embedding(user_ids)
        gmf_item_embedded = self.gmf_item_embedding(item_ids)
        gmf_vector = gmf_user_embedded * gmf_item_embedded
        
        # mlp path
        mlp_user_embedded = self.mlp_user_embedding(user_ids)
        mlp_item_embedded = self.mlp_item_embedding(item_ids)
        mlp_concatenated = torch.cat([mlp_user_embedded, mlp_item_embedded], dim=-1)
        
        mlp_vector = mlp_concatenated
        for layer in self.mlp_layers_neumf:
            mlp_vector = layer(mlp_vector)
            
        # fusion
        fused_vector = torch.cat([gmf_vector, mlp_vector], dim=-1)
        prediction = self.fusion_layer(fused_vector)
        return prediction.squeeze(-1)