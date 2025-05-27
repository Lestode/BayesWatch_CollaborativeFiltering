import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import MSELoss

class BPRWishlistPlusMSELoss(nn.Module):
    def __init__(self, lambda_bpr: float = 0.4):
        super().__init__()
        self.lambda_bpr = lambda_bpr
        self.mse_loss_fn = nn.MSELoss()

    def forward(self,
                predictions_explicit: torch.Tensor,
                targets_explicit: torch.Tensor,
                model: nn.Module, # model is passed to get scores for implicit feedback
                user_ids_implicit: torch.Tensor = None,
                positive_item_ids_implicit: torch.Tensor = None,
                negative_item_ids_implicit: torch.Tensor = None):

        mse_val = self.mse_loss_fn(predictions_explicit, targets_explicit)

        bpr_val = torch.tensor(0.0, device=predictions_explicit.device)

        if self.lambda_bpr > 0 and \
           user_ids_implicit is not None and user_ids_implicit.numel() > 0 and \
           positive_item_ids_implicit is not None and positive_item_ids_implicit.numel() > 0 and \
           negative_item_ids_implicit is not None and negative_item_ids_implicit.numel() > 0:

            score_positive = model(user_ids_implicit, positive_item_ids_implicit)

            num_neg_samples = negative_item_ids_implicit.size(1)

            user_ids_expanded = user_ids_implicit.unsqueeze(1).expand(-1, num_neg_samples).reshape(-1)
            negative_item_ids_flat = negative_item_ids_implicit.reshape(-1)

            score_negative_flat = model(user_ids_expanded, negative_item_ids_flat)
            score_negative = score_negative_flat.view(user_ids_implicit.size(0), num_neg_samples)

            score_positive_expanded = score_positive.unsqueeze(1)

            diff = score_positive_expanded - score_negative

            log_likelihood = F.logsigmoid(diff)

            bpr_val = -log_likelihood.mean()
        else:
            pass

        total_loss = mse_val + self.lambda_bpr * bpr_val
        return total_loss