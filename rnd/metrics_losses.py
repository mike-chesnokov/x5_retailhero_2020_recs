import torch
import torch.nn as nn
import numpy as np


class BCELogitsLoss(nn.Module):
    __name__ = 'bce_logits_loss'
    
    def __init__(self,):
        super().__init__()

    def forward(self, inputs, targets):
        
        loss = nn.BCEWithLogitsLoss()
        
        return loss(inputs, targets)

    
# Metrics calculations
def average_precision(actual, recommended, k=30):
    ap_sum = 0
    hits = 0
    for i in range(k):
        product_id = recommended[i] if i < len(recommended) else None
        if product_id is not None and product_id in actual:
            hits += 1
            ap_sum += hits / (i + 1)
    return ap_sum / k


def normalized_average_precision(actual, recommended, k=30):
    actual = set(actual)
    if len(actual) == 0:
        return 0.0
    
    ap = average_precision(actual, recommended, k=k)
    ap_ideal = average_precision(actual, list(actual)[:k], k=k)
    return ap / ap_ideal


def get_actual_lists(batch_targets):
    """
    batch_targets: torch tensor of shape=(batch_size, num_products)
    
    return: list of lists of indeces with non zero elements
    """
    actual_batch = []
    row_inds = []
    control = 0

    for row, ind in torch.nonzero(batch_targets):
        if row == control:
            row_inds.append(ind.item())
        else:
            actual_batch.append(row_inds)
            row_inds = []
            row_inds.append(ind.item())
            control += 1
    actual_batch.append(row_inds)
    
    return actual_batch

def compute_nmap_batch(batch_targets, batch_preds, num_recommended=30):
    """
    batch_targets: list of torch tensors of shape=(batch_size, num_products)
    batch_preds: torch tensor of shape=(batch_size, num_products)
    """
    # process inputs
    #batch_targets = torch.stack(batch_targets, dim=0)
    bs, num_products = batch_targets.shape
    batch_preds = torch.sigmoid(batch_preds.view(bs, num_products))
    
    batch_targets = get_actual_lists(batch_targets)
    batch_preds = torch.argsort(batch_preds, dim=1, descending=True)[:, :num_recommended].cpu().numpy()
    
    aps = []
    for acts, preds in zip(batch_targets, batch_preds):
        aps.append(normalized_average_precision(acts, preds))

    return np.mean(aps)
