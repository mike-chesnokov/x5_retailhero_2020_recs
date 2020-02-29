import os
import random
from collections import defaultdict

import torch
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt


def _fix_seeds(seed=7):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
def plot_lgb_feature_importance(lgb_model, max_features):
    """
    Method to plot LightGBM feature importance
    :param lgb_model: LightGBM trained model
    :param  max_features: number of features to plot
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    lgb.plot_importance(lgb_model, max_num_features=max_features, height=0.8, ax=ax, importance_type='gain')
    ax.grid(False)
    plt.title("LightGBM - Feature Importance", fontsize=20)
    plt.show()
    
    
def plot_loss_metrics(loss_history, metrics_history):
    """
    loss_history: dict of lists, contains train and valid loss
    metrics_history: dict of 2 dicts of lists, metrics on train and validation
    """
    fig, ax = plt.subplots(nrows=len(metrics_history) + 1, ncols=1, figsize=(10, 6 * len(metrics_history) + 1))
    ax[0].set_title('Loss')
    ax[0].plot(loss_history['train'], label='train')
    ax[0].plot(loss_history['valid'], label='valid')
    ax[0].legend()
    
    for ind, metric in enumerate(metrics_history):
        ax[ind + 1].set_title(metric)
        ax[ind + 1].plot(metrics_history[metric]['train'], label='train')
        ax[ind + 1].plot(metrics_history[metric]['valid'], label='valid')
        ax[ind + 1].legend()

    plt.show()


def blend_product_lists(*cands, num_candidates=30):
    cnt_dict = defaultdict(int)
    for cand in cands:
        #print(cands)
        for ind, pr in enumerate(cand):
            cnt_dict[pr] += ind + 1

    # add rank for items not any of set
    #for cand in cands:
    #    cand = set(cand)
    max_rank = max([len(el) for el in cands])
    
    for pr in cnt_dict:
        for cand in cands:
            if pr not in cand:
                cnt_dict[pr] += max_rank

    sorted_list = sorted(cnt_dict.items(), key=lambda x: x[1], reverse=False)

    return [el[0] for el in sorted_list[:num_candidates]]


# user-item mtrix preparation
def filter_user_item_pairs(user_items_pairs, users_to_drop, items_to_drop):
    """
    Params:
        user_items_pairs: set of pairs (client, product) or dict (client, product) -> cnt
        users_to_drop: list, clients
        items_to_drop: list, products
    """
    users_to_drop = set(users_to_drop)
    items_to_drop = set(items_to_drop)
    
    if type(user_items_pairs) is set:
        return {pair for pair in user_items_pairs \
                    if pair[0] not in users_to_drop and \
                       pair[1] not in items_to_drop}
    else:
        return {pair: user_items_pairs[pair] for pair in user_items_pairs \
                    if pair[0] not in users_to_drop and \
                       pair[1] not in items_to_drop}


def get_user_item_matrix(user_items_pairs, data_format='bin'):
    """
    Method for making user_item csr_matrix from user_items_pairs.
    2 data formats: 
        - 'bin': clinet-product -> 1 or 0 (user_items_pairs can be st or dict)
        - 'cnt': client-product -> count of such pair (user_items_pairs should be dict)
    
    Params:
        user_items_pairs: set of pairs (client, product) or dict (client, product) -> cnt
        data_format: str in ['bin', 'cnt']
    """
    # create user-row and item-col dicts
    users = []
    items = []
    data = []

    user_index = {}
    user_ind = 0

    item_index = {}
    item_ind = 0

    for pair in user_items_pairs:
        if pair[0] not in user_index:
            user_index[pair[0]] = user_ind
            user_ind += 1

        if pair[1] not in item_index:
            item_index[pair[1]] = item_ind
            item_ind += 1

        users.append(user_index[pair[0]])
        items.append(item_index[pair[1]])
        if data_format == 'bin':
            data.append(1)
        elif data_format == 'cnt':
            data.append(user_items_pairs[pair])
            
    return user_index, item_index, csr_matrix((data, (users, items)))
