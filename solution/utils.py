from collections import defaultdict

import numpy as np


def blend_product_lists(*candidates, num_candidates=30):
    """
    Blend based on products ranks in lists
    :param candidates: list of lists of product candidates
    :param num_candidates:
    :return:
    """
    cnt_dict = defaultdict(int)
    for cand in candidates:
        for ind, pr in enumerate(cand):
            cnt_dict[pr] += ind + 1

    # add rank for items not any of set
    max_rank = max([len(el) for el in candidates])
    for pr in cnt_dict:
        for cand in candidates:
            if pr not in cand:
                cnt_dict[pr] += max_rank

    sorted_list = sorted(cnt_dict.items(), key=lambda x: x[1], reverse=False)

    return [el[0] for el in sorted_list[:num_candidates]]


def get_user_vector(user_products_, item_index_, user_vector_):
    user_vector = user_vector_.copy()

    for pr in user_products_:
        if pr in item_index_:
            user_vector[0, item_index_[pr]] = user_products_[pr]

    return user_vector


def get_trans_dates_features(tr_dates, target_dt_):
    trans_days_diff = [(t - s).days for s, t in zip(tr_dates, tr_dates[1:])]

    first_last_days_diff = (tr_dates[-1] - tr_dates[0]).days
    trans_days_diff_avg = np.round(np.mean(trans_days_diff), 2)
    # trans_days_diff_std = round(np.std(trans_days_diff), 2)

    last_tr_days_diff = (target_dt_ - tr_dates[-1]).days
    last_tr_days_ratio = round(last_tr_days_diff / trans_days_diff_avg, 6) if trans_days_diff_avg > 0 else 0

    return first_last_days_diff, trans_days_diff_avg, \
        last_tr_days_diff, last_tr_days_ratio


def get_prod_dates_features(pr_dates_, target_dt_):
    if len(pr_dates_) > 1:
        pr_days_diff = [(t - s).days for s, t in zip(pr_dates_, pr_dates_[1:])]

        first_last_pr_days_diff = (pr_dates_[-1] - pr_dates_[0]).days
        pr_days_diff_avg = round(np.mean(pr_days_diff), 2) if len(pr_days_diff) > 0 else 0
        pr_days_diff_std = round(np.std(pr_days_diff), 2) if len(pr_days_diff) > 0 else 0

        last_pr_days_diff = (target_dt_ - pr_dates_[-1]).days
        last_pr_days_ratio = round(last_pr_days_diff / pr_days_diff_avg, 6) if pr_days_diff_avg > 0 else 0

        return first_last_pr_days_diff, pr_days_diff_avg, pr_days_diff_std, \
            last_pr_days_diff, last_pr_days_ratio
    else:
        return 0, 0, 0, 0, 0


def add_candidates(cands, cands_to_add, num_cands_threshold=30):
    """
    Add candidates if there is not enough
    """
    rec_len = len(cands)
    if rec_len < num_cands_threshold:
        items_add = [item for item in cands_to_add
                     if item not in cands]
        candidates_ = cands + items_add[:num_cands_threshold - rec_len]
    else:
        candidates_ = cands.copy()
    return candidates_
