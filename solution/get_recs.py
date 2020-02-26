import pickle
from datetime import datetime
from collections import defaultdict

import gensim
import numpy as np
import lightgbm as lgb
from scipy.sparse import csr_matrix

from solution.utils import (
    blend_product_lists,
    get_user_vector,
    get_prod_dates_features,
    get_trans_dates_features,
    add_candidates,
)
from solution.settings import (
    NUM_CANDIDATES,
    DATETIME_FORMAT,
    baseline_items,
    lgb_models,
)

# load artifacts
with open('model_artifacts/item_index_02_01_3.pckl', 'rb') as f:
    item_index = pickle.load(f)
index_item = {v: k for k, v in item_index.items()}
num_items = len(item_index)

with open('model_artifacts/model_implicit_nn_tfidf_02_01_3.pckl', 'rb') as f:
    model_nn = pickle.load(f)

with open('model_artifacts/incremental_encoder_dict_01_23_01.pckl', 'rb') as f:
    inc_encoder = pickle.load(f)

with open('model_artifacts/product_features_01_23_01.pckl', 'rb') as f:
    product_features = pickle.load(f)

with open('model_artifacts/baseline_popular_feb.pckl', 'rb') as f:
    baseline_popular_feb = pickle.load(f)
baseline_popular_feb_cur = baseline_popular_feb[:NUM_CANDIDATES].copy()

w2v_file = 'model_artifacts/w2v_01_29_2_10_20_20_2.wv'
model_w2v = gensim.models.Word2Vec.load(w2v_file)
W2V_SIZE = int(w2v_file.split('/')[-1].split('_')[-3])

lgb_models = [lgb.Booster(model_file=lgb_file) for lgb_file in lgb_models]

prodcut_index = inc_encoder['product_id']
level1_index = inc_encoder['level_1']
level2_index = inc_encoder['level_2']
level3_index = inc_encoder['level_3']
level4_index = inc_encoder['level_4']
brand_index = inc_encoder['brand_id']
vendor_index = inc_encoder['vendor_id']


def predict(query_data):
    """
    Main method for prediction

    :param query_data: dict with "transaction_history", "age", "gender", "query_time"
    :return candidates: list or product ids
    """
    transaction_history = query_data.get("transaction_history", [])
    cnt_trans = len(transaction_history)
    query_dt = datetime.strptime(query_data['query_time'], DATETIME_FORMAT)

    # age processing
    client_age = query_data.get('age', 0)
    if client_age < 14 or client_age > 100:
        client_age = 0

    # transaction history processing and feature creation
    if cnt_trans > 1:
        sorted_transactions = sorted(transaction_history,
                                     key=lambda x: datetime.strptime(x['datetime'], DATETIME_FORMAT))

        last_tr_dt = datetime.strptime(sorted_transactions[-1]['datetime'], DATETIME_FORMAT)
        first_tr_dt = datetime.strptime(sorted_transactions[0]['datetime'], DATETIME_FORMAT)
        coef1 = (query_dt - first_tr_dt).days + 1
        coef2 = (query_dt - last_tr_dt).days

        user_products = defaultdict(float)
        user_products_exp = defaultdict(float)
        user_product_dates = defaultdict(list)
        user_product_cnt_in_tr = defaultdict(list)

        # brand features
        user_brand = defaultdict(float)

        # vendor features
        user_vendor = defaultdict(float)

        # level features
        user_level4 = defaultdict(float)
        user_level3 = defaultdict(float)

        user_vector = np.zeros(shape=(1, num_items), dtype=np.float32)
        purchase_sums = []
        trans_dates = []

        for transaction in sorted_transactions:
            # transaction level features
            purchase_sums.append(float(transaction['purchase_sum']))
            # dt of transaction
            tr_dt = datetime.strptime(transaction['datetime'], DATETIME_FORMAT)
            trans_dates.append(tr_dt)

            cur_days_diff = (query_dt - tr_dt).days

            pr_cnt_in_tr = len(transaction['products'])

            for pr in transaction['products']:
                # for 1lvl model
                user_products[pr['product_id']] += 1. / cnt_trans
                # for 2lvl model
                user_products_exp[pr['product_id']] += np.exp(-(cur_days_diff - coef2) / coef1) / cnt_trans
                user_product_dates[pr['product_id']].append(tr_dt)
                user_product_cnt_in_tr[pr['product_id']].append(pr_cnt_in_tr)

                # brand features
                brand_id = product_features[pr['product_id']]['brand_id']
                user_brand[brand_id] += 1. / cnt_trans

                # vendor features
                vendor_id = product_features[pr['product_id']]['vendor_id']
                user_vendor[vendor_id] += 1. / cnt_trans

                # level features
                level4 = product_features[pr['product_id']]['level_4']
                user_level4[level4] += 1. / cnt_trans
                level3 = product_features[pr['product_id']]['level_3']
                user_level3[level3] += 1. / cnt_trans

        # get dummy candidates
        ups = sorted(user_products.items(), key=lambda x: x[1], reverse=True)
        items_ranked = [item[0] for item in ups if item[1] > 1. / cnt_trans]

        # add candidates to get NUM_CANDIDATES_1LEVEL items
        candidates_dummy = add_candidates(items_ranked,
                                          baseline_popular_feb_cur,
                                          num_cands_threshold=NUM_CANDIDATES)

        # get model nn candidates
        user_vector = get_user_vector(user_products, item_index, user_vector)
        raw_recs = model_nn.recommend(userid=0, user_items=csr_matrix(user_vector),
                                      N=NUM_CANDIDATES,
                                      filter_already_liked_items=False, recalculate_user=True)
        candidates_nn = [index_item[ind] for (ind, score) in raw_recs]

        # blend 1lvl models: dummy and implicit_nn_tfidf
        candidates_product = blend_product_lists(*[candidates_nn, candidates_dummy],
                                                 num_candidates=NUM_CANDIDATES)
        # get some features for product
        trans_sum_avg = np.mean(purchase_sums)

        # trans dates features
        first_last_days_diff, trans_days_diff_avg, \
            last_tr_days_diff, last_tr_days_ratio = get_trans_dates_features(trans_dates, query_dt)

        # collect NUM_CANDIDATES_1LEVEL rows for every client
        user_features = []

        for lvl1_ind, pr in enumerate(candidates_product):
            if pr in product_features:
                pr_dates = user_product_dates[pr]
                first_last_pr_days_diff, \
                    pr_days_diff_avg, \
                    pr_days_diff_std, \
                    last_pr_days_diff, \
                    last_pr_days_ratio = get_prod_dates_features(pr_dates, query_dt)

                row_ = [
                    prodcut_index[pr],

                    level1_index[product_features[pr]['level_1']],
                    level2_index[product_features[pr]['level_2']],
                    level3_index[product_features[pr]['level_3']],
                    level4_index[product_features[pr]['level_4']],
                    brand_index[product_features[pr]['brand_id']],
                    vendor_index[product_features[pr]['vendor_id']],
                    product_features[pr]['netto'],
                    product_features[pr]['is_own_trademark'],

                    round(user_products[pr], 6),
                    cnt_trans,
                    client_age,

                    first_last_days_diff,
                    trans_days_diff_avg,
                    last_tr_days_diff, last_tr_days_ratio,

                    first_last_pr_days_diff,
                    pr_days_diff_avg, pr_days_diff_std,
                    last_pr_days_diff, last_pr_days_ratio,

                    np.round(trans_sum_avg, 2),
                    round(user_products_exp[pr], 4),
                    round(np.mean(user_product_cnt_in_tr[pr]) if pr in user_product_cnt_in_tr else 0, 4),
                    lvl1_ind,

                    # other cat features
                    round(user_brand[product_features[pr]['brand_id']], 4),
                    round(user_vendor[product_features[pr]['vendor_id']], 4),
                    round(user_level4[product_features[pr]['level_4']], 4),
                    round(user_level3[product_features[pr]['level_3']], 4),
                ]
                if pr in model_w2v.wv.vocab:
                    row_add = model_w2v.wv[pr].tolist()
                else:
                    row_add = [0.] * W2V_SIZE
                row_.extend(row_add)

                user_features.append(row_)

        # get predictions from lightgbm
        user_preds = [lgb_model.predict(np.array(user_features)) for lgb_model in lgb_models]
        candidates_ = []

        for user_pred in user_preds:
            cand_dict = {pr: pred for pr, pred in zip(candidates_product, user_pred.tolist())}
            temp_cands = sorted(cand_dict.items(), key=lambda x: x[1], reverse=True)
            candidates_.append([x[0] for x in temp_cands])

        candidates_ = blend_product_lists(*candidates_, num_candidates=NUM_CANDIDATES)
        # delete outdated items
        candidates_ = [item for item in candidates_ if item in baseline_popular_feb]

        # add candidates to get 30 items
        candidates = add_candidates(candidates_[:30].copy(),
                                    baseline_items,
                                    num_cands_threshold=30)

    # for clients with 1 transaction - dummy solution
    elif cnt_trans == 1:
        user_products = defaultdict(int)
        for transaction in transaction_history:
            for product in transaction['products']:
                user_products[product['product_id']] += 1

        # get dummy candidates
        ups = sorted(user_products.items(), key=lambda x: x[1], reverse=True)
        items_ranked = [item[0] for item in ups if item[1] > 1][:30]

        # add candidates to get 30 items
        candidates = add_candidates(items_ranked,
                                    baseline_items,
                                    num_cands_threshold=30)

    # for clients without transactions - baseline items
    else:
        candidates = baseline_items.copy()

    return candidates
