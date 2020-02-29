import json
import random
from datetime import datetime
from collections import OrderedDict

from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def dataloader_collate(batch):
    """
    Method for using dataloader with LSTM
    """
    # sort batch by sequencw length
    sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
    sequences = [x[0] for x in sorted_batch]
    
    # store lengths
    lengths = torch.LongTensor([len(x) for x in sequences])
    # pad sequences
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0.)
    # get targets
    targets = [x[1] for x in sorted_batch]

    return sequences_padded, lengths, targets


class TransactionsDataset(Dataset):
    """
    Class for transaction processing to LSTM input
    """
    def __init__(self,  chunk_paths, valid_time, item_index, clients_to_drop):
        """
        chunk_paths: list of str paths
        """
        cur_client_ind = 0
        index_client = {}
        client_tr_inds = OrderedDict()
        
        for path in chunk_paths:
            print(path)
            with open(path, 'r') as chunk:
                for row in tqdm(chunk):

                    client_id, transaction_history = row.split('\t')
                    client_id, transaction_history = json.loads(client_id), json.loads(transaction_history)

                    if client_id in clients_to_drop:
                        continue

                    tr_history = [transaction_history[tr] for tr in transaction_history]
                    sorted_transactions = sorted(tr_history, 
                                             key=lambda x: datetime.strptime(x['datetime'], '%Y-%m-%d %H:%M:%S'))

                    # find candidates for target transactions, if no - skip iteration
                    split_candidates = [datetime.strptime(tr['datetime'], '%Y-%m-%d %H:%M:%S') \
                                            for tr in sorted_transactions
                                            if datetime.strptime(tr['datetime'], 
                                                                 '%Y-%m-%d %H:%M:%S') > valid_time]
                    if len(split_candidates) == 0:
                        continue
                    #print(client_id) 
                    # for random split get validation query
                    split_time = random.choice(split_candidates)

                    train_trans_history = [tr for tr in sorted_transactions
                                            if datetime.strptime(
                                                       tr['datetime'], '%Y-%m-%d %H:%M:%S'
                                                   ) < split_time]
                    if len(train_trans_history) > 1:
                        target_transaction = [tr for tr in sorted_transactions
                                                  if datetime.strptime(tr['datetime'], 
                                                                       '%Y-%m-%d %H:%M:%S') == split_time][0]
                        target_product_inds = [item_index[pr['product_id']]
                                                    for pr in target_transaction['products']
                                                    if pr['product_id'] in item_index]
                        client_non_zero_inds = []
                        for transaction in train_trans_history:
                            # process transaction
                            cur_trans = []
                            for pr in transaction['products']:
                                if pr['product_id'] in item_index:
                                    cur_trans.append(item_index[pr['product_id']])
                                # add transaction to client history
                            client_non_zero_inds.append(cur_trans)

                        #print(client_id, cur_client_ind)
                        index_client[cur_client_ind] = client_id
                        cur_client_ind += 1
                        client_tr_inds[client_id] = (client_non_zero_inds, target_product_inds)
                
        self.client_tr_inds = client_tr_inds
        self.index_client = index_client
        self.num_products = len(item_index)
    
    def __len__(self):
        return len(self.client_tr_inds)
    
    def __getitem__(self, idx):
        """
        Return train transactions (2d tensor shape: cnt_client_trans*num_products) 
        and target transaction (1d tensor shape: 1*num_products)
        """
        client_non_zero_inds, target_product_inds = self.client_tr_inds[self.index_client[idx]]
        
        # train transactions
        zero_vec = torch.zeros(self.num_products)
        client_trans = []
        
        for pr_inds in client_non_zero_inds:
            pr_inds_t = torch.tensor(pr_inds)
            client_trans.append(zero_vec.index_fill(0, pr_inds_t, 1))
        
        return torch.stack(client_trans, dim=0), \
               zero_vec.index_fill(0, torch.tensor(target_product_inds), 1)
