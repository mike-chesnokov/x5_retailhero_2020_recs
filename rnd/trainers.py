import sys
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from utils import _fix_seeds
from metrics_losses import compute_nmap_batch


class TransactionTrainer:
    """
    Input sequence contains data, data lengths, targets
    """
    def __init__(self, 
                 path_to_save_model,
                 train_loader, valid_loader,
                 loss, 
                 num_epochs, 
                 device):
        self.path_to_save_model = path_to_save_model
        self.num_epochs = num_epochs
        self.device = device
        
        self.model = None
        self.loss = loss
        self.best_loss = None
        self.optimizer = None
        self.scheduler = None
        
        self.loaders = {
            'train': train_loader,
            'valid': valid_loader
        }
        # metrics history
        self.loss_history = {
            'train': [],
            'valid': []
        }
        self.nmap_history = {
            'train': [],
            'valid': []
        }
        self.nmap_history_batches = {
            'train': [],
            'valid': []
        }        
        
    def forward_train(self, sequence):
        """
        Forward pass for train phase
        """
        _, _, targets = sequence
        bs, num_products = targets.shape
        
        preds = self.model.forward(sequence)
        loss_ = self.loss(preds.view(bs, num_products), targets)
        
        self.optimizer.zero_grad()
        loss_.backward()
        self.optimizer.step()
        
        return loss_, preds


    def forward_valid(self, sequence):
        """
        Forward pass for validation phase
        """
        with torch.no_grad():
            _, _, targets = sequence
            bs, num_products = targets.shape
            
            preds = self.model.forward(sequence)
            loss_ = self.loss(preds.view(bs, num_products), targets)
            
        return loss_, preds      

    
    def _format_logs(self, logs):
        """
        logs - dict with metrics values
        """
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s


    def run_epoch(self, phase, data_loader):
        """
        Run single epoch of train or validation
        
        phase: str, 'train' or 'valid'
        """        
        # enter mode
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()
            
        # fix seeds
        _fix_seeds()

        data_len = len(data_loader)
        
        logs = {}
        nmap_batches = []
        epoch_loss = 0
        
        # iterate over data
        with tqdm(data_loader, desc=phase, file=sys.stdout) as iterator:
            for (batch_seq, batch_len, batch_targets) in iterator:
                
                _fix_seeds()
                # to gpu
                batch_seq = batch_seq.to(self.device)
                batch_targets = torch.stack(batch_targets, dim=0)
                batch_targets = batch_targets.to(self.device)
        
                if phase == 'train':
                    batch_loss, batch_preds = self.forward_train((batch_seq, batch_len, batch_targets))
                else:                   
                    batch_loss, batch_preds = self.forward_valid((batch_seq, batch_len, batch_targets))   
                
                epoch_loss += batch_loss.item()
                
                nmap_batch = compute_nmap_batch(batch_targets, batch_preds)
                nmap_batches.append(nmap_batch)
                logs['nmap_batch'] = nmap_batch
                
                del batch_targets, batch_seq, batch_preds      
                #torch.cuda.empty_cache()
                
                # save current batch loss value for output
                loss_logs = {self.loss.__name__: batch_loss.item()}
                logs.update(loss_logs)
                s = self._format_logs(logs)
                iterator.set_postfix_str(s)
        
        epoch_loss /= data_len
        
        if self.scheduler is not None:
            self.scheduler.step()
            #self.scheduler.step(epoch_loss)
        
        torch.cuda.empty_cache()
        
        return epoch_loss, np.mean(nmap_batches), nmap_batches


    def run_model(self, model, optimizer, scheduler):
        """
        Iterate through epochs with both train and validation pahses
        """
        _fix_seeds()
        cur_time = datetime.now().strftime("%H:%M:%S")
        # initialize model, learning rate and loss for each run
        self.best_loss = float("inf")
        self.model = model
        self.model.to(self.device)
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.init_lr = [pg['lr'] for pg in self.optimizer.param_groups][0]
        
        for g in self.optimizer.param_groups:
            g['lr'] = self.init_lr
        
        for epoch in range(self.num_epochs):
            _fix_seeds()
            
            print(f"Starting epoch: {epoch} | time: {cur_time}")
            print('LR:',[pg['lr'] for pg in self.optimizer.param_groups])
            for phase in ['train', 'valid']:
                
                epoch_all_metrics = self.run_epoch(phase, self.loaders[phase])
                self.loss_history[phase].append(epoch_all_metrics[0])
                self.nmap_history[phase].append(epoch_all_metrics[1])
                self.nmap_history_batches[phase].extend(epoch_all_metrics[2])
                
                if phase == 'valid':
                    print(f'Valid avg loss: {round(epoch_all_metrics[0], 6)}')
                    print(f'Valid avg nmap: {round(epoch_all_metrics[1], 6)}')
                
                del epoch_all_metrics
                torch.cuda.empty_cache()
                
            if self.best_loss > self.loss_history['valid'][-1]:
                self.best_loss = self.loss_history['valid'][-1]
                torch.save(self.model, self.path_to_save_model)
                print('*** Model saved! ***\n')
                
            cur_time = datetime.now().strftime("%H:%M:%S")
