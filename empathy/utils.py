from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
import torch.nn.functional as F
import torch
import logging
import os
from torch import nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset


class TokenizersCollateFn:
    ''' implementation of CollateFN to do tokenization and batches of sequences '''
    def __init__(self, max_tokens=512):

        #RoBERTa uses the BPE tokenizer, similarly to GPT-2
        t = ByteLevelBPETokenizer(
            "tokenizer/vocab.json",
            "tokenizer/merges.txt"
        )
        t._tokenizer.post_processor = BertProcessing(
            ("</s>", t.token_to_id("</s>")),
            ("<s>", t.token_to_id("<s>")),
        )
        t.enable_truncation(max_tokens)
        t.enable_padding(pad_id=t.token_to_id("<pad>"))
        self.tokenizer = t

    def __call__(self, batch):
        encoded = self.tokenizer.encode_batch([x[0] for x in batch])
        sequences_padded = torch.tensor([enc.ids for enc in encoded])
        attention_masks_padded = torch.tensor([enc.attention_mask for enc in encoded])
        labels = torch.tensor([x[1] for x in batch])

        # print(sequences_padded.requires_grad)
        # print(attention_masks_padded.requires_grad)
        # print(labels.requires_grad)

        return (sequences_padded, attention_masks_padded), labels
    
class EarlyStopper:
    """
    Custom early stopping class to stop training if validation loss
    doesn't improve after a given patience.
    """
    def __init__(self, patience=1, verbose=False, min_delta=0):
        """
        Args:
        patience (int): How long to wait after last time validation loss improved.
        Default: 7
        verbose (bool): If True, prints a message for each validation loss improvement.
        Default: False
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        Default: 0
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.min_validation_loss = np.inf
        self.best_model_path = os.path.join("saved_models", "checkpoint_model.pt")

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}\n\n')
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}\n\n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        ''' Saves model when validation loss decreases. '''
        if self.verbose:
            logging.info(f'Validation loss decreased ({self.min_validation_loss:.6f} --> {val_loss:.6f}). Saving model...\n\n')
            # print(f'Validation loss decreased ({self.min_validation_loss:.6f} --> {val_loss:.6f}). Saving model...\n\n')
        
        checkpoint_path = self.best_model_path
        torch.save(model.state_dict(), checkpoint_path)
        logging.info(f'Checkpoint model saved to {checkpoint_path}\n\n')
        self.min_validation_loss = val_loss

    '''
    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
        self.min_validation_loss = validation_loss
        self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
        self.counter += 1
        if self.counter >= self.patience:
            return True
        return False
    '''
    
# using Mish activation function
# (from https://github.com/digantamisra98/Mish/blob/b5f006660ac0b4c46e2c6958ad0301d7f9c59651/Mish/Torch/mish.py)

@torch.jit.script
def mish(input):
    return input * torch.tanh(F.softplus(input))

class Mish(nn.Module):
    def forward(self, input):
        return mish(input)
    

#create a dictionary which associates each string label to an integer value
labels = ["weak", "strong"]
label2int = dict(zip(labels, list(range(len(labels)))))
print(label2int)

class EmpathyDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.data_column = "text"
        self.class_column = "class"
        self.data = pd.read_csv(path, sep=";", header=None, names=[self.data_column, self.class_column],
                               engine="python")

    def __getitem__(self, idx):
        return self.data.loc[idx, self.data_column], label2int[self.data.loc[idx, self.class_column]]

    def __len__(self):
        return self.data.shape[0]