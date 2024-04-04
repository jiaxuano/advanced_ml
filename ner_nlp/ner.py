import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd


# if a word is not in your vocabulary use len(vocabulary) as the encoding
class NERDataset(Dataset):
    def __init__(self, df_enc, window_size=5):
        self.window_size = window_size
        self.df = df_enc

    def __len__(self):
        """ Length of the dataset """
        ### BEGIN SOLUTION
        self.y = self.df['label'].values
        L = len(self.y) - (self.window_size - 1)
        ### END SOLUTION
        return L

    def __getitem__(self, idx):
        """ returns x[idx], y[idx] for this dataset
        
        x[idx] should be a numpy array of shape (5,)
        """
        ### BEGIN SOLUTION
        self.x = self.df['word'].values
        self.y = self.df['label'].values
        x = self.x[idx:idx+self.window_size]
        y = self.y[idx+(self.window_size//2)]
        ### END SOLUTION
        return x, y 


def label_encoding(cat_arr):
    """ Given a numpy array of strings returns a dictionary with label encodings.

    First take the array of unique values and sort them (as strings). 
    """
    ### BEGIN SOLUTION
    unique_vals = np.sort(np.unique(cat_arr))
    vocab2index = {val: idx for idx, val in enumerate(unique_vals)}
    ### END SOLUTION
    return vocab2index


def dataset_encoding(df, vocab2index, label2index):
    """Apply vocab2index to the word column and label2index to the label column

    Replace columns "word" and "label" with the corresponding encoding.
    If a word is not in the vocabulary give it the index V=(len(vocab2index))
    """
    V = len(vocab2index)
    df_enc = df.copy()
    ### BEGIN SOLUTION
    df_enc['word'] = df_enc['word'].apply(lambda x: vocab2index.get(x, V))
    df_enc['label'] = df_enc['label'].apply(lambda x: label2index[x])
    ### END SOLUTION
    return df_enc


class NERModel(nn.Module):
    def __init__(self, vocab_size, n_class, emb_size=50, seed=3):
        """Initialize an embedding layer and a linear layer
        """
        super(NERModel, self).__init__()
        torch.manual_seed(seed)
        ### BEGIN SOLUTION
        self.embedding = nn.Embedding(vocab_size + 1, emb_size)
        self.linear = nn.Linear(emb_size * 5, n_class)
        ### END SOLUTION
        
    def forward(self, x):
        """Apply the model to x
        
        1. x is a (N,5). Lookup embeddings for x
        2. reshape the embeddings (or concatenate) such that x is N, 5*emb_size 
           .flatten works
        3. Apply a linear layer
        """
        ### BEGIN SOLUTION
        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        ### END SOLUTION
        return x

def get_optimizer(model, lr = 0.01, wd = 0.0):
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    return optim

def train_model(model, optimizer, train_dl, valid_dl, epochs=10):
    for i in range(epochs):
        ### BEGIN SOLUTION
        model.train()
        total = 0
        sum_loss = 0
        for x, y in train_dl:
            batch = y.shape[0]
            x = x.long()
            y = y.long()
            out = model(x)
            loss = F.cross_entropy(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += batch
            sum_loss += batch * loss.item()
        train_loss = sum_loss / total
        ### END SOLUTION
        valid_loss, valid_acc = valid_metrics(model, valid_dl)
        print("train loss  %.3f val loss %.3f and accuracy %.3f" % (
            train_loss, valid_loss, valid_acc))

def valid_metrics(model, valid_dl):
    ### BEGIN SOLUTION
    model.eval()
    total = 0
    sum_loss = 0
    correct = 0
    for x, y in valid_dl:
        batch = y.shape[0]
        x = x.long()
        y = y.long()
        out = model(x)
        loss = F.cross_entropy(out, y)
        sum_loss += batch * loss.item()
        total += batch
        pred = torch.max(out, 1)[1]
        correct += (pred == y).float().sum().item()
    val_loss = sum_loss / total
    val_acc = correct / total
    ### END SOLUTION
    return val_loss, val_acc

