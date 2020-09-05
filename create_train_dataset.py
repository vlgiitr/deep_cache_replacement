import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from glob import glob
from torch.utils.data import Dataset, DataLoader
import torch
from embed_lstm_32 import ByteEncoder
from embed_lstm_32 import Token

class miss_dataset(Dataset):
    def __init__(self, train_x , train_y):
        self.train_x = train_x
        self.train_y = train_y

    def __len__(self):
        return self.train_x.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        return [self.train_x[idx] , self.train_y[idx]]

def create_inout_sequences(input_x,freq,rec,tw):
    L = input_x.shape[0]
    x = torch.zeros(L-tw,tw,2)
    y = torch.zeros(L-tw,3)

    for i in range(L-tw):
        x[i] = input_x[i:i+tw]  
        y[i,0] = input_x[i+tw:i+tw+1,1]
        y[i,1] = freq[i+tw:i+tw+1]
        y[i,2] = rec[i+tw:i+tw+1]

    return x,y




def get_miss_dataloader(batch_size,window_size,n_files):
    PATH = "/home/deku/Coding/AAAI/deep_cache_replacement/data/csv_data/cse240_project_ucsd/misses"
    EXT = "*.csv"
    all_csv_files = [file
                    for path, subdir, files in os.walk(PATH)
                    for file in glob(os.path.join(path, EXT))]


    total_len = 0
    count_files = 0
    for file in all_csv_files:
        count_files +=1

        df = pd.read_csv(file)
        total_len = total_len + len(df.index) - window_size
        if(count_files>n_files):
            break

    train_x = torch.zeros(total_len , window_size, 2)
    train_y = torch.zeros(total_len , 3)


    i = 0
    count_files = 0

    for file in all_csv_files:
        count_files+=1
        df = pd.read_csv(file)
        df['Address'] = df['Address'].apply(int, base=16)
        df['PC'] = df['PC'].apply(int, base=16)


        pc = torch.tensor(df['PC'].astype(np.float32)).unsqueeze(1)
        addr = torch.tensor(df['Address'].astype(np.float32)).unsqueeze(1)
        freq = torch.tensor(df['Frequency'].astype(np.float32))


        input_x = torch.cat([pc,addr], dim = -1)

        rec = torch.tensor(df['Recency'].astype(np.float32))

        x,y = create_inout_sequences(input_x,freq,rec,window_size)
        L = x.shape[0]
        train_x[i:i+L] = x
        train_y[i:i+L] = y
        i+=L

        if(count_files>n_files):
            break

    
    dataset = miss_dataset(train_x,train_y)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=0,drop_last = True)

    return dataloader