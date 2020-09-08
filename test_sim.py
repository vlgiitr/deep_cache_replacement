import numpy as np
from collections import deque, defaultdict
from tqdm import tqdm
import torch
from cache_model_train import DeepCache
from cache_lecar import LeCaR
import csv
import glob
import os
import argparse
from embed_lstm_32 import ByteEncoder
from sklearn.neighbors import KernelDensity
import pandas as pd

EMBED_ENCODER = torch.load("w2vec_checkpoints/byte_encoder_32.pt")
EMB_SIZE = 40

def create_inout_sequences(input_x,tw):
    L = input_x.shape[0]
    x = torch.zeros(1,L,2)
    x[0] = input_x[0:]  
    return x

def get_test_data_from_list(addresses,pcs,window_size):
   
    df = pd.DataFrame(list(zip(pcs, addresses)), 
                   columns =['PC', 'Address']) 
    df['Address'] = df['Address'].apply(int, base=16)
    df['PC'] = df['PC'].apply(int, base=16)
    pc = torch.tensor(df['PC'].astype(np.float32)).unsqueeze(1)
    addr = torch.tensor(df['Address'].astype(np.float32)).unsqueeze(1)
    input_x = torch.cat([pc,addr], dim = -1)
    x = create_inout_sequences(input_x,window_size)
    return x

def get_embeddings(addresses,pcs,deepcache):

    input = get_test_data_from_list(addresses,pcs,len(addresses))
    pc      = input[:,:,0:1] 
    address = input[:,:,1:2] # Address value in decimal
    pc_embed = deepcache.get_embed_pc(pc) # Convert decimal address to 4 byte embeddings using pretrained embeddings
    addr_embed = deepcache.get_embed_addr(address)
    # time distributed MLP because we need to apply it on every element of the sequence
    embeddings_pc = deepcache.time_distributed_encoder_mlp(pc_embed) # Convert 4byte embedding to a single address embedding using an MLP
    embeddings_address = deepcache.time_distributed_encoder_mlp(addr_embed)
    # concat pc and adress emeddings
    embeddings = torch.cat([embeddings_pc,embeddings_address] ,dim=-1)

    return embeddings

def get_freq_rec(self, probs, dist_vector,deepcache):

    freq_rec = deepcache.get_freq_rec(probs,dist_vector) # get freq and rec estimate from prediced probs and distribution vector
    freq = freq_rec[:,0]
    rec = freq_rec[:,1]
    return freq,rec

def get_dist(self, input,deepcache):
    dist_vector = deepcache.get_distribution_vector(input)
    return dist_vector

def get_probs(x,decoder):
    probs,_ = decoder.lstm_decoder(x)
    return probs

def get_prefetch(misses_address,misses_pc,deepcache):

    hidden_cell = (torch.zeros(1, len(misses_address), deepcache.hidden_size), # reinitialise hidden state for each new sample
                            torch.zeros(1, len(misses_address), deepcache.hidden_size))
    embeddings = get_embeddings(misses_address,misses_pc,deepcache)
    _,hidden_cell = deepcache.lstm(embeddings, hidden_cell)
    probs,_ = deepcache.lstm_decoder(hidden_cell[0])
    print(probs)
    return probs


def test_cache_sim(cache_size, addresses, pcs, misses_window, miss_history_length):

    emb_size = 40
    hidden_size = 40
    cache_address = set()
    cache_pc = set()
    num_hit, num_miss,total_miss = 0, 0, 0
    miss_addresses = []
    pc_misses = []
    rec = None
    freq = None
    deepcache = DeepCache(input_size=2*emb_size, hidden_size=hidden_size, output_size=256)
    
    for i in tqdm(range(len(addresses))):
        address = addresses[i]
        pc = pcs[i]
        if address in cache_address: # If address is in cache increment the num_hit
            num_hit += 1
            continue
            
        elif len(cache_address) < cache_size: # If address is not in cache and the cache is not full yet then increment the num_miss 
            cache_address.add(address)
            cache_pc.add(pc)
            num_miss += 1
            total_miss+=1
            miss_addresses.append(address)
            pc_misses.append(pc)
            if num_miss == miss_history_length: # Calculate freq and rec for every 10 misses
                num_miss = 0
                if len(miss_addresses) >= miss_history_length:
                    prefetch = get_prefetch(miss_addresses[-misses_window:],pc_misses[-misses_window:],deepcache)
                else:
                    prefetch = get_prefetch(miss_addresses,pc_misses,deepcache)
        else:
            num_miss += 1
            total_miss+=1
            miss_addresses.append(address)
            pc_misses.append(pc)
            if num_miss == miss_history_length: # Calculate freq and rec for every 10 misses
                num_miss = 0
                if len(miss_addresses) >= miss_history_length:
                    prefetch = prefetch(miss_addresses[-misses_window:],pc_misses[-misses_window:],deepcache)
                else:
                    prefetch = prefetch(miss_addresses,pc_misses,deepcache)
            e = get_embeddings(list(cache_address),list(cache_pc),deepcache)
            dist_vector = get_dist(e,deepcache)
            probs = get_probs(x=e,decoder=deepcache)
            freq,rec = get_freq_rec(decoder=deepcache,dist_vector=dist_vector,x=probs)
            cache_address.add(address)
            cache_pc.add(pc)
    
    hitrate = num_hit / (num_hit + total_miss)
    return hitrate


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Test cache")
    parser.add_argument("--r", required=True,
    help="path to test csv file")
    args =  parser.parse_args()

    count = 0
    addresses = []
    pcs = []
    with open(args.r,'r') as file:
        reader = csv.reader(file)
        for row in reader:
            count+=1
            if count == 1:
                continue
            else:
                pcs.append(row[1])
                addresses.append(row[2]) 
    
    print('Count: {}'.format(count))
    print('Testing Started')
    hitrate = test_cache_sim(cache_size=32,addresses=addresses,pcs=pcs,misses_window=30,miss_history_length=10)
    print('---------------------------')
    print('Testing Complete')
    print('HitRate: {}'.format(hitrate))
    print('---------------------------')