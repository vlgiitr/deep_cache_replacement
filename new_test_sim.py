import numpy as np
import pandas as pd
from collections import deque, defaultdict
from tqdm import tqdm
import torch
from cache_model_train import DeepCache
from cache_lecar import LeCaR
import csv
import argparse

def create_input(addresses,pcs):
    """
    Function to convert the lists into an inputable form for the deepcache model
    """
    tw = 30
    addr = torch.tensor(addresses.astype(np.float32)).unsqueeze(1)
    pc = torch.tensor(pcs.astype(np.float32)).unsqueeze(1)
    input_x = torch.cat([pc,addr], dim = -1)
    L = input_x.shape[0]
    x = torch.zeros(L-tw,tw,2)

    for i in range(L-tw):
        x[i] = input_x[i:i+tw]  
    return x


def encoder(addresses,pcs,predictor):
    """
    model that predicts the future and recency of every address in the cache
    Inputs: List of last [n] miss addresses and pc so far
            predictor- instance of the deepcache model
    Outputs: recencies and frequencies list for the addresses  
    """
    input = create_input(addresses,pcs)
    hidden_cell = (torch.zeros(1, 1, predictor.hidden_size), # reinitialise hidden state for each new sample
                torch.zeros(1, 1, predictor.hidden_size))
    _,_,freq,rec = predictor(input = input,hidden_cell=hidden_cell)
    return freq,rec

# TODO: complete this function
def eviction_policy(cache,leCaR):
    """
    evcition carried out
    Inputs: cache,leCaR model
    Outputs: removes a address from the cache and returns the new cache 
    """
    cache = leCaR()
    return cache


def test_cache_sim(cache_size, addresses, pcs, misses_window, miss_history_length):
    """
    Function to run a test simulation on the cache
    Inputs: cache_size- size of cache (int)
            addresses- list of addresses available to store the data
            pc- list of pc
            misses_window- the number of recent misses to be passed to the deepcache model
            miss_history_length- the number of misses after which freq and rec are calculated
    Output: the hitrate 
    """
    emb_size = 40
    hidden_size = 40

    #cache = set()
    cache = pd.DataFrame(columns = ['Address','Frequency','Recency'])

    num_hit, num_miss,total_miss = 0, 0, 0
    miss_addresses = []
    pc_misses = []
    rec = None
    freq = None
    decoder = DeepCache(input_size=2*emb_size, hidden_size=hidden_size, output_size=256)
    lecar = LeCaR(cache_size,cache)
    
    for i in tqdm(range(len(addresses))):
        address = addresses[i]
        pc = pcs[i]
        if address in cache['Address']: # If address is in cache increment the num_hit
            num_hit += 1
            continue
            
        elif len(cache) < cache_size: # If address is not in cache and the cache is not full yet the increment the num_miss 

            #cache.add(address)
            cache = cache.append({'Address': address, 'Frequency': float('inf'), 'Recency': float('inf')}, ignore_index = True)

            num_miss += 1
            total_miss+=1
            miss_addresses.append(address)
            pc_misses.append(pc)
            if num_miss == miss_history_length: # Calculate freq and rec for every 10 misses
                num_miss = 0
                # _,_,freq,rec = encoder(addresses=miss_addresses[-misses_window:],pcs=pc_misses[-misses_window:],predictor = decoder)
                _,_,freq,rec = encoder(addresses=miss_addresses[-misses_window:],pcs=pc_misses[-misses_window:],predictor = decoder)

                """
                Update freqs /recs values for addresses in cache
                write functions to update recency / freq for addresses in cache
                """
                cache['Frequency'], cache['Recency'] = freq, rec 
        else:
            num_miss += 1
            total_miss+=1
            miss_addresses.append(address)
            pc_misses.append(pc)
            if num_miss == miss_history_length: # Calculate freq and rec for every 10 misses
                num_miss = 0
                _,_,freq,rec = encoder(addresses=miss_addresses[-misses_window:],pcs=pc_misses[-misses_window:],predictor = decoder)

                """
                Update freqs /recs values for addresses in cache
                write functions to update recency / freq for addresses in cache
                """
                cache['Frequency'], cache['Recency'] = freq, rec

            # cache = eviction_policy(cache=cache,leCaR=leCaR) 
            # cache.add(address)

            """
            Perform eviction using lecar
            Returns :is_miss (bool), updated cache (containing admitted address) and evicted address 
            """

            lecar.cache = cache # update lecars self.cache
            is_miss, evicted_address, cache = lecar.run(address)
    
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