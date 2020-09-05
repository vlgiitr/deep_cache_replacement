import numpy as np
from collections import deque, defaultdict
from tqdm import tqdm

def encoder(addresses,predictor):
    """
    model that predicts the future and recency of every address in the cache
    Inputs: List of addresses in a cache
    Outputs: recencies and frequencies list for the addresses  
    """
    recencies, frequencies = predictor(addresses)
    return recencies,frequencies

def eviction_policy(cache, misses_list, frequencies, recencies,leCaR):
    """
    evcition carried out
    Inputs: cache, misses_list(the [miss_history_length] most recent ones), frequencies, recencies
    Outputs: removes a address from the cache and returns the new cache 
    """
    cache = leCaR(cache, misses_list, frequencies, recencies)
    return cache


def test_model(cache_size, addresses, pcs, misses_window, miss_history_length,predictor,leCaR):
    
    cache = set()
    num_hit, num_miss = 0, 0
    miss_addresses = []
    pc_misses = []
    recencies = None
    frequencies = None
    
    for i in tqdm(range(len(addresses))):
        address = addresses[i]
        pc = pcs[i]
        if address in cache:
            num_hit += 1
            continue
            
        elif len(cache) < cache_size:
            cache.add(address)
            num_miss += 1
            miss_addresses.append(address)
            pc_misses.append(pc)
            if num_miss == miss_history_length:
                num_miss = 0
                recencies,frequencies = encoder(list(cache),predictor)
        else:
            num_miss += 1
            miss_addresses.append(address)
            pc_misses.append(pc)
            if num_miss == miss_history_length:
                num_miss = 0
                recencies,frequencies = encoder(list(cache),predictor)

            cache = eviction_policy(cache=cache,misses_list=miss_addresses[-misses_window:],
                                    frequencies=frequencies,recencies=recencies,leCaR=leCaR)
            cache.add(address)
    
    hitrate = num_hit / (num_hit + num_miss)
    print('---------------------------')
    print('Testing Complete')
    print('HitRate: {}'.format(hitrate))
    print('---------------------------')

if __name__=='__main__':
    # Call test_model here   