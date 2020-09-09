import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import cache_lecar


### Define Custom Policy Functions
def embed_policy(address) :
    """
    returns embedding for single addresss

    input : addres
    output: embedding
    """
    return f'emb{address}'


# Create Class for cache
class Cache() :
    def __init__(self, cache_size, embed_policy) :
        self.cache_size = cache_size
        vals = np.stack((np.random.randint(10,30,5), np.random.randint(0,10,5), np.random.randint(0,10,5)), 0)
        vals = np.transpose(vals)
        self.data = pd.DataFrame(vals, columns = ['Address','Frequency','Recency'])
        #self.data['Embeddings'] = [embed_policy(x) for x in self.data['Address']]
        self.lecar = cache_lecar.LeCaR(self.cache_size, self.data)
        self.num_hit, self.num_miss = 0, 0
       
       
    def __str__(self) :
        return self.data.to_string()
    
    def admit(self, req) :
        """
        evicts element using evict_policy(), 
        admits element at the removed index if allowed by admit_policy(), 
        updates embeddings by update_embed() 

        input : request (address)
        output: NaN , cache is modified
        """

        miss, evicted, self.data = self.lecar.run(req)
        if miss :
            self.num_miss += 1
        else : 
            self.num_hits += 1
        # print(f'EVICTED : {evicted}\n') 
        #self.update_embed(idx, embed_policy)
    
    def update_embed(self, idx, embed_policy) :
        """
        updates embeddings of all elements in cache
        """
        self.data['Embeddings'][idx] = embed_policy(self.data['Address'][idx])




if __name__ == "__main__" :
    cache_size = 5

    stream = np.random.randint(10,30,100) # input srequest stream
    
    cache = Cache(cache_size, embed_policy) # create instance for Cache class
    print(stream)
    print(cache)

    # freqs, recs = cache.data['Frequency'], cache.data['Recency']

    for r in tqdm(range(len(stream))) :
        cache.admit(stream[r]) #evicts and admits request
        # print(cache)
    print(cache)
    hitrate = cache.num_hit / (cache.num_hit + cache.num_miss)
    print('---------------------------')
    print('Testing Complete')
    print('HitRate: {}'.format(hitrate))
    print('---------------------------')
