
import pandas as pd
import numpy as np
import random


### Define Custom Policy Functions
def admit_policy(request) :
    """
    admit policy
    Here: admits if address is positive
    """
    if request > 0 :
        return True

def evict_policy(cache) :
    """
    removes and prints random element form cache

    input : cache (array, df)
    output: evicted element index, modified cache addresses
    """
    idx = np.random.randint(0, len(cache))
    removed = cache[idx]
    cache = cache.drop([idx]).astype(int)
    print(f'Removed : {removed}\n\n')
    return idx, cache

def embed_policy(address) :
    """
    returns embedding for single addresss

    input : addres
    output: embedding
    """
    return f'emb{address}'


# Create Class for cache
class Cache() :
    def __init__(self, cache_size, admit_policy, evict_policy, embed_policy) :
        self.cache_size = cache_size
        self.addresses = np.random.randint(1,10,(self.cache_size)) # enter addresses
        self.pc_addresses = np.random.randint(1,10,(self.cache_size)) # enter pc_address
        self.embeddings = [embed_policy(x) for x in self.addresses] # get embeddings
        self.data = pd.DataFrame({'Address': self.addresses, 'PC-Address' : self.pc_addresses, 'Embeddings' : self.embeddings})
       
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
        if admit_policy(req) : # if allowed by admit policy
            idx, self.data['Address'] = self.evict() # perform eviction
            self.data['Address'][idx] = req # admit request
            self.update_embed(idx, embed_policy) # upate embeddings for admitted element

    def evict(self) :
        """
        calls evict_policy() which returns thrown index and updated cache and prints evicted element
        """
        idx, cache = evict_policy(self.data['Address'])
        return idx, cache
    
    def update_embed(self, idx, embed_policy) :
        """
        updates embeddings of all elements in cache
        """
        self.data['Embeddings'][idx] = embed_policy(self.data['Address'][idx])




if __name__ == "__main__" :

    stream = np.random.randint(-20,100,10) # input srequest stream

    cache = Cache(5, admit_policy, evict_policy, embed_policy) # create instance for Cache class

    print(stream)
    print(cache)

    for r in stream :
        cache.admit(r) #evicts and admits request
        print(cache)

