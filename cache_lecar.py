from utils.lib.dequedict import DequeDict
from utils.lib.heapdict import HeapDict
import time
import numpy as np
import pandas as pd

class LeCaR:
    # kwargs: We're using keyword arguments so that they can be passed down as
    #         needed. Please note that cache_size is a required argument and not
    #         optional like all the kwargs are.
    def __init__(self, cache_size, cache,  **kwargs):
        # Randomness and Time
        np.random.seed(123)
        self.time = time.time()
        self.cache = cache

        # Cache
        self.cache_size = cache_size
        self.lru = DequeDict()
        self.lfu = HeapDict()

        # Histories
        self.history_size = cache_size
        self.lru_hist = pd.DataFrame({'Address': np.ones(self.cache_size), 'Time': np.zeros(self.cache_size)})
        self.lfu_hist = pd.DataFrame({'Address': np.ones(self.cache_size), 'Time': np.zeros(self.cache_size)})

        # Decision Weights Initilized
        self.initial_weight = 0.5

        # Fixed Learning Rate
        self.learning_rate = 0.45

        # Fixed Discount Rate
        self.discount_rate = 0.005**(1 / self.cache_size)

        # Decision Weights
        self.W = np.array([self.initial_weight, 1 - self.initial_weight],
                          dtype=np.float32)

    # True if oblock is in cache (which LRU can represent)
    def __contains__(self, oblock):
        return oblock in self.lru

    def cacheFull(self):
        return len(self.lru) == self.cache_size

    # Add Entry to cache with given frequency
    def addToCache(self, oblock):
        # self.cache['Address'][idx] = oblock
        self. cache = self.cache.append({'Address': oblock, 'Frequency': np.random.randint(0,10), 'Recency': np.random.randint(0,10)}, ignore_index=True)

    
    def addToHistory(self, x, policy):
        """
        Add Entry to history dictated by policy
        policy: 0, Add Entry to LRU History
                1, Add Entry to LFU History
        """
        # Use reference to policy_history to reduce redundant code
        policy_history = None
        if policy == 0:
            policy_history = self.lru_hist    # Initialize policy histriy to original history
        elif policy == 1:
            policy_history = self.lfu_hist
        elif policy == -1:
            return

        # Evict from history is it is full
        if len(policy_history) == self.history_size:
            idx, evicted = self.get_first(policy_history)
            # del policy_history[evicted.oblock]
            policy_history = policy_history.drop([idx]).reset_index(drop = True) #drop that entry
        policy_history = policy_history.append({'Address': x, 'Time': time.time()}, ignore_index=True)

        if policy == 0:
            self.lru_hist = policy_history    # Initialize policy histriy to original history
        elif policy == 1:
            self.lfu_hist = policy_history
    
    def get_first(self, history) :
        """
        Returns id and address of first entered element
        """
        idx =  history[['Time']].idxmin()[0]
        return idx, history['Address'][idx]

    def getLRU(self):
        """
        Get the LRU item in the given frequency preds
        """
        idx = self.cache[['Recency']].idxmin()[0]
        return self.cache['Address'][idx]

    def getLFU(self):
        """
        Get index LRU item in the given frequency preds
        return correspoding address
        """
        idx = self.cache[['Frequency']].idxmin()[0] 
        return self.cache['Address'][idx]
    
    def getChoice(self):
        """
        Get the random eviction choice based on current weights

        Returns 0 / 1
        """
        return 0 if np.random.rand() < self.W[0] else 1

    # Evict an entry from cache
    # policy : 0 --> lru, 1 --> lfu,  -1 --> same
    def evict(self):
        """
        Gets lru = 0 / lfu = 1 element
        Chooses policy
        Adds to resp policy history

        input :
        output: returns address of evicted and policy bool
        """

        lru = self.getLRU() # returns address to remove from cache
        lfu = self.getLFU()

        evicted = lru # removed address
        policy = self.getChoice()

        # print('POLICY !!!! ', policy)
        # Since we're using Entry references, we use is to check
        # that the LRU and LFU Entries are the same Entry
        if lru == lfu:
            evicted, policy = lru, -1
        elif policy == 0:
            evicted = lru
        else:
            evicted = lfu

        # Drop the evicited item from cache
        idx = self.cache[self.cache['Address'] == evicted].index.values[0]
        self.cache = self.cache.drop([idx]).reset_index(drop = True)

        self.addToHistory(evicted, policy) # add address to history of policy
        return evicted, policy

    # Adjust the weights based on the given rewards for LRU and LFU
    def adjustWeights(self, rewardLRU, rewardLFU):
        reward = np.array([rewardLRU, rewardLFU], dtype=np.float32)
        self.W = self.W * np.exp(self.learning_rate * reward)
        self.W = self.W / np.sum(self.W)

        if self.W[0] >= 0.99:
            self.W = np.array([0.99, 0.01], dtype=np.float32)
        elif self.W[1] >= 0.99:
            self.W = np.array([0.01, 0.99], dtype=np.float32)

    # Cache Miss
    def miss(self, oblock):
        """
        Check if req belongs to policy hist and update resp policy
        Perform eviction if cache is full
        Add to cache

        evict() should return index of which to remove
        add should add to that cache
        """
        evicted = None


        # if in hist --> cal reward and then remove 
        if oblock in self.lru_hist['Address']:
            idx = self.lru_hist[self.lru_hist['Adddress'] == oblock].index.values[0]
            reward_lru = -(self.discount_rate**(self.time - self.lru_hist['Time'][idx]))
            self.lru_hist = self.lru_hist.drop([idx])
            self.adjustWeights(reward_lru, 0)
        elif oblock in self.lfu_hist:
            idx = self.lfu_hist[self.lfu_hist['Adddress'] == oblock].index.values[0]
            reward_lfu = -(self.discount_rate**(self.time - self.lfu_hist['Time'][idx]))
            self.lfu_hist = self.lfu_hist.drop([idx])
            self.adjustWeights(0, reward_lfu)

        # If the cache is full, evict
        if len(self.cache) == self.cache_size:
            evicted, policy = self.evict()

        self.addToCache(oblock)
        return evicted

    # Process and access request for the given oblock    
    def run(self, oblock) :
        miss = True
        evicted = None

        self.time += 1

        if oblock in self.cache['Address']:
            miss = False
        else:
            evicted = self.miss(oblock)

        return miss, evicted, self.cache
