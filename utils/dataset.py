import os
from glob import glob
import numpy as np
import pandas as pd
from collections import Counter, deque, defaultdict
from tqdm import tqdm as tqdm 


maxpos = 1000000000000

num_params = 3

cache_size = 50 # default cache size
sampling_freq = 1 # number of samples skipped
eviction = int(0.7 * cache_size)  


lruCorrect = 0
lruIncorrect = 0

lfuCorrect = 0
lfuIncorrect = 0


X = np.array([], dtype=np.int64).reshape(0,num_params)
Y = np.array([], dtype=np.int64).reshape(0,1)


def get_complete_data_padded():
    PATH = "csv_data/cse240_project_ucsd"
    EXT = "*.csv"
    all_csv_files = [file
                    for path, subdir, files in os.walk(PATH)
                    for file in glob(os.path.join(path, EXT))]
    len_list = []
    
    for path in all_csv_files:
        df = pd.read_csv(path)
        len_list.append(df.shape[0])

    max_len = np.max(len_list)
    num_datasets = len(len_list)

    dataset = np.zeros((num_datasets,max_len,2), dtype = "int")
    print(dataset.shape)


    for i,path in enumerate(all_csv_files):
        df = pd.read_csv(path)
        pc = (df['PC'].apply(int, base=16)).zfill(35)
        address = df['Address'].apply(int, base=16)
        dataset[i,:len(address),0]  = pc
        dataset[i,:len(address),0]  = address
    
    return dataset

def lruPredict(C,LRUQ,Y_OPT):
    global lruCorrect, lruIncorrect
    Y_current = []
    KV = defaultdict(int)
    for i in range(len(LRUQ)):
        KV[LRUQ[i]] = len(LRUQ) - i
    KV_sorted = Counter(KV)
    evict_dict = dict(KV_sorted.most_common(eviction))
    for e in C:
        if e in evict_dict:
            Y_current.append(1)
        else:
            Y_current.append(0)
    for i in range(len(Y_current)):
        if Y_current[i] is Y_OPT[i]:
            lruCorrect+=1
        else:
            lruIncorrect+=1
    return Y_current

# returns sequence of blocks in prioirty order

def Y_getBlockSeq(Y_pred_prob):
    x = []
    for i in range(len(Y_pred_prob)):
        x.append(Y_pred_prob[i][0])
    x = np.array(x)
    idx = np.argsort(x)
    idx = idx[:eviction]
    return idx


def Y_getMinPredict(Y_pred_prob):
    x = []
    for i in range(len(Y_pred_prob)):
        x.append(Y_pred_prob[i][0])
    x = np.array(x)
    idx = np.argpartition(x, eviction)
    
    Y_pred = np.zeros(len(Y_pred_prob), dtype=int)
    for i in range(eviction):
        Y_pred[idx[i]] = 1
    assert(Counter(Y_pred)[1] == eviction)
    return Y_pred


def lfuPredict(C,LFUDict,Y_OPT):
    global lfuCorrect, lfuIncorrect
    Y_current = []
    KV = defaultdict()
    for e in C:
        KV[e] = LFUDict[e]
    KV_sorted = Counter(KV)
    evict_dict = dict(KV_sorted.most_common(eviction))
    for e in C:
        if e in evict_dict:
            Y_current.append(1)
        else:
            Y_current.append(0)
    for i in range(len(Y_current)):
        if Y_current[i] is Y_OPT[i]:
            lfuCorrect+=1
        else:
            lfuIncorrect+=1
    return Y_current

# return "eviction" blocks that are being accessed furthest
# from the cache that was sent to us.

def getY(C,D):
    assert(len(C) == len(D))
    Y_current = []
    KV_sorted = Counter(D)
    evict_dict = dict(KV_sorted.most_common(eviction))

    assert(len(evict_dict) == eviction)
    all_vals = evict_dict.values()
    for e in C:
        if e in evict_dict.values():
            Y_current.append(1)
        else:
            Y_current.append(0)
    #print (Y_current.count(1))
    assert(Y_current.count(1) == eviction)
    assert((set(all_vals)).issubset(set(C)))
    return Y_current

def getLFURow(LFUDict, C):
    x_lfurow = []
    for e in C:
        x_lfurow.append(LFUDict[e])
    norm = x_lfurow / np.linalg.norm(x_lfurow)
    return norm
    
def getLRURow(LRUQ, C):
    x_lrurow = []
    KV = defaultdict(int)
    for i in range(len(LRUQ)):
        KV[LRUQ[i]] = i
    for e in C:
        x_lrurow.append(KV[e])
    norm = x_lrurow / np.linalg.norm(x_lrurow)
    return norm

def normalize(feature, blocks):
    x_feature = []
    for i in range(len(blocks)):
        x_feature.append(feature[blocks[i]])
    return x_feature / np.linalg.norm(x_feature)

def getX(LRUQ, LFUDict, C):
#def getX(LRUQ, LFUDict, C, CacheTS, CachePID):   
    X_lfurow = getLFURow(LFUDict, C)
    X_lrurow = getLRURow(LRUQ, C)
    X_bno    = C / np.linalg.norm(C)
#     X_ts     = normalize(CacheTS, C)
#     X_pid    = normalize(CachePID, C)
    return (np.column_stack((X_lfurow, X_lrurow, X_bno)))
    
    
def populateData(LFUDict, LRUQ, C, D):
#def populateData(LFUDict, LRUQ, C, D, CacheTS, CachePID):
    global X,Y
    C = list(C)
    Y_current = getY(C, D)
    #X_current = getX(LRUQ, LFUDict, C, CacheTS, CachePID)
    X_current = getX(LRUQ, LFUDict, C)

    Y = np.append(Y, Y_current)
    X = np.concatenate((X,X_current))
    assert(Y_current.count(1) == eviction)
    return Y_current

def belady_opt(blocktrace, frame):
    global maxpos
    
    OPT = defaultdict(deque)
    D = defaultdict(int)
    LFUDict = defaultdict(int)
    LRUQ = []
    #CacheTS = defaultdict(int)
    #CachePID = defaultdict(int)

    for i, block in enumerate(tqdm(blocktrace, desc="OPT: building index")):
        OPT[block].append(i)

    hit, miss = 0, 0

    C = []
    #count=0
    #seq_number = 0
    for seq_number, block in enumerate(tqdm(blocktrace, desc="OPT")):
#    for block in blocktrace: 
        LFUDict[block] +=1

        if len(OPT[block]) is not 0 and OPT[block][0] == seq_number:
            OPT[block].popleft()
        #CacheTS [blocktrace[seq_number]] = timestamp[seq_number]
        #CachePID [blocktrace[seq_number]] = pid[seq_number]
        if block in C:
            hit+=1
            LRUQ.remove(block)
            LRUQ.append(block)
            assert( seq_number in D)
            del D[seq_number]
            if len(OPT[block]) is not 0:
                D[OPT[block][0]] = block
                OPT[block].popleft()
            else:
                D[maxpos] = block
                maxpos -= 1
        else:
            miss+=1
            if len(C) == frame:
                assert(len(D) == frame)
                evictpos = max(D)
                
                if (seq_number % sampling_freq +1 == sampling_freq):
                    #Y_OPT = populateData(LFUDict, LRUQ, C, D, CacheTS, CachePID)
                    Y_OPT = populateData(LFUDict, LRUQ, C, D)
                    lruPredict(C,LRUQ,Y_OPT)
                    lfuPredict(C,LFUDict,Y_OPT)
                
                C[C.index(D[evictpos])] = block
                LRUQ.remove(D[evictpos])
                #del CacheTS [D[evictpos]]
                #del CachePID [D[evictpos]]
                del D[evictpos]
            else:
                C.append(block)
                
            if len(OPT[block]) is not 0:
                D[OPT[block][0]] = block
                OPT[block].popleft()
            else:
                D[maxpos] = block
                maxpos -= 1
            LRUQ.append(block)


    hitrate = hit / (hit + miss)
    #print(hitrate)
    return hitrate, X,Y