import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv
import os
from glob import glob
import numpy as np
import pandas as pd
from collections import Counter, deque, defaultdict

def hex_to_bin(string):
    scale = 16
    res = bin(int(string, scale)).zfill(32)
    return str(res) 

def get_data():
    """
    Extract data from 
    """
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
        addresses = []
        pcs = []
        count = 0
        with open(path,'r') as file:
            reader = csv.reader(file)
            for row in reader:
                count+=1
                if count == 1:
                    continue
                else:
                    pcs.append(hex_to_bin(row[1]))
                    addresses.append(hex_to_bin(row[2]))
        dataset[i,:len(pcs),0] = pcs
        dataset[i,:len(addresses),1] = addresses
    
    return dataset

class Token:
<<<<<<< HEAD
    first_address_set = None
    second_address_set = None
    third_address_set = None
    fourth_address_set = None
    first_pc_set = None
    second_pc_set = None
    third_pc_set = None
    fourth_pc_set = None
    
    first_address_ix = None
    second_address_ix = None
    third_address_ix = None
    fourth_address_ix = None
    first_pc_ix = None
    second_pc_ix = None
    third_pc_ix = None
    fourth_pc_ix = None

    def __init__(self):
        self.first_address_set = set()
        self.second_address_set = set()
        self.third_address_set = set()
        self.fourth_address_set = set()
        self.first_pc_set = set()
        self.second_pc_set = set()
        self.third_pc_set = set()
        self.fourth_pc_set = set()

        self.first_address_ix = {}
        self.second_address_ix = {}
        self.third_address_ix = {}
        self.fourth_address_ix = {}
        self.first_pc_ix = {}
        self.second_pc_ix = {}
        self.third_pc_ix = {}
        self.fourth_pc_ix = {}

    def pc_tokens(self,pc):
        pc_bytes = [pc[:8], pc[8:16], pc[16:24], pc[24:32]]
        self.first_pc_set.add(pc_bytes[0])
        self.second_pc_set.add(pc_bytes[1])
        self.third_pc_set.add(pc_bytes[2])
        self.fourth_pc_set.add(pc_bytes[3])
        for pc in pc_bytes:
            if pc[0] not in self.first_pc_ix.keys():
                self.first_pc_ix[pc[0]] = len(self.first_pc_ix)        
            if pc[1] not in self.second_pc_ix.keys():
                self.second_pc_ix[pc[1]] = len(self.second_pc_ix)
            if pc[2] not in self.third_pc_ix.keys():
                self.third_pc_ix[pc[2]] = len(self.third_pc_ix)        
            if pc[3] not in self.fourth_pc_ix.keys():
                self.fourth_pc_ix[pc[3]] = len(self.fourth_pc_ix)  

    def address_tokens(self,address):
        address_bytes = [address[:8], address[8:16], address[16:24], address[24:32]]
        self.first_address_set.append(address_bytes[0])
        self.second_address_set.append(address_bytes[1])
        self.third_address_set.append(address_bytes[2])
        self.fourth_address_set.append(address_bytes[3])
        for address in address_bytes:
            if address[0] not in self.first_address_ix.keys():
                self.first_address_ix[address[0]] = len(self.first_address_ix)        
            if address[1] not in self.second_address_ix.keys():
                self.second_address_ix[address[1]] = len(self.second_address_ix)
            if address[2] not in self.third_address_ix.keys():
                self.third_address_ix[address[2]] = len(self.third_address_ix)        
            if address[3] not in self.fourth_address_ix.keys():
                self.fourth_address_ix[address[3]] = len(self.fourth_address_ix)     

class ByteEncoder(nn.Module):
    first_address_embedding = None
    second_address_embedding = None
    third_address_embedding = None
    fourth_address_embedding = None
    first_pc_embedding = None
    second_pc_embedding = None
    third_pc_embedding = None
    fourth_pc_embedding = None
    
    linear_first_address_1 = None
    linear_second_address_1 = None
    linear_third_address_1 = None
    linear_fourth_address_1 = None

    linear_first_address_2 = None
    linear_second_address_2 = None
    linear_third_address_2 = None
    linear_fourth_address_2 = None
=======
    address_sets = None
    pc_sets = None
    
    address_ixs = None
    pc_ixs = None

    def __init__(self):
        self.address_sets = [set()]*4
        self.pc_sets = [set()]*4

        self.address_ixs = [{}]*4
        self.pc_ixs = [{}]*4

    def pc_tokens(self,pc):
        pc_bytes = [pc[:8], pc[8:16], pc[16:24], pc[24:32]]
        self.pc_sets[0].add(pc_bytes[0])
        self.pc_sets[1].add(pc_bytes[1])
        self.pc_sets[2].add(pc_bytes[2])
        self.pc_sets[3].add(pc_bytes[3])
        for pc in pc_bytes:
            if pc[0] not in self.pc_ixs[0].keys():
                self.pc_ixs[0][pc[0]] = len(self.pc_ixs[0])        
            if pc[1] not in self.pc_ixs[1].keys():
                self.pc_ixs[1][pc[1]] = len(self.pc_ixs[1])
            if pc[2] not in self.pc_ixs[2].keys():
                self.pc_ixs[2][pc[2]] = len(self.pc_ixs[2])        
            if pc[3] not in self.pc_ixs[3].keys():
                self.pc_ixs[3][pc[3]] = len(self.pc_ixs[3])  

    def address_tokens(self,address):
        address_bytes = [address[:8], address[8:16], address[16:24], address[24:32]]
        self.address_sets[0].add(address_bytes[0])
        self.address_sets[1].add(address_bytes[1])
        self.address_sets[2].add(address_bytes[2])
        self.address_sets[3].add(address_bytes[3])
        for address in address_bytes:
            if address[0] not in self.address_ixs[0].keys():
                self.address_ixs[0][address[0]] = len(self.address_ixs[0])        
            if address[1] not in self.address_ixs[1].keys():
                self.address_ixs[1][address[1]] = len(self.address_ixs[1])
            if address[2] not in self.address_ixs[2].keys():
                self.address_ixs[2][address[2]] = len(self.address_ixs[2])        
            if address[3] not in self.address_ixs[3].keys():
                self.address_ixs[3][address[3]] = len(self.address_ixs[3])     

class ByteEncoder(nn.Module):
    address_embeddings = None
    pc_embeddings = None
    
    linears_address_1 = None
    linears_address_2 = None

    linears_pc_1 = None
    linears_pc_2 = None

    ##Alternate path
    # self.linear_address_2 = None
    # self.linear_ps_2 = None
>>>>>>> 1947fc61fb955b099b0a49b07b573b5c57cd2127

    embedding_size = 32
    token = None

    def __init__(self,pcs,addresses):
        super(ByteEncoder,self).__init__()

        self.token = Token()

        for pc,address in zip(pc,address):
            self.token.pc_tokens(pc = pc)
            self.token.address_tokens(address = address)

<<<<<<< HEAD
        self.first_address_embedding = nn.Embedding(256,self.embedding_size)
        self.second_address_embedding = nn.Embedding(256,self.embedding_size)
        self.third_address_embedding = nn.Embedding(256,self.embedding_size)
        self.fourth_address_embedding = nn.Embedding(256,self.embedding_size)
        
        self.first_pc_embedding = nn.Embedding(256,self.embedding_size)
        self.second_pc_embedding = nn.Embedding(256,self.embedding_size)
        self.third_pc_embedding = nn.Embedding(256,self.embedding_size)
        self.fourth_pc_embedding = nn.Embedding(256,self.embedding_size)

        self.linear_first_address_1 = nn.Linear(self.embedding_size,8)
        self.linear_second_address_1 = nn.Linear(self.embedding_size,8)
        self.linear_third_address_1 = nn.Linear(self.embedding_size,8)
        self.linear_fourth_address_1 = nn.Linear(self.embedding_size,8)

        self.linear_first_address_2 = nn.Linear(8,2)
        self.linear_second_address_2 = nn.Linear(8,2)
        self.linear_third_address_2 = nn.Linear(8,2)
        self.linear_fourth_address_2 = nn.Linear(8,2)

        ##Alternate path
        # self.linear_address_2 = nn.Linear(8,4)
        # self.linear_ps_2 = nn.Linear(8,4)

        self.linear_first_pc_1 = nn.Linear(self.embedding_size,8)
        self.linear_second_pc_1 = nn.Linear(self.embedding_size,8)
        self.linear_third_pc_1 = nn.Linear(self.embedding_size,8)
        self.linear_fourth_pc_1 = nn.Linear(self.embedding_size,8)

        self.linear_first_pc_2 = nn.Linear(8,2)
        self.linear_second_pc_2 = nn.Linear(8,2)
        self.linear_third_pc_2 = nn.Linear(8,2)
        self.linear_fourth_pc_2 = nn.Linear(8,2)         

    def forward(self, inputs):
        
        first_address_input = torch.tensor([self.token.first_address_ix[ad[0:8]] for ad in inputs[1]], dtype=torch.long)
        second_address_input = torch.tensor([self.token.second_address_ix[ad[8:16]] for ad in inputs[1]], dtype=torch.long)
        third_address_input = torch.tensor([self.token.third_address_ix[ad[16:24]] for ad in inputs[1]], dtype=torch.long)
        fourth_address_input = torch.tensor([self.token.fourth_address_ix[ad[24:32]] for ad in inputs[1]], dtype=torch.long)
        
        first_pc_input = torch.tensor([self.token.first_pc_ix[pc[0:8]] for pc in inputs[0]], dtype=torch.long)
        second_pc_input = torch.tensor([self.token.second_pc_ix[pc[8:16]] for pc in inputs[0]], dtype=torch.long)
        third_pc_input = torch.tensor([self.token.third_pc_ix[pc[16:24]] for pc in inputs[0]], dtype=torch.long)
        fourth_pc_input = torch.tensor([self.token.fourth_pc_ix[pc[24:32]] for pc in inputs[0]], dtype=torch.long)

        first_address_embed = self.first_address_embedding(first_address_input)
        second_address_embed = self.second_address_embedding(second_address_input)
        third_address_embed = self.third_address_embedding(third_address_input)
        fourth_address_embed = self.fourth_address_embedding(fourth_address_input)

        first_pc_embed = self.first_pc_embedding(first_pc_input)
        second_pc_embed = self.second_pc_embedding(second_pc_input)
        third_pc_embed = self.third_pc_embedding(third_pc_input)
        fourth_pc_embed = self.fourth_pc_embedding(fourth_pc_input)

        ad_first_out_1 = F.relu(self.linear_first_address_1(first_address_embed))
        ad_second_out_1 = F.relu(self.linear_second_address_1(second_address_embed))
        ad_third_out_1 = F.relu(self.linear_third_address_1(third_address_embed))
        ad_fourth_out_1 = F.relu(self.linear_fourth_address_1(fourth_address_embed))

        pc_first_out_1 = F.relu(self.linear_first_pc_1(first_pc_embed))
        pc_second_out_1 = F.relu(self.linear_first_pc_1(second_pc_embed))
        pc_third_out_1 = F.relu(self.linear_first_pc_1(third_pc_embed))
        pc_fourth_out_1 = F.relu(self.linear_first_pc_1(fourth_pc_embed))

        ad_first_out_2 = F.relu(self.linear_first_address_2(ad_first_out_1))
        ad_second_out_2 = F.relu(self.linear_first_address_2(ad_second_out_1))
        ad_third_out_2 = F.relu(self.linear_first_address_2(ad_third_out_1))
        ad_fourth_out_2 = F.relu(self.linear_first_address_2(ad_fourth_out_1))

        pc_first_out_2 = F.relu(self.linear_first_pc_2(pc_first_out_1))
        pc_second_out_2 = F.relu(self.linear_first_pc_2(pc_second_out_1))
        pc_third_out_2 = F.relu(self.linear_first_pc_2(pc_third_out_1))
        pc_fourth_out_2 = F.relu(self.linear_first_pc_2(pc_fourth_out_1))

        ad_out = torch.cat(ad_first_out_2, ad_second_out_2, ad_third_out_2, ad_fourth_out_2,axis=0)
        pc_out = torch.cat(pc_first_out_2, pc_second_out_2, pc_third_out_2, pc_fourth_out_2,axis=0)

        ## Alternate path
        # ad_out_2 = torch.mul(ad_first_out_1,ad_second_out_1,ad_third_out_1,ad_fourth_out_1)
        # ps_out_2 = torch.mul(ps_first_out_1,ps_second_out_1,ps_third_out_1,ps_fourth_out_1)
=======
        self.address_embeddings = [nn.Embedding(256,self.embedding_size)]*4
        self.pc_embeddings = [nn.Embedding(256,self.embedding_size)]*4

        self.linears_address_1 = [nn.Linear(self.embedding_size,8)]*4
        self.linears_address_2 = [nn.Linear(8,2)]*4

        self.linears_pc_1 = [nn.Linear(self.embedding_size,8)]*4
        self.linears_pc_2 = [nn.Linear(8,2)]*4 

        ##Alternate path
        # self.linear_address_2 = nn.Linear(8,4)
        # self.linear_ps_2 = nn.Linear(8,4)       

    def forward(self, inputs):
        
        address_inputs = []
        address_inputs.append(torch.tensor([self.token.first_address_ix[ad[0:8]] for ad in inputs[1]], dtype=torch.long))
        address_inputs.append(torch.tensor([self.token.second_address_ix[ad[8:16]] for ad in inputs[1]], dtype=torch.long))
        address_inputs.append(torch.tensor([self.token.third_address_ix[ad[16:24]] for ad in inputs[1]], dtype=torch.long))
        address_inputs.append(torch.tensor([self.token.fourth_address_ix[ad[24:32]] for ad in inputs[1]], dtype=torch.long))
        
        pc_inputs = []
        pc_inputs.append(torch.tensor([self.token.first_pc_ix[pc[0:8]] for pc in inputs[0]], dtype=torch.long))
        pc_inputs.append(torch.tensor([self.token.second_pc_ix[pc[8:16]] for pc in inputs[0]], dtype=torch.long))
        pc_inputs.append(torch.tensor([self.token.third_pc_ix[pc[16:24]] for pc in inputs[0]], dtype=torch.long))
        pc_inputs.append(torch.tensor([self.token.fourth_pc_ix[pc[24:32]] for pc in inputs[0]], dtype=torch.long))

        address_embeds = []
        address_embeds.append(self.address_embeddings[0](address_inputs[0]))
        address_embeds.append(self.address_embeddings[1](address_inputs[1]))
        address_embeds.append(self.address_embeddings[2](address_inputs[2]))
        address_embeds.append(self.address_embeddings[3](address_inputs[3]))

        pc_embeds = []
        pc_embeds.append(self.pc_embeddings[0](pc_inputs[0]))
        pc_embeds.append(self.pc_embeddings[1](pc_inputs[1]))
        pc_embeds.append(self.pc_embeddings[2](pc_inputs[2]))
        pc_embeds.append(self.pc_embeddings[3](pc_inputs[3]))

        address_outs_1 = []
        address_outs_1.append(F.relu(self.linears_address_1[0](address_embeds[0])))
        address_outs_1.append(F.relu(self.linears_address_1[1](address_embeds[1])))
        address_outs_1.append(F.relu(self.linears_address_1[2](address_embeds[2])))
        address_outs_1.append(F.relu(self.linears_address_1[3](address_embeds[3])))

        pc_outs_1 = []
        pc_outs_1.append(F.relu(self.linears_pc_1[0](pc_embeds[0])))
        pc_outs_1.append(F.relu(self.linears_pc_1[1](pc_embeds[1])))
        pc_outs_1.append(F.relu(self.linears_pc_1[2](pc_embeds[2])))
        pc_outs_1.append(F.relu(self.linears_pc_1[3](pc_embeds[3])))

        address_outs_2 = []
        address_outs_2.append(F.relu(self.linears_address_2[0](address_outs_1[0])))
        address_outs_2.append(F.relu(self.linears_address_2[1](address_outs_1[1])))
        address_outs_2.append(F.relu(self.linears_address_2[2](address_outs_1[2])))
        address_outs_2.append(F.relu(self.linears_address_2[3](address_outs_1[3])))

        pc_outs_2 = []
        pc_outs_2.append(F.relu(self.linears_pc_2[0](pc_outs_1[0])))
        pc_outs_2.append(F.relu(self.linears_pc_2[1](pc_outs_1[1])))
        pc_outs_2.append(F.relu(self.linears_pc_2[2](pc_outs_1[2])))
        pc_outs_2.append(F.relu(self.linears_pc_2[3](pc_outs_1[3])))

        ad_out = torch.cat(address_outs_2[0], address_outs_2[1], address_outs_2[2], address_outs_2[3],axis=0)
        pc_out = torch.cat(pc_outs_2[0], pc_outs_2[1], pc_outs_2[2], pc_outs_2[3],axis=0)

        ## Alternate path
        # ad_out_2 = torch.mul(address_outs_1[0], address_outs_1[1], address_outs_1[2], address_outs_1[3])
        # ps_out_2 = torch.mul(pc_outs_1[0], pc_outs_1[1], pc_outs_1[2], pc_outs_1[3])
>>>>>>> 1947fc61fb955b099b0a49b07b573b5c57cd2127
        
        # ad_out = F.relu(self.linear_address_2(ad_out_2))
        # ps_out = F.relu(self.linear_ps_2(ps_out_2))

        lstm_input = torch.cat(ad_out,pc_out)

        return lstm_input
