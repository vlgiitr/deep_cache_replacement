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
import argparse
from tqdm import tqdm

def hex_to_bin(string):
    scale = 16
    res = bin(int(string, scale)).zfill(32)
    return str(res) 

def get_data(p):
    """
    Extract data from 
    """
    PATH = p
    all_csvs = []
    len_list = []
    files_list = [files for path, subdir, files in os.walk(PATH)]
    for files in files_list:
        for file in files:
            all_csvs.append(file)
    all_csv_files = [PATH + '/' +file for file in all_csvs]

    for path in all_csv_files:
        df = pd.read_csv(path)
        len_list.append(df.shape[0])

    max_len = np.max(len_list)
    num_datasets = len(len_list)

    dataset = []
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

        dataset.append((pcs,addresses))
    
    return dataset

class Token:

    address_sets = None # list containing the address sets for 4 parts (8 bits each) of the binary address
    pc_sets = None # list containing the pc sets for 4 parts (8 bits each) of the binary PC
  
    address_ixs = None # dict for allotting an index to each address
    pc_ixs = None # dict for allotting an index to each PC

    def __init__(self):
        self.address_sets = [set()]*4
        self.pc_sets = [set()]*4

        self.address_ixs = [{}]*4
        self.pc_ixs = [{}]*4

    def pc_tokens(self,pc):
        pc_bytes = [pc[:8], pc[8:16], pc[16:24], pc[24:32]] # calculate bytes
        for i in range(4):
            self.pc_sets[i].add(pc_bytes[i]) # add bytes to corresponding sets

        # allot an index to the byte if not already done 
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
        for i in range(4):
            self.address_sets[i].add(address_bytes[i])
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
    address_embeddings = [] # list containing the address embedding layers
    pc_embeddings = [] # list containing the PC embedding layers
    
    linears_address_1 = None # list containing the first set of address linear layers
    linears_address_2 = None # list containing the second set of address linear layers

    linears_pc_1 = None # list containing the first set of PC linear layers
    linears_pc_2 = None # list containing the second set of PC linear layers

    embedding_size = 32
    vocab_sizes_pc = [] # list containing the vocab sizes for PCs
    vocab_sizes_address = [] # list containing the vocab sizes for addresses
    context_size = None
    token = None

    def __init__(self,token,context_size):
        super(ByteEncoder,self).__init__()

        self.token = token
        for i in range(4):
            self.vocab_sizes_address.append(len(self.token.address_sets[i]))
            self.vocab_sizes_pc.append(len(self.token.pc_sets[i]))
        self.context_size = context_size

        for i in range(4):
            self.address_embeddings.append(nn.Embedding(self.vocab_sizes_address[i],self.embedding_size))
            self.pc_embeddings.append(nn.Embedding(self.vocab_sizes_pc[i],self.embedding_size))

        self.linears_address_1 = [nn.Linear(self.embedding_size * self.context_size,32)]*4
        self.linears_pc_1 = [nn.Linear(self.embedding_size * self.context_size,32)]*4

        for i in range(4):
            self.linears_address_2.append(nn.Embedding(32,self.vocab_sizes_address[i]))
            self.linears_pc_2.append(nn.Embedding(32,self.vocab_sizes_pc[i]))   

    def forward(self, inputs):
        
        # 4 inputs (4 bytes) for each address
        address_inputs = []
        for i in range(4):
            # convert each byte to its index (input to the embedding)
            address_inputs.append(torch.tensor([self.token.first_address_ix[ad[8*i:8*(i+1)]] for ad in inputs[1]], dtype=torch.long))
        
        # 4 inputs (4 bytes) for each PC
        pc_inputs = []
        for i in range(4):
            # convert each byte to its index (input to the embedding)
            pc_inputs.append(torch.tensor([self.token.first_pc_ix[pc[8*i:8*(i+1)]] for pc in inputs[0]], dtype=torch.long))

        # Embedding Calculation for address
        address_embeds = []
        for i in range(4):
            address_embeds.append(self.address_embeddings[i](address_inputs[i]).view((1, -1)))

        # Embedding Calculation for PC
        pc_embeds = []
        for i in range(4):
            pc_embeds.append(self.pc_embeddings[i](pc_inputs[i]).view((1, -1)))

        # outputs by 1st set of linear layers for address
        address_outs_1 = []
        for i in range(4):
            address_outs_1.append(F.relu(self.linears_address_1[i](address_embeds[i])))

        # outputs by 1st set of linear layers for PC
        pc_outs_1 = []
        for i in range(4):
            pc_outs_1.append(F.relu(self.linears_pc_1[i](pc_embeds[i])))

        # outputs by 2nd set of linear layers for address
        address_outs_2 = []
        for i in range(4):
            address_outs_2.append(F.relu(self.linears_address_2[i](address_outs_1[i])))
        
        # outputs by 2nd set of linear layers for PC
        pc_outs_2 = []
        for i in range(4):
            pc_outs_2.append(F.relu(self.linears_pc_2[i](pc_outs_1[i])))

        # Concatenate the 4 outputs for address and PC separately
        ad_out = torch.cat(address_outs_2[0], address_outs_2[1], address_outs_2[2], address_outs_2[3],axis=0)
        pc_out = torch.cat(pc_outs_2[0], pc_outs_2[1], pc_outs_2[2], pc_outs_2[3],axis=0)

        # Concatenate Outputs from PC and Address W2Vec
        out = torch.cat(pc_out,ad_out)

        #Calculate log_probs
        log_probs = F.log_softmax(out, dim=2)

        return log_probs

def train(inputs,epochs):
    pcs = inputs[0]
    addresses = inputs[1]
    
    token = Token()

    # initialize the index for each byte
    for pc,address in zip(pc,address):
        token.pc_tokens(pc = pc)
        token.address_tokens(address = address)
    
    losses = []
    encoder = ByteEncoder(token=token,context_size=2)
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(encoder.parameters(), lr=0.001)
    best_loss = 1000000000000
    
    address_trigrams = []
    pc_trigrams = []

    # Calculate 3 consecutive values for address and pc respectively
    for i in range(4):
        address_trigrams.append([([addresses[i], addresses[i + 1]], addresses[i + 2])
            for i in range(len(addresses) - 2)])
        pc_trigrams.append([([pcs[i], pcs[i + 1]], pcs[i + 2])
            for i in range(len(pcs) - 2)])

    trigrams = (pc_trigrams,address_trigrams)

    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for pc_trigram,address_trigram in trigrams:

            inputs = (pc_trigram[0],address_trigram[0]) # input to the model are 2 consecutive values for each pc and address

            encoder.zero_grad()

            log_probs = encoder(inputs) 

            ## Overlook this 
            # pc_target = torch.cat(token.pc_ixs[pc_trigram[1][0:8]],token.pc_ixs[pc_trigram[1][8:16]],
            #             token.pc_ixs[pc_trigram[1][16,24]],token.pc_ixs[pc_trigram[1][24:32]])
            
            # address_target = torch.cat(token.address_ixs[address_trigram[1][0:8]],token.address_ixs[address_trigram[1][8:16]],
            #             token.pc_ixs[address_trigram[1][16,24]],token.address_ixs[address_trigram[1][24:32]])
            
            target = torch.cat(pc_trigram[1],address_trigram[1])

            loss = loss_function(log_probs, target)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        losses.append(total_loss)
        print('-----------------------------')
        print('Epoch {} loss: {}'.format(epoch+1,total_loss))
        print('-----------------------------')
        if total_loss < best_loss:
            torch.save(encoder, 'byte_encoder.pt')
            print('Saved at epoch : {}'.format(epoch+1))

def main(args):
    dataset = get_data(args.path)

    for i in range(len(dataset)):
        if i==0: # train on only one file for now
            train(dataset[i])
            break



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='HTMLPhish')
    parser.add_argument('--path', type=str, required=True,
                        help='path to dir containing the csv files')
    args = parser.parse_args()

    main(args)