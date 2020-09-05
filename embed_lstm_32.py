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
from torchsummary import summary

def hex_to_bin(string):
    scale = 16
    res = bin(int(string, scale)).split('0b')[1].zfill(32)
    return str(res) 

def get_vocab_bytes():
    count = 0
    byte_list = []
    with open('bytes.csv','r') as file:
        reader = csv.reader(file)
        for row in reader:
            count+=1
            if count == 1:
                continue
            else:
                byte_list.append(row[1])
    return byte_list

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

    csv_files = [PATH+ '/' +file for file in all_csvs]
    dataset = []

    for i,path in enumerate(csv_files):
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
    print('Number of csvs read: {}'.format(len(dataset)))
    
    return dataset

class Token:

    address_sets = [] # list containing the address sets for 4 parts (8 bits each) of the binary address
    pc_sets = [] # list containing the pc sets for 4 parts (8 bits each) of the binary PC
  
    address_ixs = [] # dict for allotting an index to each address
    pc_ixs = [] # dict for allotting an index to each PC

    def __init__(self):
        for i in range(4):
            self.address_sets.append(set())
            self.pc_sets.append(set())

        for i in range(4):
            self.address_ixs.append({})
            self.pc_ixs.append({})

    def pc_tokens(self,pc):
        pc_bytes = []
        for i in range(4):
            pc_bytes.append(pc) # calculate bytes
        for i in range(4):
            self.pc_sets[i].add(pc_bytes[i]) # add bytes to corresponding sets

        # allot an index to the byte if not already done
        for i in range(4): 
            if pc_bytes[i] not in self.pc_ixs[i].keys():
                self.pc_ixs[i][pc_bytes[i]] = len(self.pc_ixs[i])        

    def address_tokens(self,address):
        address_bytes = []
        for i in range(4):
            address_bytes.append(address) # calculate bytes
        
        for i in range(4):
            self.address_sets[i].add(address_bytes[i]) # add bytes to corresponding sets
        
        # allot an index to the byte if not already done
        for i in range(4): 
            if address_bytes[i] not in self.address_ixs[i].keys():
                self.address_ixs[i][address_bytes[i]] = len(self.address_ixs[i])      

class ByteEncoder(nn.Module):
   
    def __init__(self,vocab_sizes_address,vocab_sizes_pc,context_size,embedding_size,hidden_size):
        super(ByteEncoder,self).__init__()

        self.address_embeddings = [] # list containing the address embedding layers
        self.pc_embeddings = [] # list containing the PC embedding layers
        
        self.linears_address_1 = [] # list containing the first set of address linear layers
        self.linears_address_2 = [] # list containing the second set of address linear layers

        self.linears_pc_1 = [] # list containing the first set of PC linear layers
        self.linears_pc_2 = [] # list containing the second set of PC linear layers

    token = None


<<<<<<< Updated upstream
    def __init__(self,token,vocab_sizes_address,vocab_sizes_pc,context_size,embedding_size,hidden_size):
        super(ByteEncoder,self).__init__()
        self.token = token
=======
>>>>>>> Stashed changes

        for i in range(4):
            self.address_embeddings.append(nn.Embedding(vocab_sizes_address[i],embedding_size * context_size))
            self.pc_embeddings.append(nn.Embedding(vocab_sizes_pc[i],embedding_size * context_size))

        for i in range(4):
            self.linears_address_1.append(nn.Linear(embedding_size * context_size,hidden_size))
            self.linears_pc_1.append(nn.Linear(embedding_size * context_size,hidden_size))

        for i in range(4):
            self.linears_address_2.append(nn.Linear(hidden_size,vocab_sizes_address[i]))
            self.linears_pc_2.append(nn.Linear(hidden_size,vocab_sizes_pc[i]))
        
        self.address_embeddings = nn.ModuleList(self.address_embeddings)
        self.linears_address_1 = nn.ModuleList(self.linears_address_1)
        self.linears_address_2 = nn.ModuleList(self.linears_address_2)
        
        self.pc_embeddings = nn.ModuleList(self.pc_embeddings)
        self.linears_pc_1 = nn.ModuleList(self.linears_pc_1)
        self.linears_pc_2 = nn.ModuleList(self.linears_pc_2)

               

    def forward(self, inputs,token):
        
        # 4 inputs (4 bytes) for each address
        address_inputs = []
        for i in range(4):
            # convert each byte to its index (input to the embedding)
            address_inputs.append(torch.tensor([token.address_ixs[i][ad[8*i:8*(i+1)]] for ad in inputs[1]], dtype=torch.long))
        
        # 4 inputs (4 bytes) for each PC
        pc_inputs = []
        for i in range(4):
            # convert each byte to its index (input to the embedding)
            pc_inputs.append(torch.tensor([token.pc_ixs[i][pc[8*i:8*(i+1)]] for pc in inputs[0]], dtype=torch.long))

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

        #Calculate log_probs
        log_probs = []
        for i in range(4):
            log_probs.append(F.log_softmax(pc_outs_2[i], dim=1))
        for i in range(4):
            log_probs.append(F.log_softmax(address_outs_2[i], dim=1))

        return log_probs

def w2vec_loss(outputs,targets):
    loss = torch.tensor(0,dtype=torch.float)
    loss_function = nn.NLLLoss()
    for output,target in zip(outputs,targets):
        loss+=loss_function(output,target)
    return loss

class Trainer:
    model = None
    best_loss = None
    optimizer = None
    def __init__(self,model,best_loss,optimizer):
        self.model = model
        self.best_loss = best_loss
        self.optimizer = optimizer    

def train(trainer,inputs,tkn,arguments,num):
    pcs = inputs[0]
    addresses = inputs[1]
    
    token = tkn
    best_epoch = 0
    
    address_trigrams = []
    pc_trigrams = []

    size = int(args.context_size/2)

    # Calculate 3 consecutive values for address and pc respectively
    address_trigrams.append([([addresses[j] for j in range(i-size,i-size+1)], addresses[i])
        for i in range(len(addresses) - size)])
    pc_trigrams.append([([pcs[j] for j in range(i-size,i-size+1)], pcs[i])
        for i in range(len(pcs) - size)])
    
    trigrams = []
    for i in range(len(pc_trigrams)):
        trigrams.append((pc_trigrams[i],address_trigrams[i]))
    

    for epoch in range(args.epochs):
        total_loss = 0
        for trigram in trigrams:

            inputs = (trigram[0][0][0],trigram[1][0][0]) # input to the model are 2 consecutive values for each pc and address
            trainer.model.zero_grad()

            log_probs = trainer.model(inputs,token) 

            pc_trigram = trigram[0][1]
            address_trigram = trigram[1][1]

            targets = []
            for i in range(4):
                targets.append(torch.tensor([token.pc_ixs[i][pc_trigram[1][8*i:8*(i+1)]]],dtype=torch.long))
            
            for i in range(4):
                targets.append(torch.tensor([token.address_ixs[i][address_trigram[1][8*i:8*(i+1)]]],dtype=torch.long))

            loss = w2vec_loss(log_probs,targets)

            loss.backward()
            trainer.optimizer.step()

            total_loss += loss.item()

        if (epoch+1)%20 == 0:
            print('Epoch {} with loss: {}'.format(epoch+1,total_loss))
            print('----------')
        if total_loss < trainer.best_loss:
            trainer.best_loss = total_loss
            best_epoch = epoch+1
            torch.save(trainer.model, 'w2vec_checkpoints/byte_encoder_32.pt')
            print('Saved at epoch {} with loss: {} for dataset: {}'.format(epoch+1,total_loss,num))
            print('----------')
    print('Best Epoch: {}'.format(best_epoch))

def get_address(index,model,outputs):
    address = ''
    for i in range(4):
        address += list(model.token.address_ixs.keys()[list(model.token.address_ixs.values()).index(torch.argmax(outputs[i+4]))])
    return address

def get_address(index,model,outputs):
    pc = ''
    for i in range(4):
        pc += list(model.token.address_ixs.keys())[list(model.token.address_ixs.values()).index(torch.argmax(outputs[i]))]
    return pc

def main(args):
    dataset = get_data(args.path)

    if not os.path.exists('w2vec_checkpoints'):
        os.makedirs('w2vec_checkpoints')

    bytes_list = get_vocab_bytes()
    token = Token()
    
    print('Preparing Tokens')
    print('------------------------------------')
    # initialize the index for each byte
    for b in bytes_list:
        token.pc_tokens(pc = b)
        token.address_tokens(address = b)

    vocab_sizes_pc = [] # list containing the vocab sizes for PCs
    vocab_sizes_address = [] # list containing the vocab sizes for addresses
    
    for i in range(4):
        vocab_sizes_address.append(len(token.address_sets[i]))
        vocab_sizes_pc.append(len(token.pc_sets[i]))
    
    encoder = ByteEncoder(token=token,vocab_sizes_address=vocab_sizes_address,vocab_sizes_pc=vocab_sizes_pc,context_size=args.context_size,
                            embedding_size=args.embed_dim,hidden_size=args.hidden_size)
    # print(summary(encoder))
    best_loss = 1e10
    optimizer = optim.Adam(encoder.parameters(),lr=3e-3,weight_decay=1e-3)

    trainer = Trainer(model=encoder,optimizer=optimizer,best_loss=best_loss)

    for i in range(len(dataset)):
        print('Training for dataset: {}'.format(i+1))
        train(trainer=trainer,inputs=dataset[i],tkn=token,arguments=args,num=i+1)
        print('Best Loss: {}'.format(trainer.best_loss))
        print('----------------------------------------')




if __name__=='__main__':
    parser = argparse.ArgumentParser(description='HTMLPhish')
    parser.add_argument('--path', type=str, required=True,
                        help='path to dir containing the csv files')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--embed_dim', type=int, default=32,
                        help='embedding dimension')
    parser.add_argument('--context_size', type=int, default=2,
                        help='context_size')       
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='dimension of hidden layer')                  
    args = parser.parse_args()

    main(args)