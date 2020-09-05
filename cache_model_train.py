import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import utils.dataset as dataset
import random
import utils.dataset
from embed_lstm_32 import ByteEncoder
from embed_lstm_32 import Token
from sklearn.neighbors import KernelDensity
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from create_train_dataset import get_miss_dataloader


def get_bytes(x):
    
    x = x.long()
    bytes = torch.zeros(4) 
    byte_list = list(x.item().to_bytes(4,byteorder='big'))
    for i in range(4):
        bytes[i] = torch.tensor(byte_list[i], dtype = torch.long)
    return bytes

def get_pred_loss(pred, target, xe_loss):
    total_loss = 0
    
    target_batch =  torch.zeros(target.shape[0],4, dtype = torch.long)
    
    for i in range(target.shape[0]):
        target_batch[i] = get_bytes(target[i])


    for i in range(4):
        # print(pred[i].shape)
        logits = pred[i].squeeze(0)
        logits = logits
        total_loss+=xe_loss(logits,target_batch[:,i])

    return total_loss



class Decoder(nn.Module):
    def __init__(self, d_in):
        super(Decoder,self).__init__()
        self.linear1 = nn.Linear(d_in, 10)

        self.linear2 = nn.Linear(10, 2)
    
    def forward(self, input):
        x = F.relu(self.linear1(input))
        x = self.linear2(x)
        return x

class Decoder_lstm(nn.Module):
    def __init__(self,d_in,d_out):
        super(Decoder_lstm,self).__init__()
        self.linear1 = nn.Linear(d_in,d_out)
        self.linear2 = nn.Linear(d_in,d_out)
        self.linear3 = nn.Linear(d_in,d_out)
        self.linear4 = nn.Linear(d_in,d_out)
        self.temperature = 0.001
    def forward(self,x):
        x1 = self.linear1(x)
        x2 = self.linear2(x)
        x3 = self.linear3(x)
        x4 = self.linear4(x)
        logits = [x1,x2,x3,x4]
        return [ F.softmax(x/self.temperature , dim=1) for x in logits], logits



class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y
    


def get_bytes_2d(x):
    out = torch.zeros((x.shape[0],4) , dtype =torch.long)
    for i in range(x.shape[0]):
        out[i] = get_bytes(x[i])

    return out

class Encoder(nn.Module):
    def __init__(self,emb_size):
        super(Encoder,self).__init__()
        self.linear = nn.Linear(emb_size*4, emb_size)
    
    def forward(self,x):
        x = self.linear(x)
        x = F.sigmoid(x)
        return x

    


class DeepCache(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DeepCache,self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first = True)
        self.lstm_decoder = Decoder_lstm(self.hidden_size, self.output_size)
       

        self.rec_freq_decoder = Decoder((input_size//2)*3)
        
        self.embed_encoder = torch.load("w2vec_checkpoints/byte_encoder_32.pt")
        for param in self.embed_encoder.parameters():
             param.requires_grad = False
        self.encoder_mlp = Encoder(emb_size)
        self.time_distributed_encoder_mlp = TimeDistributed(self.encoder_mlp,batch_first=True)
        


    def get_freq_rec(self, x, dist_vector):
        
        byte_embeddings = []
        

        for i in range(4):
            byte_embeddings.append(torch.matmul(x[i], self.embed_encoder.address_embeddings[i].weight))  
        
        final_embedding = torch.cat(byte_embeddings , dim=-1)
        final_embedding = self.encoder_mlp(final_embedding).squeeze(0)


        final_embedding = final_embedding.float()
        dist_vector = dist_vector.float()
        final_embedding = torch.cat([final_embedding , dist_vector] , dim=-1)
        output = self.rec_freq_decoder(final_embedding)
        return output

    def get_distribution_vector(self, input):

        dist_vector = torch.zeros(input.shape[0],input.shape[2])

        for i in range(input.shape[0]):
            kde = KernelDensity()
            try :
                kde.fit(input[i].detach())
            except:
                print(self.embed_encoder.address_embeddings[0].weight)
                print("i:",i)
                print(input[i])
                exit()
            n_samples = 200

            random_samples = kde.sample(n_samples)
            random_samples = torch.from_numpy(random_samples.astype(float))
            dist_vector[i] = torch.mean(random_samples , axis = 0)

        return dist_vector

    def get_embed_pc(self, address):
        b,s,_ = list(address.shape)
        embeddings = torch.zeros(b*s,emb_size*4)

        address =address.view(-1,(address.shape[-1]))
        address_bytes = get_bytes_2d(address)

        for i in range(4) :
            
            temp = self.embed_encoder.pc_embeddings[i](address_bytes[:,i])
            if torch.isnan(temp).any():
                print(i)
                print(self.embed_encoder.pc_embeddings[i].weight)
                print(temp)
                print(address_bytes[:,i])
                print("GOT NANS")
                exit()
            embeddings[:,i*emb_size:(i+1)*emb_size] = temp

        embeddings = embeddings.view(b,s,emb_size*4)

        return embeddings

    def get_embed_addr(self, address):
        b,s,_ = list(address.shape)
        embeddings = torch.zeros(b*s,emb_size*4)

        address =address.view(-1,(address.shape[-1]))
        address_bytes = get_bytes_2d(address)

        for i in range(4) :
            temp = self.embed_encoder.address_embeddings[i](address_bytes[:,i])
            if torch.isnan(temp).any():
                print(i)
                print(self.embed_encoder.address_embeddings[i].weight)
                print(temp)
                print(address_bytes[:,i])
                print("GOT NANS")
                exit()
            embeddings[:,i*emb_size:(i+1)*emb_size] = temp

        embeddings = embeddings.view(b,s,emb_size*4)

        return embeddings

    def forward(self, input, hidden_cell):

        pc      = input[:,:,0:1]
        address = input[:,:,1:2]
        
        pc_embed = self.get_embed_pc(pc)
        addr_embed = self.get_embed_addr(address)

        if torch.isnan(pc_embed).any():
            print("GOT NANS")
            exit()

        if torch.isnan(addr_embed).any():
            print("GOT NANS1")
            exit()

        embeddings_pc = self.time_distributed_encoder_mlp(pc_embed)
        embeddings_address = self.time_distributed_encoder_mlp(addr_embed)

        if torch.isnan(embeddings_pc).any():
            print(self.encoder_mlp.linear.weight)
            print("GOT NANS2")
            exit()

        if torch.isnan(embeddings_address).any():
            
            print("GOT NANS3")
            exit()

        embeddings = torch.cat([embeddings_pc,embeddings_address] ,dim=-1)
        dist_vector = self.get_distribution_vector(embeddings)

        lstm_out, hidden_cell = self.lstm(embeddings, hidden_cell)
        probs , logits = self.lstm_decoder(hidden_cell[0])

        freq_rec = self.get_freq_rec(probs,dist_vector)

        freq = freq_rec[:,0]
        rec = freq_rec[:,1]

        return [probs , logits , freq , rec]

       
n_files = 1
seq_len = 200
emb_size = 20
label_size = 3
vocab_size = 500
window_size = 30
hidden_size = 20
n_bytes = 4
epochs = 100
alpha = 0.33
beta = 0.33
batch_size =256

model = DeepCache(input_size=2*emb_size, hidden_size=hidden_size, output_size=256)

for i in range(4):
    print(model.embed_encoder.address_embeddings[i].weight)
    print(model.embed_encoder.pc_embeddings[i].weight)
exit()
# print()
# for i in range(4):
#     if torch.isnan(model.embed_encoder.address_embeddings[i].weight).any() or torch.isnan(model.embed_encoder.pc_embeddings[i].weight).any() :
#         print("AAAAAAAAAAAA")
#         exit()
xe_loss = nn.CrossEntropyLoss()
mse_loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
dataloader = get_miss_dataloader(batch_size, window_size, n_files)




for epoch in tqdm(range(epochs)):
   
   for i_batch, (seq,labels) in enumerate(dataloader):
            print("i_batch: ",i_batch)
            optimizer.zero_grad()
            hidden_cell = (torch.zeros(1, batch_size, model.hidden_size),
                            torch.zeros(1, batch_size, model.hidden_size))
            probs, logits, freq, rec = model(seq,hidden_cell)


            loss_address = get_pred_loss(logits,labels[:,0], xe_loss)

            freq_address = mse_loss(freq, labels[:,1].float())
            rec_address = mse_loss(rec, labels[:,2].float())

            total_loss = (alpha)*loss_address + (beta)*freq_address + (1-alpha-beta)*rec_address

            total_loss.backward()
            optimizer.step()





