import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import utils.dataset as dataset
import random
import utils.dataset
from embed_lstm_32 import ByteEncoder
from sklearn.neighbors import KernelDensity

class Decoder(nn.Module):
    def __init__(self, d_in):
        super(Decoder,self).__init__()
        self.linear1 = nn.Linear(40, 10)
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
    


class DeepCache(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DeepCache,self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size, hidden_size)
        self.lstm_decoder = Decoder_lstm(self.hidden_size, self.output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_size),
                            torch.zeros(1,1,self.hidden_size))

        self.rec_freq_decoder = Decoder(input_size)
        
        self.embed_encoder = torch.load("w2vec_checkpoints/byte_encoder_32.pt")
        self.encoder_mlp = nn.Linear(emb_size*4 , emb_size)
        self.time_distributed_encoder_mlp = TimeDistributed(self.encoder_mlp,batch_first=True)
        self.dist_vector = torch.zeros(emb_size)

        self.embedding_matrix = torch.rand(4,256,20)        

    def get_freq_rec(self, input):
        
        byte_embeddings = []
        
        for i in range(4):
            # byte_embeddings.append(torch.matmul(input[i], self.embed_encoder.address_embeddings[i].weight))  
            byte_embeddings.append(torch.matmul(input[i], self.embedding_matrix[i]))  
        
        final_embedding = torch.cat(byte_embeddings , dim=-1)
        final_embedding = self.encoder_mlp(final_embedding)

        print("final_embedding_shape: ",final_embedding.shape)
        dist_vector = self.dist_vector.expand(final_embedding.shape[0],1,-1)
        print("dist_vector.shape:",dist_vector.shape)
        final_embedding = torch.cat([final_embedding , dist_vector] , dim=-1)
        print("final_embedding_shape: ",final_embedding.shape)
        output = self.rec_freq_decoder(final_embedding)
        return output

    def get_distribution_vector(self, input):
        kde = KernelDensity()
        kde.fit(input)
        n_samples = 200

        random_samples = kde.sample(n_samples)

        self.dist_vector = torch.mean(random_samples , axis = 0)

        



    def forward(self, input):
        print("input_shape :",input.shape)

        embeddings = self.time_distributed_encoder_mlp(input)
        embeddings = embeddings.view(len(embeddings) ,1, -1)
        print("embeddings_shape: ",embeddings.shape)

        lstm_out, self.hidden_cell = self.lstm(embeddings.view(len(embeddings) ,1, -1), self.hidden_cell)
        print("self.hidden_cell: ",self.hidden_cell[0].shape)
        # exit()
        probs , logits = self.lstm_decoder(self.hidden_cell[0])
        print("probs_shape: ",probs[0].shape)

        freq_rec = self.get_freq_rec(probs)
        print("freq_req: ",freq_rec.shape)
        freq = freq_rec[:,:,0]
        rec = freq_rec[:,:,1]

        return [probs , logits , freq , rec]

       
n_files = 10
seq_len = 200
emb_size = 20
label_size = 3
vocab_size = 500
window_size = 50
hidden_size = 20
n_bytes = 4

embedding_matrix = torch.rand(vocab_size , emb_size)

train_x_byte = torch.rand(n_files,seq_len,n_bytes,emb_size)
train_x = train_x_byte.view(n_files,seq_len,-1)
train_y = torch.randint(0,3000,(n_files,seq_len,label_size))

train_seq_x = []
train_seq_y = []

def create_inout_sequences(input_data, labels,tw):
    inout_seq = []
    L = len(input_data)
    x = torch.zeros(L-tw,tw,input_data[0].shape[-1] )
    y = []
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = (labels[i+tw:i+tw+1 , 0] , labels[i+tw:i+tw+1 , 1] ,labels[i+tw:i+tw+1 , 2])
        x[i] = train_seq
        y.append(train_label)
    return x,y

for i in range(n_files):
    x,y = create_inout_sequences(train_x[i] , train_y[i], window_size)
    train_seq_x.append(x)
    train_seq_y.append(y)

epochs = 100

model = DeepCache(input_size=emb_size, hidden_size=hidden_size, output_size=256)
xe_loss = nn.CrossEntropyLoss()
mse_loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

alpha = 0.33
beta = 0.33

def get_bytes(x):
    
    binary = bin(x).replace("0b","") 
    byte_list = list(x.item().to_bytes(4,byteorder='big'))
    byte_list = [torch.tensor(x) for x in byte_list]
    return byte_list

def get_pred_loss(pred, target, xe_loss):
    total_loss = 0
    
    target_byte = get_bytes(target)

    for i in range(4):
        
        logits = pred[i].squeeze(1)
        target = target_byte[i].unsqueeze(0)
        total_loss+=xe_loss(logits,target)

    return total_loss

for epoch in range(epochs):
   
    for i in range(n_files):            
        
        inputs = train_seq_x[i]
        labels = train_seq_y[i]

        for seq, labels in zip(inputs , labels):
        
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_size),
                            torch.zeros(1, 1, model.hidden_size))
            probs, logits, freq, rec = model(seq)

            print("len(logits) : ",len(logits))
            print("logits: ",logits[0].shape)
            

            loss_address = get_pred_loss(logits,labels[0], xe_loss)
            freq_address = mse_loss(freq, labels[1])
            rec_address = mse_loss(rec, labels[2])

            total_loss = (alpha)*loss_address + (beta)*freq_address + (1-alpha-beta)*rec_address

            total_loss.backward()
            optimizer.step()





