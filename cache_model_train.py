import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import utils.dataset as dataset
import random
import utils.dataset
from embed_lstm_32 import ByteEncoder

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
        return [ lambda x: F.softmax(x/self.temperature , dim=1) for x in logits], logits


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
        self.encoder_mlp = nn.linear(emb_size*4 , emb_size)
        self.time_distributed_encoder_mlp = TimeDistributed(self.encoder_mlp,batch_first=True)
        

    def get_freq_rec(self, input):
        
        byte_embeddings = []

        for i in range(4):
            byte_embeddings.append(torch.matmul(input[i], self.embed_encoder.address_embeddings[i].weight))  
        
        final_embedding = torch.cat(byte_embeddings , dim=-1)
        final_embedding = self.encoder_mlp(final_embedding)

        output = self.rec_freq_decoder(final_embedding)
        return output

    def forward(self, input):
        embeddings = self.time_distributed_encoder_mlp(input)
        lstm_out, self.hidden_cell = self.lstm(embeddings.view(len(embeddings) ,1, -1), self.hidden_cell)
        probs , logits = self.lstm_decoder(lstm_out)
        freq, rec = self.get_freq_rec(probs)

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

train_x = torch.rand(n_files,seq_len,n_bytes,emb_size)
train_x = train_x.view(n_files,seq_len,-1)
train_y = torch.rand(n_files,seq_len,label_size)

train_seq_x = []
train_seq_y = []

for i in range(n_files):
    x,y = utils.dataset.create_inout_sequences(train_x[i] , train_y[i], window_size)
    train_seq_x.append(x)
    train_seq_y.append(y)

epochs = 100

model = DeepCache(input_size=window_size, hidden_size=hidden_size, output_size=256)
xe_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

alpha = 0.33
beta = 0.33

for epoch in range(epochs):
   
    for i in range(n_files):            
        
        inputs = train_seq_x[i]
        labels = train_seq_y[i]

        for seq, labels in zip(inputs , labels):
        
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_size),
                            torch.zeros(1, 1, model.hidden_size))
            probs, logits, freq, rec = model(inputs)

            loss_address = xe_loss(probs, labels[0])
            freq_address = xe_loss(freq, labels[1])
            rec_address = xe_loss(rec, labels[2])

            total_loss = (alpha)*loss_address + (beta)*freq_address + (1-alpha-beta)*rec_address

            total_loss.backward()
            optimizer.step()





