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

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first = True)
        self.lstm_decoder = Decoder_lstm(self.hidden_size, self.output_size)
       

        self.rec_freq_decoder = Decoder(input_size)
        
        self.embed_encoder = torch.load("w2vec_checkpoints/byte_encoder_32.pt")
        self.encoder_mlp = nn.Linear(emb_size*4 , emb_size)
        self.time_distributed_encoder_mlp = TimeDistributed(self.encoder_mlp,batch_first=True)
        


<<<<<<< HEAD
    def get_freq_rec(self, input, dist_vector):
        
        byte_embeddings = []
        
        # print(len(input))

        for i in range(4):
            byte_embeddings.append(torch.matmul(input[i], self.embed_encoder.address_embeddings[i].weight))  
=======

    def get_freq_rec(self, input):
        
        byte_embeddings = []
        for i in range(4):
            # byte_embeddings.append(torch.matmul(input[i], self.embed_encoder.address_embeddings[i].weight))  
            byte_embeddings.append(torch.matmul(input[i], self.embedding_matrix[i]))  

>>>>>>> bd7aaaaab5ec919f3be9178064ad7e61d3837777
        
        final_embedding = torch.cat(byte_embeddings , dim=-1)
        final_embedding = self.encoder_mlp(final_embedding).squeeze(0)
        

        final_embedding = final_embedding.float()
        dist_vector = dist_vector.float()
        final_embedding = torch.cat([final_embedding , dist_vector] , dim=-1)
        output = self.rec_freq_decoder(final_embedding)
        return output

    def get_distribution_vector(self, input):
        # print(input.shape)
        dist_vector = torch.zeros(input.shape[0],input.shape[2])

        for i in range(input.shape[0]):
            kde = KernelDensity()
            kde.fit(input[i].detach())
            n_samples = 200

            random_samples = kde.sample(n_samples)
            random_samples = torch.from_numpy(random_samples.astype(float))
            dist_vector[i] = torch.mean(random_samples , axis = 0)

<<<<<<< HEAD
        return dist_vector

=======
>>>>>>> bd7aaaaab5ec919f3be9178064ad7e61d3837777


    def forward(self, input, hidden_cell):

        
        embeddings = self.time_distributed_encoder_mlp(input)
        dist_vector = self.get_distribution_vector(embeddings)

        lstm_out, hidden_cell = self.lstm(embeddings, hidden_cell)
        probs , logits = self.lstm_decoder(hidden_cell[0])


        freq_rec = self.get_freq_rec(probs,dist_vector)
    

        freq = freq_rec[:,0]
        rec = freq_rec[:,1]


        return [probs , logits , freq , rec]

       
n_files = 10
seq_len = 200
emb_size = 20
label_size = 3
vocab_size = 500
window_size = 50
hidden_size = 20
n_bytes = 4


train_x = torch.rand(n_files,seq_len,emb_size*4).float()
train_y = torch.randint(0,3000,(n_files,seq_len,label_size))

train_seq_x = torch.zeros(n_files,seq_len-window_size, window_size, emb_size*4)
train_seq_y = torch.zeros(n_files,seq_len-window_size, 3)

def create_inout_sequences(input_data, labels,tw):
    L = len(input_data)
    x = torch.zeros(L-tw,tw,input_data[0].shape[-1] )
    y = torch.zeros(L-tw,3)

    for i in range(L-tw):
        x[i] = input_data[i:i+tw]
        y[i] = labels[i+tw:i+tw+1]

    return x,y

for i in range(n_files):
    x,y = create_inout_sequences(train_x[i] , train_y[i], window_size)
    train_seq_x[i] = x
    train_seq_y[i] = y

train_seq_x = train_seq_x.view(n_files*(seq_len-window_size),window_size,emb_size*4)
train_seq_y = train_seq_y.view(n_files*(seq_len-window_size),3)

epochs = 100

model = DeepCache(input_size=emb_size, hidden_size=hidden_size, output_size=256)
xe_loss = nn.CrossEntropyLoss()
mse_loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

alpha = 0.33
beta = 0.33

def get_bytes(x):
    x = x.int()
    binary = bin(x).replace("0b","")
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

<<<<<<< HEAD
class miss_dataset(Dataset):
    def __init__(self, train_x , train_y):
        self.train_x = train_x
        self.train_y = train_y
=======

for epoch in range(epochs):
   
    for i in range(n_files):            
        
        inputs = train_seq_x[i]
        labels = train_seq_y[i]
>>>>>>> bd7aaaaab5ec919f3be9178064ad7e61d3837777

    def __len__(self):
        return self.train_x.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        return [self.train_x[idx] , self.train_y[idx]]


batch_size =256

miss_dataset = miss_dataset(train_seq_x,train_seq_y)
dataloader = DataLoader(miss_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0,drop_last = True)


for epoch in tqdm(range(epochs)):
   
   for i_batch, (seq,labels) in enumerate(dataloader):

            # print(seq.shape)
            # print(labels.shape)
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





