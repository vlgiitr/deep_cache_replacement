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
from torchsummary import summary
import argparse
import os
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_bytes(x):
    
    x = x.long()
    bytes = torch.zeros(4).to(device) 
    byte_list = list(x.item().to_bytes(4,byteorder='big'))
    for i in range(4):
        bytes[i] = torch.tensor(byte_list[i], dtype = torch.long)
    return bytes

def get_pred_loss(pred, target, xe_loss):
    total_loss = 0
    
    target_batch =  torch.zeros(target.shape[0],4, dtype = torch.long)
    for i in range(target.shape[0]):
        target_batch[i] = get_bytes(target[i]) # convert dec to bytes ( since target is in byte (0-255))


    for i in range(4):
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
        x1 = self.linear1(x) #1st byte
        x2 = self.linear2(x) #2nd byte
        x3 = self.linear3(x) #3rd byte
        x4 = self.linear4(x) #4th byte
        logits = [x1,x2,x3,x4]
        return [ torch.softmax(x/self.temperature , dim=2) for x in logits], logits



class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        # if torch.isnan(x).any().item():
        #     print('TimeDistributed INPUT is NaN')

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
        # if torch.isnan(y).any().item():
        #     print('TimeDistributed OUTPUT is NaN')

        return y
    


def get_bytes_2d(x):
    out = torch.zeros((x.shape[0],4) , dtype =torch.long).to(device)
    for i in range(x.shape[0]):
        out[i] = get_bytes(x[i])
    return out

class Encoder(nn.Module):
    def __init__(self,emb_size):
        super(Encoder,self).__init__()
        self.linear = nn.Linear(emb_size*4, emb_size)
    
    def forward(self,x):
        # if torch.isnan(x).any().item():
        #     print('Encoder Input is NaN')
        x = self.linear(x)
        x = torch.sigmoid(x)
        # if torch.isnan(x).any().item():
        #     print(self.linear.weight)
        #     print('Encoder Output is NaN')
        #     exit()
        return x

    


class DeepCache(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DeepCache,self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.emb_size = int(input_size/2)

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first = True) #lstm model
        self.lstm_decoder = Decoder_lstm(self.hidden_size, self.output_size) # decoder to get address predictions
       

        self.rec_freq_decoder = Decoder((input_size//2)*3) # decoder to get freq and rec
        
        self.embed_encoder = torch.load("checkpoints/byte_encoder_32.pt") # byte -> embedding encoder
        for param in self.embed_encoder.parameters(): 
             param.requires_grad = False
        self.encoder_mlp = Encoder(int(self.input_size/2)) # 4 byte embeddings -> address embeddings
        self.time_distributed_encoder_mlp = TimeDistributed(self.encoder_mlp,batch_first=True) # wrapper function to make encoder time distributed


    def get_freq_rec(self, x, dist_vector):
        
        byte_embeddings = []
        
        # multiply predicted probs (with temperature) with embedding matrix to get embeddings in a differentiable manner
        for i in range(4):
            byte_embeddings.append(torch.matmul(x[i], self.embed_encoder.address_embeddings[i].weight))  
        
        final_embedding = torch.cat(byte_embeddings , dim=-1) # concatenate all bytes' embeddings
        final_embedding = self.encoder_mlp(final_embedding).squeeze(0) # get address embedding from 4 byte embeddings


        final_embedding = final_embedding.float()
        dist_vector = dist_vector.float()
        final_embedding = torch.cat([final_embedding , dist_vector] , dim=-1) # concatenate address embedding with dist vector
        output = self.rec_freq_decoder(final_embedding) # predict freq, rec using MLP
        return torch.sigmoid(output)

    def get_distribution_vector(self, input):
        # if torch.isnan(input).any().item():
        #     print('-----------------------------------')
        #     print('get_distribution_vector INPUT is NaN')

        dist_vector = torch.zeros(input.shape[0],input.shape[2]) # initilise the dist vector

        for i in range(input.shape[0]):
            kde = KernelDensity()    # fit KDE
            try :
                kde.fit(input[i].detach())
            except:
                print("i:",i)
                print('-----------------------------------')
                exit()
            n_samples = 200
            
            # sample from distribution and take mean to get estimate of true mean ie. dist vector

            random_samples = kde.sample(n_samples)
            random_samples = torch.from_numpy(random_samples.astype(float))
            dist_vector[i] = torch.mean(random_samples , axis = 0)

        return dist_vector

    def get_embed_pc(self, address):
        b,s,_ = list(address.shape)
        embeddings = torch.zeros(b*s,self.emb_size*4).to(device) # initialise the byte embeddings

        address =address.view(-1,(address.shape[-1]))
        address_bytes = get_bytes_2d(address) # convert input decimal into 4 bytes
        for i in range(4) :
            
            temp = self.embed_encoder.pc_embeddings[i](address_bytes[:,i]) # get embeddings of each byte 
            embeddings[:,i*self.emb_size:(i+1)*self.emb_size] = temp

        embeddings = embeddings.view(b,s,self.emb_size*4)

        return embeddings

    def get_embed_addr(self, address):
        b,s,_ = list(address.shape)
        embeddings = torch.zeros(b*s,self.emb_size*4) # initialise the byte embeddings

        address =address.view(-1,(address.shape[-1])).to(device)
        address_bytes = get_bytes_2d(address)  # convert input decimal into 4 bytes

        for i in range(4) :
            temp = self.embed_encoder.address_embeddings[i](address_bytes[:,i]) # get embeddings of each byte 
            embeddings[:,i*self.emb_size:(i+1)*self.emb_size] = temp

        embeddings = embeddings.view(b,s,self.emb_size*4)

        return embeddings

    def forward(self, input, hidden_cell):  
        # if torch.isnan(input).any().item():
        #     print('Forward INPUT is NaN')
        pc      = input[:,:,0:1] 
        address = input[:,:,1:2] # Address value in decimal
        
        pc_embed = self.get_embed_pc(pc) # Convert decimal address to 4 byte embeddings using pretrained embeddings
        addr_embed = self.get_embed_addr(address)
        # time distributed MLP because we need to apply it on every element of the sequence
        embeddings_pc = self.time_distributed_encoder_mlp(pc_embed) # Convert 4byte embedding to a single address embedding using an MLP
        embeddings_address = self.time_distributed_encoder_mlp(addr_embed)
        # if torch.isnan(embeddings_pc).any().item():
        #     print('embeddings_pc is NaN')
        # if torch.isnan(embeddings_address).any().item():
        #     print('embeddings_address is NaN')
        # concat pc and adress emeddings
        embeddings = torch.cat([embeddings_pc,embeddings_address] ,dim=-1)
        # get distribution vector using KDE
        dist_vector = self.get_distribution_vector(embeddings)

        lstm_out, hidden_cell = self.lstm(embeddings, hidden_cell)
        probs , logits = self.lstm_decoder(hidden_cell[0]) # get prediction logits and probs

        freq_rec = self.get_freq_rec(probs,dist_vector) # get freq and rec estimate from prediced probs and distribution vector

        freq = freq_rec[:,0]
        rec = freq_rec[:,1]

        return [probs , logits , freq , rec]

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Deep Cache')
    parser.add_argument('--epochs', type=int, default=2,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')                       
    args = parser.parse_args()

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    
    writer = SummaryWriter('runs/deepcache')

    n_files = 1
    seq_len = 200
    emb_size = 80
    label_size = 3
    window_size = 30
    hidden_size = 40
    n_bytes = 4
    epochs = args.epochs
    alpha = 0.33
    beta = 0.33
    batch_size = args.batch_size

    print('Creating Model')
    # model = DeepCache(input_size=2*emb_size,hidden_size=hidden_size,output_size=256)
    model = torch.load('checkpoints/deep_cache_testgen_sigmoid_3.pt')
    model.to(device)

    xe_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print('Loading Data')
    dataloader = get_miss_dataloader(batch_size, window_size, n_files)
    print('Num_Batches: {}'.format(len(dataloader)))
    print('------------------------------------')
    best_loss = 1e30
    for epoch in range(epochs):
        losses = []
        i = 0
        for (seq,labels) in tqdm(dataloader):
            i+=1
            optimizer.zero_grad()
            hidden_cell = (torch.zeros(1, batch_size, model.hidden_size).to(device), # reinitialise hidden state for each new sample
                            torch.zeros(1, batch_size, model.hidden_size).to(device))
            probs, logits, freq, rec = model(input = seq.to(device),hidden_cell=hidden_cell)
            # print('Freq: {}'.format(freq))
            # print('Rec: {}'.format(rec))
            add_target = labels[:,0].to(device)
            loss_address = get_pred_loss(logits,add_target, xe_loss) # Cross entropy loss with address predictions
            freq_target = labels[:,1].float().to(device)
            freq_target = (freq_target - torch.min(freq_target))/(torch.max(freq_target)-torch.min(freq_target))           
            rec_target = labels[:,2].float().to(device)
            rec_target = (rec_target - torch.min(rec_target))/(torch.max(rec_target)-torch.min(rec_target))
            freq_address = mse_loss(freq, freq_target) #MSE loss with frequency
            rec_address = mse_loss(rec, rec_target) #MSE loss with recency
            loss = (alpha)*loss_address + (beta)*freq_address + (1-alpha-beta)*rec_address
            loss.backward()
            losses.append(loss.item())
            # ...log the running loss
            writer.add_scalar('loss/train/', loss.item(), epoch*len(dataloader) + i-1)
            writer.add_scalar('loss/address/', loss_address, epoch*len(dataloader) + i-1)
            writer.add_scalar('loss/freq/', freq_address, epoch*len(dataloader) + i-1)
            writer.add_scalar('loss/rec/', rec_address, epoch*len(dataloader) + i-1)
            optimizer.step()

        print('Epoch {} with loss: {}'.format(epoch+1,np.mean(losses)))
        print('-------------------------')
            
        if np.mean(losses) < best_loss:
            best_loss = np.mean(losses)
            best_epoch = epoch+1
            torch.save(model, 'checkpoints/deep_cache_testgen_sigmoid_4.pt')
            print('Saved at epoch {} with loss: {}'.format(epoch+1,np.mean(losses)))
            print('---------------------')
    print('---------------------')
    print('Best Epoch: {}'.format(best_epoch))
    print('---------------------')