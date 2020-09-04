import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import utils.dataset as dataset
import random

train_x = torch.rand(10,200,20)
train_y = torch.rand(10,200,3)

class Decoder(nn.Module):
    def __init__(self, d_in, pretrained_embeddings):
        super(Decoder,self).__init__()
        self.linear1 = nn.Linear(d_in, 10)
        self.linear2 = nn.Linear(10, 2)
        self.embedding_layer = torch.nn.Embedding.from_pretrained(pretrained_embeddings)
    
    def get_embedding(self,input):
        return None

    def forward(self, input):
        address = [lambda : F.gumbel_softmax(x, tau =1, hard = True) for x in input]
        embeddings = self.get_embedding(address)
        x = self.linear1(embeddings)
        x = self.linear2(x)
        return x

class Decoder_lstm(nn.Module):
    def __init__(self,d_in,d_out):
        super(Decoder_lstm,self).__init__()
        self.linear1 = nn.Linear(d_in,d_out)
        self.linear2 = nn.Linear(d_in,d_out)
        self.linear3 = nn.Linear(d_in,d_out)
        self.linear4 = nn.Linear(d_in,d_out)

    def forward(self,x):
        x1 = self.linear1(x)
        x2 = self.linear2(x)
        x3 = self.linear3(x)
        x4 = self.linear4(x)
        logits = [x1,x2,x3,x4]
        return [ lambda x: F.softmax(x) for x in logits], logits
    


class DeepCache(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, ):
        super(DeepCache,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTMCell(self.input_size, self.hidden_size)
        self.lstm_decoder = Decoder_lstm(self.hidden_size, self.output_size)
        self.decoder = Decoder(input_size)



    def get_freq_rec(self, input):
        input = (input>0.5)
        embedding = self.get_embedding(input)
        output = self.decoder(embedding)
        return output

    def forward(self, input,future =0,y=None):

        outputs_seq = []
        outputs_freq = []
        outputs_rec = []
        # reset the state of LSTM
        # the state is kept till the end of the sequence
        h_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float32)
        c_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float32)
        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm(input_t, (h_t, c_t))
            output_seq = self.lstm_decoder(h_t)
            output_freq , output_rec = self.get_freq_rec(output_seq)
            outputs_seq += [output_seq]
            outputs_freq += [output_freq]
            outputs_rec += [output_rec]

        for i in range(future):
            if y is not None and random.random() > 0.5:
                output = y[:, [i]]  # teacher forcing
            h_t, c_t = self.lstm(output, (h_t, c_t))
            output_seq = self.lstm_decoder(h_t)
            outputs_seq += [output_seq]
        outputs_seq = torch.stack(outputs_seq, 1).squeeze(2)
        return outputs_seq


        

    
        
        



