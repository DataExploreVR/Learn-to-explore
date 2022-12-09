#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 16:00:00 2021

@author: caoyukun
"""

from utils import activation_func
import torch
import torch.nn.functional as F

# ======================Embedding=========================
# input_loading
class InputLoading(torch.nn.Module): # sharing
    def __init__(self,vector_dim,embedding_dim):
        super(InputLoading, self).__init__()
        self.vector_dim=vector_dim
        self.F=torch.nn.LeakyReLU(0.001)

        self.embedding_dim = embedding_dim
        self.embedding_layer = torch.nn.Linear(in_features=self.vector_dim, out_features=self.embedding_dim)
      
    def forward(self, x1):
        emb=self.F(self.embedding_layer(x1))
        return  emb


# query embedding
class QueryEmbedding(torch.nn.Module):
    def __init__(self, n_layer, in_dim, embedding_dim, activation='relu'):
        super(QueryEmbedding, self).__init__()
        self.input_size = in_dim

        fcs = []
        last_size = self.input_size
        hid_dim = int(self.input_size/2)

        for i in range(n_layer - 1):
            linear_model = torch.nn.Linear(last_size, hid_dim)
            linear_model.bias.data.fill_(0.0)
            fcs.append(linear_model)
            last_size = hid_dim
            fcs.append(activation_func(activation))

        self.fc = torch.nn.Sequential(*fcs)

        finals = [torch.nn.Linear(last_size, embedding_dim), activation_func(activation)]
        self.final_layer = torch.nn.Sequential(*finals)

    def forward(self, x):
        x = self.fc(x)
        out = self.final_layer(x)
        return out

# tuple embedding
class TupleEmbedding(torch.nn.Module):
    def __init__(self, n_layer, in_dim, embedding_dim, activation='relu'):
        super(TupleEmbedding, self).__init__()
        self.input_size = in_dim

        fcs = []
        last_size = self.input_size
        hid_dim = int(self.input_size / 2)

        for i in range(n_layer - 1):
            linear_model = torch.nn.Linear(last_size, hid_dim)
            linear_model.bias.data.fill_(0.0)
            fcs.append(linear_model)
            last_size = hid_dim
            fcs.append(activation_func(activation))

        self.fc = torch.nn.Sequential(*fcs)

        finals = [torch.nn.Linear(last_size, embedding_dim), activation_func(activation)]
        self.final_layer = torch.nn.Sequential(*finals)

    def forward(self, x):
        x = self.fc(x)
        out = self.final_layer(x)
        return out

# Classification_model
class CFMAM(torch.nn.Module):
    def __init__(self, embedding_dim, n_y, n_layer, activation='relu', classification=True):
        super(CFMAM, self).__init__()
        self.input_size = embedding_dim * 2

        self.mem_layer = torch.nn.Linear(self.input_size, self.input_size)
        self.flag=classification

        fcs = []
        last_size = self.input_size

        for i in range(n_layer - 1):
            out_dim = int(last_size / 2)
            linear_model = torch.nn.Linear(last_size, out_dim)
            fcs.append(linear_model)
            last_size = out_dim
            fcs.append(activation_func(activation))

        self.fc = torch.nn.Sequential(*fcs)

        if classification:
            finals = [torch.nn.Linear(last_size, n_y),activation_func('d')]
        else:
            finals = [torch.nn.Linear(last_size, 1),activation_func('sigmoid')]
            
        self.final_layer = torch.nn.Sequential(*finals)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], 1)
        out0 = self.mem_layer(x)
        out = self.fc(out0)
        out = self.final_layer(out)

        return out
