#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 16:14:51 2021

@author: caoyukun
"""

from utils import activation_func
import torch

class FeatureMem:
    def __init__(self, n_k, q_emb_dim, base_model, device):
        self.n_k = n_k
        self.base_model = base_model
        self.p_memory = torch.randn(n_k, q_emb_dim, device=device).normal_() 
        u_param, _, _ = base_model.get_weights()
        self.q_memory = []
        for i in range(n_k):
            bias_list = []
            for param in u_param:
                bias_list.append(param.normal_(std=0.05))
            self.q_memory.append(bias_list)
        self.att_values = torch.zeros(n_k).to(device)
        self.device = device

    def read_head(self, p_q, alpha, train=True):
        att_model = Attention(self.n_k).to(self.device)
        
        attention_values = att_model(p_q, self.p_memory).to(self.device)  
        personalized_mq = get_mq(attention_values, self.q_memory, self.base_model, self.device)
        transposed_att = attention_values.reshape(self.n_k, 1)
        product = torch.mm(transposed_att, p_q)
        if train:
            self.p_memory = alpha * product + (1-alpha) * self.p_memory
        self.att_values = attention_values
        return personalized_mq, attention_values

    def write_head(self, q_grads, lr):
        update_mq(self.att_values, self.q_memory, q_grads, lr)


class QueryMem:
    def __init__(self, n_k, emb_dim, device):
        self.n_k = n_k
        self.memory_QT = torch.rand(n_k, emb_dim *2, emb_dim*2, device=device).normal_()
        self.att_values = torch.zeros(n_k)

    def read_head(self, att_values):
        self.att_values = att_values
        return get_mqt(att_values, self.memory_QT, self.n_k)

    def write_head(self, q_mqt, lr):
        update_values = update_mqt(self.att_values, self.n_k, q_mqt)
        self.memory_QT = lr* update_values + (1-lr) * self.memory_QT


def cosine_similarity(input1, input2):
    query_norm = torch.sqrt(torch.sum(input1**2+0.00001, 1))
    doc_norm = torch.sqrt(torch.sum(input2**2+0.00001, 1))

    prod = torch.sum(torch.mul(input1, input2), 1)
    norm_prod = torch.mul(query_norm, doc_norm)

    cos_sim_raw = torch.div(prod, norm_prod)
    return cos_sim_raw

class Attention(torch.nn.Module):
    def __init__(self, n_k, activation='relu'):
        super(Attention, self).__init__()
        self.n_k = n_k
        self.fc_layer = torch.nn.Linear(self.n_k, self.n_k, activation_func(activation))
        self.soft_max_layer = torch.nn.Softmax()

    def forward(self, pu, mp):
        
        expanded_pu = pu.repeat(1, len(mp)).view(len(mp), -1)  
        inputs = cosine_similarity(expanded_pu, mp)
        fc_layers = self.fc_layer(inputs)
        attention_values = self.soft_max_layer(fc_layers)
        return attention_values


def get_mq(att_values, mq, model, device):
    mq0,_,_ = model.get_zero_weights()
    attention_values = att_values.reshape(len(mq),1)
    for i in range(len(mq)):
        for j in range(len(mq[i])):
            mq0[j] += attention_values[i] * mq[i][j].to(device)
    return mq0


def update_mq(att_values, mq, grads, lr):
    att_values = att_values.reshape(len(mq), 1)
    for i in range(len(mq)):
        for j in range(len(mq[i])):
            mq[i][j] = lr * att_values[i] * grads[j] + (1-lr) * mq[i][j]


def get_mqt(att_values, mqt, n_k):
    attention_values = att_values.reshape(n_k, 1, 1)
    attend_mui = torch.mul(attention_values, mqt)
    q_mqt = attend_mui.sum(dim=0)
    return q_mqt


def update_mqt(att_values, n_k, q_mqt):
    repeat_q_mqt = q_mqt.unsqueeze(0).repeat(n_k, 1, 1)
    attention_tensor = att_values.reshape(n_k, 1, 1)
    attend_q_mqt = torch.mul(attention_tensor, repeat_q_mqt)
    return attend_q_mqt
