
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 13:57:06 2021

@author: caoyukun
"""
import torch
import pickle
from numpy import float32,int64,float16
from utils import to_torch,load_query_info,load_offline_query_info,UserDataLoader,get_params,get_zeros_like_params,init_params,init_q_mem_params,init_qt_mem_params,get_grad,update_parameters,Evaluation
from utils import Evaluation2
from torch.utils.data import DataLoader
import matplotlib.pylab as plt
import numpy as np

    

class BASEModel(torch.nn.Module):
    def __init__(self, input1_loading,input2_loading,embedding1_module, embedding2_module, classification_module):
        super(BASEModel, self).__init__()
        self.query_loading=input1_loading
        self.tuple_loading=input2_loading

        self.query_embedding = embedding1_module
        self.tuple_embedding = embedding2_module
        self.classification_model = classification_module

    def forward(self, query_vector, tuple_vector):
        ql=self.query_loading(query_vector)
        tl=self.tuple_loading(tuple_vector)
        qe=self.query_embedding(ql)
        te=self.tuple_embedding(tl)
 
        classification_value= self.classification_model(qe, te)
        return classification_value

    def get_weights(self):
        q_emb_params = get_params(self.query_embedding.parameters())
        t_emb_params = get_params(self.tuple_embedding.parameters())
        cf_params = get_params(self.classification_model.parameters())
        return q_emb_params, t_emb_params, cf_params

    def get_zero_weights(self):
        zeros_like_q_emb_params = get_zeros_like_params(self.query_embedding.parameters())
        zeros_like_t_emb_params = get_zeros_like_params(self.tuple_embedding.parameters())
        zeros_like_cf_params = get_zeros_like_params(self.classification_model.parameters())
        return zeros_like_q_emb_params, zeros_like_t_emb_params, zeros_like_cf_params

    def init_weights(self, q_emb_para, t_emb_para, cf_para):
        init_params(self.query_embedding.parameters(),q_emb_para)
        init_params(self.tuple_embedding.parameters(), t_emb_para)
        init_params(self.classification_model.parameters(), cf_para)

    def get_grad(self):
        q_grad = get_grad(self.query_embedding.parameters())
        t_grad = get_grad(self.tuple_embedding.parameters())
        c_grad = get_grad(self.classification_model.parameters())
        return q_grad, t_grad, c_grad

    def init_q_mem_weights(self, q_emb_para, mu, tao, t_emb_para, cf_para):
        init_q_mem_params(self.query_embedding.parameters(), q_emb_para, mu, tao)
        init_params(self.tuple_embedding.parameters(), t_emb_para)
        init_params(self.classification_model.parameters(), cf_para)

    def init_qt_mem_weights(self, att_values, query_mem):
        # init the weights only for the mem layer
        q_mui = query_mem.read_head(att_values)
        init_qt_mem_params(self.classification_model.mem_layer.parameters(), q_mui )

    def get_qt_mem_weights(self):
        return get_params(self.classification_model.mem_layer.parameters())


class LOCALUpdate:
    def __init__(self, your_model, q_idx, sup_size, que_size, bt_size, n_loop, update_lr, path,device):
       
        self.s_q_vector, self.s_t_vector, self.s_y, self.q_q_vector, self.q_t_vector, self.q_y = load_query_info(q_idx,
                                                                                                              sup_size,
                                                                                                              que_size,
                                                                                                              path,
                                                                                                              device)
        print(self.s_t_vector.size())
        user_data = UserDataLoader(self.s_q_vector, self.s_t_vector, self.s_y)
        self.user_data_loader = DataLoader(user_data, batch_size=bt_size)
        del  user_data 
        
        self.model = your_model

        self.update_lr = update_lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.update_lr)

        
        self.loss_fn = torch.nn.CrossEntropyLoss()
        #self.loss_fn = torch.nn.BCEWithLogitsLoss() 

        self.n_loop = n_loop

        self.device = device
        self.s_q_vector, self.s_t_vector, self.s_y = self.s_q_vector.to(self.device), self.s_t_vector.to(self.device), self.s_y.to(self.device)
        self.q_q_vector, self.q_t_vector, self.q_y = self.q_q_vector.to(self.device), self.q_t_vector.to(self.device), self.q_y.to(self.device)

        
    def train(self):
        for i in range(self.n_loop):

            # on support set
            for i_batch, (qv, tv, y) in enumerate(self.user_data_loader):
                qv, tv, y = qv.to(self.device), tv.to(self.device), y.to(self.device)
                pred_y = self.model(qv, tv)
                loss = self.loss_fn(pred_y, y)
                self.optimizer.zero_grad()
                loss.backward()  # local theta updating
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optimizer.step()

        q_pred_y = self.model(self.q_q_vector, self.q_t_vector)
        self.optimizer.zero_grad()
        print(q_pred_y[:5])
        print(torch.argmax(q_pred_y, dim=1).cpu())
        loss = self.loss_fn(q_pred_y, self.q_y)
        print("global:",loss.item())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)

        q_grad, t_grad, c_grad = self.model.get_grad()
        return q_grad, t_grad, c_grad

    def test(self):
        for i in range(self.n_loop):
            # on support set
            for i_batch, (qv, tv, y) in enumerate(self.user_data_loader):
                qv, tv, y = qv.to(self.device), tv.to(self.device), y.to(self.device)
               
                pred_y = self.model(qv, tv)
                
                loss = self.loss_fn(pred_y, y)
                self.optimizer.zero_grad()
                loss.backward()  # local theta updating
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optimizer.step()

        q_pred_y = self.model(self.q_q_vector, self.q_t_vector)  # on query set
        Evaluation(self.q_y, q_pred_y)

class LOCALUpdate_offline:
    def __init__(self, your_model, group_id,q_idx, sup_size,que_size, bt_size, n_loop, update_lr, path,device,dim):
    
        self.s_q_vector, self.s_t_vector, self.s_y, self.q_q_vector, self.q_t_vector, self.q_y= load_offline_query_info(group_id,
                                                                                                                q_idx,
                                                                                                              sup_size,
                                                                                                              que_size,
                                                                                                              path,
                                                                                                              device,
                                                                                                              dim)
             
        user_data = UserDataLoader(self.s_q_vector, self.s_t_vector, self.s_y)
        self.user_data_loader = DataLoader(user_data, batch_size=bt_size)
        del  user_data 
        
        self.model = your_model

        self.update_lr = update_lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.update_lr)

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.n_loop = n_loop
        self.device = device
        
        
        self.s_q_vector, self.s_t_vector, self.s_y = self.s_q_vector.to(self.device), self.s_t_vector.to(self.device), self.s_y.to(self.device)
        self.q_q_vector, self.q_t_vector, self.q_y = self.q_q_vector.to(self.device), self.q_t_vector.to(self.device), self.q_y.to(self.device)
        
        self.center_index=[]
        for i in range(sup_size-5):
            if self.s_y[i]==1:
                self.center_index.append(i)


    def test(self):
        for i in range(self.n_loop):
            # on support set
            for i_batch, (qv, tv, y) in enumerate(self.user_data_loader):
                qv, tv, y = qv.to(self.device), tv.to(self.device), y.to(self.device)
                pred_y = self.model(qv, tv)
                loss = self.loss_fn(pred_y, y)
                self.optimizer.zero_grad()
                loss.backward()  # local theta updating
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optimizer.step()

        print(self.q_q_vector.shape)
        print(self.q_t_vector.shape)
        q_pred_y = self.model(self.q_q_vector, self.q_t_vector)  # on query set 
        labels,Accuracy,Recall,Precision,F1_score=Evaluation(self.q_y, q_pred_y)
        return self.q_y.numpy(),labels,Accuracy,Recall,Precision,F1_score
   

class LOCALUpdate_offline_select:
    def __init__(self, your_model, group_id,q_idx, sup_size,que_size, bt_size, n_loop, update_lr, path,device,dim,old_t,old_l,incre_t,incre_l):
    
        self.s_q_vector, _, self.s_y, self.q_q_vector, self.q_t_vector, self.q_y= load_offline_query_info(group_id,
                                                                                                              q_idx,
                                                                                                              sup_size,
                                                                                                              que_size,
                                                                                                              path,
                                                                                                              device,
                                                                                                              dim)
        
        if incre_t is not None:
            print(old_t.shape)
            print(incre_t.shape)
            
            self.new_t=np.vstack((old_t,incre_t))
            self.new_l=np.append(old_l,incre_l)
            self.new_t=self.new_t.astype(float32)
            self.new_l=self.new_l.astype(int64)
        else:
            self.new_t=old_t
            self.new_l=old_l
            
           
            
        s_qv=np.tile(self.s_q_vector[0], (len(self.new_l), 1))
        user_data = UserDataLoader(s_qv,self.new_t, self.new_l)
        self.user_data_loader = DataLoader(user_data, batch_size=bt_size)
        del  user_data 
        
        self.model = your_model
        self.update_lr = update_lr
        print(self.update_lr)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.update_lr)

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.n_loop = n_loop
        self.device = device
        self.s_q_vector= self.s_q_vector.to(self.device)
        self.q_q_vector, self.q_t_vector, self.q_y = self.q_q_vector.to(self.device), self.q_t_vector.to(self.device), self.q_y.to(self.device)
        
        self.center_index=[]
        for i in range(sup_size-5):
            if self.s_y[i]==1:
                self.center_index.append(i)


    def test(self):
        for i in range(self.n_loop):
            # on support set
            for i_batch, (qv, tv, y) in enumerate(self.user_data_loader):
                qv, tv, y = qv.to(self.device), tv.to(self.device), y.to(self.device)
                pred_y = self.model(qv, tv)
                loss = self.loss_fn(pred_y, y)
                self.optimizer.zero_grad()
                loss.backward()  # local theta updating
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optimizer.step()

        print(self.q_q_vector.shape)
        print(self.q_t_vector.shape)
        q_pred_y = self.model(self.q_q_vector, self.q_t_vector)  # on query set
        
        q_n=torch.sigmoid(q_pred_y).cpu()
        log_q = torch.log(q_n)
        self.en=-q_n[:,0]*log_q[:,0]-q_n[:,1]*log_q[:,1]
        self.qp=q_pred_y
        labels,Accuracy,Recall,Precision,F1_score=Evaluation(self.q_y, q_pred_y)
        return self.q_y.numpy(),labels,Accuracy,Recall,Precision,F1_score


def maml_train(raw_phi_q, raw_phi_t, raw_phi_c, q_grad_list, t_grad_list, c_grad_list, global_lr):
    phi_q = update_parameters(raw_phi_q, q_grad_list, global_lr)
    phi_t = update_parameters(raw_phi_t, t_grad_list, global_lr)
    phi_c = update_parameters(raw_phi_c, c_grad_list, global_lr)
    return phi_q, phi_t, phi_c


def user_mem_init(q_id, path, device, feature_mem, loading_model, alpha):
    q_data = pickle.load(open('{}/sample_{}_qv.p'.format(path, str(q_id)), 'rb'))
    q_v = to_torch([q_data]).to(device)

    pq = loading_model(q_v)
    personalized_bias_term, att_values = feature_mem.read_head(pq, alpha)
    del q_data , q_v, pq
    return personalized_bias_term, att_values

def user_mem_init_offline(q_id, group_id,path, device, feature_mem, loading_model, alpha,dim):
    if dim==None:
        q_data = pickle.load(open('{}/group_{}_'.format(path,group_id)+'sample_'+str(q_id)+'_qv.p', 'rb'))
    else:
        q_data = pickle.load(open('{}/group_{}_D{}_'.format(path,group_id,dim)+'sample_'+str(q_id)+'_qv.p', 'rb'))
    q_v = to_torch([q_data]).to(device)
    pq = loading_model(q_v)
    personalized_bias_term, att_values = feature_mem.read_head(pq, alpha)
    del q_data , q_v, pq
    return personalized_bias_term, att_values


