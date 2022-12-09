
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 09:42:57 2021

@author: caoyukun
"""

from shapely.geometry import Point,MultiPoint,Polygon
import numpy as np
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
import torch
import pickle
#from configsnew import taskGenerate_configs,mamexplore_configs,OfflineTaskGenerate_configs
from torch.utils.data import Dataset
from copy import deepcopy

import torch.nn as nn
import torch.nn.functional as F
'''
一些会用到到类
'''

class CFMAM(torch.nn.Module):
    def __init__(self, embedding_dim, n_y, n_layer, activation='relu', classification=True):
        super(CFMAM, self).__init__()
        self.input_size = embedding_dim * 2

        self.mem_layer = torch.nn.Linear(self.input_size, self.input_size)

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
            finals = [torch.nn.Linear(last_size, n_y)]
        else:
            finals = [torch.nn.Linear(last_size, 1),activation_func('sigmoid')]
        self.final_layer = torch.nn.Sequential(*finals)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], 1)
        out0 = self.mem_layer(x)
        out = self.fc(out0)
        out = self.final_layer(out)
        #out = out.squeeze(dim=-1)
        return out


class ClassifyModel(nn.Module):
    def __init__(self, tuple_vector_dim, tuple_loading_embedding_dim,n_layer,tuple_embedding_dim,n_y,activation='relu'):
        super(ClassifyModel, self).__init__()
        self.tuple_vector_dim=tuple_vector_dim

        self.tuple_loading_embedding_dim = tuple_loading_embedding_dim
        self.embedding_layer = torch.nn.Linear(in_features=self.tuple_vector_dim, out_features=self.tuple_loading_embedding_dim)
        self.tuple_embedding_dim=tuple_embedding_dim
        fcs1 = []
        last_size1 = self.tuple_loading_embedding_dim
        hid_dim1 = int(self.tuple_loading_embedding_dim / 2)

        for i in range(n_layer - 1):
            linear_model1 = torch.nn.Linear(last_size1, hid_dim1)
            linear_model1.bias.data.fill_(0.0)
            fcs1.append(linear_model1)
            last_size1 = hid_dim1
            fcs1.append(activation_func(activation))

        self.fc1 = torch.nn.Sequential(*fcs1)

        finals1 = [torch.nn.Linear(last_size1, self.tuple_embedding_dim), activation_func(activation)]
        self.final_layer1 = torch.nn.Sequential(*finals1)

        self.input_size2 = self.tuple_embedding_dim
        
        self.mem_layer = torch.nn.Linear(self.input_size2, self.input_size2)

        fcs2 = []
        last_size2 = self.input_size2

        for i in range(n_layer - 1):
            out_dim2 = int(last_size2 / 2)
            linear_model2 = torch.nn.Linear(last_size2, out_dim2)
            fcs2.append(linear_model2)
            last_size2 = out_dim2
            fcs2.append(activation_func(activation))

        self.fc2 = torch.nn.Sequential(*fcs2)

        finals2 = [torch.nn.Linear(last_size2, n_y)]
        #finals2 = [torch.nn.Linear(last_size2, 1),activation_func('sigmoid')]
        self.final_layer2 = torch.nn.Sequential(*finals2)
        
      

    def forward(self, x1):
        emb=F.relu(self.embedding_layer(x1))
        x = self.fc1(emb)
        out = self.final_layer1(x)
        out = self.mem_layer(out)
        out = self.fc2(out)
        out = self.final_layer2(out)
        #out = out.squeeze(dim=-1)
   

        return out


class ConvexSpace():
    '''
    凸包空间模型，ci
    在此类中的点均使用shapely模块Point表示
    '''

    def __init__(self, init_pos):

        self.pos_points = init_pos
        self.pos_region = MultiPoint(self.pos_points).convex_hull

    def in_pos_region(self, point):
        '''是否在正区域'''
        if not isinstance(point, Point):
            point = Point(point)
        return self.pos_region.contains(point) or self.pos_region.touches(point)


    def get_point_region(self, point):
        '''判断一个点所在的区域
        正区域为1'''
        if not isinstance(point, Point):
            point = Point(point)
        if self.in_pos_region(point):
            return 1


    def add_pos_point(self, point):
        '''增加正值点'''
        if not isinstance(point, Point):
            point = Point(point)
        #更新正值点集与正区域
        self.pos_points.append(point)
        self.pos_region = MultiPoint(self.pos_points).convex_hull
        
class PolygonSpace():
    '''
    直接在抽样点上创建最大包含多边形，很大可能为非凸
    在此类中的点均使用shapely模块Point表示
    '''

    def __init__(self, init_pos):

        self.pos_points = init_pos
        self.pos_region = Polygon(self.pos_points)

    def in_pos_region(self, point):
        '''是否在正区域'''
        if not isinstance(point, Point):
            point = Point(point)
        return self.pos_region.contains(point) or self.pos_region.touches(point)


    def get_point_region(self, point):
        '''判断一个点所在的区域
        正区域为1'''
        if not isinstance(point, Point):
            point = Point(point)
        if self.in_pos_region(point):
            return 1


    def add_pos_point(self, point):
        '''增加正值点'''
        if not isinstance(point, Point):
            point = Point(point)
        #更新正值点集与正区域
        self.pos_points.append(point)
        self.pos_region = Polygon(self.pos_points)
                


def to_torch(in_list):
    return torch.from_numpy(np.array(in_list))

def load_query_info(query_id, support_size, query_size, path,device):

    qv = pickle.load(open('{}/sample_{}_qv.p'.format(path, str(query_id)), 'rb'))
            
    s_tv = pickle.load(open('{}/sample_{}_s_tv.p'.format(path, str(query_id)), 'rb'))
    s_y = pickle.load(open('{}/sample_{}_s_y.p'.format(path, str(query_id)), 'rb'))
    q_tv = pickle.load(open('{}/sample_{}_q_tv.p'.format(path, str(query_id)), 'rb'))
    q_y = pickle.load(open('{}/sample_{}_q_y.p'.format(path, str(query_id)), 'rb'))
    
    s_qv=np.tile(qv, (support_size, 1))
    q_qv=np.tile(qv, (query_size, 1))
        
    s_q_vector = to_torch(s_qv)
    s_t_vector = to_torch(s_tv)
    s_label=to_torch(s_y)
    
    q_q_vector = to_torch(q_qv)
    q_t_vector = to_torch(q_tv)
    q_label=to_torch(q_y)
    

    return s_q_vector, s_t_vector, s_label, q_q_vector, q_t_vector, q_label

#def load_offline_query_info(query_id, support_size, query_size,path,device=torch.device('cpu')):
#    #path='task_datas_offline_10_20_st_20'
#    print(path)
#    flag=mamexplore_configs['flag']
#    
#    
#    if flag==1:
#        
#        qv = pickle.load(open('{}/sample_{}_qv.p'.format(path, str(query_id)), 'rb'))
#        s_tv = pickle.load(open('{}/sample_s_tv.p'.format(path), 'rb'))
#        s_y = pickle.load(open('{}/sample_{}_s_y.p'.format(path, str(query_id)), 'rb'))
#    
#       
#        q_tv = pickle.load(open('{}/sample_q_tv.p'.format(path), 'rb'))
#        q_y = pickle.load(open('{}/sample_{}_q_y.p'.format(path, str(query_id)), 'rb'))
#        
#    elif flag==2:
#        tit=pickle.load(open('{}/task_index_tuples.p'.format(path), 'rb'))
#        qv = pickle.load(open('{}/sample_{}_qv.p'.format(path, str(query_id)), 'rb'))
#        
#        s_tv = pickle.load(open('{}/sample_{}_s_tv.p'.format(path, str(query_id)), 'rb'))
#        s_y = pickle.load(open('{}/sample_{}_s_y.p'.format(path, str(query_id)), 'rb'))
#        
#        
#        q_tv = pickle.load(open('{}/sample_q_tv.p'.format(path), 'rb'))
##        q_tv = pickle.load(open('{}/sample_{}_q_tv.p'.format(path,str(query_id)), 'rb')) 
#        
#        
#        q_y = pickle.load(open('{}/sample_{}_q_y.p'.format(path, str(query_id)), 'rb'))
#    
#    
#    
#    
#    
#    s_qv=np.tile(qv, (support_size, 1))
#    q_qv=np.tile(qv, (query_size, 1))
#    tit_qv=np.tile(qv,(len(tit),1))
#        
#    s_q_vector = to_torch(s_qv)
#    s_t_vector = to_torch(s_tv)
#    s_label =to_torch(s_y)
#    tit = to_torch(tit)
#    q_q_vector = to_torch(q_qv)
#    q_t_vector = to_torch(q_tv)
#    q_label=to_torch(q_y)
#    tit_qv = to_torch(tit_qv)
#    
#
#    return s_q_vector, s_t_vector, s_label, q_q_vector, q_t_vector, q_label,tit,tit_qv
def load_offline_query_info(group_id,query_id,support_size, query_size,path,device,dim):
    #path='task_datas_offline_10_20_st_20'
    print(path)

    if group_id==None:
#        tit=pickle.load(open('{}/task_index_tuples.p'.format(path), 'rb'))
        qv = pickle.load(open('{}/sample_{}_qv.p'.format(path, str(query_id)), 'rb'))
            
        s_tv = pickle.load(open('{}/sample_{}_s_tv.p'.format(path, str(query_id)), 'rb'))
        s_y = pickle.load(open('{}/sample_{}_s_y.p'.format(path, str(query_id)), 'rb'))
            
        q_tv = pickle.load(open('{}/sample_q_tv.p'.format(path), 'rb'))
           
        q_y = pickle.load(open('{}/sample_{}_q_y.p'.format(path, str(query_id)), 'rb'))
        
        
        
    else:
        if dim==None:
            qv = pickle.load(open('{}/group_{}_'.format(path,group_id)+'sample_'+str(query_id)+'_qv.p', 'rb'))
    
            s_tv = pickle.load(open('{}/group_{}_'.format(path,group_id)+'sample_'+str(query_id)+'_s_tv_trans.p', 'rb'))
            s_y = pickle.load(open('{}/group_{}_'.format(path,group_id)+'sample_'+str(query_id)+'_s_y.p', 'rb'))
                
            q_tv = pickle.load(open('{}/group_{}_'.format(path,group_id)+'sample_q_tv_trans.p', 'rb'))
               
                
                
            q_y = pickle.load(open('{}/group_{}_'.format(path,group_id)+'sample_'+str(query_id)+'_q_y.p', 'rb'))
        else:
            
            qv = pickle.load(open('{}/group_{}_D{}_'.format(path,group_id,dim)+'sample_'+str(query_id)+'_qv.p', 'rb'))
    
            s_tv = pickle.load(open('{}/group_{}_'.format(path,group_id)+'sample_'+str(query_id)+'_s_tv_trans.p', 'rb'))
            s_y = pickle.load(open('{}/group_{}_D{}_'.format(path,group_id,dim)+'sample_'+str(query_id)+'_s_y.p', 'rb'))
                
            q_tv = pickle.load(open('{}/group_{}_'.format(path,group_id)+'sample_q_tv_trans.p', 'rb'))
               
            q_y = pickle.load(open('{}/group_{}_D{}_'.format(path,group_id,dim)+'sample_'+str(query_id)+'_q_y.p', 'rb'))
            
    
    
    
    #print(q_tv.shape)
    #print(q_y.shape)
    s_qv=np.tile(qv, (support_size, 1))
    q_qv=np.tile(qv, (query_size, 1))
        
    s_q_vector = to_torch(s_qv)
    s_t_vector = to_torch(s_tv)
    s_label =to_torch(s_y)
   
    q_q_vector = to_torch(q_qv)
    q_t_vector = to_torch(q_tv)
    q_label=to_torch(q_y)
  
    

    return s_q_vector, s_t_vector, s_label, q_q_vector, q_t_vector, q_label

#def load_offline_task_info(task_id,path,device=torch.device('cpu')):
#    #path='task_datas_offline_10_20_st_20'
#    print(path)
#    
#    tiv = pickle.load(open('{}/sample_{}_tiv.p'.format(path, str(task_id)), 'rb'))
#    qps = pickle.load(open('{}/sample_{}_qps.p'.format(path, str(task_id)), 'rb'))
#    
#    return tiv,qps
    

# ==========================================
class UserDataLoader(Dataset):
    def __init__(self, s_q_vector, s_t_vector, s_label,transform=None):
        self.s_q_vector = s_q_vector
        self.s_t_vector = s_t_vector
        self.s_label = s_label
        self.transform = transform

    def __len__(self):
        return len(self.s_label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        query_info = self.s_q_vector[idx]
        tuple_info = self.s_t_vector[idx]
        label = self.s_label[idx]
       
        return query_info, tuple_info, label

# =============================================
def get_params(param_list):
    params = []
    count = 0
    for param in param_list:
        if count % 2 == 0:
            value = deepcopy(param.data)
            params.append(value)
            del value
        count += 1
    return params


def get_zeros_like_params(param_list):
    zeros_like_params = []
    count = 0
    for param in param_list:
        if count % 2 == 0:
            value = deepcopy(torch.zeros_like(param.data))
            zeros_like_params.append(value)
        count += 1
    return zeros_like_params


def init_params(param_list, init_values):
    count = 0
    init_count = 0
    for param in param_list:
        if count % 2 == 0:
            param.data.copy_(init_values[init_count])
            init_count += 1
        count += 1


def init_q_mem_params(param_list, init_values, bias_term, tao):
    count = 0
    init_count = 0
    for param in param_list:
        if count % 2 == 0:
            param.data.copy_(init_values[init_count]-tao*bias_term[init_count])
            init_count += 1
        count += 1


def init_qt_mem_params(param_list, init_values):
    count = 0
    for param in param_list:
        if count % 2 == 0:
            param.data.copy_(init_values)
        count += 1


def get_grad(param_list):
    count = 0
    param_grads = []
    for param in param_list:
        if count % 2 == 0:
            value = deepcopy(param.grad)
            param_grads.append(value)
            del value
        count += 1
    return param_grads
#def get_params(param_list):
#    params = []
#    count = 0
#    for param in param_list:
#        if 1:
#            value = deepcopy(param.data)
#            params.append(value)
#            del value
#        count += 1
#    return params
#
#
#def get_zeros_like_params(param_list):
#    zeros_like_params = []
#    count = 0
#    for param in param_list:
#        if 1:
#            value = deepcopy(torch.zeros_like(param.data))
#            zeros_like_params.append(value)
#        count += 1
#    return zeros_like_params
#
#
#def init_params(param_list, init_values):
#    count = 0
#    init_count = 0
#    for param in param_list:
#        if 1:
#            param.data.copy_(init_values[init_count])
#            init_count += 1
#        count += 1
#
#
#def init_q_mem_params(param_list, init_values, bias_term, tao):
#    count = 0
#    init_count = 0
#    for param in param_list:
#        if 1 :
#            param.data.copy_(init_values[init_count]-tao*bias_term[init_count])
#            init_count += 1
#        count += 1
#
#
#def init_qt_mem_params(param_list, init_values):
#    count = 0
#    for param in param_list:
#        if 1 :
#            param.data.copy_(init_values)
#        count += 1
#
#
#def get_grad(param_list):
#    count = 0
#    param_grads = []
#    for param in param_list:
#        if 1:
#            value = deepcopy(param.grad)
#            param_grads.append(value)
#            del value
#        count += 1
#    return param_grads

def grads_sum(raw_grads_list, new_grads_list):
    return [raw_grads_list[i]+new_grads_list[i] for i in range(len(raw_grads_list))]


def update_parameters(params, grads, lr):
    return [params[i] - lr*grads[i] for i in range(len(params))]


# ===============================================
def activation_func(name):
    name = name.lower()
    if name == "sigmoid":
        return torch.nn.Sigmoid()
    elif name == "tanh":
        return torch.nn.Tanh()
    elif name == "relu":
        return torch.nn.ReLU()
    elif name == 'softmax':
        return torch.nn.Softmax()
    elif name == 'leaky_relu':
        return torch.nn.LeakyReLU(5e-6)
    else:
        return torch.nn.Sequential()


# ===============================================
def Evaluation(ground_truth, test_result):
    if len(ground_truth) > 0:
        ground_truth=ground_truth.cpu()
        pred_y = torch.argmax(test_result, dim=1).cpu()
        print(pred_y)
        print(ground_truth)
        print("Accuracy:",accuracy_score(ground_truth,pred_y))
        print("Recall:",recall_score(ground_truth,pred_y))
        print("Precision:",precision_score(ground_truth,pred_y))
        print("F1_score:",f1_score(ground_truth,pred_y))
    return pred_y,accuracy_score(ground_truth,pred_y),recall_score(ground_truth,pred_y),precision_score(ground_truth,pred_y),f1_score(ground_truth,pred_y)
    


def Evaluation2(ground_truth, test_result):
    if len(ground_truth) > 0:
        ground_truth=ground_truth.cpu()
        pred_y = test_result.ge(0.5).cpu()
        print(pred_y)
        print(ground_truth)
        print("Accuracy:",accuracy_score(ground_truth,pred_y))
        print("Recall:",recall_score(ground_truth,pred_y))
        print("Precision:",precision_score(ground_truth,pred_y))
        print("F1_score:",f1_score(ground_truth,pred_y))
    return pred_y,accuracy_score(ground_truth,pred_y),recall_score(ground_truth,pred_y),precision_score(ground_truth,pred_y),f1_score(ground_truth,pred_y)
    

