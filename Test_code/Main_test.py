#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 14:25:45 2021

@author: caoyukun
"""
import sys
sys.path.append("..")
sys.path.append("../Train_code")
from TaskGenerator_Offline import TaskGeneratorOffline
from GlobalConfigs import OfflineTaskGenerate_configs
import pandas as pd
import torch

data=pd.read_csv('../Data/newdata.csv')                  
DSmodel=torch.load("../Trained_models/DS_model_nomal_service.bin")

    
mode_list={'1':{'attr_list':[['price', 'powerPS']],
                'u_flag':None}}
    
for m in mode_list.keys():
    TGmodel=TaskGeneratorOffline(
            DSmodel,
            OfflineTaskGenerate_configs['task_num'],
            OfflineTaskGenerate_configs['support_tuple_num'],
            OfflineTaskGenerate_configs['intrval_flag'],
            OfflineTaskGenerate_configs['space_list'],
            OfflineTaskGenerate_configs['path']+"_mode"+str(m),
            OfflineTaskGenerate_configs['task_complexity'],
            OfflineTaskGenerate_configs['taskspace_topk'],
            OfflineTaskGenerate_configs['queryspace_topk'],
            OfflineTaskGenerate_configs['multi_intrval'],
            mode_list[m],
            30)
        
    TGmodel.build(None)

#Quick generation for different labeling budgets    
#label_num=[100]
#for num in  label_num:
#   DSmodel=torch.load("DS_model_nomal_service_"+str(num)+".bin")
#   DSmodel.DataCenters_queryspace_all=DSmodel.query_space_model.cluster_centers_.copy()
#   for i in range(DSmodel.attr_dim):
#       DSmodel.DataCenters_queryspace_all[:,i]=DSmodel.DataCenters_queryspace_all[:,i]*(DSmodel.normal_interval[i][1]-DSmodel.normal_interval[i][0])+DSmodel.normal_interval[i][0]
#   for m in mode_list.keys():
#       qps=pickle.load(open('{}/qps.p'.format(OfflineTaskGenerate_configs['path']+"_mode"+str(m)), 'rb'))
#       TGmodel=TaskGeneratorOffline(
#                DSmodel,
#                OfflineTaskGenerate_configs['task_num'],
#                num+5,
#                OfflineTaskGenerate_configs['intrval_flag'],
#                OfflineTaskGenerate_configs['space_list'],
#                OfflineTaskGenerate_configs['path']+"_mode"+str(m)+'_'+str(num),
#                OfflineTaskGenerate_configs['task_complexity'],
#                OfflineTaskGenerate_configs['taskspace_topk'],
#                OfflineTaskGenerate_configs['queryspace_topk'],
#                OfflineTaskGenerate_configs['multi_intrval'],
#                mode_list[m],
#                30)      
#       TGmodel.build(qps)
   

