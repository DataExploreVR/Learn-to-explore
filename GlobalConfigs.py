#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 09:03:50 2021

@author: caoyukun
"""

dataspace_configs={
        'cluster_method':'Kmeans',
        'cluster_queryspace_num':25,
        'cluster_task_space_num':100,
        'cluster_queryset_space_num':200,
        'represent_method':'nomal',
        'online_split_list':[['price', 'powerPS']],
        'offline_split_list':[['price', 'powerPS']],
        'cluster_sample_rate':0.8,
        'JK_GMM_sample_rate':0.8,
        }



taskGenerate_configs={
        'train_task_num':20000,
        'test_task_num':5, # These tasks are not related to meta-training and are only used as intermediate test results, which can be set to 0.
        'support_tuple_num':dataspace_configs['cluster_queryspace_num']+5,
        'query_tuple_num':dataspace_configs['cluster_queryset_space_num'],
        'split_list':dataspace_configs['online_split_list'],
        'task_complexity':4, # α in UIS modes of paper
        'taskspace_topk':20, # ψ in UIS modes of paper
        'queryspace_topk':int(dataspace_configs['cluster_task_space_num']*0.1), # l in paper
        }

#The path to the offline storage of the pre-trained meta-tasks
taskGenerate_configs['path']= 'train_task_root_'+str(taskGenerate_configs['train_task_num'])+"_"+str(taskGenerate_configs['task_complexity'])+"_"+str(taskGenerate_configs['taskspace_topk'])+"_"+str(dataspace_configs['cluster_queryspace_num'])+"_"+str(dataspace_configs['represent_method'])



#The settings here can be found in the MAMO‘s settings : https://github.com/dongmanqing/Code-for-MAMO 
mamexplore_configs={
     'support_size':taskGenerate_configs['support_tuple_num'],
     'query_size':taskGenerate_configs['query_tuple_num']+int(taskGenerate_configs['query_tuple_num']/2), # Here just set 'query_size' greater than or equal to 'query_tuple_num'
     'train_task_num':taskGenerate_configs['train_task_num'],
     'test_task_num':taskGenerate_configs['test_task_num'],
     'n_epoch':4,
     'n_inner_loop':30,
     'batch_size':16,
     'n_layer':2,
     'embedding_dim':100,
     'query_loading_dim':100,
     'tuple_loading_dim':100,
     'rho': 5e-5,
     'lamda':5e-5,
     'tao': 5e-5,
     'cuda_option':'cpu',
     'n_k':6,
     'alpha': 5e-5,
     'beta': 5e-6,
     'gamma': 5e-5,
     'active_func': 'leaky_relu',
     'path':taskGenerate_configs['path'],
     'query_vector_dim':dataspace_configs['cluster_task_space_num'],
     'tuple_vector_dim':None #Depends on the different data sets
     }
 
# Generate settings for the benchmark tasks used for testing
OfflineTaskGenerate_configs={
        'task_num':50,
        'intrval_flag':False,
        'space_stype':'pre_define', # "pre_define" or "random"
        'space_list':dataspace_configs['offline_split_list'],
        'task_complexity':2, 
        'taskspace_topk':20, 
        'queryspace_topk':10,
        }

#The path to the offline storage of the test tasks
OfflineTaskGenerate_configs['path']='test_offline_task_'+str(OfflineTaskGenerate_configs['task_num'])+"_"+str(OfflineTaskGenerate_configs['task_complexity'])+"_"+str(OfflineTaskGenerate_configs['taskspace_topk'])

#Quick parameter setting for different labeling budgets
if OfflineTaskGenerate_configs['intrval_flag']:
    OfflineTaskGenerate_configs['support_tuple_num']=taskGenerate_configs['support_tuple_num']+40
    OfflineTaskGenerate_configs['multi_intrval']=[taskGenerate_configs['support_tuple_num'],taskGenerate_configs['support_tuple_num']+20,taskGenerate_configs['support_tuple_num']+40]
else:
    OfflineTaskGenerate_configs['support_tuple_num']=taskGenerate_configs['support_tuple_num']
    OfflineTaskGenerate_configs['multi_intrval']=[taskGenerate_configs['support_tuple_num']]

