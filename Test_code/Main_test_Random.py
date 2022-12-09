#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 10:30:58 2021

@author: caoyukun
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 14:25:45 2021

@author: caoyukun
"""
import sys
sys.path.append("..")
sys.path.append("../Train_code")
from MAMexplore import MAMexplore
from DataSpace import DataSpace
from TaskGenerator_Offline_Random import TaskGeneratorOffline
from GlobalConfigs import OfflineTaskGenerate_configs
import pandas as pd
import torch
import pickle


data=pd.read_csv('../Data/SDSS_10w.csv')  
data=data[['rowc', 'colc','sky_u','sky_g','ra','dec','nDetect','nEdge']].astype('float32') 

DSmodel=torch.load("../Trained_models/DS2_model_nomal_service.bin")


mode_list = {'1':{'attr_list': [['rowc', 'colc'],['sky_u','sky_g'],['ra', 'dec']],
                   'u_flag': None,
                   'dim': [1,1,1],
                   'reg': [[1, 15], [1, 20], [1, 25], [1, 30]]},
             '2':{'attr_list': [['rowc', 'colc'],['sky_u','sky_g']],
                   'u_flag': None,
                   'dim': [2,1],
                   'reg': [[1, 15], [1, 20], [1, 25], [1, 30]]},
             '3':{'attr_list': [['sky_u','sky_g'],['ra', 'dec']],
                   'u_flag': None,
                   'dim': [2,1],
                   'reg': [[1, 15], [1, 20], [1, 25], [1, 30]]},
             '4':{'attr_list': [['rowc', 'colc'],['sky_u','sky_g']],
                   'u_flag': None,
                   'dim': [1,2],
                   'reg': [[1, 15], [1, 20], [1, 25], [1, 30]]},
             '5':{'attr_list': [['sky_u','sky_g'],['ra', 'dec']],
                   'u_flag': None,
                   'dim': [1,2],
                   'reg': [[1, 15], [1, 20], [1, 25], [1, 30]]},
             '6':{'attr_list': [['rowc', 'colc']],
                   'u_flag': None,
                   'dim': [3],
                   'reg': [[1, 15], [1, 20], [1, 25], [1, 30]]},
             '7':{'attr_list': [['sky_u','sky_g']],
                   'u_flag': None,
                   'dim': [3],
                   'reg': [[1, 15], [1, 20], [1, 25], [1, 30]]},
             '8':{'attr_list': [['rowc', 'colc'],['sky_u','sky_g'],['ra', 'dec']],
                   'u_flag': None,
                   'dim': [1,1,2],
                   'reg': [[1, 15], [1, 20], [1, 25], [1, 30]]},
             '9':{'attr_list': [['rowc', 'colc'],['sky_u','sky_g'],['ra', 'dec']],
                   'u_flag': None,
                   'dim': [1,2,1],
                   'reg': [[1, 15], [1, 20], [1, 25], [1, 30]]},
             '10':{'attr_list': [['rowc', 'colc'],['sky_u','sky_g'],['ra', 'dec']],
                   'u_flag': None,
                   'dim': [2,1,1],
                   'reg': [[1, 15], [1, 20], [1, 25], [1, 30]]},
             '11':{'attr_list': [['rowc', 'colc'],['sky_u','sky_g']],
                   'u_flag': None,
                   'dim': [2,2],
                   'reg': [[1, 15], [1, 20], [1, 25], [1, 30]]},
             '12':{'attr_list': [['sky_u','sky_g'],['ra', 'dec']],
                   'u_flag': None,
                   'dim': [2,2],
                   'reg': [[1, 15], [1, 20], [1, 25], [1, 30]]},
             '13':{'attr_list': [['rowc', 'colc'],['sky_u','sky_g']],
                   'u_flag': None,
                   'dim': [1,3],
                   'reg': [[1, 15], [1, 20], [1, 25], [1, 30]]},
             '14':{'attr_list': [['sky_u','sky_g'],['ra', 'dec']],
                   'u_flag': None,
                   'dim': [1,3],
                   'reg': [[1, 15], [1, 20], [1, 25], [1, 30]]},
             '15':{'attr_list': [['rowc', 'colc'],['sky_u','sky_g']],
                   'u_flag': None,
                   'dim': [3,1],
                   'reg': [[1, 15], [1, 20], [1, 25], [1, 30]]},
             '16':{'attr_list': [['sky_u','sky_g'],['ra', 'dec']],
                   'u_flag': None,
                   'dim': [3,1],
                   'reg': [[1, 15], [1, 20], [1, 25], [1, 30]]},
             '17':{'attr_list': [['rowc', 'colc']],
                   'u_flag': None,
                   'dim': [4],
                   'reg': [[1, 15], [1, 20], [1, 25], [1, 30]]},
             '18':{'attr_list': [['sky_u','sky_g']],
                   'u_flag': None,
                   'dim': [4],
                   'reg': [[1, 15], [1, 20], [1, 25], [1, 30]]},
             '19':{'attr_list': [['rowc', 'colc'],['sky_u','sky_g']],
                   'u_flag': None,
                   'dim': [1,1],
                   'reg': [[1, 15], [1, 20], [1, 25], [1, 30]]},
             '20':{'attr_list': [['sky_u','sky_g'],['ra', 'dec']],
                   'u_flag': None,
                   'dim': [1,1],
                   'reg': [[1, 15], [1, 20], [1, 25], [1, 30]]},
             '21':{'attr_list': [['rowc', 'colc']],
                   'u_flag': None,
                   'dim': [2],
                   'reg': [[1, 15], [1, 20], [1, 25], [1, 30]]},
             '22':{'attr_list': [['sky_u','sky_g']],
                   'u_flag': None,
                   'dim': [2],
                   'reg': [[1, 15], [1, 20], [1, 25], [1, 30]]},
             '23':{'attr_list': [['rowc', 'colc']],
                   'u_flag': None,
                   'dim': [1],
                   'reg': [[1, 15], [1, 20], [1, 25], [1, 30]]},
             '24':{'attr_list': [['sky_u','sky_g']],
                   'u_flag': None,
                   'dim': [1],
                   'reg': [[1, 15], [1, 20], [1, 25], [1, 30]]},
             '25':{'attr_list': [['ra', 'dec']],
                   'u_flag': None,
                   'dim': [1],
                   'reg': [[1, 15], [1, 20], [1, 25], [1, 30]]},
            }



random_sample_num=30
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
            random_sample_num)
        
    TGmodel.build(None)  
    
#label_num=[50,75,100]
#for num in  label_num:
#   DSmodel=torch.load("../Trained_models/DS2_model_nomal_service_"+str(num)+".bin")
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
#            
#       TGmodel.build(qps)
#       
    

