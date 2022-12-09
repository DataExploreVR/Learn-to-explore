#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 10:08:15 2021

@author: caoyukun
"""
import sys
sys.path.append("..")
sys.path.append("../Train_code")
from sklearn.svm import SVC
import pickle
from numpy import float32,int64,float16
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from utils import Evaluation,load_offline_query_info,UserDataLoader,ClassifyModel,to_torch
import torch
from MAMexplore import MAMexplore
from DataSpace import DataSpace
from GlobalConfigs import dataspace_configs,taskGenerate_configs,mamexplore_configs,OfflineTaskGenerate_configs
from Submodule_models import InputLoading,QueryEmbedding,TupleEmbedding,CFMAM
from Main_models import BASEModel
from torch.utils.data import DataLoader
import numpy as np

class Modeltest():
    def __init__(self,DSmodel,mode,offline_task_ids,path,support_size,query_size,batch_size,n_inner_loop,lr,device,label_num):
        self.mode=mode
        self.offline_task_ids=offline_task_ids
        self.path=path
        self.support_size=support_size
        self.query_size=query_size
        self.batch_size=batch_size
        self.n_inner_loop=n_inner_loop
        self.device=device
        self.lr=lr
        self.label_num=label_num
        self.MAM_acc=None
        self.MAM_rec=None
        self.MAM_pre=None
        self.MAM_f1=None 
        self.nom_acc=[]
        self.nom_rec=[]
        self.nom_pre=[]
        self.nom_f1=[]
        self.svm1_acc=[]
        self.svm1_rec=[]
        self.svm1_pre=[]
        self.svm1_f1=[]
        self.svm2_acc=[]
        self.svm2_rec=[]
        self.svm2_pre=[]
        self.svm2_f1=[]        
        self.nom2_acc=[]
        self.nom2_rec=[]
        self.nom2_pre=[]
        self.nom2_f1=[]
        self.tag=mamexplore_configs['path']

    
    def run(self):
        
    
        '''
        Meta
        '''
        
        mode_attr=[]
        for i in self.mode['attr_list']:
            for j in i:
                pos=DSmodel.raw.columns.to_list().index(j)
                if pos not in mode_attr:
                    mode_attr.append(pos)
        mode_attr=sorted(mode_attr)  
        self.mode['mode_attr']=mode_attr
        
        print(self.mode['attr_list'])
        
        attr_list_id=[]
        
        for i in self.mode['attr_list']:
            attr_list_id.append(DSmodel.off_split_list.index(i))
        
        self.mode['attr_list_id']=attr_list_id
        print(self.mode['mode_attr'])
        
        solve={}
        MAMmodel_list={}
        for n in range(len(self.mode['attr_list'])):
        
            attrg_id=self.mode['attr_list_id'][n]
            if attrg_id < len(DSmodel.on_split_list):
                m_path='../Trained_models/Model_train_task_root_'+str(taskGenerate_configs['train_task_num'])+"_"+str(taskGenerate_configs['task_complexity'])+"_"+str(taskGenerate_configs['taskspace_topk'])+"_"+str(self.label_num)+"_"+str(dataspace_configs['represent_method'])
                MAMmodel=torch.load(m_path+"_"+str(attrg_id), map_location=torch.device('cpu'))
                MAMmodel.device=torch.device('cpu')
                MAMmodel.FeatureMEM.device=torch.device('cpu')
                MAMmodel.rho=self.lr
                MAMmodel_list[n]=MAMmodel
                solve[n]=MAMmodel.test_with_offline_task(self.offline_task_ids,n,self.path,self.support_size,self.query_size,self.batch_size,self.n_inner_loop,attrg_id,None)

        self.part_test_labels=[]
        self.cand_test_labels=[]
        for i in self.offline_task_ids:
            temp_part_test_labels=[]
            for n in range(len(self.mode['attr_list'])):
                if n ==0:
                    temp_test_labels=solve[n][i]['pred'].numpy()
                else:
                    temp_test_labels=temp_test_labels & solve[n][i]['pred'].numpy()
                if self.mode['u_flag']!= None and n in self.mode['u_flag']:
                    temp_part_test_labels.append(temp_test_labels)
                    temp_test_labels= np.ones(len(temp_test_labels)).astype(int64)
            temp_part_test_labels.append(temp_test_labels) 
            self.part_test_labels.append(temp_part_test_labels)    
            self.cand_test_labels.append(temp_test_labels)
         
        self.final_test_labels=[]
        if self.mode['u_flag']!= None:
            for i in self.offline_task_ids:
                index3=0
                for j in self.part_test_labels[i]:
                    if index3==0:
                        total_test_labels=j
                    else:
                        total_test_labels = total_test_labels | j
                    index3+=1
                self.final_test_labels.append(total_test_labels)
        else:
            self.final_test_labels=self.cand_test_labels
        
        final_f1=[]
        for i in  self.offline_task_ids:
            ture_final_labes=pickle.load(open('{}/'.format(self.path)+'sample_'+str(i)+'_q_y.p', 'rb'))
            final_f1.append(f1_score(ture_final_labes,self.final_test_labels[i]))
        self.final_f1=final_f1


        '''
        Basic
        '''  
        solve2={}
        for n in range(len(self.mode['attr_list'])):
            solve2[n]={}
            for j in self.offline_task_ids:
                solve2[n][j]={}
        for j in self.offline_task_ids:
            for n in range(len(self.mode['attr_list'])):
                
                query_vector_loading, tuple_vector_loading = InputLoading(MAMmodel_list[n].query_vector_dim,MAMmodel_list[n].query_loading_dim).to(self.device), \
                InputLoading(MAMmodel_list[n].tuple_vector_dim,MAMmodel_list[n].tuple_loading_dim).to(self.device)
                QEmb = QueryEmbedding(MAMmodel_list[n].n_layer, MAMmodel_list[n].query_loading_dim ,
                MAMmodel_list[n].embedding_dim, activation=MAMmodel_list[n].active_func).to(self.device)
                TEmb = TupleEmbedding(MAMmodel_list[n].n_layer, MAMmodel_list[n].tuple_loading_dim,
                MAMmodel_list[n].embedding_dim, activation=MAMmodel_list[n].active_func).to(self.device)
        
                # Classification_model 
                cf_model = CFMAM(MAMmodel_list[n].embedding_dim,MAMmodel_list[n].n_y, MAMmodel_list[n].n_layer, activation=MAMmodel_list[n].active_func).to(self.device)
                normal_model = BASEModel(query_vector_loading,tuple_vector_loading,  QEmb,TEmb, cf_model).to(self.device)
                s_q_vector, s_t_vector, s_y, q_q_vector, q_t_vector, q_y = load_offline_query_info(n,j,self.support_size,self.query_size,self.path,self.device,None)
                s_q_vector, s_t_vector, s_y = s_q_vector.to(self.device), s_t_vector.to(self.device), s_y.to(self.device)
                q_q_vector,  q_t_vector, q_y = q_q_vector.to(self.device), q_t_vector.to(self.device), q_y.to(self.device)
                user_data = UserDataLoader(s_q_vector,s_t_vector,s_y)
                user_data_loader = DataLoader(user_data, batch_size=self.batch_size)
                
                loss_fn = torch.nn.CrossEntropyLoss()
                update_lr= self.lr
                optimizer = torch.optim.Adam(normal_model.parameters(), lr=update_lr)
               
                for i in range(self.n_inner_loop):
                    for i_batch, (qv, tv, y) in enumerate(user_data_loader):
                        qv, tv, y = qv.to(self.device), tv.to(self.device), y.to(self.device)
                        pred_y = normal_model(qv, tv)
                        loss = loss_fn(pred_y, y)
                        optimizer.zero_grad()
                        loss.backward()  
                        torch.nn.utils.clip_grad_norm_(normal_model.parameters(), 5.)
                        optimizer.step()
                q_pred_y = normal_model(q_q_vector, q_t_vector) 
                result,Accuracy,Recall,Precision,F1_score=Evaluation(q_y, q_pred_y)
            
                solve2[n][j]['pred']=result
                solve2[n][j]['Accuracy']=Accuracy
                solve2[n][j]['Recall']=Recall
                solve2[n][j]['Precision']=Precision
                solve2[n][j]['F1_score']=F1_score

        part_test_labels2=[]
        cand_test_labels2=[]
        for i in self.offline_task_ids:
            temp_part_test_labels=[]
            for n in range(len(self.mode['attr_list'])):
                if n ==0:
                    temp_test_labels=solve2[n][i]['pred'].numpy()
                else:
                    temp_test_labels=temp_test_labels & solve2[n][i]['pred'].numpy()
                if self.mode['u_flag']!= None and n in self.mode['u_flag']:
                    temp_part_test_labels.append(temp_test_labels)
                    temp_test_labels= np.ones(len(temp_test_labels)).astype(int64)
            temp_part_test_labels.append(temp_test_labels) 
            part_test_labels2.append(temp_part_test_labels)    
            cand_test_labels2.append(temp_test_labels)
        
           
        final_test_labels2=[]
        if self.mode['u_flag']!= None:
            for i in self.offline_task_ids:
                index3=0
                for j in part_test_labels2[i]:
                    if index3==0:
                        total_test_labels=j
                    else:
                        total_test_labels = total_test_labels | j
                    index3+=1
                final_test_labels2.append(total_test_labels)
        else:
            final_test_labels2=cand_test_labels2
        
        final_f12=[]
        for i in  self.offline_task_ids:
            ture_final_labes=pickle.load(open('{}/'.format(self.path)+'sample_'+str(i)+'_q_y.p', 'rb'))
            final_f12.append(f1_score(ture_final_labes,final_test_labels2[i]))
        self.final_f12=final_f12
        self.solve2=solve2 
        self.solve=solve
        
        '''
        Meta*
        '''

        solve3={}
        MAMmodel_list3={}
        for n in range(len(self.mode['attr_list'])):
        
            attrg_id=self.mode['attr_list_id'][n]
            if attrg_id < len(DSmodel.on_split_list):
                m_path='../Trained_models/Model_train_task_root_'+str(taskGenerate_configs['train_task_num'])+"_"+str(taskGenerate_configs['task_complexity'])+"_"+str(taskGenerate_configs['taskspace_topk'])+"_"+str(self.label_num)+"_"+str(dataspace_configs['represent_method'])
                MAMmodel=torch.load(m_path+"_"+str(attrg_id), map_location=torch.device('cpu'))
                MAMmodel.device=torch.device('cpu')
                MAMmodel.FeatureMEM.device=torch.device('cpu')
                MAMmodel.rho=self.lr
                MAMmodel_list3[n]=MAMmodel
                solve3[n]=MAMmodel.test_with_offline_task_opt(self.offline_task_ids,n,self.path,self.support_size,self.query_size,self.batch_size,self.n_inner_loop,attrg_id,None)
                
                
                         
        self.part_test_labels3=[]
        self.cand_test_labels3=[]
        for i in self.offline_task_ids:
            temp_part_test_labels=[]
            for n in range(len(self.mode['attr_list'])):
                if n ==0:
                    temp_test_labels=solve3[n][i]['pred'].numpy()
                else:
                    temp_test_labels=temp_test_labels & solve3[n][i]['pred'].numpy()
                if self.mode['u_flag']!= None and n in self.mode['u_flag']:
                    temp_part_test_labels.append(temp_test_labels)
                    temp_test_labels= np.ones(len(temp_test_labels)).astype(int64)
            temp_part_test_labels.append(temp_test_labels) 
            self.part_test_labels3.append(temp_part_test_labels)    
            self.cand_test_labels3.append(temp_test_labels)
         
            
        
        self.final_test_labels3=[]
        if self.mode['u_flag']!= None:
            for i in self.offline_task_ids:
                index3=0
                for j in self.part_test_labels[i]:
                    if index3==0:
                        total_test_labels=j
                    else:
                        total_test_labels = total_test_labels | j
                    index3+=1
                self.final_test_labels3.append(total_test_labels)
        else:
            self.final_test_labels3=self.cand_test_labels3
        
        final_f16=[]
        for i in  self.offline_task_ids:
            ture_final_labes=pickle.load(open('{}/'.format(self.path)+'sample_'+str(i)+'_q_y.p', 'rb'))
            final_f16.append(f1_score(ture_final_labes,self.final_test_labels3[i]))
        self.final_f16=final_f16

        self.final_f13=[]
        self.final_f14=[]
        for i in self.offline_task_ids:
            
            q_y=pickle.load(open('{}/'.format(self.path)+'sample_'+str(i)+'_q_y.p', 'rb'))
            s_t_vector=pickle.load(open('{}/'.format(self.path)+'sample_'+str(i)+'_s_tv_trans.p', 'rb'))
            print(s_t_vector.shape)
            
            s_y=pickle.load(open('{}/'.format(self.path)+'sample_'+str(i)+'_s_y.p', 'rb'))
            q_t_vector=pickle.load(open('{}/'.format(self.path)+'sample_q_tv_trans.p', 'rb'))
           

            '''
            SVM^r
            '''                        
            clf = SVC(kernel='rbf',gamma='scale')
            print(s_t_vector.shape)
            print(len(set(s_y)))
            if len(set(s_y))==1:
                self.final_f14.append(0)
            else: 
                clf.fit(s_t_vector, s_y) 
                svm_p=clf.predict(q_t_vector)
                print(f1_score(q_y,svm_p))
                self.final_f14.append(f1_score(q_y,svm_p))
                
        '''
        SVM
        '''
        self.final_f15=[]    
        self.final_f17=[]    
        
        for i in self.offline_task_ids:
            q_y=pickle.load(open('{}/'.format(self.path)+'sample_'+str(i)+'_q_y.p', 'rb'))
            s_t=pickle.load(open('{}/'.format(self.path)+'sample_'+str(i)+'_s_tv.p', 'rb'))
            #print(s_t)
            
            s_y=pickle.load(open('{}/'.format(self.path)+'sample_'+str(i)+'_s_y.p', 'rb'))
            q_t=pickle.load(open('{}/'.format(self.path)+'sample_q_tv.p', 'rb'))
            clf = SVC(kernel='rbf',gamma='scale')
            if len(set(s_y))==1:
                self.final_f15.append(0)
            else:
                clf.fit(s_t, s_y) 
                svm_p=clf.predict(q_t)
                print(f1_score(q_y,svm_p))
                self.final_f15.append(f1_score(q_y,svm_p))
            
            s_t_nom=s_t.copy()
            q_t_nom=q_t.copy()
            for i in range(len(self.mode['mode_attr'])):
                s_t_nom[:,i]= s_t_nom[:,i]*(DSmodel.normal_interval[self.mode['mode_attr'][i]][1]-DSmodel.normal_interval[self.mode['mode_attr'][i]][0])+DSmodel.normal_interval[self.mode['mode_attr'][i]][0]
                q_t_nom[:,i]= q_t_nom[:,i]*(DSmodel.normal_interval[self.mode['mode_attr'][i]][1]-DSmodel.normal_interval[self.mode['mode_attr'][i]][0])+DSmodel.normal_interval[self.mode['mode_attr'][i]][0]

            clf = SVC(kernel='rbf',gamma='scale')
            if len(set(s_y))==1:
                 self.final_f17.append(0)
            else:
                
                clf.fit(s_t_nom, s_y) 
                svm_p=clf.predict(q_t_nom)
                print(f1_score(q_y,svm_p))
                self.final_f17.append(f1_score(q_y,svm_p))
        
        self.cand_test_labels=None
        self.final_test_labels= None
        self.part_test_labels=None
        self.cand_test_labels3=None
        self.final_test_labels3= None
        self.part_test_labels3=None

        self.solve=None
        self.solve2=None
        

            
        
if __name__ == "__main__":        
    DSmodel=torch.load("../Trained_models/DS_model_nomal_service.bin",map_location=torch.device('cpu'))    
        
    mode_list={'1':{'attr_list':[['price', 'powerPS']],
                    'u_flag':None}
        }
    for md in mode_list.keys():
        for i in OfflineTaskGenerate_configs['multi_intrval']: 
            if OfflineTaskGenerate_configs['intrval_flag']==True:
                a=Modeltest(DSmodel,mode_list[md],range(OfflineTaskGenerate_configs['task_num']),OfflineTaskGenerate_configs['path']+"_mode"+md+"_st_"+str(i),i,len(DSmodel.numpy_sample_raw),16,50,0.001,'cpu',25)
                a.run()
            else:
                a=Modeltest(DSmodel,mode_list[md],range(OfflineTaskGenerate_configs['task_num']),OfflineTaskGenerate_configs['path']+"_mode"+md,i,len(DSmodel.numpy_sample_raw),16,50,0.001,'cpu',25)
                a.run()
                pickle.dump(a,open("solve"+OfflineTaskGenerate_configs['path']+"_mode"+md+"_"+str(a.lr)+"_"+a.tag,'wb'))               
            
    
#label_num=[100]    
#for num in  label_num:
#    DSmodel=torch.load("DS_model_nomal_service_"+str(num)+".bin")
#    for md in mode_list.keys():
#        for i in OfflineTaskGenerate_configs['multi_intrval']: 
#            if OfflineTaskGenerate_configs['intrval_flag']==True:
#                a=Modeltest(DSmodel,mode_list[md],range(OfflineTaskGenerate_configs['task_num']),OfflineTaskGenerate_configs['path']+"_mode"+md+"_"+str(num)+"_st_"+str(i),num+5,len(DSmodel.numpy_sample_raw),16,50,0.01,'cpu',num)
#            else:
#                a=Modeltest(DSmodel,mode_list[md],range(OfflineTaskGenerate_configs['task_num']),OfflineTaskGenerate_configs['path']+"_mode"+md+"_"+str(num),num+5,len(DSmodel.numpy_sample_raw),16,50,5e-5,'cpu',num)
#                pickle.dump(a,open("solve"+OfflineTaskGenerate_configs['path']+"_mode"+md+"_"+str(num)+"_"+str(a.lr)+"_"+a.tag,'wb'))               

