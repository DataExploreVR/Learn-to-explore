

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 09:23:54 2021

@author: caoyukun
"""

import pandas as pd
import torchvision.transforms as transforms
import math
import shapely
from shapely.geometry import Point, LineString, MultiPoint
from numpy import float32,int64,float16
import pickle
import os.path
import numpy as np
import random
from math import sqrt, acos, pi
from numpy import cross
from utils import *
from DataSpace import *
import matplotlib.pylab as plt

   
        
   
class TaskGeneratorOffline:

    def __init__(self, dataspace,task_num,support_tuple_num,intrval_flag,space_list,path,task_complexity,taskspace_topk,queryspace_topk,multi_intrval,mode,random_sample_num):

        self.dataspace= dataspace #数据空间对象
        self.task_num = task_num #要构造的训练task数目
      
        self.n_way = 2
        self.k_shot = support_tuple_num#支撑集的样本个数，这里与train数据空间的聚类数相同
        self.intrval_flag=intrval_flag
        self.space_list=space_list
        self.path=path #生成的task的数据文件的存储地址
        self.attr_dim=self.dataspace.raw.shape[1]
        self.task_complexity=task_complexity 
        self.taskspace_topk=taskspace_topk
        self.queryspace_topk=queryspace_topk
        self.mode=mode
        self.multi_intrval=multi_intrval
        self.random_sample_num=random_sample_num
        
        
        mode_attr=[]
        for i in mode['attr_list']:
            for j in i:
                pos=self.dataspace.raw.columns.to_list().index(j)
                if pos not in mode_attr:
                    mode_attr.append(pos)
        mode_attr=sorted(mode_attr)  
        self.mode['mode_attr']=mode_attr
        
        
        attr_list_id=[]
        
     
        for i in mode['attr_list']:
            attr_list_id.append(self.dataspace.off_split_list.index(i))
        
        
        self.mode['attr_list_id']=attr_list_id

        
 
    
    def task_2D(self,attr_id,reg):
        random.seed()
        if reg==None:
            region_num=self.task_complexity
            topK=self.taskspace_topk
        else:
            region_num=reg[0]
            topK=reg[1]
        
       
     
        region_num=self.task_complexity
            
        CovSpaces=[]
        indexlist=np.array(range(self.dataspace.cluster_task_space_num))
        for i in range(region_num):
            center=random.sample(list(indexlist),1)

            neighbors=self.dataspace.taskspace_centers_neighbors[attr_id][center[0]][:topK+1]
            
            condition_data=self.dataspace.DataCenters_task_space[attr_id][neighbors]
            
            CovSpace=ConvexSpace(condition_data)
            CovSpaces.append(CovSpace)
            indexlist=np.setdiff1d(indexlist, neighbors)
            
        return CovSpaces
            
            
    def Generate_single(self,mode,qp):
            if qp==None:
                
                query_patterns=[]
                for i in mode['attr_list_id']:
                    if self.taskspace_topk=='random':
                        reg=random.choice(mode['reg'])
                        query_patterns.append(self.task_2D(i,reg))
                    else:
                        query_patterns.append(self.task_2D(i,None))
            else:
                query_patterns=qp
                          
            
            temp_task={}
            
            ori_hot_list=[]
            for n in range(len(query_patterns)):
                temp_task[n]={}
            for n in range(len(query_patterns)):
                
                query_vector_temp=[]
                temp_task[n]['support_tuples']=[]
                temp_task[n]['train_labels']=[]
                temp_task[n]['test_labels']=[]
                temp_task[n]['support_tuples_sample']=[]
                temp_task[n]['train_labels_sample']=[]
                for i in self.dataspace.DataCenters_queryspace[mode['attr_list_id'][n]]:
                    pos_data=i
                    result=False
                    for j in query_patterns[n]:
                        if j.in_pos_region(pos_data):
                            result=True
                            break
                    query_vector_temp.append(result)
                    temp_task[n]['support_tuples'].append(pos_data)
                    temp_task[n]['train_labels'].append(result)
               
                ori_hot_list.append(np.array(query_vector_temp))
           
                
            
            part_hot_list=[] 
            for i in range(len(ori_hot_list)):
                
                tem_pos=np.argsort(~ori_hot_list[i])
                new_list=ori_hot_list[i][tem_pos].copy()
                temp_task[i]['train_labels_total']=list(new_list)
                if qp==None:
                    if np.sum(ori_hot_list[i]==1)==0:
                        return None
                else:
                    if np.sum(ori_hot_list[i]==1)==0:
                       ori_hot_list[i][random.randint(0,len(ori_hot_list[i]-1))]=1 
                    
                if i ==0:
                    temp_hot_list=new_list
                    total_temp=np.array(temp_task[i]['support_tuples'])[tem_pos]
                    total_temp_orioder=np.array(temp_task[i]['support_tuples'])
                else:
                    temp_hot_list=temp_hot_list & new_list
                    total_temp=np.hstack((total_temp,np.array(temp_task[i]['support_tuples'])[tem_pos]))
                    total_temp_orioder=np.hstack((total_temp_orioder,np.array(temp_task[i]['support_tuples'])))
                    
                if mode['u_flag']!= None and i in mode['u_flag']:
                    part_hot_list.append(temp_hot_list)
                    temp_hot_list= np.ones(len(temp_hot_list)).astype(int64)
                
                
            part_hot_list.append(temp_hot_list)
            
            temp_task['total_support_tuples']=total_temp
            
            temp_task['total_support_tuples_orioder']=total_temp_orioder
            if mode['u_flag']!= None:
                index3=0
                for j in part_hot_list:
                    if index3==0:
                        total_hot_list=j.copy()
                    else:
                        total_hot_list = total_hot_list | j
                    index3+=1
            else:
                total_hot_list=temp_hot_list
                
            if qp==None:   
                if np.sum(total_hot_list==1)==0:
                    return None
                
            
                         
            for i in range(len(ori_hot_list)):
                part_query_vector=np.zeros(self.dataspace.cluster_task_space_num)
                for j in range(len(ori_hot_list[i])):
                    if ori_hot_list[i][j]==1:
                        part_query_vector[self.dataspace.queryspace_centers_neighbors[mode['attr_list_id'][i]][j][:self.queryspace_topk]]=True
                temp_task[i]['queryspace_vector']=part_query_vector
            
            
            
            
            
            
            temp_task['total_support_tuples_sample']=[]
            for i in range(self.random_sample_num):
                flag=random.randint(0,len(ori_hot_list)-1)
                index=random.choice(np.where(ori_hot_list[flag]==1)[0])
                
                temp_data=self.dataspace.numpy_sample_raw[np.where(self.dataspace.DataLabels_queryspace[mode['attr_list_id'][flag]]==index)]
                temp_tuple=random.sample(list(temp_data),1)
                temp_task['total_support_tuples_sample'].append(temp_tuple[0][mode['mode_attr']])
                
                #print(temp_data)
               
                for n in range(len(query_patterns)):
                    
                    temp_task[n]['support_tuples_sample'].append(temp_tuple[0][self.dataspace.pos_list_offline[mode['attr_list_id'][n]]]) 
                    result=False
                    for j in query_patterns[n]:
                        if j.in_pos_region(temp_tuple[0][self.dataspace.pos_list_offline[mode['attr_list_id'][n]]]):
                            result=True
                            break
                    temp_task[n]['train_labels_sample'].append(result)

                            
            
            
            for i in range(self.k_shot-self.dataspace.cluster_queryspace_num):
                
                for n in range(len(query_patterns)):
                    index=random.choice(np.where(ori_hot_list[n]==1)[0])
                    temp_data=self.dataspace.numpy_sample_raw[:,self.dataspace.pos_list_offline[mode['attr_list_id'][n]]][np.where(self.dataspace.DataLabels_queryspace[mode['attr_list_id'][n]]==index)]
                    temp_tuple=random.sample(list(temp_data),1)
                    
                    temp_task[n]['support_tuples'].append(temp_tuple[0]) 
                    result=False
                    for j in query_patterns[n]:
                        if j.in_pos_region(temp_tuple[0]):
                            result=True
                            break
                    temp_task[n]['train_labels'].append(result)
                    if n==0:
                        total=temp_tuple[0]
                    else:
                        total=np.hstack((total,temp_tuple[0]))
                    temp_task[n]['train_labels_total'].append(result)
                temp_task['total_support_tuples']=np.vstack((temp_task['total_support_tuples'],total))
                temp_task['total_support_tuples_orioder']=np.vstack((temp_task['total_support_tuples_orioder'],total))
            
                
                    
                    
            for i in range(len(self.dataspace.numpy_sample_raw)):
            
                for n in range(len(query_patterns)):
                    result=False
                    for j in query_patterns[n]:
                        if j.in_pos_region(self.dataspace.numpy_sample_raw[i][self.dataspace.pos_list_offline[mode['attr_list_id'][n]]]):
                            result=True
                            break
                    temp_task[n]['test_labels'].append(result)


            temp_task['U_train_labels_list']=[]
            temp_task['U_test_labels_list']=[]
            temp_task['U_sample_labels_list']=[]
            
            index2=0
            for n in range(len(query_patterns)):
                temp_task[n]['train_labels']=np.array(temp_task[n]['train_labels'])
                temp_task[n]['train_labels_total']=np.array(temp_task[n]['train_labels_total'])
                temp_task[n]['test_labels']=np.array(temp_task[n]['test_labels'])
                temp_task[n]['queryspace_vector']=np.array(temp_task[n]['queryspace_vector'])
                temp_task[n]['train_labels_sample']=np.array(temp_task[n]['train_labels_sample'])
                
                if index2==0:
                    temp_train_labels=temp_task[n]['train_labels_total'].copy()
                    temp_test_labels=temp_task[n]['test_labels'].copy()
                    temp_sample_labels=temp_task[n]['train_labels_sample'].copy()
                else:
                    temp_train_labels=temp_train_labels & temp_task[n]['train_labels_total']
                    temp_test_labels= temp_test_labels & temp_task[n]['test_labels']
                    temp_sample_labels= temp_sample_labels & temp_task[n]['train_labels_sample']
                if mode['u_flag']!= None and index2 in mode['u_flag']:
                    temp_task['U_train_labels_list'].append(temp_train_labels)
                    temp_task['U_test_labels_list'].append(temp_test_labels)
                    temp_task['U_sample_labels_list'].append(temp_sample_labels)
                    
                    temp_train_labels= np.ones(len(temp_train_labels)).astype(int64)
                    temp_test_labels= np.ones(len(temp_test_labels)).astype(int64)
                    temp_sample_labels = np.ones(len(temp_sample_labels)).astype(int64)
                index2+=1
            temp_task['U_train_labels_list'].append(temp_train_labels)
            temp_task['U_test_labels_list'].append(temp_test_labels)
            temp_task['U_sample_labels_list'].append(temp_sample_labels)
   
            if mode['u_flag']!= None :
                index3=0
                
                for j in temp_task['U_train_labels_list']:
                    if index3==0:
                        total_train_labels=j.copy()
                    else:
                        total_train_labels = total_train_labels | j
                    index3+=1
                
                temp_task['total_train_labels']=total_train_labels
    
                index4=0
                
                for j in temp_task['U_test_labels_list']:
                    if index4==0:
                        total_test_labels=j.copy()
                    else:
                        total_test_labels = total_test_labels | j
                    index4+=1
                
                temp_task['total_test_labels']=total_test_labels
                
                index5=0
                
                for j in temp_task['U_sample_labels_list']:
                    if index5==0:
                        total_test_labels=j.copy()
                    else:
                        total_test_labels = total_test_labels | j
                    index5+=1
                
                temp_task['total_sample_labels']=total_test_labels
                
                
            else:
                temp_task['total_train_labels']=temp_train_labels
                temp_task['total_test_labels']=temp_test_labels
                temp_task['total_sample_labels']=temp_sample_labels
                
            temp_task['total_support_tuples']=np.array(temp_task['total_support_tuples'])
            temp_task['total_support_tuples_orioder']=np.array(temp_task['total_support_tuples_orioder'])
            

            all_attrs=np.array(self.dataspace.raw.columns.to_list())
            
            
            Temp_train_tuples1=np.zeros([len(temp_task['total_support_tuples']),len(all_attrs)])
            
           
    
            for i in range(len(mode['mode_attr'])):
                Temp_train_tuples1[:,mode['mode_attr'][i]]= temp_task['total_support_tuples'][:,i]

            
            
            Temp_train_tuples=np.zeros([len(temp_task['total_support_tuples_orioder']),len(all_attrs)])
            
           
    
            for i in range(len(mode['mode_attr'])):
                Temp_train_tuples[:,mode['mode_attr'][i]]= temp_task['total_support_tuples_orioder'][:,i]

            
            
            
            Temp_train_tuples1=self.dataspace.model.transform(pd.DataFrame(Temp_train_tuples1,columns=self.dataspace.raw.columns))
            Temp_train_tuples=self.dataspace.model.transform(pd.DataFrame(Temp_train_tuples,columns=self.dataspace.raw.columns))
          
            
            index=0
            for i in all_attrs[mode['mode_attr']]:
                begin_index=0
                for j in self.dataspace.model.meta:
                    if j['name']==i:
                        break
                    else:
                        begin_index+=j['output_dimensions']
                
                
                if index==0:
                    final_train_tuples=Temp_train_tuples1[:,begin_index:begin_index+j['output_dimensions']]
                    
                else:
                    final_train_tuples=np.column_stack((final_train_tuples,Temp_train_tuples1[:,begin_index:begin_index+j['output_dimensions']]))
                index+=1
            temp_task['total_support_tuples_trans']=final_train_tuples
            
            
            for n in range(len(mode['attr_list'])):
                
            
                index=0
                for i in mode['attr_list'][n]:
                    begin_index=0
                    for j in self.dataspace.model.meta:
                        if j['name']==i:
                            break
                        else:
                            begin_index+=j['output_dimensions']
                    
                    
                    if index==0:
                        final_train_tuples=Temp_train_tuples[:,begin_index:begin_index+j['output_dimensions']]
                        
                    else:
                        final_train_tuples=np.column_stack((final_train_tuples,Temp_train_tuples[:,begin_index:begin_index+j['output_dimensions']]))
                    index+=1
                temp_task[n]['support_tuples_trans']=final_train_tuples
                
            temp_task['qp']=query_patterns
           
            return temp_task

                            
 
    def  Generate_tasks(self,mode,qps):
        '''
        数据的组织方式：'query_id': {
        'part_id'
        {'querysapce_vector':[]
        'support_tuples':[[]] or None
        'support_labels':[]
        'query_tuples':[[]] or None
        'query_labels':[]}
    
        }
        'U_train_labels_list':
        'U_test_labels_list':
        'total_train_labels':
        'total_test_labels':
        
        '''
        
        
        Task_dataset={}
        Task_dataset['qps']=[]
        

        if qps==None:
            index=0
            while index<self.task_num:
                print(index)
                temp_task=self.Generate_single(mode,None)
                if temp_task==None:
                    continue
                if self.intrval_flag==True:
                    cond= (0.005<=np.sum(temp_task['total_test_labels'][:self.multi_intrval[0]]==1)/len(temp_task['total_test_labels'][:self.multi_intrval[0]])<=0.99)
                else:
                    cond= (0.005<=np.sum(temp_task['total_test_labels']==1)/len(temp_task['total_test_labels'])<=0.99)
                    
                if cond:   
                    Task_dataset[index]=temp_task
                    Task_dataset['qps'].append(temp_task['qp'])
                    index+=1
        else:
            index=0
            while index<self.task_num:
                print(index)
                temp_task=self.Generate_single(mode,qps[index])
    
                Task_dataset[index]=temp_task
                index+=1
            
        return Task_dataset
                
                
            
                
                                
                        
    def build(self,qps):
        query_tuples_all=self.dataspace.numpy_sample_raw
        
        total_query_tuples=query_tuples_all[:,self.mode['mode_attr']]
        part_query_tuples={}
        for n in range(len(self.mode['attr_list'])):
            part_query_tuples[n]=query_tuples_all[:,self.dataspace.pos_list_offline[self.mode['attr_list_id'][n]]]
        
                
            
        Temp_train_tuples=self.dataspace.model.transform(pd.DataFrame(query_tuples_all,columns=self.dataspace.raw.columns))
          
            
        index=0
        for i in self.dataspace.raw.columns[self.mode['mode_attr']]:
            begin_index=0
            for j in self.dataspace.model.meta:
                if j['name']==i:
                    break
                else:
                    begin_index+=j['output_dimensions']
                
                
            if index==0:
                final_train_tuples=Temp_train_tuples[:,begin_index:begin_index+j['output_dimensions']]
                    
            else:
                final_train_tuples=np.column_stack((final_train_tuples,Temp_train_tuples[:,begin_index:begin_index+j['output_dimensions']]))
            index+=1
        total_query_tuples_trans=final_train_tuples
        
        
            
        part_query_tuples_trans={}   
        for n in range(len(self.mode['attr_list'])):
             
            index=0
            for i in self.mode['attr_list'][n]:
                begin_index=0
                for j in self.dataspace.model.meta:
                    if j['name']==i:
                        break
                    else:
                        begin_index+=j['output_dimensions']
                    
                    
                if index==0:
                    final_train_tuples=Temp_train_tuples[:,begin_index:begin_index+j['output_dimensions']]
                        
                else:
                    final_train_tuples=np.column_stack((final_train_tuples,Temp_train_tuples[:,begin_index:begin_index+j['output_dimensions']]))
                index+=1
            part_query_tuples_trans[n]=final_train_tuples
        
               

        

        if self.intrval_flag==False:
            if not os.path.exists('{}/'.format(self.path)):
                os.mkdir('{}/'.format(self.path))
            print("Generate begin!")
            if qps==None:
                
                Task_dataset=self.Generate_tasks(self.mode,None)
            else:
                Task_dataset=self.Generate_tasks(self.mode,qps)
                
            print("Generate over!")
            
            
            if qps==None:
                pickle.dump(Task_dataset['qps'], open('{}/'.format(self.path)+'qps.p','wb'))
                
            pickle.dump(total_query_tuples.astype(float32), open('{}/'.format(self.path)+'sample_q_tv.p','wb'))
            pickle.dump(total_query_tuples_trans.astype(float32), open('{}/'.format(self.path)+'sample_q_tv_trans.p','wb'))
            for i in range(len(self.mode['attr_list'])):
                pickle.dump(part_query_tuples[i].astype(float32), open('{}/group_{}_'.format(self.path,i)+'sample_q_tv.p','wb'))
                pickle.dump(part_query_tuples_trans[i].astype(float32), open('{}/group_{}_'.format(self.path,i)+'sample_q_tv_trans.p','wb'))
           
            
            
            
            
            print(Task_dataset.keys())
            for i in Task_dataset.keys() :
                print(i)
                
                if i != "qps":
                    print(Task_dataset[i]['total_support_tuples'])
                    pickle.dump(Task_dataset[i]['total_support_tuples'].astype(float32),open('{}/'.format(self.path)+'sample_'+str(i)+'_s_tv.p','wb'))
                    pickle.dump(Task_dataset[i]['total_support_tuples_trans'].astype(float32),open('{}/'.format(self.path)+'sample_'+str(i)+'_s_tv_trans.p','wb'))
                    pickle.dump(Task_dataset[i]['total_train_labels'].astype(int64),open('{}/'.format(self.path)+'sample_'+str(i)+'_s_y.p','wb'))
                    pickle.dump(Task_dataset[i]['total_test_labels'].astype(int64),open('{}/'.format(self.path)+'sample_'+str(i)+'_q_y.p','wb'))
                    pickle.dump(Task_dataset[i]['total_test_labels'].astype(int64),open('{}/'.format(self.path)+'sample_'+str(i)+'_q_y.p','wb'))
                    pickle.dump(np.array(Task_dataset[i]['total_support_tuples_sample']).astype(float32),open('{}/'.format(self.path)+'sample_'+str(i)+'_ss_t.p','wb'))
                    pickle.dump(Task_dataset[i]['total_sample_labels'].astype(int64),open('{}/'.format(self.path)+'sample_'+str(i)+'_ss_y.p','wb'))
                    
    #                pickle.dump(Task_dataset[i]['querysapce_vector'].astype(float32),open('{}/'.format(self.path)+'sample_'+str(i)+'_qv.p','wb'))
                    
                    for j in range(len(self.mode['attr_list'])):
                        pickle.dump(np.array(Task_dataset[i][j]['support_tuples']).astype(float32),open('{}/group_{}_'.format(self.path,j)+'sample_'+str(i)+'_s_tv.p','wb'))
                        pickle.dump(np.array(Task_dataset[i][j]['support_tuples_trans']).astype(float32),open('{}/group_{}_'.format(self.path,j)+'sample_'+str(i)+'_s_tv_trans.p','wb'))
                        pickle.dump(np.array(Task_dataset[i][j]['train_labels']).astype(int64),open('{}/group_{}_'.format(self.path,j)+'sample_'+str(i)+'_s_y.p','wb'))
                        pickle.dump(np.array(Task_dataset[i][j]['test_labels']).astype(int64),open('{}/group_{}_'.format(self.path,j)+'sample_'+str(i)+'_q_y.p','wb'))
                        pickle.dump(np.array(Task_dataset[i][j]['queryspace_vector']).astype(float32),open('{}/group_{}_'.format(self.path,j)+'sample_'+str(i)+'_qv.p','wb'))
                        
                        pickle.dump(np.array(Task_dataset[i][j]['support_tuples_sample']).astype(float32),open('{}/group_{}_'.format(self.path,j)+'sample_'+str(i)+'_ss_t.p','wb'))
                        pickle.dump(np.array(Task_dataset[i][j]['train_labels_sample']).astype(int64),open('{}/group_{}_'.format(self.path,j)+'sample_'+str(i)+'_ss_y.p','wb'))
                    
        else:
            print("Generate begin!")
            if qps==None:
                
                Task_dataset=self.Generate_tasks(self.mode,None)
            else:
                Task_dataset=self.Generate_tasks(self.mode,qps)

            print("Generate over!")
            
            if qps==None:
                pickle.dump(Task_dataset['qps'], open('{}/'.format(self.path)+'qps.p','wb'))

            for it in self.multi_intrval:
                if not os.path.exists('{}_st_{}/'.format(self.path,it )):
                    os.mkdir('{}_st_{}/'.format(self.path,it))
                
                pickle.dump(total_query_tuples.astype(float32), open('{}_st_{}/'.format(self.path,it )+'sample_q_tv.p','wb'))
                pickle.dump(total_query_tuples_trans.astype(float32), open('{}_st_{}/'.format(self.path,it )+'sample_q_tv_trans.p','wb'))
                for i in range(len(self.mode['attr_list'])):
                    pickle.dump(part_query_tuples[i].astype(float32), open('{}_st_{}//group_{}_'.format(self.path,it,i)+'sample_q_tv.p','wb'))
                    pickle.dump(part_query_tuples_trans[i].astype(float32), open('{}_st_{}//group_{}_'.format(self.path,it,i)+'sample_q_tv_trans.p','wb'))
               
                
                
                for i in Task_dataset.keys():
                    if i != "qps":
                        pickle.dump(Task_dataset[i]['total_support_tuples'][:it].astype(float32),open('{}_st_{}/'.format(self.path,it )+'sample_'+str(i)+'_s_tv.p','wb'))
                        pickle.dump(Task_dataset[i]['total_support_tuples_trans'][:it].astype(float32),open('{}_st_{}/'.format(self.path,it )+'sample_'+str(i)+'_s_tv_trans.p','wb'))
                        pickle.dump(Task_dataset[i]['total_train_labels'][:it].astype(int64),open('{}_st_{}/'.format(self.path,it )+'sample_'+str(i)+'_s_y.p','wb'))
                        pickle.dump(Task_dataset[i]['total_test_labels'].astype(int64),open('{}_st_{}/'.format(self.path,it )+'sample_'+str(i)+'_q_y.p','wb'))
                        pickle.dump(np.array(Task_dataset[i]['total_support_tuples_sample']).astype(float32),open('{}_st_{}/'.format(self.path,it )+'sample_'+str(i)+'_ss_t.p','wb'))
                        pickle.dump(Task_dataset[i]['total_sample_labels'].astype(int64),open('{}_st_{}/'.format(self.path,it )+'sample_'+str(i)+'_ss_y.p','wb'))
                        
    #                    pickle.dump(Task_dataset[i]['querysapce_vector'].astype(float32),open('{}_st_{}/'.format(self.path,it )+'sample_'+str(i)+'_qv.p','wb'))
                        
                        for j in range(len(self.mode['attr_list'])):
                            pickle.dump(np.array(Task_dataset[i][j]['support_tuples'])[:it].astype(float32),open('{}_st_{}//group_{}_'.format(self.path,it,j)+'sample_'+str(i)+'_s_tv.p','wb'))
                            pickle.dump(np.array(Task_dataset[i][j]['support_tuples_trans'])[:it].astype(float32),open('{}_st_{}//group_{}_'.format(self.path,it,j)+'sample_'+str(i)+'_s_tv_trans.p','wb'))
                            pickle.dump(np.array(Task_dataset[i][j]['train_labels'])[:it].astype(int64),open('{}_st_{}//group_{}_'.format(self.path,it,j)+'sample_'+str(i)+'_s_y.p','wb'))
                            pickle.dump(np.array(Task_dataset[i][j]['test_labels']).astype(int64),open('{}_st_{}//group_{}_'.format(self.path,it,j)+'sample_'+str(i)+'_q_y.p','wb'))
                            pickle.dump(np.array(Task_dataset[i][j]['queryspace_vector']).astype(float32),open('{}_st_{}//group_{}_'.format(self.path,it,j)+'sample_'+str(i)+'_qv.p','wb'))
                    
                            pickle.dump(np.array(Task_dataset[i][j]['support_tuples_sample']).astype(float32),open('{}_st_{}//group_{}_'.format(self.path,it,j)+'sample_'+str(i)+'_ss_t.p','wb'))
                            pickle.dump(np.array(Task_dataset[i][j]['train_labels_sample']).astype(int64),open('{}_st_{}//group_{}_'.format(self.path,it,j)+'sample_'+str(i)+'_ss_y.p','wb'))
                        
          