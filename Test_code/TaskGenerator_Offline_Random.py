
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
        '''
        task_complexity指构建任务的复杂性：
        n：指子空间上凸查询区域的最大值
        '''
        self.taskspace_topk=taskspace_topk
        self.queryspace_topk=queryspace_topk
        self.mode=mode
        self.multi_intrval=multi_intrval
        self.random_sample_num=random_sample_num
        self.range=[[-2.972521,1445.488037,5.036839,2039.900024],[0.561423,272.201996,0.768829,170.067200],[0.009221,359.998444,-24.207066,84.823318],[1.000000,40.000000,0.000000,27.000000]]
        
        mode_attr=[]
        for i in mode['attr_list']:
            for j in i:
                pos=self.dataspace.raw.columns.to_list().index(j)
                if pos not in mode_attr:
                    mode_attr.append(pos)
        #print(mode_attr)
        mode_attr=sorted(mode_attr)  
        self.mode['mode_attr']=mode_attr
        
        
        
        attr_list_id=[]
        
     
        for i in mode['attr_list']:
            attr_list_id.append(self.dataspace.off_split_list.index(i))
        
        
        self.mode['attr_list_id']=attr_list_id

        
 
    
    def task_2D(self,attr_id,reg,indexl):
        random.seed()
        if reg==None:
            region_num=self.task_complexity
            topK=self.taskspace_topk
        else:
            region_num=reg[0]
            topK=reg[1]
            
        CovSpaces=[]
        indexlist=indexl
        
        '''
        其实只执行了一次
        '''
        for i in range(region_num):
            center=random.sample(list(indexlist),1)

            neighbors=self.dataspace.taskspace_centers_neighbors[attr_id][center[0]][:topK+1]
            
            condition_data=self.dataspace.DataCenters_task_space[attr_id][neighbors]
            #print(condition_data)
            
            CovSpace=ConvexSpace(condition_data)
            CovSpaces.append(CovSpace)
            indexlist=np.setdiff1d(indexlist, neighbors)
            
#        
#        plotdata=[]    
#        
#        for i in range(int(self.dataspace.numpy_sample_raw.shape[0]/10)):
#            for j in CovSpaces:
#                if j.in_pos_region(self.dataspace.numpy_sample_raw[:,self.dataspace.pos_list[attr_id]][i]):
#                    plotdata.append(self.dataspace.numpy_sample_raw[:,self.dataspace.pos_list[attr_id]][i])
#                    break
#        plt.axis(self.range[attr_id])
#        plt.scatter(np.array(plotdata)[:,0],np.array(plotdata)[:,1],s=15,color='black',marker='o')
#        plt.scatter(self.dataspace.DataCenters_queryspace[attr_id][:,0],self.dataspace.DataCenters_queryspace[attr_id][:,1],s=15,color='red',marker='o')
#        plt.scatter(self.dataspace.DataCenters_task_space[attr_id][:,0],self.dataspace.DataCenters_task_space[attr_id][:,1],s=15,color='green',marker='o')
#        plt.show()
       
        return CovSpaces,center
            
            
    def Generate_single(self,mode,qp):
            #print(mode['attr_list_id'])
            
            if qp ==None:
                query_patterns={}
                for i in range(len(mode['attr_list_id'])):
                    query_patterns[mode['attr_list_id'][i]]=[]
                    
                    for j in range(mode['dim'][i]):
                        if j==0:
                            indexl=np.array(range(self.dataspace.cluster_task_space_num))
                            if self.taskspace_topk=='random':
                                reg=random.choice(mode['reg'])
                                qptmp,center=self.task_2D(mode['attr_list_id'][i],reg,indexl)
                            else:
                                qptmp,center=self.task_2D(mode['attr_list_id'][i],None,indexl)
                            query_patterns[mode['attr_list_id'][i]].append(qptmp)
                        else:
                            nei_num=random.randint(10,20)
                            indexl=self.dataspace.taskspace_centers_neighbors[mode['attr_list_id'][i]][center[0]][1:nei_num+1]
                            if self.taskspace_topk=='random':
                                reg=random.choice(mode['reg'])
                                query_patterns[mode['attr_list_id'][i]].append(self.task_2D(mode['attr_list_id'][i],reg,indexl)[0])
                            else:
                                query_patterns[mode['attr_list_id'][i]].append(self.task_2D(mode['attr_list_id'][i],None,indexl)[0])
            else:
                query_patterns=qp
                

                    
                    
                    
                   
                      
            
            temp_task={}
            
            ori_hot_list=[]
            ori_query_vec=[]
            
            for n in mode['attr_list_id']:
                temp_task[n]={}
            for n in mode['attr_list_id']:
                
                
                temp_task[n]['support_tuples']=[]
                temp_task[n]['train_labels']=[]
                temp_task[n]['test_labels']=[]
                temp_task[n]['support_tuples_sample']=[]
                temp_task[n]['train_labels_sample']=[]
                
                for m in range(len(query_patterns[n])):
                    query_vector_temp=[]
                    train_labels_temp=[]
                    for i in self.dataspace.DataCenters_queryspace[n]:
                        pos_data=i
                        if m==0:
                            temp_task[n]['support_tuples'].append(pos_data)
                        
                        result=False
                        for j in query_patterns[n][m]:
                            if j.in_pos_region(pos_data):
                                result=True
                                break
                        query_vector_temp.append(result)
                        train_labels_temp.append(result)
                    temp_task[n]['train_labels'].append(train_labels_temp)
                    if m==0:
                        query_vector_part=np.array(query_vector_temp)
                    else:
                        query_vector_part=query_vector_part & np.array(query_vector_temp)
                ori_query_vec.append(train_labels_temp)
                ori_hot_list.append(query_vector_part)
            
            #print("&&&&")
            #print(len(temp_task[0]['support_tuples']))
                
            
           
            #print(len(ori_hot_list)) 
            #print(temp_task[n]['train_labels'])
             
             
#            for i in self.dataspace.DataCenters_queryspace_all:     
#                temp_task['total_support_tuples'].append(i[mode['mode_attr']])
                    
                    
            
#            plt.axis([0, 80000, 0, 500])
#            plt.scatter(np.array(temp_task['total_support_tuples'])[:,0],np.array(temp_task['total_support_tuples'])[:,1])
#            plt.show()
#            
#            print(temp_task[0]['train_labels'])  
#            print(np.argsort(temp_task[0]['train_labels']))
#            print(np.array(temp_task[0]['train_labels'])[np.argsort(temp_task[0]['train_labels'])])
            
    
            #print('1')
            #print(ori_hot_list)
            print(len(ori_hot_list[0]-1))
            print(len(ori_hot_list[0])-1)
            print(qp)
            for i in range(len(ori_hot_list)):
                
                tem_pos=np.argsort(~ori_hot_list[i])
                new_list=ori_hot_list[i][tem_pos].copy()
                temp_task[mode['attr_list_id'][i]]['train_labels_total']=list(new_list)
                
                if qp == None:
                    if np.sum(ori_hot_list[i]==1)==0:
                        return None
                else:
                    print("DD")
                    if np.sum(ori_hot_list[i]==1)==0:
                       ori_hot_list[i][random.randint(0,len(ori_hot_list[i])-1)]=1 

                if i ==0:
                    temp_hot_list=new_list
                    total_temp=np.array(temp_task[mode['attr_list_id'][i]]['support_tuples'])[tem_pos]
                    total_temp_orioder=np.array(temp_task[mode['attr_list_id'][i]]['support_tuples'])
                else:
                    temp_hot_list=temp_hot_list & new_list
                    total_temp=np.hstack((total_temp,np.array(temp_task[mode['attr_list_id'][i]]['support_tuples'])[tem_pos]))
                    total_temp_orioder=np.hstack((total_temp_orioder,np.array(temp_task[mode['attr_list_id'][i]]['support_tuples'])))
                    
     
                
            
#            for i in range(len(ori_hot_list)):
#                print(sum(ori_hot_list[i]))
#                print(sum(temp_task[i]['train_labels_total']))
#                print(temp_task[i]['train_labels_total'])
            #print(total_temp)
           # print(total_temp.shape)
            #print(total_temp_orioder)
            temp_task['total_support_tuples']=total_temp
            
            temp_task['total_support_tuples_orioder']=total_temp_orioder
            

            total_hot_list=temp_hot_list
                
            if qp==None:    
                if np.sum(total_hot_list==1)==0:
                    print("&")
                    return None
            
            
                         
            #print(len(ori_hot_list))
            for i in range(len(ori_hot_list)):
                part_query_vector=np.zeros(self.dataspace.cluster_task_space_num)
                for j in range(len(ori_hot_list[i])):
                    if ori_hot_list[i][j]==1:
                        part_query_vector[self.dataspace.queryspace_centers_neighbors[mode['attr_list_id'][i]][j][:self.queryspace_topk]]=True
                temp_task[mode['attr_list_id'][i]]['queryspace_vector_part']=part_query_vector
            
            
            
            for n in mode['attr_list_id']:
                temp_task[n]['queryspace_vector']=[]
                for m in range(len(temp_task[n]['train_labels'])):
                    part_query_vector=np.zeros(self.dataspace.cluster_task_space_num)
                    for j in range(len(temp_task[n]['train_labels'][m])):
                        if temp_task[n]['train_labels'][m][j]==1:
                            part_query_vector[self.dataspace.queryspace_centers_neighbors[n][j][:self.queryspace_topk]]=True
                    temp_task[n]['queryspace_vector'].append(part_query_vector)
                
            
            
            
            
            
            temp_task['total_support_tuples_sample']=[]
            temp_total_data=[]
            for i in range(self.random_sample_num):
                flag=random.randint(0,len(ori_hot_list)-1)
                index=random.choice(np.where(ori_hot_list[flag]==1)[0])
                
                temp_data=self.dataspace.numpy_sample_raw[np.where(self.dataspace.DataLabels_queryspace[mode['attr_list_id'][flag]]==index)]
                temp_tuple=random.sample(list(temp_data),1)
                temp_task['total_support_tuples_sample'].append(temp_tuple[0][mode['mode_attr']])
                temp_total_data.append(temp_tuple[0])
            
            
            for n in mode['attr_list_id']:
                for m in range(len(query_patterns[n])):
                    train_labels_temp=[]
                    for data in temp_total_data:
                        #print(self.dataspace.pos_list_offline[n])
                        if m==0:
                            temp_task[n]['support_tuples_sample'].append(data[self.dataspace.pos_list_offline[n]]) 
                        result=False
                        for j in query_patterns[n][m]:
                            if j.in_pos_region(data[self.dataspace.pos_list_offline[n]]):
                                result=True
                                break
                        train_labels_temp.append(result)  
                    temp_task[n]['train_labels_sample'].append(train_labels_temp)


            #print(temp_task[n]['train_labels_sample'])
            
            '''
            随机从训练与测试的聚类中抽样样本，进行标记,每个聚簇抽样一个,这里是从全局进行抽样，之后拆解到每个子空间上，一定要注意
            '''    
            
            for i in range(self.k_shot-self.dataspace.cluster_queryspace_num):
#                index=random.choice(np.where(total_hot_list)[0])
#                
#                temp_data=self.dataspace.numpy_sample_raw[np.where(self.dataspace.DataLabels_queryspace[0]==index)]
#                temp_tuple=random.sample(list(temp_data),1)
#                temp_task['total_support_tuples'].append(temp_tuple[0][mode['mode_attr']])
#                
                #print(temp_data)
                
                for n in range(len(mode['attr_list_id'])):
                    index=random.choice(np.where(ori_hot_list[n]==1)[0])
                    temp_data=self.dataspace.numpy_sample_raw[:,self.dataspace.pos_list_offline[mode['attr_list_id'][n]]][np.where(self.dataspace.DataLabels_queryspace[mode['attr_list_id'][n]]==index)]
                    temp_tuple=random.sample(list(temp_data),1)
                    
                    temp_task[mode['attr_list_id'][n]]['support_tuples'].append(temp_tuple[0]) 
                   
                    for m in range(len(query_patterns[mode['attr_list_id'][n]])):
                        result=False
                        for j in query_patterns[mode['attr_list_id'][n]][m]:
                            if j.in_pos_region(temp_tuple[0]):
                                result=True
                                break
                        temp_task[mode['attr_list_id'][n]]['train_labels'][m].append(result)
                        if m==0:
                            temp_label=result
                        else:
                            temp_label=temp_label & result
                    temp_task[mode['attr_list_id'][n]]['train_labels_total'].append(temp_label)  
                    
                    
                    if n==0:
                        total=temp_tuple[0]
                    else:
                        total=np.hstack((total,temp_tuple[0]))
                    
                    
                   
                #print(total.shape)
                temp_task['total_support_tuples']=np.vstack((temp_task['total_support_tuples'],total))
                temp_task['total_support_tuples_orioder']=np.vstack((temp_task['total_support_tuples_orioder'],total))
            #print(temp_task['total_support_tuples'])
            #print(np.array(temp_task['total_support_tuples']).shape)
            
            for n in range(len(mode['attr_list_id'])):
                for m in range(len(query_patterns[mode['attr_list_id'][n]])):
                    temp_task[mode['attr_list_id'][n]]['test_labels'].append([])
                       
                    
                    
            #print(len(temp_task[0]['support_tuples']))
            #print(len(temp_task['total_support_tuples']))
            for i in range(len(self.dataspace.numpy_sample_raw)):
                
                for n in range(len(mode['attr_list_id'])):
                
                    for m in range(len(query_patterns[mode['attr_list_id'][n]])):
                        result=False
                        for j in query_patterns[mode['attr_list_id'][n]][m]:
                            if j.in_pos_region(self.dataspace.numpy_sample_raw[i][self.dataspace.pos_list_offline[mode['attr_list_id'][n]]]):
                                result=True
                                break
                        temp_task[mode['attr_list_id'][n]]['test_labels'][m].append(result)







#            print(temp_task.keys())
#            print(temp_task[0].keys())
#            print(len(temp_task[0]['train_labels']))
#            print(len(temp_task[0]['train_labels'][0]))
#            print(len(temp_task[0]['train_labels_total']))
#            print(len(temp_task[0]['queryspace_vector_part']))
#            print(len(temp_task[0]['queryspace_vector']))
#            print(len(temp_task[0]['queryspace_vector'][0]))
#            print(len(temp_task[0]['test_labels']))
#            print(len(temp_task[0]['test_labels'][0]))
#            print(len(temp_task[0]['train_labels_sample']))
#            print(len(temp_task[0]['train_labels_sample'][0]))
#            print(len(temp_task[0]['support_tuples']))
#            print(len(temp_task[0]['support_tuples_sample']))
#            
#            

            
            for n in range(len(mode['attr_list_id'])):
                for m in range(len(query_patterns[mode['attr_list_id'][n]])):
                    temp_task[mode['attr_list_id'][n]]['test_labels'][m]=np.array(temp_task[mode['attr_list_id'][n]]['test_labels'][m])
                    temp_task[mode['attr_list_id'][n]]['train_labels_sample'][m]=np.array(temp_task[mode['attr_list_id'][n]]['train_labels_sample'][m])
                    if n==0 and m==0:
                        temp_test_labels=temp_task[mode['attr_list_id'][n]]['test_labels'][m]
                        temp_sample_labels=temp_task[mode['attr_list_id'][n]]['train_labels_sample'][m]
                        
                        
                    else:
                        temp_test_labels=temp_test_labels & temp_task[mode['attr_list_id'][n]]['test_labels'][m]
                        temp_sample_labels=temp_sample_labels & temp_task[mode['attr_list_id'][n]]['train_labels_sample'][m]
            

                        
            temp_task['total_test_labels']=temp_test_labels
            temp_task['total_sample_labels']=temp_sample_labels


            
            for n in range(len(mode['attr_list_id'])):
                temp_task[mode['attr_list_id'][n]]['train_labels_total']=np.array(temp_task[mode['attr_list_id'][n]]['train_labels_total'])
                if n==0:
                    temp_train_labels= temp_task[mode['attr_list_id'][n]]['train_labels_total']
                else:
                    
                    temp_train_labels= temp_train_labels & temp_task[mode['attr_list_id'][n]]['train_labels_total']

            temp_task['total_train_labels']=temp_train_labels
            
            
            for n in range(len(mode['attr_list_id'])):
                for m in range(len(query_patterns[mode['attr_list_id'][n]])):
                    if  m==0:
                        temp_test_labels=temp_task[mode['attr_list_id'][n]]['test_labels'][m]
                    else:
                        temp_test_labels=temp_test_labels & temp_task[mode['attr_list_id'][n]]['test_labels'][m]
                temp_task[mode['attr_list_id'][n]]['test_labels_part']=temp_test_labels
                
                
                
            for n in range(len(mode['attr_list_id'])):
                for m in range(len(query_patterns[mode['attr_list_id'][n]])):
                    temp_task[mode['attr_list_id'][n]]['train_labels'][m]=np.array(temp_task[mode['attr_list_id'][n]]['train_labels'][m])

                    if  m==0:
                        temp_train_labels=temp_task[mode['attr_list_id'][n]]['train_labels'][m]
                    else:
                        temp_train_labels=temp_train_labels & temp_task[mode['attr_list_id'][n]]['train_labels'][m]
                temp_task[mode['attr_list_id'][n]]['train_labels_part']=temp_train_labels
              
        
#            print(sum(temp_task[mode['attr_list_id'][0]]['test_labels_total']))
#            print(sum(temp_task[mode['attr_list_id'][1]]['test_labels_total']))
#            print(sum(temp_task[mode['attr_list_id'][0]]['test_labels_total']&temp_task[mode['attr_list_id'][1]]['test_labels_total']))
#            print(sum(temp_task['total_test_labels']))
#
            
            #print(temp_task['total_train_labels'])
            temp_task['total_support_tuples']=np.array(temp_task['total_support_tuples'])
            temp_task['total_support_tuples_orioder']=np.array(temp_task['total_support_tuples_orioder'])
            
            #print(temp_task['total_support_tuples'].shape)

            all_attrs=np.array(self.dataspace.raw.columns.to_list())
            
            
            Temp_train_tuples1=np.zeros([len(temp_task['total_support_tuples']),len(all_attrs)])
            
           
    
            for i in range(len(mode['mode_attr'])):
                #print(mode['mode_attr'][i])
                Temp_train_tuples1[:,mode['mode_attr'][i]]= temp_task['total_support_tuples'][:,i]

            
            
            ##print(Temp_train_tuples)
            Temp_train_tuples=np.zeros([len(temp_task['total_support_tuples_orioder']),len(all_attrs)])
            
           
    
            for i in range(len(mode['mode_attr'])):
                Temp_train_tuples[:,mode['mode_attr'][i]]= temp_task['total_support_tuples_orioder'][:,i]

            
            #print(mode['attr_list_id'])
            #print(mode['attr_list'])
            
            #print(Temp_train_tuples)
           # print(Temp_train_tuples)
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
                temp_task[mode['attr_list_id'][n]]['support_tuples_trans']=final_train_tuples
            
            temp_task['qp']=query_patterns
           
            #print(len(temp_task['total_support_tuples_trans']))
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
           
            
            
            
            
            
            for i in Task_dataset.keys():
                if i != "qps":
                    pickle.dump(Task_dataset[i]['total_support_tuples'].astype(float32),open('{}/'.format(self.path)+'sample_'+str(i)+'_s_tv.p','wb'))
                    pickle.dump(Task_dataset[i]['total_support_tuples_trans'].astype(float32),open('{}/'.format(self.path)+'sample_'+str(i)+'_s_tv_trans.p','wb'))
                    pickle.dump(Task_dataset[i]['total_train_labels'].astype(int64),open('{}/'.format(self.path)+'sample_'+str(i)+'_s_y.p','wb'))
                    pickle.dump(Task_dataset[i]['total_test_labels'].astype(int64),open('{}/'.format(self.path)+'sample_'+str(i)+'_q_y.p','wb'))
                    pickle.dump(np.array(Task_dataset[i]['total_support_tuples_sample']).astype(float32),open('{}/'.format(self.path)+'sample_'+str(i)+'_ss_t.p','wb'))
                    pickle.dump(Task_dataset[i]['total_sample_labels'].astype(int64),open('{}/'.format(self.path)+'sample_'+str(i)+'_ss_y.p','wb'))
                    
    #                pickle.dump(Task_dataset[i]['querysapce_vector'].astype(float32),open('{}/'.format(self.path)+'sample_'+str(i)+'_qv.p','wb'))
                    
                    for j in range(len(self.mode['attr_list'])):
                        pickle.dump(np.array(Task_dataset[i][self.mode['attr_list_id'][j]]['support_tuples']).astype(float32),open('{}/group_{}_'.format(self.path,j)+'sample_'+str(i)+'_s_tv.p','wb'))
                        pickle.dump(np.array(Task_dataset[i][self.mode['attr_list_id'][j]]['support_tuples_trans']).astype(float32),open('{}/group_{}_'.format(self.path,j)+'sample_'+str(i)+'_s_tv_trans.p','wb'))
                        pickle.dump(np.array(Task_dataset[i][self.mode['attr_list_id'][j]]['queryspace_vector_part']).astype(float32),open('{}/group_{}_'.format(self.path,j)+'sample_'+str(i)+'_qv.p','wb'))
                        pickle.dump(np.array(Task_dataset[i][self.mode['attr_list_id'][j]]['support_tuples_sample']).astype(float32),open('{}/group_{}_'.format(self.path,j)+'sample_'+str(i)+'_ss_t.p','wb'))
                        pickle.dump(np.array(Task_dataset[i][self.mode['attr_list_id'][j]]['train_labels_part']).astype(int64),open('{}/group_{}_'.format(self.path,j)+'sample_'+str(i)+'_s_y.p','wb'))
                        pickle.dump(np.array(Task_dataset[i][self.mode['attr_list_id'][j]]['test_labels_part']).astype(int64),open('{}/group_{}_'.format(self.path,j)+'sample_'+str(i)+'_q_y.p','wb'))
    
                        for d in range(self.mode['dim'][j]):
                            pickle.dump(np.array(Task_dataset[i][self.mode['attr_list_id'][j]]['train_labels'][d]).astype(int64),open('{}/group_{}_D{}_'.format(self.path,j,d)+'sample_'+str(i)+'_s_y.p','wb'))
                            pickle.dump(np.array(Task_dataset[i][self.mode['attr_list_id'][j]]['test_labels'][d]).astype(int64),open('{}/group_{}_D{}_'.format(self.path,j,d)+'sample_'+str(i)+'_q_y.p','wb'))
                            pickle.dump(np.array(Task_dataset[i][self.mode['attr_list_id'][j]]['queryspace_vector'][d]).astype(float32),open('{}/group_{}_D{}_'.format(self.path,j,d)+'sample_'+str(i)+'_qv.p','wb'))
                            pickle.dump(np.array(Task_dataset[i][self.mode['attr_list_id'][j]]['train_labels_sample'][d]).astype(int64),open('{}/group_{}_D{}_'.format(self.path,j,d)+'sample_'+str(i)+'_ss_y.p','wb'))
                    
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
                            pickle.dump(np.array(Task_dataset[i][self.mode['attr_list_id'][j]]['support_tuples'])[:it].astype(float32),open('{}_st_{}//group_{}_'.format(self.path,it,j)+'sample_'+str(i)+'_s_tv.p','wb'))
                            pickle.dump(np.array(Task_dataset[i][self.mode['attr_list_id'][j]]['support_tuples_trans'])[:it].astype(float32),open('{}_st_{}//group_{}_'.format(self.path,it,j)+'sample_'+str(i)+'_s_tv_trans.p','wb'))
                            pickle.dump(np.array(Task_dataset[i][self.mode['attr_list_id'][j]]['queryspace_vector_part']).astype(float32),open('{}_st_{}//group_{}_'.format(self.path,it,j)+'sample_'+str(i)+'_qv.p','wb'))
                            pickle.dump(np.array(Task_dataset[i][self.mode['attr_list_id'][j]]['support_tuples_sample']).astype(float32),open('{}_st_{}//group_{}_'.format(self.path,it,j)+'sample_'+str(i)+'_ss_t.p','wb'))
                            pickle.dump(np.array(Task_dataset[i][self.mode['attr_list_id'][j]]['train_labels_part'])[:it].astype(int64),open('{}_st_{}//group_{}_'.format(self.path,it,j)+'sample_'+str(i)+'_s_y.p','wb'))
                            pickle.dump(np.array(Task_dataset[i][self.mode['attr_list_id'][j]]['test_labels_part'])[:it].astype(int64),open('{}_st_{}//group_{}_'.format(self.path,it,j)+'sample_'+str(i)+'_q_y.p','wb'))
    
                            for d in range(self.mode['dim'][j]):
                                pickle.dump(np.array(Task_dataset[i][self.mode['attr_list_id'][j]]['train_labels'][d])[:it].astype(int64),open('{}_st_{}//group_{}_D{}_'.format(self.path,it,j,d)+'sample_'+str(i)+'_s_y.p','wb'))
                                pickle.dump(np.array(Task_dataset[i][self.mode['attr_list_id'][j]]['test_labels'][d]).astype(int64),open('{}_st_{}//group_{}_D{}_'.format(self.path,it,j,d)+'sample_'+str(i)+'_q_y.p','wb'))
                                pickle.dump(np.array(Task_dataset[i][self.mode['attr_list_id'][j]]['queryspace_vector'][d]).astype(float32),open('{}_st_{}//group_{}_D{}_'.format(self.path,it,j,d)+'sample_'+str(i)+'_qv.p','wb'))
                                pickle.dump(np.array(Task_dataset[i][self.mode['attr_list_id'][j]]['train_labels_sample'][d]).astype(int64),open('{}_st_{}//group_{}_D{}_'.format(self.path,it,j,d)+'sample_'+str(i)+'_ss_y.p','wb'))
                            
              