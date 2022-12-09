
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



        
        
class TaskGenerator:

    def __init__(self, dataspace,train_task_num,test_task_num,support_tuple_num,query_tuple_num,split_list,path,task_complexity,taskspace_topk,queryspace_topk):

        self.dataspace= dataspace #数据空间对象
        self.train_task_num = train_task_num #要构造的训练task数目
        self.test_task_num =test_task_num #要构造的测试task数目
        
        self.n_way = 2
        self.k_shot = support_tuple_num#支撑集的样本个数，这里包括两部分聚类中心与随机抽样元组
        self.k_query= query_tuple_num #查询集的样本个数，这里与test数据空间的聚类数相同
        self.path=path #生成的task的数据文件的存储地址
        self.split_list=split_list 
        self.attr_dim=self.dataspace.raw.shape[1]
        self.task_sample_num=None #在子任务上构建凸空间或者多边形空间抽样的样本个数，要小于支撑集的样本个数
        self.task_complexity=task_complexity 
        self.taskspace_topk=taskspace_topk
        self.queryspace_topk=queryspace_topk


    
    def task_2D(self,attr_id):
       
     
        region_num=self.task_complexity
            
        CovSpaces=[]
        indexlist=np.array(range(self.dataspace.cluster_task_space_num))
        for i in range(region_num):
            center=random.sample(list(indexlist),1)
            neighbors=self.dataspace.taskspace_centers_neighbors[attr_id][center[0]][:self.taskspace_topk+1]
            condition_data=self.dataspace.DataCenters_task_space[attr_id][neighbors]
            CovSpace=ConvexSpace(condition_data)
            CovSpaces.append(CovSpace)
            indexlist=np.setdiff1d(indexlist, neighbors)
        
        
        return CovSpaces
            
            
            
    def Generate_single(self,attr_gid):
        
            query_vector_ori=[]
            query_vector=np.zeros(self.dataspace.cluster_task_space_num)
            train_labels=[] 
            test_labels=[]
            train_tuples=[]
            test_tuples=[]     
     

            query_patterns=self.task_2D(attr_gid)
            hot_index=0
            hot_list=[]
            for i in self.dataspace.DataCenters_queryspace[attr_gid]:
                result=False
                for j in query_patterns:
                    if j.in_pos_region(i):
                        result=True
                        break
                query_vector_ori.append(result)
                train_tuples.append(i)
                train_labels.append(result)
                if result==True:
                    hot_list.append(hot_index)    
                hot_index+=1
                
            if len(hot_list)==0:
                return None,None,None,None,None,None,None
            
        
            for i in hot_list:
                query_vector[self.dataspace.queryspace_centers_neighbors[attr_gid][i][:self.queryspace_topk]]=True
                
         

          
            '''
            随机从训练与测试的聚类中抽样样本，进行标记,每个聚簇抽样一个
            '''    
            
            for i in range(self.k_shot-self.dataspace.cluster_queryspace_num):
                index=random.choice(hot_list)
                temp_data=self.dataspace.numpy_sample_raw[np.where(self.dataspace.DataLabels_queryspace[attr_gid]==index)][:,self.dataspace.pos_list[attr_gid]]
                temp_tuple=random.sample(list(temp_data),1)                
                train_tuples.append(temp_tuple[0]) 
                result=False
                for j in query_patterns:
                    if j.in_pos_region(temp_tuple[0]):
                        result=True
                        break
                train_labels.append(result)
                
            for i in range(self.k_query):
                temp_data=self.dataspace.numpy_sample_raw[np.where(self.dataspace.DataLabels_queryset_space[attr_gid]==i)][:,self.dataspace.pos_list[attr_gid]]
                temp_tuple=random.sample(list(temp_data),1)
                test_tuples.append(temp_tuple[0]) 
                result=False
                for j in query_patterns:
                    if j.in_pos_region(temp_tuple[0]):
                        result=True
                        break
                test_labels.append(result)
            
            for i in range(int(self.k_query/2)):
                index=random.choice(hot_list)
                temp_data=self.dataspace.numpy_sample_raw[np.where(self.dataspace.DataLabels_queryspace[attr_gid]==index)][:,self.dataspace.pos_list[attr_gid]]
                temp_tuple=random.sample(list(temp_data),1) 
                test_tuples.append(temp_tuple[0]) 
                result=False
                for j in query_patterns:
                    if j.in_pos_region(temp_tuple[0]):
                        result=True
                        break
                test_labels.append(result)


            train_tuples=np.array(train_tuples)
            
            
            test_tuples=np.array(test_tuples)
            train_labels=np.array(train_labels)
            test_labels=np.array(test_labels)
            
            query_vector=np.array(query_vector)
            train_rate=np.sum(train_labels==True)/len(train_labels)
            test_rate=np.sum(test_labels==True)/len(test_labels)
            all_attrs=self.dataspace.raw.columns.to_list()
            
            Temp_train_tuples=np.zeros([len(train_tuples),len(all_attrs)])
            
            Temp_test_tuples=np.zeros([len(test_tuples),len(all_attrs)])
    
            for i in range(len(self.split_list[attr_gid])):
                Temp_train_tuples[:,self.dataspace.pos_list[attr_gid][i]]= train_tuples[:,i]
                Temp_test_tuples[:,self.dataspace.pos_list[attr_gid][i]]= test_tuples[:,i]
            Temp_train_tuples=self.dataspace.model.transform(pd.DataFrame(Temp_train_tuples,columns=self.dataspace.raw.columns))
            Temp_test_tuples=self.dataspace.model.transform(pd.DataFrame(Temp_test_tuples,columns=self.dataspace.raw.columns))
            
            
            index=0
            for i in self.split_list[attr_gid]:

                begin_index=0
                for j in self.dataspace.model.meta:
                    if j['name']==i:
                        break
                    else:
                        begin_index+=j['output_dimensions']
                if index==0:
                    final_train_tuples=Temp_train_tuples[:,begin_index:begin_index+j['output_dimensions']]
                    final_test_tuples=Temp_test_tuples[:,begin_index:begin_index+j['output_dimensions']]
                else:
                    final_train_tuples=np.column_stack((final_train_tuples,Temp_train_tuples[:,begin_index:begin_index+j['output_dimensions']]))
                    final_test_tuples=np.column_stack((final_test_tuples,Temp_test_tuples[:,begin_index:begin_index+j['output_dimensions']]))
                index+=1
                #print(final_train_tuples.shape)            
            return query_vector,final_train_tuples,final_test_tuples,train_labels,test_labels,train_rate,test_rate


        
        
 
        
    def  Generate_tasks(self,attr_gid):
        '''
        数据的组织方式：'query_id': {
        'querysapce_vector':[]
        'support_tuples':[[]] or None
        'support_labels':[]
        'query_tuples':[[]] or None
        'query_labels':[]
        }
        
        '''
        
        Task_dataset={}
        Task_dataset['train']={}
        Task_dataset['test']={}
       
        index=0
        while index<self.train_task_num:
            query_vector,train_tuples,test_tuples,train_labels,test_labels,train_rate,test_rate = self.Generate_single(attr_gid)
            
            if train_rate != None:
                if  0.05<=train_rate<=0.95 and  0.01<=test_rate<=0.99:
                    Task_dataset['train'][index]={}
                    Task_dataset['train'][index]['querysapce_vector']=query_vector.astype(float32)
                    Task_dataset['train'][index]['support_tuples']=train_tuples.astype(float32)
                    Task_dataset['train'][index]['support_labels']=train_labels.astype(int64)
                    Task_dataset['train'][index]['query_tuples']=test_tuples.astype(float32)
                    Task_dataset['train'][index]['query_labels']=test_labels.astype(int64)
                    index+=1
                    print("Train task ID:",index)
                    
             
                
        while index<self.train_task_num+self.test_task_num:
          query_vector,train_tuples,test_tuples,train_labels,test_labels,train_rate,test_rate = self.Generate_single(attr_gid)
          if train_rate != None:
              if  0.05<=train_rate<=0.95 and  0.01<=test_rate<=0.99:
                    Task_dataset['test'][index]={}
                    Task_dataset['test'][index]['querysapce_vector']=query_vector.astype(float32)
                    Task_dataset['test'][index]['support_tuples']=train_tuples.astype(float32)
                    Task_dataset['test'][index]['support_labels']=train_labels.astype(int64)
                    Task_dataset['test'][index]['query_tuples']=test_tuples.astype(float32)
                    Task_dataset['test'][index]['query_labels']=test_labels.astype(int64)
                    index+=1
                    
        return Task_dataset
                  
    def build(self):
        
        
        if not os.path.exists('{}/'.format(self.path)):
            os.mkdir('{}/'.format(self.path))
        
        for id in range(len(self.split_list)):
            if not os.path.exists('{}/attr_group_{}/'.format(self.path,id)):
                os.mkdir('{}/attr_group_{}/'.format(self.path,id))
            print("Group "+str(id)+" generate begin!")
            Task_dataset=self.Generate_tasks(id)
            print("Group "+str(id)+" generate over!")
     
            for i in Task_dataset['train'].keys():
                pickle.dump(Task_dataset['train'][i]['querysapce_vector'], open('{}/attr_group_{}/'.format(self.path,id)+'sample_'+str(i)+'_qv.p', 'wb'))
                pickle.dump(Task_dataset['train'][i]['support_tuples'], open('{}/attr_group_{}/'.format(self.path,id)+'sample_'+str(i)+'_s_tv.p', 'wb'))
                pickle.dump(Task_dataset['train'][i]['support_labels'], open('{}/attr_group_{}/'.format(self.path,id)+'sample_'+str(i)+'_s_y.p', 'wb'))
                pickle.dump(Task_dataset['train'][i]['query_tuples'], open('{}/attr_group_{}/'.format(self.path,id)+'sample_'+str(i)+'_q_tv.p', 'wb'))
                pickle.dump(Task_dataset['train'][i]['query_labels'], open('{}/attr_group_{}/'.format(self.path,id)+'sample_'+str(i)+'_q_y.p', 'wb'))
            for i in Task_dataset['test'].keys():
                pickle.dump(Task_dataset['test'][i]['querysapce_vector'], open('{}/attr_group_{}/'.format(self.path,id)+'sample_'+str(i)+'_qv.p', 'wb'))
                pickle.dump(Task_dataset['test'][i]['support_tuples'], open('{}/attr_group_{}/'.format(self.path,id)+'sample_'+str(i)+'_s_tv.p', 'wb'))
                pickle.dump(Task_dataset['test'][i]['support_labels'], open('{}/attr_group_{}/'.format(self.path,id)+'sample_'+str(i)+'_s_y.p', 'wb'))
                pickle.dump(Task_dataset['test'][i]['query_tuples'], open('{}/attr_group_{}/'.format(self.path,id)+'sample_'+str(i)+'_q_tv.p', 'wb'))
                pickle.dump(Task_dataset['test'][i]['query_labels'], open('{}/attr_group_{}/'.format(self.path,id)+'sample_'+str(i)+'_q_y.p', 'wb'))
          
      