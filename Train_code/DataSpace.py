#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 19:06:41 2021

@author: caoyukun
"""
#用来读取多维数据，对数据进行无监督表示后构建聚类空间的类


import random
from sklearn.cluster import KMeans
from JKTransformer import JKTransformer
from GMMTransformer import GMMTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import warnings
warnings.filterwarnings('ignore')

class DataSpace:

    def __init__(self, raw, cluster_method,cluster_task_space_num,cluster_queryspace_num,cluster_queryset_space_num,represent_method, on_split_list,off_split_list,cluster_sample_rate,JG_rate):

        self.raw = raw #Raw data read in pandas format
        self.cluster_method=cluster_method #Methods of clustering
        #Used to create a global data space
        self.cluster_queryspace_num=cluster_queryspace_num #查询空间结构向量的聚类中心个数,以及支撑集元组抽样空间的聚类中心个数  
        #Used to create a subspaces
        self.cluster_task_space_num=cluster_task_space_num #task生成空间的聚类中心个数
        self.cluster_queryset_space_num=cluster_queryset_space_num #查询集元组抽样空间的聚类中心个数       
        self.represent_method=represent_method #对数据元组表征的方法
        self.represent_dim=None #对数据元组表征的维度
        self.cluster_sample_rate=cluster_sample_rate #参与聚类的数据元组的比例，这样是为了加快速度
        self.JK_GMM_sample_rate=JG_rate #用来进行JK/GMM nomal的样本比例，为了加快速度
        self.off_split_list=off_split_list
        self.on_split_list=on_split_list
        self.represent_data=None #转化后的数据
        self.DataLabels_task_space={} #聚类的簇标签,对于task生成空间
        self.DataCenters_task_space={} #聚类后的数据中心,对于task生成空间
        self.DataLabels_queryspace={} #聚类的簇标签,对于查询空间结构向量/支撑集元组抽样空间
        self.DataCenters_queryspace={}#聚类后的数据中心,对于查询空间结构向量/支撑集元组抽样空间
        self.DataLabels_queryset_space={} #聚类的簇标签,对于查询集元组抽样空间
        self.DataCenters_queryset_space={} #聚类后的数据中心，对于查询集元组抽样空间
        
        self.DataLabels_task_space_offline={} #聚类的簇标签,对于task生成空间
        self.DataCenters_task_space_offline={} #聚类后的数据中心,对于task生成空间
        self.DataLabels_queryspace_offline={} #聚类的簇标签,对于查询空间结构向量/支撑集元组抽样空间
        self.DataCenters_queryspace_offline={}#聚类后的数据中心,对于查询空间结构向量/支撑集元组抽样空间
        self.DataLabels_queryset_space_offline={} #聚类的簇标签,对于查询集元组抽样空间
        self.DataCenters_queryset_space_offline={} #聚类后的数据中心，对于查询集元组抽样空间
       
        self.model=None #存储表征方式的转化模型
        self.numpy_sample_raw=None #存储抽样出来进行聚类的原始样本，为numpy格式
        
    
        
        #选择近邻聚类中心的格式
        
        
        self.attr_dim=self.raw.shape[1]
        self.GMM_raw=None
        self.task_space_model={}       
        self.query_space_model={}
        self.queryset_space_model={}
        self.taskspace_centers_neighbors={}
        self.queryspace_centers_neighbors={}
        self.query_space_model_offline={}
        self.task_space_model_offline={}       
        self.queryset_space_model_offline={}
        self.taskspace_centers_neighbors_offline={}     
        self.queryspace_centers_neighbors_offline={}       
        self.pos_list=[]
        self.pos_list_offline=[]
        self.GMM_trans_pos=[]
        for i in self.off_split_list:
            temp=[]
            pos_index1=self.raw.columns.to_list().index(i[0])
            pos_index2=self.raw.columns.to_list().index(i[1])
            temp.append(pos_index1)
            temp.append(pos_index2)
            if i in self.on_split_list:
                self.pos_list.append(temp)
            self.pos_list_offline.append(temp)
            
            
    def DataRepresent(self):
        
        
        '''
        这里实现多种表征方法，其中：
        ‘nomal’: 最简单的表征方法，直接将数据元组各个属性下的值归一化，得到一个长度为数据元组维度的向量
        ‘GMM_nomal’: 利用GMM_based 归一化得到元组的特征表示
        ‘JK_nomal’: 利用JK_based 归一化得到元组的特征表示
        '''
        
        
        if self.JK_GMM_sample_rate==None:
            JG_samples=self.raw.copy()
        else:
            JG_samples=self.raw.copy().sample(n=int(len(self.raw)*self.JK_GMM_sample_rate),replace=False)
            
        '''
        默认使用GMM模型做表征
        '''
        self.model=GMMTransformer()
        self.model.fit(JG_samples)

         
        if self.represent_method=='nomal':
            self.normal_interval=[]
            #转成numpy格式
            self.represent_data=self.raw.copy().to_numpy()
            #归一化
            for i in range(self.attr_dim):
                self.normal_interval.append([self.represent_data[:,i].min(),self.represent_data[:,i].max()])
                self.represent_data[:,i]=(self.represent_data[:,i]-self.represent_data[:,i].min())/(self.represent_data[:,i].max()-self.represent_data[:,i].min())
                
        if self.represent_method=='JK_nomal':  
            self.jk_model=JKTransformer()
            self.jk_model.fit(JG_samples,False)
            self.represent_data=self.jk_model.transform(self.raw)
            
        if self.represent_method=='GMM_nomal':  
            self.represent_data=self.model.transform(self.raw)  
            
        for i in self.off_split_list:
            temp_pos=[]
            for attr in i:
                pos=[]
                begin_index=0
                for j in self.model.meta:
                    if j['name']==attr:
                        break
                    else:
                        begin_index+=j['output_dimensions']
                pos.append(begin_index)
                pos.append(begin_index+j['output_dimensions'])
                temp_pos.append(pos)
            
            self.GMM_trans_pos.append(temp_pos)
 
            
    def Cluster(self):

        if self.cluster_sample_rate==None:
            self.cluster_samples=self.represent_data.copy()
        else:
            self.cluster_samples=np.array(random.sample(list(self.represent_data),int(len(self.represent_data)*self.cluster_sample_rate)))
        
        if self.represent_method=='nomal':
            self.numpy_sample_raw=self.cluster_samples.copy()
            for i in range(self.attr_dim):
                self.numpy_sample_raw[:,i]=self.numpy_sample_raw[:,i]*(self.normal_interval[i][1]-self.normal_interval[i][0])+self.normal_interval[i][0]
 
        elif self.represent_method=='GMM_nomal':
            self.numpy_sample_raw=self.model.inverse_transform(self.cluster_samples,None).to_numpy()
        elif self.represent_method=='JK_nomal':
            self.numpy_sample_raw=self.jk_model.inverse_transform(self.cluster_samples,None).to_numpy()
            
            
            
            
            
        if self.cluster_method=='Kmeans':
            
            if self.represent_method=='nomal':
                
                index=0
                index1=0
                for i in self.pos_list_offline:
            
                    kmodel_qs= KMeans(n_clusters=self.cluster_queryspace_num)
                    kmodel_qs.fit(self.cluster_samples[:,i])
                    
                    if i in self.pos_list:
                        self.query_space_model[index1]=kmodel_qs
                        index1+=1
                    self.query_space_model_offline[index]=kmodel_qs  
                    index+=1
    
                index=0
                index1=0
                for i in self.pos_list_offline:
                    print(i)
                    kmodel_ts=KMeans(n_clusters=self.cluster_task_space_num)
                    kmodel_ts.fit(self.cluster_samples[:,i])
                    
                    if i in self.pos_list:
                        self.task_space_model[index1]=kmodel_ts
                        index1+=1
                    self.task_space_model_offline[index]=kmodel_ts   
                    index+=1
    
                index=0
                index1=0
                for i in self.pos_list_offline:
                    
                    
                    kmodel_qss=KMeans(n_clusters=self.cluster_queryset_space_num)
                    kmodel_qss.fit(self.cluster_samples[:,i])   
                    #print(kmodel_qss.cluster_centers_)
                              
                    if i in self.pos_list:
                        self.queryset_space_model[index1]=kmodel_qss
                        index1+=1
                    self.queryset_space_model_offline[index]=kmodel_qss
                    index+=1

                for i in range(len(self.pos_list_offline)):
                    if i<len(self.pos_list):
                    
                        self.DataLabels_queryset_space[i]=self.queryset_space_model[i].labels_
                        self.DataCenters_queryset_space[i]=self.queryset_space_model[i].cluster_centers_.copy()
                        self.DataLabels_task_space[i]=self.task_space_model[i].labels_
                        self.DataCenters_task_space[i]=self.task_space_model[i].cluster_centers_.copy()
                        self.DataLabels_queryspace[i]=self.query_space_model[i].labels_
                        self.DataCenters_queryspace[i]=self.query_space_model[i].cluster_centers_.copy()
              
                        for j in range(len(self.pos_list[i])):
                            self.DataCenters_task_space[i][:,j]=self.DataCenters_task_space[i][:,j]*(self.normal_interval[self.pos_list[i][j]][1]-self.normal_interval[self.pos_list[i][j]][0])+self.normal_interval[self.pos_list[i][j]][0]                     
                            self.DataCenters_queryset_space[i][:,j]=self.DataCenters_queryset_space[i][:,j]*(self.normal_interval[self.pos_list[i][j]][1]-self.normal_interval[self.pos_list[i][j]][0])+self.normal_interval[self.pos_list[i][j]][0]
                            self.DataCenters_queryspace[i][:,j]=self.DataCenters_queryspace[i][:,j]*(self.normal_interval[self.pos_list[i][j]][1]-self.normal_interval[self.pos_list[i][j]][0])+self.normal_interval[self.pos_list[i][j]][0]
                            
                        #print(self.DataCenters_task_space[i])
                        
                    self.DataLabels_queryset_space_offline[i]=self.queryset_space_model_offline[i].labels_
                    self.DataCenters_queryset_space_offline[i]=self.queryset_space_model_offline[i].cluster_centers_.copy()
                    self.DataLabels_task_space_offline[i]=self.task_space_model_offline[i].labels_
                    self.DataCenters_task_space_offline[i]=self.task_space_model_offline[i].cluster_centers_.copy()
                    self.DataLabels_queryspace_offline[i]=self.query_space_model_offline[i].labels_
                    self.DataCenters_queryspace_offline[i]=self.query_space_model_offline[i].cluster_centers_.copy()
                    for j in range(len(self.pos_list_offline[i])):
                        self.DataCenters_task_space_offline[i][:,j]=self.DataCenters_task_space_offline[i][:,j]*(self.normal_interval[self.pos_list_offline[i][j]][1]-self.normal_interval[self.pos_list_offline[i][j]][0])+self.normal_interval[self.pos_list_offline[i][j]][0]
                        self.DataCenters_queryset_space_offline[i][:,j]=self.DataCenters_queryset_space_offline[i][:,j]*(self.normal_interval[self.pos_list_offline[i][j]][1]-self.normal_interval[self.pos_list_offline[i][j]][0])+self.normal_interval[self.pos_list_offline[i][j]][0]
                        self.DataCenters_queryspace_offline[i][:,j]=self.DataCenters_queryspace_offline[i][:,j]*(self.normal_interval[self.pos_list_offline[i][j]][1]-self.normal_interval[self.pos_list_offline[i][j]][0])+self.normal_interval[self.pos_list_offline[i][j]][0]
                
            if self.represent_method=='JK_nomal':   
                

                for i in range(len(self.pos_list)):
                    Temp_DataCenters_queryspace=np.around(self.jk_model.inverse_transform(self.query_space_model.cluster_centers_,None).to_numpy(),decimals=0)
                    
                    
                    Temp_DataCenters_queryspace=np.around(self.jk_model.inverse_transform(self.query_space_model.cluster_centers_,None).to_numpy(),decimals=0)
                    self.DataCenters_queryspace[i]=Temp_DataCenters_queryspace[:,self.pos_list[i]]
                    self.DataLabels_queryspace[i]=self.query_space_model.labels_
                
                
                
                for i in range(len(self.pos_list)):
                    self.DataLabels_task_space[i]=self.task_space_model[i].labels_
                    self.DataCenters_task_space[i]=np.around(self.jk_model.inverse_transform(self.task_space_model[i].cluster_centers_,None).to_numpy(),decimals=0)
            
                    self.DataLabels_queryset_space[i]=self.queryset_space_model[i].labels
                    self.DataCenters_queryset_space[i]=np.around(self.jk_model.inverse_transform(self.queryset_space_model[i].cluster_centers_,None).to_numpy(),decimals=0)
    
    
            if self.represent_method=='GMM_nomal':
                
                
                np.column_stack
                
                index=0
                index1=0
                for i in self.pos_list_offline:
                
                    temp_data=self.cluster_samples[:,self.GMM_trans_pos[index][0][0]:self.GMM_trans_pos[index][0][1]]
                    temp_data=np.column_stack((temp_data,self.cluster_samples[:,self.GMM_trans_pos[index][1][0]:self.GMM_trans_pos[index][1][1]]))
                   
                        
                    kmodel_ts=KMeans(n_clusters=self.cluster_task_space_num)
                    kmodel_ts.fit(temp_data)
                    
                    if i in self.pos_list:
                        self.task_space_model[index1]=kmodel_ts
                        index1+=1
                    self.task_space_model_offline[index]=kmodel_ts   
                    index+=1
    
                index=0
                index1=0
                for i in self.pos_list_offline:
                    temp_data=self.cluster_samples[:,self.GMM_trans_pos[index][0][0]:self.GMM_trans_pos[index][0][1]]
                    temp_data=np.column_stack((temp_data,self.cluster_samples[:,self.GMM_trans_pos[index][1][0]:self.GMM_trans_pos[index][1][1]]))
                    
                    
                    kmodel_qss=KMeans(n_clusters=self.cluster_queryset_space_num)
                    kmodel_qss.fit(temp_data)   
                    #print(kmodel_qss.cluster_centers_)
                              
                    if i in self.pos_list:
                        self.queryset_space_model[index1]=kmodel_qss
                        index1+=1
                    self.queryset_space_model_offline[index]=kmodel_qss
                    index+=1

            
                    
        

                
                for i in range(len(self.pos_list_offline)):
                    Temp_DataCenters_queryspace=np.zeros([len(self.query_space_model.cluster_centers_),self.model.output_dimensions])
                    index=0
                    for j in self.GMM_trans_pos[i]:
                        Temp_DataCenters_queryspace[:,j[0]:j[1]]=self.query_space_model.cluster_centers_[:,index:index+(j[1]-j[0])]
                        index+=index+(j[1]-j[0])
                    Temp_DataCenters_queryspace=np.around(self.model.inverse_transform(Temp_DataCenters_queryspace,None).to_numpy(),decimals=0)
                    if i<len(self.pos_list):
                        self.DataCenters_queryspace[i]=Temp_DataCenters_queryspace[:,self.pos_list[i]]
                        self.DataLabels_queryspace[i]=self.query_space_model.labels_
                
                    self.DataCenters_queryspace_offline[i]=Temp_DataCenters_queryspace[:,self.pos_list_offline[i]]
                    self.DataLabels_queryspace_offline[i]=self.query_space_model.labels_
                
                
                for i in range(len(self.pos_list_offline)):
                    Temp_DataCenters_task_space=np.zeros([len(self.task_space_model_offline[i].cluster_centers_),self.model.output_dimensions])
                    index=0
                    for j in self.GMM_trans_pos[i]:
                        Temp_DataCenters_task_space[:,j[0]:j[1]]=self.task_space_model_offline[i].cluster_centers_[:,index:index+(j[1]-j[0])]
                        index+=index+(j[1]-j[0])
                        
                        
                    Temp_DataCenters_task_space=np.around(self.model.inverse_transform(Temp_DataCenters_task_space,None).to_numpy(),decimals=0)
                    
                    Temp_DataCenters_queryset_space=np.zeros([len(self.queryset_space_model_offline[i].cluster_centers_),self.model.output_dimensions])
                    index=0
                    for j in self.GMM_trans_pos[i]:
                        Temp_DataCenters_queryset_space[:,j[0]:j[1]]=self.queryset_space_model_offline[i].cluster_centers_[:,index:index+(j[1]-j[0])]
                        index+=index+(j[1]-j[0])
                        
                        
                    Temp_DataCenters_queryset_space=np.around(self.model.inverse_transform(Temp_DataCenters_queryset_space,None).to_numpy(),decimals=0)
                    
                    
                    
                    if i<len(self.pos_list):
                        self.DataLabels_task_space[i]=self.task_space_model[i].labels_
                
                        self.DataCenters_task_space[i]=Temp_DataCenters_task_space[:,self.pos_list[i]]
                        
                        self.DataLabels_queryset_space[i]=self.queryset_space_model[i].labels_
                        self.DataCenters_queryset_space[i]=Temp_DataCenters_queryset_space[:,self.pos_list[i]]
                        
                    self.DataLabels_task_space_offline[i]=self.task_space_model_offline[i].labels_
                
                    self.DataCenters_task_space_offline[i]=Temp_DataCenters_task_space[:,self.pos_list_offline[i]]
                        
                    self.DataLabels_queryset_space_offline[i]=self.queryset_space_model_offline[i].labels_
                    self.DataCenters_queryset_space_offline[i]=Temp_DataCenters_queryset_space[:,self.pos_list_offline[i]]
                   
                    
    def build(self):

  
        self.DataRepresent()
        self.Cluster() 
        
        
        for i in range(len(self.pos_list_offline)):
            if i<len(self.pos_list):
                tasksapce_cos=euclidean_distances(self.task_space_model[i].cluster_centers_,self.task_space_model[i].cluster_centers_)
                    
                if self.represent_method=='nomal':
                    tasksapce_cos=euclidean_distances(self.task_space_model[i].cluster_centers_,self.task_space_model[i].cluster_centers_)
                    querysapce_cos=euclidean_distances(self.query_space_model[i].cluster_centers_,self.task_space_model[i].cluster_centers_)
                    self.taskspace_centers_neighbors[i]=np.argsort(tasksapce_cos)
                    self.queryspace_centers_neighbors[i]=np.argsort(querysapce_cos)

                    
                elif self.represent_method=='GMM_nomal': 
                    tasksapce_cos=cosine_similarity(self.task_space_model[i].cluster_centers_,self.task_space_model[i].cluster_centers_)
                    
                    temp_data=self.query_space_model.cluster_centers_[:,self.GMM_trans_pos[i][0][0]:self.GMM_trans_pos[i][0][1]]
                    temp_data=np.column_stack((temp_data,self.query_space_model.cluster_centers_[:,self.GMM_trans_pos[i][1][0]:self.GMM_trans_pos[i][1][1]]))
 
                    querysapce_cos=cosine_similarity(temp_data,self.task_space_model[i].cluster_centers_)
                    self.taskspace_centers_neighbors[i]=np.argsort(-tasksapce_cos)
                    self.queryspace_centers_neighbors[i]=np.argsort(-querysapce_cos)
                                       
                else:
                    Temp=self.jk_model.transformpd.DataFrame(self.DataCenters_queryspace[i],columns=self.split_list[i])
                    querysapce_cos=cosine_similarity(Temp,self.task_space_model[i].cluster_centers_)
    
    
                    
            if self.represent_method=='nomal':
                tasksapce_cos=euclidean_distances(self.task_space_model_offline[i].cluster_centers_,self.task_space_model_offline[i].cluster_centers_)

                querysapce_cos=euclidean_distances(self.query_space_model[i].cluster_centers_,self.task_space_model_offline[i].cluster_centers_)
                self.taskspace_centers_neighbors_offline[i]=np.argsort(tasksapce_cos)
                self.queryspace_centers_neighbors_offline[i]=np.argsort(querysapce_cos)
   
                    
            elif self.represent_method=='GMM_nomal': 
                tasksapce_cos=cosine_similarity(self.task_space_model_offline[i].cluster_centers_,self.task_space_model_offline[i].cluster_centers_)
                
                                    
                temp_data=self.query_space_model.cluster_centers_[:,self.GMM_trans_pos[i][0][0]:self.GMM_trans_pos[i][0][1]]
                temp_data=np.column_stack((temp_data,self.query_space_model.cluster_centers_[:,self.GMM_trans_pos[i][1][0]:self.GMM_trans_pos[i][1][1]]))

                querysapce_cos=cosine_similarity(temp_data,self.task_space_model_offline[i].cluster_centers_)
                self.taskspace_centers_neighbors[i]=np.argsort(-tasksapce_cos)
                self.queryspace_centers_neighbors[i]=np.argsort(-querysapce_cos)
                    
            else:
                Temp=self.jk_model.transformpd.DataFrame(self.DataCenters_queryspace[i],columns=self.split_list_offline[i])
                querysapce_cos=cosine_similarity(Temp,self.task_space_model_offline[i].cluster_centers_)
    
        print("数据空间创建完毕！")
        print("表征方法为:",self.represent_method)
        print("聚类方法为:",self.cluster_method)
            
            
            
   
