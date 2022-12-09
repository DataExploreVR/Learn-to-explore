#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 09:35:36 2021

@author: caoyukun
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import jenkspy

'''
实现JK-based表征
'''
class JKTransformer(object):
    """
    这里的n_cluster不同于后面的聚类中心个数，这里指的是JK的划分区间的个数
    """
    
    def __init__(self, n_clusters=10):
        self.n_clusters = n_clusters

    def _fit_continuous(self, column, data):
        
        JK=jenkspy.jenks_breaks(data,nb_class=self.n_clusters)
        components=[]
        for i in range(self.n_clusters):
            components.append(1)
            
        num_components = len(components) 
        

        return {
            'name': column,
            'model':JK ,
            'components': components,
            'output_info': [(1, 'tanh'), (num_components, 'softmax')],
            'output_dimensions': 1 + num_components,
        }
    
        
    def _fit_discrete(self, column, data):
        ohe = OneHotEncoder(sparse=False)
        ohe.fit(data)
        categories = len(ohe.categories_[0])

        return {
            'name': column,
            'encoder': ohe,
            'output_info': [(categories, 'softmax')],
            'output_dimensions': categories
        }

    def fit(self, data,meta,discrete_columns=tuple()):
        self.output_info = []
        self.output_dimensions = 0

        if not isinstance(data, pd.DataFrame):
            self.dataframe = False
            data = pd.DataFrame(data)
        else:
            self.dataframe = True

        self.dtypes = data.infer_objects().dtypes
        if meta==False:
            self.meta = []
            for column in data.columns:
                column_data = data[[column]].values
                if column in discrete_columns:
                    meta = self._fit_discrete(column, column_data)
                else:
                    meta = self._fit_continuous(column, column_data)
    
                self.output_info += meta['output_info']
                self.output_dimensions += meta['output_dimensions']
                self.meta.append(meta)
        else:
            self.meta=meta

    def _transform_continuous(self, column_meta, data):
        #components = column_meta['components']
        model = column_meta['model']
        interval=[]
        for i in range(1,len(model)):
            interval.append([model[i-1],model[i]])
        
        
        probs_onehot=np.zeros([len(data),self.n_clusters])
        norm=np.zeros([len(data),1])
        
            
        for i in range(len(data)):
            mark=0
            for j in range(len(interval)):
                if data[i]>=interval[j][0] and data[i] <interval[j][1]:
                    mark=j
                    break
            probs_onehot[i,mark]=1
            norm[i,0]=(data[i]-interval[j][0])/(interval[j][1]-interval[j][0])
            


        return [norm, probs_onehot]

    def _transform_discrete(self, column_meta, data):
        encoder = column_meta['encoder']
        return encoder.transform(data)

    def transform(self, data):
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        values = []
        for meta in self.meta:
            column_data = data[[meta['name']]].values
            if 'model' in meta:
                values += self._transform_continuous(meta, column_data)
            else:
                values.append(self._transform_discrete(meta, column_data))

        return np.concatenate(values, axis=1).astype(np.float64)

    def _inverse_transform_continuous(self, meta, data):
        model = meta['model']
        #components = meta['components']
        interval=[]
        for i in range(1,len(model)):
            interval.append([model[i-1],model[i]])
            
        norm=data[:,0]
        porb=data[:,1:].argmax(axis=1)
        
        column=np.zeros([len(data)])
        
        for i in range(len(data)):
            column[i]=norm[i]*(interval[porb[i]][1]-interval[porb[i]][0])+interval[porb[i]][0]
        
    

        return column

    def _inverse_transform_discrete(self, meta, data):
        encoder = meta['encoder']
        return encoder.inverse_transform(data)

    def inverse_transform(self, data):
        
        start = 0
        output = []
        column_names = []
        for meta in self.meta:
            dimensions = meta['output_dimensions']
            columns_data = data[:, start:start + dimensions]

            if 'model' in meta:
                inverted = self._inverse_transform_continuous(meta, columns_data)
            else:
                inverted = self._inverse_transform_discrete(meta, columns_data)

            output.append(inverted)
            column_names.append(meta['name'])
            start += dimensions

        output = np.column_stack(output)
        output = pd.DataFrame(output, columns=column_names).astype(self.dtypes)
        if not self.dataframe:
            output = output.values

        return output

    def save(self, path):
        with open(path + "/data_transform.pl", "wb") as f:
            pickle.dump(self, f)

    def covert_column_name_value_to_id(self, column_name, value):
        discrete_counter = 0
        column_id = 0
        for info in self.meta:
            if info["name"] == column_name:
                break
            if len(info["output_info"]) == 1:  # is discrete column
                discrete_counter += 1
            column_id += 1

        return {
            "discrete_column_id": discrete_counter,
            "column_id": column_id,
            "value_id": np.argmax(info["encoder"].transform([[value]])[0])
        }

    @classmethod
    def load(cls, path):
        with open(path + "/data_transform.pl", "rb") as f:
            return pickle.load(f)

