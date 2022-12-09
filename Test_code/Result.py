#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 18:22:30 2021

@author: caoyukun
"""

import pickle
from Test import Modeltest

f=open('solvetest_offline_task_50_2_20_mode1_0.001_train_task_root_20000_4_20_25_nomal','rb')

solve=pickle.load(f)
print("Meta*:",sum(solve.final_f16)/50)
print("Meta:",sum(solve.final_f1)/50)
print("Basic:",sum(solve.final_f12)/50)
print("SVM^r:",sum(solve.final_f14)/50)
print("SVM:",sum(solve.final_f15)/50)
