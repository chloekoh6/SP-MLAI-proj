# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 16:50:21 2023

@author: chloe
"""

import os 
import shutil
from sklearn.model_selection import train_test_split


src = 'C:/MLAI/dataset/lemon'
lemonPic = os.listdir(src)

src1 = 'C:/MLAI/dataset/tomato'
tomatoPic = os.listdir(src1)

src2 = 'C:/MLAI/dataset/unknown'
unknownPic = os.listdir(src2)

#splitting lemon dataset
trainl, lemonRem = train_test_split(lemonPic, train_size=0.8)
validl, testl = train_test_split(lemonRem, test_size=0.5)

#splitting tomato dataset
traint, tomatoRem = train_test_split(tomatoPic, train_size=0.8)
validt, testt = train_test_split(tomatoRem, test_size=0.5)

#splitting unknown dataset
trainu, unknownRem = train_test_split(unknownPic, train_size=0.8)
validu, testu = train_test_split(unknownRem, test_size=0.5)


for image in trainl:
    shutil.copy(src+'/'+image, 'C:/MLAI/train/lemon')
    
for image in validl:
    shutil.copy(src+'/'+image, 'C:/MLAI/valid/lemon')
    
for image in testl:
    shutil.copy(src+'/'+image, 'C:/MLAI/test/lemon')


for image in traint:
    shutil.copy(src1+'/'+image, 'C:/MLAI/train/tomato')
    
for image in validt:
    shutil.copy(src1+'/'+image, 'C:/MLAI/valid/tomato')
    
for image in testt:
    shutil.copy(src1+'/'+image, 'C:/MLAI/test/tomato')
    
    
for image in trainu:
    shutil.copy(src2+'/'+image, 'C:/MLAI/train/unknown')
    
for image in validu:
    shutil.copy(src2+'/'+image, 'C:/MLAI/valid/unknown')
    
for image in testu:
    shutil.copy(src2+'/'+image, 'C:/MLAI/test/unknown')

