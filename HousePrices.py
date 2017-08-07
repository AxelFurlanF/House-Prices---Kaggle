# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 10:46:39 2017

@author: afurlan
"""

#Librerias
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Datasets
train = pd.read_csv('train.csv', header = 0)
test  = pd.read_csv('test.csv' , header = 0)
full_data = [train, test]

#histogram
sns.distplot(train['SalePrice']);
#correlation matrix
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
plt.yticks(rotation=0) 
plt.xticks(rotation=90)

passengers_pred=test["Id"]

"""--------Análisis de features--------"""
print (train[['MSSubClass', 'SalePrice']].groupby(['MSSubClass'], as_index=False).mean())
#Son simplemente categorías, no guardan ninguna correlación entre sus códigos y qué tan caro se pueden vender.
print(train['MSSubClass'].corr(train['SalePrice'], method='pearson'))
print(train['MSSubClass'].corr(train['SalePrice'], method='kendall'))
#Casi nula

print (train[['MSZoning', 'SalePrice']].groupby(['MSZoning'], as_index=False).mean())

#Asigno numeros random dentro del espectro del campo
for dataset in full_data:
    lt_avg 	   = dataset['LotFrontage'].mean()
    lt_std 	   = dataset['LotFrontage'].std()
    lt_null_count = dataset['LotFrontage'].isnull().sum()
    
    lt_null_random_list = np.random.randint(lt_avg - lt_std, lt_avg + lt_std, size=lt_null_count)
    dataset['LotFrontage'][np.isnan(dataset['LotFrontage'])] = lt_null_random_list
    dataset['LotFrontage'] = dataset['LotFrontage'].astype(int)

train['CategoricalLotFrontage'] = pd.qcut(train['LotFrontage'], 5)
print (train[['CategoricalLotFrontage', 'SalePrice']].groupby(['CategoricalLotFrontage'], as_index=False).mean())
print(train['LotFrontage'].corr(train['SalePrice'], method='pearson'))
print(train['LotFrontage'].corr(train['SalePrice'], method='kendall'))
#LotFrontage tiene una correlación proporcional leve que afecta el precio de venta

train['CategoricalLotArea'] = pd.qcut(train['LotArea'], 10)
print (train[['CategoricalLotArea', 'SalePrice']].groupby(['CategoricalLotArea'], as_index=False).mean())
print(train['LotArea'].corr(train['SalePrice'], method='pearson'))
print(train['LotArea'].corr(train['SalePrice'], method='kendall'))
#LotArea tiene una correlación no lineal por kendall

for dataset in full_data:
    # Mapping Street
    dataset['Street'] = dataset['Street'].map( {'Grvl': 0, 'Pave': 1} ).astype(int)

print (train[['Street', 'SalePrice']].groupby(['Street'], as_index=False).mean())
print(train['Street'].corr(train['SalePrice'], method='pearson'))
print(train['Street'].corr(train['SalePrice'], method='kendall'))
#Casi nula

for dataset in full_data:
    # Mapping Street
    dataset['Alley'] = dataset['Alley'].map( {'Grvl': 0, 'Pave': 1} )
    dataset['Alley'] = dataset['Alley'].fillna(2).astype(int)

    
print (train['Alley'].isnull().sum())
print (train[['Alley', 'SalePrice']].groupby(['Alley'], as_index=False).mean())
print(train['Alley'].corr(train['SalePrice'], method='pearson'))
print(train['Alley'].corr(train['SalePrice'], method='kendall'))
#Alley tiene una correlación lineal de más del 10%


for dataset in full_data:
    # Mapping Street
    dataset['LotShape'] = dataset['LotShape'].map( {'Reg': 0, 'IR1': 1, 'IR2' : 2, 'IR3' : 3} )

print (train[['LotShape', 'SalePrice']].groupby(['LotShape'], as_index=False).mean())
print(train['LotShape'].corr(train['SalePrice'], method='kendall'))

for dataset in full_data:
    # Mapping Street
    dataset['LandContour'] = dataset['LandContour'].map( {'Bnk': 0, 'HLS': 4, 'Low' : 3, 'Lvl' : 2} )

print (train[['LandContour', 'SalePrice']].groupby(['LandContour'], as_index=False).mean())
print(train['LandContour'].corr(train['SalePrice'], method='kendall'))

