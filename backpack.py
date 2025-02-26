# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 22:52:08 2025

@author: User
"""

import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV, KFold


train = pd.read_csv("D:\\Kaggle\\Backpack Prediction\\train_mod.csv")

test = pd.read_csv("D:\\Kaggle\\Backpack Prediction\\test_mod.csv")


train.info()

train = train.drop('Style', axis = 1)

test = test.drop('Style', axis = 1)


categorical_columns_subset = ['Brand', 'Material', 'Size', 'Laptop.Compartment','Waterproof', 'Color', 'weight_bins']

train[categorical_columns_subset] = train[categorical_columns_subset].astype("category")

test[categorical_columns_subset] = test[categorical_columns_subset].astype("category")


model = CatBoostRegressor()



kfold = KFold(3, shuffle = True)

params = {
    'iterations': [500],
    'depth': [7],
    'learning_rate': [0.05],
    'eval_metric': ['RMSE']
}

# Instantiate the grid search model
grid_search = GridSearchCV(estimator = model,param_grid = params, 
                          cv = kfold, verbose = 2)

grid_search.fit(train.drop(['id', 'Price', 'Weight.Capacity..kg.'], axis = 1), train['Price'], cat_features=categorical_columns_subset)

grid_search.best_estimator_

catpred = grid_search.predict(test.drop(['id', 'Weight.Capacity..kg.'], axis = 1))

catpred = pd.DataFrame(catpred)


pred = pd.concat([test['id'], catpred], axis=1)

pred = pred.rename(columns = {'id':'id', 0: 'Price'})


pred.to_csv('D:/Kaggle/catpred.csv', index = False)





















