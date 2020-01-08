# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 20:13:08 2020

@author: lokesh
"""

#Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Reading dataset
data = pd.read_csv("Test_Data.csv")
data.drop(['Unnamed: 0'],axis=1,inplace = True)
data.head()
data.shape

# Encoding 
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
data['Class']= label_encoder.fit_transform(data['Class']) 


# Selecting Features and Target
X = data.drop(['Class'],axis = 1)
y = data['Class']

df_OUT = X.copy()

# Outliers Treatment

def cap_data(df_OUT):
    for col in df_OUT.columns:
        
        if (((df_OUT[col].dtype)=='float64') | ((df_OUT[col].dtype)=='int64')):
            percentiles = df_OUT[col].quantile([0.10,0.9]).values
            df_OUT[col][df_OUT[col] <= percentiles[0]] = percentiles[0]
            df_OUT[col][df_OUT[col] >= percentiles[1]] = percentiles[1]
        else:
            df_OUT[col]=df_OUT[col]
    return df_OUT

#calling function
final_X=cap_data(df_OUT)


df_CORR = final_X.copy() 

# Removing correlation variables

col_corr = []  # Set of all the names of deleted columns
def correlation(df_CORR,threshold ):
      
    corr_matrix = df_CORR.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.append(colname)
                if colname in df_CORR.columns:
                    del df_CORR[colname] # deleting the column from the dataset


    return df_CORR


#calling function
final_X1 = correlation(df_CORR,0.90)



# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(final_X1, y, test_size = 0.30, random_state = 0)



# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# =============================================================================
# Classification using Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=800)
rfc.fit(X_train,y_train)

# Predictions
predictions = rfc.predict(X_test)


# Evalution of Model
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))


from sklearn import metrics as met
met.accuracy_score(y_test,predictions)


# using Random Forest Classifier Accuracy score is 0.74


 #=============================================================================
 # Using XGBoost
 
from xgboost import XGBClassifier, XGBRegressor
model = XGBClassifier(n_estimators=800)
model.fit(X_train, y_train)
 
 
 # Predictions
y_pred = model.predict(X_test)
 
 
 # Evalution of Model
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,y_pred))

from sklearn import metrics as met
met.accuracy_score(y_test,y_pred)
 
 # using XGBoost Accuracy score is 0.72
 #=============================================================================


