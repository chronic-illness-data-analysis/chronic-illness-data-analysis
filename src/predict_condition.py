
# coding: utf-8

# In[1]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[2]:


# !ls '/content/drive/My Drive/Datamining/data'


# In[3]:



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[4]:


file_name = '../../data/fd-export.csv'

## read file
df = pd.read_csv(file_name)
cond_file_name = '../../data/conditions_list.csv'
df_conditions = pd.read_csv(cond_file_name)
"""
    Function to remove abnormal age.
    This is required for reusability purpose.
    
    @Input : dataframe
    @Return : dataframe
"""
def remove_abnormal_age(df):
    df.age = df.age.fillna(-1)
    invalid_ids = set(df[ (df.age<0) | (df.age > 100) ].user_id.values)
    valid_df = df[~df.user_id.isin(invalid_ids)]
    
    print("Valid users with norma age = {}, Percentage {}".format( valid_df.user_id.unique().shape[0]
                                                                  , valid_df.user_id.unique().shape[0]/
                                                               float( df.user_id.unique().shape[0] ) ))
    
    return valid_df

## filter the user
df_processed = remove_abnormal_age(df)

df_processed.head(10)
    


# In[5]:




def filter_the_symptoms(df):
  symptoms = df[(df['trackable_type'] == "Symptom")].trackable_id.value_counts().sort_values(ascending = False)[:5000].index
  print(symptoms[0])
  df = df[df.trackable_id.isin(symptoms)]
  return df
df = filter_the_symptoms(df)


# In[6]:


df.head(100)


# In[7]:


df_conditions.sort_values(['Count']).tail(10)


# # Merge conditions

# In[8]:


df_with_conds = pd.merge(df_processed, df_conditions, how ='left', on = 'trackable_id')


# In[9]:


df_with_conds_unique = df_with_conds[df_with_conds.trackable_type == 'Condition'].drop_duplicates(['user_id', 'trackable_id'])


# In[10]:


temp_df = df_with_conds_unique['Family'].value_counts().sort_values( ascending = False)[:10]
plt.figure(figsize=(20, 6))
sns_plt = sns.barplot(y = temp_df, x = temp_df.index)


# # Prediction Task Below

# In[11]:


import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier

df['checkin_date'] = pd.to_datetime(df['checkin_date'])

def combineConditions(x):
    return set(x)

def makeList(x):
    return list(x)

def numericOr(x):
    if 1 in x.values:
        return 1
    else:
        return 0
    
def reshapeSymptoms(df):
    #reshape and one-hot the symptoms

    symptoms = pd.get_dummies(df[(df['trackable_type'] == "Symptom") & (df['trackable_value'] != 0)], columns=['trackable_name'])
    symptoms = symptoms.drop(['trackable_id', 'trackable_type', 'trackable_value'], axis=1)
    symptoms = symptoms.groupby(['user_id', 'checkin_date']).agg(numericOr).reset_index()
    return symptoms
    
def createXY(df, symptoms):
    newdf = df[df['trackable_type'] == 'Condition'].groupby(['user_id', 'checkin_date'])['trackable_name'].agg(combineConditions).reset_index()
    newdf = newdf.merge(symptoms, on=['user_id','checkin_date'])

    #newdf = newdf.drop_duplicates().drop(['user_id','checkin_date','trackable_id','trackable_type', 'trackable_value'], axis=1)
    newdf = newdf.drop(['user_id','checkin_date'], axis=1)
    X = newdf.drop('trackable_name', axis=1)
    Y = newdf['trackable_name'].apply(makeList)  # each row of Y is a list, because this is a multilabel problem
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(Y)  
    return train_test_split(X, Y, test_size=0.2, random_state=42)


# In[ ]:


symptoms = reshapeSymptoms(df)
symptoms.head(10)
# print(list(symptoms))
X_train, X_test, Y_train, Y_test = createXY(df, symptoms)


# 
# # PCA

# In[ ]:


from sklearn.decomposition import PCA
import os
def do_pca(df_train, df_test,df_train_y, df_test_y, n_components = 50,save_folder = '../..//data'):
    
    pca = PCA(n_components = n_components)
    pca.fit(df_train)
    df_train = pca.transform(df_train)
    df_test = pca.transform(df_test)
    
    
    pca_test_out_file = os.path.join(save_folder, 'pca_test.csv')
    pca_train_out_file = os.path.join(save_folder, 'pca_train.csv')
    
    pca_test_out_file_y = os.path.join(save_folder, 'pca_test_y.csv')
    pca_train_out_file_y = os.path.join(save_folder, 'pca_train_y.csv')
    
    np.savetxt(pca_train_out_file_y, df_train_y , delimiter=",")
    np.savetxt(pca_test_out_file_y, df_test_y, delimiter=",")
    
    np.savetxt(pca_train_out_file, df_train,  delimiter=",")
    np.savetxt(pca_test_out_file, df_test, delimiter=",")
    
    return df_train, df_test



# In[ ]:


## call pca
X_train, X_test= do_pca(X_train, X_test, Y_train, Y_test)


# In[ ]:




# In[ ]:


# META CODE
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import MultiLabelBinarizer

clf = OneVsRestClassifier(XGBClassifier(n_jobs=-1, max_depth=4))

# You may need to use MultiLabelBinarizer to encode your variables from arrays [[x, y, z]] to a multilabel 
# format before training.
# mlb = MultiLabelBinarizer()
# y = mlb.fit_transform(y)

clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)


# In[ ]:

import pickle as pk
from sklearn.metrics import classification_report
report = classification_report(Y_test, Y_pred)
print(report)

with open('classification_report.txt', 'wb') as f:
    pk.dump(report, f)
    

