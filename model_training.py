#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import mlflow
from sklearn.model_selection import learning_curve


# In[2]:


mlflow.set_tracking_uri("http://127.0.0.1:5000")


# In[3]:


name = "my-experiment-1"
#mlflow.create_experiment(name)


# In[4]:


mlflow.set_experiment(name)


# In[5]:


from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
bank_marketing = fetch_ucirepo(id=222) 
  
# data (as pandas dataframes) 
df = bank_marketing.data.original

df.head()


# In[6]:


df.drop(columns=['poutcome'],axis=1,inplace=True)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='most_frequent')
df = imputer.fit_transform(df)

df = pd.DataFrame(df,
                  columns=bank_marketing.data.original.columns)

df = df.astype(bank_marketing.data.original.dtypes.to_dict())


# In[7]:


# converting y to numeric
y_map = {'yes':1,'no':0}
df['y'] = df['y'].apply(lambda x: y_map[x.strip().lower()])


# In[8]:


df


# It says in the [data dictionary](#https://archive.ics.uci.edu/dataset/222/bank+marketing) that the `duration` column is the last contact duration and this column should be discarded for the predictive model.

# In[9]:


df.drop(['duration'],axis=1,inplace=True)


# In[10]:


X = df.drop(['y'],axis=1)
y = df['y']


# In[11]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[12]:


from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier,Pool


# ## Designing experiements: Feature engineering and model training

# In[25]:


with mlflow.start_run():

    mlflow.set_tag("model", "Catboost Pipeline")

    mlflow.log_param("train-data-path", "https://archive.ics.uci.edu/dataset/222/bank+marketing")


    #train catboost and get the cross validation score
    cat = CatBoostClassifier(random_state=123,cat_features=list(X.select_dtypes(include='object').columns))
    cat_pipeline=Pipeline([('catboost',cat)])
    cat_scores=cross_val_score(cat_pipeline,X_train,y_train,cv=3,scoring='f1_weighted',verbose=False)
    print("3-fold cross validation weighted f1 score:{:.3f}".format(cat_scores.mean()))
    cat_pipeline.fit(X_train,y_train,catboost__verbose=False) 

    #logging metrics
    mlflow.log_metric("f1_weighted", cat_scores.mean())
    mlflow.sklearn.log_model(cat_pipeline, artifact_path="models")
    print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")  

    cat_pipeline['catboost'].get_feature_importance(prettified=True)    



# - Bucket pdays?
# - if we can figure out how to add year column (data dictionary says that the data is ordered by date (from May 2008 to November 2010)) then we can add a column for weekend(1/0)
# - bucket campaign?

# In[26]:


with mlflow.start_run():

    mlflow.set_tag("model", "Catboost Pipeline age binned")

    mlflow.log_param("train-data-path", "https://archive.ics.uci.edu/dataset/222/bank+marketing") 

    #add age binned column and drop age column
    df_copy = df.copy()
    df_copy['age_binned'] = pd.cut(df['age'], bins=10).astype(str)
    df_copy = df_copy.drop(['age'],axis=1)    

    #recreate training and testing data split
    X = df_copy.drop(['y'],axis=1)
    y = df_copy['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)      

    #train catboost and get the cross validation score
    cat = CatBoostClassifier(random_state=123,cat_features=list(X.select_dtypes(include='object').columns))
    cat_pipeline=Pipeline([('catboost',cat)])
    cat_scores=cross_val_score(cat_pipeline,X_train,y_train,cv=3,scoring='f1_weighted',verbose=False)
    print("3-fold cross validation weighted f1 score:{:.3f}".format(cat_scores.mean()))
    cat_pipeline.fit(X_train,y_train,catboost__verbose=False) 

    #logging metrics
    mlflow.log_metric("f1_weighted", cat_scores.mean())
    mlflow.sklearn.log_model(cat_pipeline, artifact_path="models")
    print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")  

    cat_pipeline['catboost'].get_feature_importance(prettified=True)
   


# In[27]:


with mlflow.start_run():

    mlflow.set_tag("model", "Catboost Pipeline age and day_of_week binned")

    mlflow.log_param("train-data-path", "https://archive.ics.uci.edu/dataset/222/bank+marketing") 

    df_copy = df.copy()
    #add age binned column and drop age column
    df_copy['age_binned'] = pd.cut(df['age'], bins=10).astype(str)
    df_copy = df_copy.drop(['age'],axis=1)    

    #add day_of_week binned column and drop day_of_week column
    df_copy['day_of_week_binned'] = pd.cut(df['day_of_week'], bins=10).astype(str)
    df_copy.drop(['day_of_week'],axis=1,inplace=True)    

    #recreate training and testing data split
    X = df_copy.drop(['y'],axis=1)
    y = df_copy['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)      

    #train catboost and get the cross validation score
    cat = CatBoostClassifier(random_state=123,cat_features=list(X.select_dtypes(include='object').columns))
    cat_pipeline=Pipeline([('catboost',cat)])
    cat_scores=cross_val_score(cat_pipeline,X_train,y_train,cv=3,scoring='f1_weighted',verbose=False)
    print("3-fold cross validation weighted f1 score:{:.3f}".format(cat_scores.mean()))
    cat_pipeline.fit(X_train,y_train,catboost__verbose=False) 

    #logging metrics
    mlflow.log_metric("f1_weighted", cat_scores.mean())
    mlflow.sklearn.log_model(cat_pipeline, artifact_path="models")
    print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")  

    cat_pipeline['catboost'].get_feature_importance(prettified=True)


# In[28]:


cat_pipeline['catboost'].get_feature_importance(prettified=True)


# It might be useful to represent the `campaign` column both as an integer value and as a categorical feature. We can first bucket the campaign values and then create a boolean value to encode the campaign buckets and concatenate both the integer and boolean into a single array.

# In[29]:


pd.cut(df['campaign'], bins=4).astype(str).unique()


# In[30]:


def campaign_categorize(value):
    if value<20:
        return "Less than 20"
    elif (value>=20) & (value<40 ):
        return "20-40"
    elif (value>=40) & (value<60):
        return "40-60"
    else:
        return "60+"


# In[31]:


# def campaign_categorize(value):
#     if value<15:
#         return "Less than 15"
#     elif (value>=15) & (value<30):
#         return "15-30"
#     elif (value>=30) & (value<45):
#         return "30-45"
#     elif (value>=45) & (value<60):
#         return "45-60"
#     else:
#         return "60+"


# In[32]:


def appending_cat_num(campaign_row,encoded_row):
    campaign_list = []
    campaign_list.append([campaign_row,encoded_row])

    return campaign_list


# In[33]:


with mlflow.start_run():

    mlflow.set_tag("model", "Catboost Pipeline age and campaign binned")

    mlflow.log_param("train-data-path", "https://archive.ics.uci.edu/dataset/222/bank+marketing") 

    df_copy = df.copy()
    #add age binned column and drop age column
    df_copy['age_binned'] = pd.cut(df['age'], bins=10).astype(str)
    df_copy = df_copy.drop(['age'],axis=1)    

    #add campaign_bucketed binned column
    #no need to drop campaign column since we are using catboost(non-linear model)
    df_copy['campaign_bucketed'] = df_copy['campaign'].apply(lambda x: campaign_categorize(x))   

    #recreate training and testing data split
    X = df_copy.drop(['y'],axis=1)
    y = df_copy['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)      

    #train catboost and get the cross validation score
    cat = CatBoostClassifier(random_state=123,cat_features=list(X.select_dtypes(include='object').columns))
    cat_pipeline=Pipeline([('catboost',cat)])
    cat_scores=cross_val_score(cat_pipeline,X_train,y_train,cv=3,scoring='f1_weighted',verbose=False)
    print("3-fold cross validation weighted f1 score:{:.3f}".format(cat_scores.mean()))
    cat_pipeline.fit(X_train,y_train,catboost__verbose=False) 

    #logging metrics
    mlflow.log_metric("f1_weighted", cat_scores.mean())
    mlflow.sklearn.log_model(cat_pipeline, artifact_path="models")
    print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")  

    cat_pipeline['catboost'].get_feature_importance(prettified=True)


# `pdays`(number of days that passed by after the client was last contacted from a previous campaign) and `previous` (umber of contacts performed before this campaign and for this client) might work together to determine how likely they are to subscribe. For example, a short pdays combined with a high previous value might suggest strong engagement with the customer.

# In[34]:


# feature cross of pdays and previous

with mlflow.start_run():

    mlflow.set_tag("model", "Catboost Pipeline age binned and feature cross of pdays and previous")

    mlflow.log_param("train-data-path", "https://archive.ics.uci.edu/dataset/222/bank+marketing") 

    df_copy = df.copy()
    #add age binned column and drop age column
    df_copy['age_binned'] = pd.cut(df['age'], bins=10).astype(str)
    df_copy = df_copy.drop(['age'],axis=1)    

    #feature cross
    df_copy['pdays_previous_cross']= df_copy['pdays'].astype(str) + '_' + df_copy['previous'].astype(str)

    #recreate training and testing data split
    X = df_copy.drop(['y'],axis=1)
    y = df_copy['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)      

    #train catboost and get the cross validation score
    cat = CatBoostClassifier(random_state=123,cat_features=list(X.select_dtypes(include='object').columns))
    cat_pipeline=Pipeline([('catboost',cat)])
    cat_scores=cross_val_score(cat_pipeline,X_train,y_train,cv=3,scoring='f1_weighted',verbose=False)
    print("3-fold cross validation weighted f1 score:{:.3f}".format(cat_scores.mean()))
    cat_pipeline.fit(X_train,y_train,catboost__verbose=False) 

    #logging metrics
    mlflow.log_metric("f1_weighted", cat_scores.mean())
    mlflow.sklearn.log_model(cat_pipeline, artifact_path="models")
    print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")  

    cat_pipeline['catboost'].get_feature_importance(prettified=True)


# In[35]:


df_copy.head()


# In[36]:


cat_pipeline['catboost'].get_feature_importance(prettified=True)


# In[37]:


# feature cross of pdays and previous

with mlflow.start_run():

    mlflow.set_tag("model", "Catboost Pipeline age & day_of_week binned and feature cross of pdays and previous")

    mlflow.log_param("train-data-path", "https://archive.ics.uci.edu/dataset/222/bank+marketing") 

    df_copy = df.copy()
    #add age binned column and drop age column
    df_copy['age_binned'] = pd.cut(df['age'], bins=10).astype(str)
    df_copy = df_copy.drop(['age'],axis=1) 
    
      
    #add day_of_week binned column and drop day_of_week column
    df_copy['day_of_week_binned'] = pd.cut(df['day_of_week'], bins=10).astype(str)
    df_copy.drop(['day_of_week'],axis=1,inplace=True) 
    

    #feature cross
    df_copy['pdays_previous_cross']= df_copy['pdays'].astype(str) + '_' + df_copy['previous'].astype(str)

    #recreate training and testing data split
    X = df_copy.drop(['y'],axis=1)
    y = df_copy['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)      

    #train catboost and get the cross validation score
    cat = CatBoostClassifier(random_state=123,cat_features=list(X.select_dtypes(include='object').columns))
    cat_pipeline=Pipeline([('catboost',cat)])
    cat_scores=cross_val_score(cat_pipeline,X_train,y_train,cv=3,scoring='f1_weighted',verbose=False)
    print("3-fold cross validation weighted f1 score:{:.3f}".format(cat_scores.mean()))
    cat_pipeline.fit(X_train,y_train,catboost__verbose=False) 

    #logging metrics
    mlflow.log_metric("f1_weighted", cat_scores.mean())
    mlflow.sklearn.log_model(cat_pipeline, artifact_path="models")
    print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")  

    cat_pipeline['catboost'].get_feature_importance(prettified=True)


# In[38]:


cat_pipeline['catboost'].get_feature_importance(prettified=True)


# In[39]:


# feature cross of pdays and previous and binned category - Dropped the pdays and previous columns

with mlflow.start_run():

    mlflow.set_tag("model", "Catboost Pipeline age & day_of_week binned and feature cross of pdays and previous. Dropped the pdays and previous columns")

    mlflow.log_param("train-data-path", "https://archive.ics.uci.edu/dataset/222/bank+marketing") 

    df_copy = df.copy()
    #add age binned column and drop age column
    df_copy['age_binned'] = pd.cut(df['age'], bins=10).astype(str)
    df_copy = df_copy.drop(['age'],axis=1) 
    
      
    #add day_of_week binned column and drop day_of_week column
    df_copy['day_of_week_binned'] = pd.cut(df['day_of_week'], bins=10).astype(str)
    df_copy.drop(['day_of_week'],axis=1,inplace=True) 
    

    #feature cross
    df_copy['pdays_previous_cross']= df_copy['pdays'].astype(str) + '_' + df_copy['previous'].astype(str)
    df_copy.drop(['pdays','previous'],axis=1,inplace=True) 


    #recreate training and testing data split
    X = df_copy.drop(['y'],axis=1)
    y = df_copy['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)      

    #train catboost and get the cross validation score
    cat = CatBoostClassifier(random_state=123,cat_features=list(X.select_dtypes(include='object').columns))
    cat_pipeline=Pipeline([('catboost',cat)])
    cat_scores=cross_val_score(cat_pipeline,X_train,y_train,cv=3,scoring='f1_weighted',verbose=False)
    print("3-fold cross validation weighted f1 score:{:.3f}".format(cat_scores.mean()))
    cat_pipeline.fit(X_train,y_train,catboost__verbose=False) 

    #logging metrics
    mlflow.log_metric("f1_weighted", cat_scores.mean())
    mlflow.sklearn.log_model(cat_pipeline, artifact_path="models")
    print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")  

    cat_pipeline['catboost'].get_feature_importance(prettified=True)


# In[40]:


cat_pipeline['catboost'].get_feature_importance(prettified=True)


# In[41]:


import numpy as np


# In[42]:


# feature cross of pdays and previous and log transformed balance

with mlflow.start_run():

    mlflow.set_tag("model", "Catboost Pipeline age & day_of_week binned and feature cross of pdays and previous")

    mlflow.log_param("train-data-path", "https://archive.ics.uci.edu/dataset/222/bank+marketing") 

    df_copy = df.copy()
    #add age binned column and drop age column
    df_copy['age_binned'] = pd.cut(df['age'], bins=10).astype(str)
    df_copy = df_copy.drop(['age'],axis=1) 
    
      
    #add day_of_week binned column and drop day_of_week column
    df_copy['day_of_week_binned'] = pd.cut(df['day_of_week'], bins=10).astype(str)
    df_copy.drop(['day_of_week'],axis=1,inplace=True) 
    

    #feature cross
    df_copy['pdays_previous_cross']= df_copy['pdays'].astype(str) + '_' + df_copy['previous'].astype(str)

    #log transformed balance
    df_copy['balance_log']= np.log(df_copy['balance'])
    df_copy.drop(['balance'],axis=1,inplace=True) 

    #recreate training and testing data split
    X = df_copy.drop(['y'],axis=1)
    y = df_copy['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)      

    #train catboost and get the cross validation score
    cat = CatBoostClassifier(random_state=123,cat_features=list(X.select_dtypes(include='object').columns))
    cat_pipeline=Pipeline([('catboost',cat)])
    cat_scores=cross_val_score(cat_pipeline,X_train,y_train,cv=3,scoring='f1_weighted',verbose=False)
    print("3-fold cross validation weighted f1 score:{:.3f}".format(cat_scores.mean()))
    cat_pipeline.fit(X_train,y_train,catboost__verbose=False) 

    #logging metrics
    mlflow.log_metric("f1_weighted", cat_scores.mean())
    mlflow.sklearn.log_model(cat_pipeline, artifact_path="models")
    print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")  

    cat_pipeline['catboost'].get_feature_importance(prettified=True)


# In[43]:


cat_pipeline['catboost'].get_feature_importance(prettified=True)


# In[44]:


# feature cross of pdays & previous
# log transformed balance 
# feature cross of month and day_of_week

with mlflow.start_run():

    mlflow.set_tag("model", "Catboost Pipeline age & day_of_week binned and feature cross of pdays and previous and month day_of_week")

    mlflow.log_param("train-data-path", "https://archive.ics.uci.edu/dataset/222/bank+marketing") 

    df_copy = df.copy()
    #feature cross
    df_copy['pdays_previous_cross']= df_copy['pdays'].astype(str) + '_' + df_copy['previous'].astype(str)
    df_copy['month_day_of_week_cross']= df_copy['month'].astype(str) + '_' + df_copy['day_of_week'].astype(str)


    #add age binned column and drop age column
    df_copy['age_binned'] = pd.cut(df['age'], bins=10).astype(str)
    df_copy = df_copy.drop(['age'],axis=1) 
      
    #add day_of_week binned column and drop day_of_week column
    df_copy['day_of_week_binned'] = pd.cut(df['day_of_week'], bins=10).astype(str)
    df_copy.drop(['day_of_week'],axis=1,inplace=True) 

    #log transformed balance
    df_copy['balance_log']= np.log(df_copy['balance'])
    df_copy.drop(['balance'],axis=1,inplace=True) 

    #recreate training and testing data split
    X = df_copy.drop(['y'],axis=1)
    y = df_copy['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)      

    #train catboost and get the cross validation score
    cat = CatBoostClassifier(random_state=123,cat_features=list(X.select_dtypes(include='object').columns))
    cat_pipeline=Pipeline([('catboost',cat)])
    cat_scores=cross_val_score(cat_pipeline,X_train,y_train,cv=3,scoring='f1_weighted',verbose=False)
    print("3-fold cross validation weighted f1 score:{:.3f}".format(cat_scores.mean()))
    cat_pipeline.fit(X_train,y_train,catboost__verbose=False) 

    #logging metrics
    mlflow.log_metric("f1_weighted", cat_scores.mean())
    mlflow.sklearn.log_model(cat_pipeline, artifact_path="models")
    print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")  

    cat_pipeline['catboost'].get_feature_importance(prettified=True)


# In[45]:


cat_pipeline['catboost'].get_feature_importance(prettified=True)


# In[46]:


# feature cross of pdays & previous
# log transformed balance 
# feature cross of month and day_of_week
#dropping the month column

with mlflow.start_run():

    mlflow.set_tag("model", "Catboost Pipeline age & day_of_week binned and feature cross of pdays and previous and month day_of_week")

    mlflow.log_param("train-data-path", "https://archive.ics.uci.edu/dataset/222/bank+marketing") 

    df_copy = df.copy()
    #feature cross
    df_copy['pdays_previous_cross']= df_copy['pdays'].astype(str) + '_' + df_copy['previous'].astype(str)
    df_copy['month_day_of_week_cross']= df_copy['month'].astype(str) + '_' + df_copy['day_of_week'].astype(str)
    df_copy.drop(['month'],axis=1,inplace=True) 


    #add age binned column and drop age column
    df_copy['age_binned'] = pd.cut(df['age'], bins=10).astype(str)
    df_copy = df_copy.drop(['age'],axis=1) 
      
    #add day_of_week binned column and drop day_of_week column
    df_copy['day_of_week_binned'] = pd.cut(df['day_of_week'], bins=10).astype(str)
    df_copy.drop(['day_of_week'],axis=1,inplace=True) 

    #log transformed balance
    df_copy['balance_log']= np.log(df_copy['balance'])
    df_copy.drop(['balance'],axis=1,inplace=True) 

    #recreate training and testing data split
    X = df_copy.drop(['y'],axis=1)
    y = df_copy['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)      

    #train catboost and get the cross validation score
    cat = CatBoostClassifier(random_state=123,cat_features=list(X.select_dtypes(include='object').columns))
    cat_pipeline=Pipeline([('catboost',cat)])
    cat_scores=cross_val_score(cat_pipeline,X_train,y_train,cv=3,scoring='f1_weighted',verbose=False)
    print("3-fold cross validation weighted f1 score:{:.3f}".format(cat_scores.mean()))
    cat_pipeline.fit(X_train,y_train,catboost__verbose=False) 

    #logging metrics
    mlflow.log_metric("f1_weighted", cat_scores.mean())
    mlflow.sklearn.log_model(cat_pipeline, artifact_path="models")
    print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")  

    cat_pipeline['catboost'].get_feature_importance(prettified=True)


# In[50]:


# feature cross of pdays & previous
# binned balance 
# feature cross of month and day_of_week

with mlflow.start_run():

    mlflow.set_tag("model", "Catboost Pipeline age & day_of_week & balancce binned and feature cross of pdays and previous and month day_of_week")

    mlflow.log_param("train-data-path", "https://archive.ics.uci.edu/dataset/222/bank+marketing") 

    df_copy = df.copy()
    #feature cross
    df_copy['pdays_previous_cross']= df_copy['pdays'].astype(str) + '_' + df_copy['previous'].astype(str)
    df_copy['month_day_of_week_cross']= df_copy['month'].astype(str) + '_' + df_copy['day_of_week'].astype(str)


    #add age binned column and drop age column
    df_copy['age_binned'] = pd.cut(df['age'], bins=10).astype(str)
    df_copy = df_copy.drop(['age'],axis=1) 
      
    #add day_of_week binned column and drop day_of_week column
    df_copy['day_of_week_binned'] = pd.cut(df['day_of_week'], bins=10).astype(str)
    df_copy.drop(['day_of_week'],axis=1,inplace=True) 

    #add balance binned column and drop balance column
    df_copy['balance_binned'] = pd.cut(df['balance'], bins=10).astype(str)
    df_copy.drop(['balance'],axis=1,inplace=True) 


    #recreate training and testing data split
    X = df_copy.drop(['y'],axis=1)
    y = df_copy['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)      

    #train catboost and get the cross validation score
    cat = CatBoostClassifier(random_state=123,cat_features=list(X.select_dtypes(include='object').columns))
    cat_pipeline=Pipeline([('catboost',cat)])
    cat_scores=cross_val_score(cat_pipeline,X_train,y_train,cv=3,scoring='f1_weighted',verbose=False)
    print("3-fold cross validation weighted f1 score:{:.3f}".format(cat_scores.mean()))
    cat_pipeline.fit(X_train,y_train,catboost__verbose=False) 

    #logging metrics
    mlflow.log_metric("f1_weighted", cat_scores.mean())
    mlflow.sklearn.log_model(cat_pipeline, artifact_path="models")
    print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")  

    cat_pipeline['catboost'].get_feature_importance(prettified=True)


# In[54]:


cat_scores.mean()


# In[56]:


train_sizes, train_scores, validation_scores = learning_curve(
estimator = cat_pipeline['catboost'],
X = X_train,
y = y_train, 
n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 10), 
cv = 5, scoring = 'f1_weighted')


# In[57]:


import matplotlib.pyplot as plt

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

test_mean = np.mean(validation_scores, axis=1)
test_std = np.std(validation_scores, axis=1)

plt.subplots(1, figsize=(10,10))
plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()


# In[60]:


#apply regularization and finalize this model
# feature cross of pdays & previous
# binned balance 
# feature cross of month and day_of_week

with mlflow.start_run():

    mlflow.set_tag("model", "Catboost Pipeline age & day_of_week & balancce binned and feature cross of pdays and previous and month day_of_week and regularization")

    mlflow.log_param("train-data-path", "https://archive.ics.uci.edu/dataset/222/bank+marketing") 

    df_copy = df.copy()
    #feature cross
    df_copy['pdays_previous_cross']= df_copy['pdays'].astype(str) + '_' + df_copy['previous'].astype(str)
    df_copy['month_day_of_week_cross']= df_copy['month'].astype(str) + '_' + df_copy['day_of_week'].astype(str)


    #add age binned column and drop age column
    df_copy['age_binned'] = pd.cut(df['age'], bins=10).astype(str)
    df_copy = df_copy.drop(['age'],axis=1) 
      
    #add day_of_week binned column and drop day_of_week column
    df_copy['day_of_week_binned'] = pd.cut(df['day_of_week'], bins=10).astype(str)
    df_copy.drop(['day_of_week'],axis=1,inplace=True) 

    #add balance binned column and drop balance column
    df_copy['balance_binned'] = pd.cut(df['balance'], bins=10).astype(str)
    df_copy.drop(['balance'],axis=1,inplace=True) 


    #recreate training and testing data split
    X = df_copy.drop(['y'],axis=1)
    y = df_copy['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)      

    #train catboost and get the cross validation score
    cat = CatBoostClassifier(random_state=123,
                             cat_features=list(X.select_dtypes(include='object').columns),
                             l2_leaf_reg=3.0)
    cat_pipeline=Pipeline([('catboost',cat)])
    cat_scores=cross_val_score(cat_pipeline,X_train,y_train,cv=3,scoring='f1_weighted',verbose=False)
    print("3-fold cross validation weighted f1 score:{:.3f}".format(cat_scores.mean()))
    cat_pipeline.fit(X_train,y_train,catboost__verbose=False) 

    #logging metrics
    mlflow.log_metric("f1_weighted", cat_scores.mean())
    mlflow.sklearn.log_model(cat_pipeline, artifact_path="models")
    print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")  

    cat_pipeline['catboost'].get_feature_importance(prettified=True)



# In[61]:


cat_scores.mean()


# In[62]:


train_sizes, train_scores, validation_scores = learning_curve(
estimator = cat_pipeline['catboost'],
X = X_train,
y = y_train, 
n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 10), 
cv = 5, scoring = 'f1_weighted')


# In[63]:


train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

test_mean = np.mean(validation_scores, axis=1)
test_std = np.std(validation_scores, axis=1)

plt.subplots(1, figsize=(10,10))
plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()


# ## Hyperparameter Tuning for finalized model

# In[14]:


from skopt import BayesSearchCV
from skopt.space import Real, Integer

#define hyperparameter search space
param_space = {
    'catboost__depth': Integer(4, 8),
    'catboost__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'catboost__iterations': Integer(100, 300),
    'catboost__l2_leaf_reg': Real(1.0, 5.0, prior='log-uniform')
}

#limiting iterations for the bayes search to keep it efficient
n_iter_search = 25

with mlflow.start_run():

    mlflow.set_tag("model", "Catboost Pipeline with Bayesian optimization and feature engineering")
    mlflow.log_param("train-data-path", "https://archive.ics.uci.edu/dataset/222/bank+marketing") 

    df_copy = df.copy()

    # Feature cross
    df_copy['pdays_previous_cross'] = df_copy['pdays'].astype(str) + '_' + df_copy['previous'].astype(str)
    df_copy['month_day_of_week_cross'] = df_copy['month'].astype(str) + '_' + df_copy['day_of_week'].astype(str)

    #add age binned column and drop age column
    df_copy['age_binned'] = pd.cut(df['age'], bins=10).astype(str)
    df_copy = df_copy.drop(['age'], axis=1)

    #add day_of_week binned column and drop day_of_week column
    df_copy['day_of_week_binned'] = pd.cut(df['day_of_week'], bins=10).astype(str)
    df_copy.drop(['day_of_week'], axis=1, inplace=True)

    #add balance binned column and drop balance column
    df_copy['balance_binned'] = pd.cut(df['balance'], bins=10).astype(str)
    df_copy.drop(['balance'], axis=1, inplace=True)

    #recreate train and test data split
    X = df_copy.drop(['y'], axis=1)
    y = df_copy['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    cat = CatBoostClassifier(random_state=123, cat_features=list(X.select_dtypes(include='object').columns))

    cat_pipeline = Pipeline([('catboost', cat)])

    #hyperparameter tuning with BayesSearchCV
    bayes_search = BayesSearchCV(
        estimator=cat_pipeline,
        search_spaces=param_space,
        n_iter=n_iter_search,
        cv=3,
        scoring='f1_weighted',
        random_state=42
    )
    bayes_search.fit(X_train, y_train)

    #log best hyperparameters
    mlflow.log_params(bayes_search.best_params_)
    

    best_model = bayes_search.best_estimator_
    best_model.fit(X_train, y_train)
    best_cv_score = bayes_search.best_score_
    print(f"Best CV f1_weighted: {best_cv_score:.3f}")

    #logging
    mlflow.log_metric("f1_weighted", best_cv_score)
    mlflow.sklearn.log_model(best_model, artifact_path="models")
    feature_importance = best_model['catboost'].get_feature_importance(prettified=True)
    mlflow.log_text(str(feature_importance), 'feature_importance.txt')

    print(f"Default artifacts URI: '{mlflow.get_artifact_uri()}'")


# In[17]:


import numpy as np


# In[18]:


## Plotting learning curve to check that the best model didn't overfit

train_sizes, train_scores, validation_scores = learning_curve(
estimator = best_model['catboost'],
X = X_train,
y = y_train, 
n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 10), 
cv = 5, scoring = 'f1_weighted')


# In[20]:


import matplotlib.pyplot as plt


# In[21]:


train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

test_mean = np.mean(validation_scores, axis=1)
test_std = np.std(validation_scores, axis=1)

plt.subplots(1, figsize=(10,10))
plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("F1 Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()


# ### Try balancing classes

# 1. Class weights

# In[23]:


#remove 'catboost__' prefix from best_params_
cleaned_best_params = {key.split('__')[-1]: value for key, value in bayes_search.best_params_.items()}

with mlflow.start_run():

    mlflow.set_tag("model", "Catboost Pipeline with Bayesian optimization and feature engineering and class balancing")
    mlflow.log_param("train-data-path", "https://archive.ics.uci.edu/dataset/222/bank+marketing") 

    df_copy = df.copy()

    # Feature cross
    df_copy['pdays_previous_cross'] = df_copy['pdays'].astype(str) + '_' + df_copy['previous'].astype(str)
    df_copy['month_day_of_week_cross'] = df_copy['month'].astype(str) + '_' + df_copy['day_of_week'].astype(str)

    #add age binned column and drop age column
    df_copy['age_binned'] = pd.cut(df['age'], bins=10).astype(str)
    df_copy = df_copy.drop(['age'], axis=1)

    #add day_of_week binned column and drop day_of_week column
    df_copy['day_of_week_binned'] = pd.cut(df['day_of_week'], bins=10).astype(str)
    df_copy.drop(['day_of_week'], axis=1, inplace=True)

    #add balance binned column and drop balance column
    df_copy['balance_binned'] = pd.cut(df['balance'], bins=10).astype(str)
    df_copy.drop(['balance'], axis=1, inplace=True)

    #recreate train and test data split
    X = df_copy.drop(['y'], axis=1)
    y = df_copy['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    cat = CatBoostClassifier(random_state=123, cat_features=list(X.select_dtypes(include='object').columns),
    class_weights={0: 1, 1: 7.5},  # class imbalance
        **cleaned_best_params)

    cat_pipeline = Pipeline([('catboost', cat)])

    cat_scores=cross_val_score(cat_pipeline,X_train,y_train,cv=3,scoring='f1_weighted',verbose=False)
    print("3-fold cross validation weighted f1 score:{:.3f}".format(cat_scores.mean()))
    cat_pipeline.fit(X_train,y_train,catboost__verbose=False) 

    #logging metrics
    mlflow.log_metric("f1_weighted", cat_scores.mean())
    mlflow.sklearn.log_model(cat_pipeline, artifact_path="models")
    print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")  

    feature_importance = cat_pipeline['catboost'].get_feature_importance(prettified=True)
    mlflow.log_text(str(feature_importance), 'feature_importance.txt')

    print(f"Default artifacts URI: '{mlflow.get_artifact_uri()}'")


# 2. Oversampling

# In[27]:


#remove 'catboost__' prefix from best_params_
cleaned_best_params = {key.split('__')[-1]: value for key, value in bayes_search.best_params_.items()}

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

with mlflow.start_run():

    mlflow.set_tag("model", "Catboost Pipeline with Bayesian optimization and feature engineering and oversampling")
    mlflow.log_param("train-data-path", "https://archive.ics.uci.edu/dataset/222/bank+marketing") 

    df_copy = df.copy()

    # Feature cross
    df_copy['pdays_previous_cross'] = df_copy['pdays'].astype(str) + '_' + df_copy['previous'].astype(str)
    df_copy['month_day_of_week_cross'] = df_copy['month'].astype(str) + '_' + df_copy['day_of_week'].astype(str)

    #add age binned column and drop age column
    df_copy['age_binned'] = pd.cut(df['age'], bins=10).astype(str)
    df_copy = df_copy.drop(['age'], axis=1)

    #add day_of_week binned column and drop day_of_week column
    df_copy['day_of_week_binned'] = pd.cut(df['day_of_week'], bins=10).astype(str)
    df_copy.drop(['day_of_week'], axis=1, inplace=True)

    #add balance binned column and drop balance column
    df_copy['balance_binned'] = pd.cut(df['balance'], bins=10).astype(str)
    df_copy.drop(['balance'], axis=1, inplace=True)

    #recreate train and test data split
    X = df_copy.drop(['y'], axis=1)
    y = df_copy['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    oversampling_pipeline = ImbPipeline([
        ('sampler', RandomOverSampler(random_state=42)),
        ('catboost',CatBoostClassifier(
                random_state=123,
                cat_features=list(X.select_dtypes(include='object').columns),
                **cleaned_best_params
            )
        )
    ])    

    cat_scores=cross_val_score(cat_pipeline,X_train,y_train,cv=3,scoring='f1_weighted',verbose=False)
    print("3-fold cross validation weighted f1 score:{:.3f}".format(cat_scores.mean()))
    cat_pipeline.fit(X_train,y_train,catboost__verbose=False) 

    #logging metrics
    mlflow.log_metric("f1_weighted", cat_scores.mean())
    mlflow.sklearn.log_model(cat_pipeline, artifact_path="models")
    print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")  

    feature_importance = cat_pipeline['catboost'].get_feature_importance(prettified=True)
    mlflow.log_text(str(feature_importance), 'feature_importance.txt')

    print(f"Default artifacts URI: '{mlflow.get_artifact_uri()}'")


# 3. Undersampling

# In[28]:


#remove 'catboost__' prefix from best_params_
cleaned_best_params = {key.split('__')[-1]: value for key, value in bayes_search.best_params_.items()}

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

with mlflow.start_run():

    mlflow.set_tag("model", "Catboost Pipeline with Bayesian optimization and feature engineering and undersampling")
    mlflow.log_param("train-data-path", "https://archive.ics.uci.edu/dataset/222/bank+marketing") 

    df_copy = df.copy()

    # Feature cross
    df_copy['pdays_previous_cross'] = df_copy['pdays'].astype(str) + '_' + df_copy['previous'].astype(str)
    df_copy['month_day_of_week_cross'] = df_copy['month'].astype(str) + '_' + df_copy['day_of_week'].astype(str)

    #add age binned column and drop age column
    df_copy['age_binned'] = pd.cut(df['age'], bins=10).astype(str)
    df_copy = df_copy.drop(['age'], axis=1)

    #add day_of_week binned column and drop day_of_week column
    df_copy['day_of_week_binned'] = pd.cut(df['day_of_week'], bins=10).astype(str)
    df_copy.drop(['day_of_week'], axis=1, inplace=True)

    #add balance binned column and drop balance column
    df_copy['balance_binned'] = pd.cut(df['balance'], bins=10).astype(str)
    df_copy.drop(['balance'], axis=1, inplace=True)

    #recreate train and test data split
    X = df_copy.drop(['y'], axis=1)
    y = df_copy['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    oversampling_pipeline = ImbPipeline([
        ('sampler', RandomUnderSampler(random_state=42)),
        ('catboost',CatBoostClassifier(
                random_state=123,
                cat_features=list(X.select_dtypes(include='object').columns),
                **cleaned_best_params
            )
        )
    ])    

    cat_scores=cross_val_score(cat_pipeline,X_train,y_train,cv=3,scoring='f1_weighted',verbose=False)
    print("3-fold cross validation weighted f1 score:{:.3f}".format(cat_scores.mean()))
    cat_pipeline.fit(X_train,y_train,catboost__verbose=False) 

    #logging metrics
    mlflow.log_metric("f1_weighted", cat_scores.mean())
    mlflow.sklearn.log_model(cat_pipeline, artifact_path="models")
    print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")  

    feature_importance = cat_pipeline['catboost'].get_feature_importance(prettified=True)
    mlflow.log_text(str(feature_importance), 'feature_importance.txt')

    print(f"Default artifacts URI: '{mlflow.get_artifact_uri()}'")


# Balancing classes didn't improve the score so we'll finalize the hyperparameter tuned model and move it to staging.

# ## Moving model to staging

# In[29]:


best_model['catboost']


# In[38]:


from mlflow.tracking import MlflowClient

client = MlflowClient("http://127.0.0.1:5000")
runs = client.search_runs(experiment_ids='1',order_by=["metrics.f1_weighted DESC"])


# In[41]:


for run in runs:
    try:
        print(f"run id: {run.info.run_id}, model name: {run.data.tags['model']},"+
            f"f1 score: {run.data.metrics['f1_weighted']:.4f}, duration(s): {(run.info.end_time-run.info.start_time)/1000:.2f}")
    except:
        print("")


# In[42]:


run_id = "a3cabd1f4cc6434a81762deef514f951"
mlflow.register_model(
    model_uri=f"runs:/{run_id}/models",
    name='bank-classifier'
)


# In[43]:


model_name = "bank-classifier"
latest_versions = client.get_latest_versions(name=model_name)

for version in latest_versions:
    print(f"version: {version.version}, stage: {version.current_stage}")


# In[44]:


#moving model to staging
model_version = 1
new_stage = "Staging"
client.transition_model_version_stage(
    name=model_name,
    version=model_version,
    stage=new_stage,
    archive_existing_versions=False
)


# In[45]:


client.transition_model_version_stage(
    name=model_name,
    version=1,
    stage="Production",
    archive_existing_versions=True
)


# ## Predicting the target on X_test

# In[46]:


import mlflow
logged_model = 'runs:/a3cabd1f4cc6434a81762deef514f951/models'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd

y_pred = loaded_model.predict(X_test)


# In[ ]:




