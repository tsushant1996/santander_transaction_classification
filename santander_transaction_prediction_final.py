
# coding: utf-8

# <h2>Santander Customer Transaction Prediction</h2>
# <br>
# <h4>Background -</h4>
# <p>At Santander, mission is to help people and businesses prosper. We are always looking
# for ways to help our customers understand their financial health and identify which
# products and services might help them achieve their monetary goals.
# Our data science team is continually challenging our machine learning algorithms,
# working with the global data science community to make sure we can more accurately
# identify new ways to solve our most common challenge, binary classification problems
# such as: is a customer satisfied? Will a customer buy this product? Can a customer pay
# this loan?
# </p>
# <h4>Problem Statement -</h4>
# <p>In this challenge, we need to identify which customers will make a  transaction in
# the future, irrespective of the amount of money transacted.<p>
# <p>You are provided with an anonymized dataset containing numeric feature variables, the
# binary target column, and a string ID_code column. The task is to predict the value
# of target column in the test set.<p>

# <h4> As from the problem statement it is confirmed that the problem that we are going to solve is binary classification problem</h4>
# <p>In this problem we have to predict target variable which is 0 or 1?</p>

# In[1]:


#importing all the libraries 
import pandas as pd #for dataframe manipulation
import seaborn as sns #written on top of matplotlib for data visualization
import numpy as np #for Numerical computing
import random
from math import radians, cos, sin, asin, sqrt 
from sklearn.model_selection import train_test_split,GridSearchCV #fo gridsearch and train-test split
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt #for visualization

sns.set()
random.seed(113)
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
#slearn for machine learning algorithms
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from imblearn.under_sampling import NearMiss #for performing under-sampling based on NearMiss methods.
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import randint 
from sklearn.ensemble import RandomForestClassifier


# <h3>Data Collection</h3>

# In[2]:


##reading data
df_train =  pd.read_csv("train.csv");


# In[3]:


#let us get some insights of data
df_train.head()


# <h3>Data Exploration and visualisation</h3>

# In[4]:


print(df_train.shape) #Let us check the shape of dataset


# In[5]:


"""
We have 200000 rows and 202 columns in dataset
"""
#Let us have some information about the dataframe like null values and types
df_train.info()


# In[6]:


##checking for missing values
df_train.count()


# In[7]:


df_train.columns


# <h4>Now we will check for missing values </h4>

# In[8]:


#checking for null values in all the fields if any
df_train.isnull().values.any()


# 
# As we can see we don't have any null values

# In[9]:


#Dropping target and ID_code to plot the histograms
df_train_custom = df_train.drop(["target","ID_code"],axis=1)


# In[10]:


for i in df_train_custom.columns:
    ax = sns.boxplot(x=df_train_custom[i])
    plt.xlabel(i)
    plt.show()


# In[11]:


##uncomment this section of code if you want outliers to be removed

# # columns = df_train.columns

# for i in tqdm(reversed(range(202))):
# #     print( df_train.columns[i])
    
#     if(df_train.columns[i] ==  "target" or df_train.columns[i] == "ID_code" ):
#         print(i)
#     else:  
#         print("operating on {}".format(df_train.columns[i]))
#         q75,q25 = np.percentile(df_train.loc[:,df_train.columns[i]],[75,25])
#         iqr = q75-q25
#         print(i)
#         min = q25 - (iqr*1.5)
#         max = q75 + (iqr*1.5)

#         print(df_train.shape)
#         df_train = df_train[~(df_train[df_train.columns[i]] < min)]
#         df_train = df_train[~(df_train[df_train.columns[i]] > max)]

#     df =df.drop(df[df.loc[:,i] < min].index)
#     df = df.drop(df[df.loc[:,i] > max].index)
   


# In[12]:


#getting shape of df_train
df_train.shape


# In[13]:


# df_train.to_csv(r'df_train_removed_outlier.csv',index=False)


# In[14]:


# df_train = pd.read_csv('df_train_removed_outlier.csv')


# In[15]:


#checking for class labels and class label distribution
sns.countplot(df_train['target'], palette='Set3')
print(df_train['target'].value_counts())
plt.show()


# <h4>As from the diagram we can see that the data is highly imbalanced </h4>
# <p>
# we have several ways to tackle the imbalanced dataset like
# </p>
# <li>Oversampling</li>
# <li>Undersampling</li>
# <li>Or we can use hyperparameter tunning to tune the model</li>
# 
# <p>We have huge amount of data so we cannot o for oversampling hence we can undersample the data to make our model training faster 
# </p>
# 

# <br>
# <br>
# <h3>Selection of performance metrics</h3>
# <p>we cannot use accuracy because a random or a dumb model which returns 1 almost all the times can also have 90% accuracy if all the queries which are made includes 90% of class 1 input varibles
# <p>As this is classification problem with imbalanced dataset we will be using confusion metrics,precision,recall,Roc and auc curve 
# </p>

# <h4>Univariate Analysis</h4>

# In[16]:


#Now we will check the distributions of different variables

# ax = sns.distplot(df['temp'])
ax = plt.hist(df_train['var_0'])
plt.xlabel('var_0')
plt.show()


# In[17]:


# ax = sns.distplot(df['temp'])
ax = plt.hist(df_train['var_1'])
plt.xlabel('var_1')
plt.show()


# In[18]:


# ax = sns.distplot(df['temp'])
ax = plt.hist(df_train['var_3'])
plt.xlabel('var_3')
plt.show()


# In[19]:


#plotting pairplots,to get insights of data
g = sns.pairplot(df_train,hue="target", vars=["var_0", "var_1","var_2"])
plt.show()


# In[20]:


#getting some statistical information of all the coulumn
df_train.describe()


# <h4> Feature Correlation</h4>
# 

# In[21]:


"""
Also displaying correlation plot to detect the collinearity in data 
"""
f, ax = plt.subplots(figsize=(15, 8))
corr =df_train.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
plt.show()


# <h4>Multicollinearity detection</h4>
# As from this correlation plot we can say that collinarity beteen variables are very small and hence correlation is not our problem.Let's check for multicollinearity

# <p>For multicollinearity we can check for VIF (variable inflation factor)<p>

# In[22]:


##Uncomment this section of code if you want to calculate VIF for all the variables and save it to dataframe

vif = pd.DataFrame()
df_train_custom = df_train.drop(["target","ID_code"],axis=1)
df_train_custom.head()
vif["VIF Factor"] = [variance_inflation_factor(df_train_custom.values, i) for i in range(df_train_custom.shape[1])]

vif["features"] = df_train_custom.columns


# In[23]:


vif.head(200)


# As from the above data we can see we don't have huge multicollinearirt between variables that we have randomly selected

# <h4>Feature scaling</h4>
# <p>As we will go from basic to advance model we will first use logistic regression and we know logistic regression 
# is distance based method we will use scale our feature </p>

# In[24]:


##Scaling features as we will use distance base model also like Logistic regression
df_train_custom = df_train.drop(["target","ID_code"],axis=1)
df_train_custom = StandardScaler().fit_transform(df_train_custom)
df_train_custom = pd.DataFrame(df_train_custom)
df_train_custom = df_train[['ID_code','target']].join(df_train_custom)


                               
                               



# In[25]:


df_train_custom.head()


# <h4>Feature Importance</h4>

# In[26]:


##Getting feature importance to get insights of some features
##Running random forest with gridseacrhcv for hyperparameter tunning
parameters = {'min_samples_leaf': [20, 25]}
forest = RandomForestClassifier(max_depth=15, n_estimators=15)
grid = GridSearchCV(forest, parameters, cv=3, n_jobs=-1, verbose=2, scoring=make_scorer(roc_auc_score))


# In[27]:


grid.fit(df_train.drop(["target","ID_code"], axis=1).values, df_train.target.values)


# In[28]:


grid.best_score_


# In[29]:



##Displaying important features
n_top = 5
importances = grid.best_estimator_.feature_importances_
idx = np.argsort(importances)[::-1][0:n_top]
feature_names = df_train.drop("target", axis=1).columns.values

plt.figure(figsize=(20,5))
sns.barplot(x=feature_names[idx], y=importances[idx]);
plt.show()
# plt.title("What are the top important features to start with?");


# In[30]:


# fig, ax = plt.subplots(n_top,2,figsize=(20,5*n_top))

# for n in range(n_top):
#     sns.distplot(train.loc[train.target==0, feature_names[idx][n]], ax=ax[n,0], color="Orange", norm_hist=True)
#     sns.distplot(train.loc[train.target==1, feature_names[idx][n]], ax=ax[n,0], color="Red", norm_hist=True)
#     sns.distplot(test.loc[:, feature_names[idx][n]], ax=ax[n,1], color="Mediumseagreen", norm_hist=True)
#     ax[n,0].set_title("Train {}".format(feature_names[idx][n]))
#     ax[n,1].set_title("Test {}".format(feature_names[idx][n]))
#     ax[n,0].set_xlabel("")
#     ax[n,1].set_xlabel("")
    


# In[31]:


##Splitting data in to test and train with 70:30 ratio
train, test = train_test_split(df_train_custom, test_size=0.3)


# In[32]:


train.shape


# In[33]:


test.shape


# In[34]:


#dropping target and id code to make data suitable for feeding in model
x_train = train.drop(["target","ID_code"],axis=1)
x_test = test.drop(["target","ID_code"],axis=1)

y_train = train[['target']]
y_test = test[['target']]


# In[35]:


# This function plots the confusion matrices given y_i, y_i_hat.
def plot_confusion_matrix(test_y, predict_y):
    C = confusion_matrix(test_y, predict_y)
    # C = 9,9 matrix, each cell (i,j) represents number of points of class i are predicted class j
    
    A =(((C.T)/(C.sum(axis=1))).T)
    #divid each element of the confusion matrix with the sum of elements in that column
    
    # C = [[1, 2],
    #     [3, 4]]
    # C.T = [[1, 3],
    #        [2, 4]]
    # C.sum(axis = 1)  axis=0 corresonds to columns and axis=1 corresponds to rows in two diamensional array
    # C.sum(axix =1) = [[3, 7]]
    # ((C.T)/(C.sum(axis=1))) = [[1/3, 3/7]
    #                           [2/3, 4/7]]

    # ((C.T)/(C.sum(axis=1))).T = [[1/3, 2/3]
    #                           [3/7, 4/7]]
    # sum of row elements = 1
    
    B =(C/C.sum(axis=0))
    #divid each element of the confusion matrix with the sum of elements in that row
    # C = [[1, 2],
    #     [3, 4]]
    # C.sum(axis = 0)  axis=0 corresonds to columns and axis=1 corresponds to rows in two diamensional array
    # C.sum(axix =0) = [[4, 6]]
    # (C/C.sum(axis=0)) = [[1/4, 2/6],
    #                      [3/4, 4/6]] 
    plt.figure(figsize=(20,4))
    
    labels = [1,2]
    # representing A in heatmap format
    cmap=sns.light_palette("blue")
    plt.subplot(1, 3, 1)
    sns.heatmap(C, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.title("Confusion matrix")
    
    plt.subplot(1, 3, 2)
    sns.heatmap(B, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.title("Precision matrix")
    
    plt.subplot(1, 3, 3)
    # representing B in heatmap format
    sns.heatmap(A, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.title("Recall matrix")
    
    plt.show()


# In[36]:


##Running Logistic regression with diff values of alpha for  hyperparameter tunning
alpha = [10 ** x for x in range(-5, 2)] #saving alpha in array and looping model with each alpha
tr_scores = []
cv_scores = []
# y_train_predict
# y_test_predict
for i in alpha:
        lr = LogisticRegression(penalty='l2',C=i,class_weight='balanced')
        lr.fit(x_train,y_train)
        y_train_predict  = lr.predict_proba(x_train)
        y_test_predict  =  lr.predict_proba(x_test)
        print(y_train_predict.shape)
        print(y_train.shape)
        tr_scores.append(roc_auc_score(y_train,np.argmax(y_train_predict,axis=1)))
        cv_scores.append(roc_auc_score(y_test,np.argmax(y_test_predict,axis=1)))
        
        
        


# In[37]:


optimal = alpha[cv_scores.index(np.max(cv_scores))] ##getting optimal alpha from arrays of alpha


# In[38]:


##Running Logistic regression on optimal alpha
lr = LogisticRegression(penalty='l2',C=optimal,class_weight='balanced')
lr.fit(x_train,y_train)
y_train_predict  = lr.predict_proba(x_train)
y_test_predict  =  lr.predict_proba(x_test)
#         print(y_train_predict.shape)
#         print(y_train.shape)
print(roc_auc_score(y_train,np.argmax(y_train_predict,axis=1)))
print(roc_auc_score(y_test,np.argmax(y_test_predict,axis=1)))
##printing confusion matrix,precsion, recall matrix and roc_auc score
plot_confusion_matrix(y_test, np.argmax(y_test_predict,axis=1))
fpr, tpr, thresholds = roc_curve(y_test, np.argmax(y_test_predict,axis=1))


# In[39]:


##plotting ROC curve
plt.plot(fpr, tpr, label='ROC curve')
plt.xlabel("False postive rate")
plt.ylabel("True Positive  rate")
plt.title("ROC curve")
plt.show()


# In[40]:


train,test = train_test_split(df_train, test_size=0.3)
x_train =  train.drop(["target","ID_code"],axis=1)
y_train = train[['target']]

x_test =  test.drop(["target","ID_code"],axis=1)
y_test = test[['target']]



# In[41]:


x_train.shape


# In[42]:


x_test.shape


# In[43]:


# Creating the hyperparameter grid   and running decision tree with gridsearchcv for hyperparamter tunning
param_dist = {"max_depth": [3,4,5], 
              "criterion": ["entropy"]} 
  
# Instantiating Decision Tree classifier 
tree = DecisionTreeClassifier(class_weight='balanced') 
  

tree_cv = GridSearchCV(tree, param_dist) 
  
tree_cv.fit(x_train, y_train)
y_train_predict = tree_cv.predict_proba(x_train)
y_test_predict = tree_cv.predict_proba(x_test)
##printing confusion matrix,precsion, recall matrix and roc_auc score
print(roc_auc_score(y_train,np.argmax(y_train_predict,axis=1)))
print(roc_auc_score(y_test,np.argmax(y_test_predict,axis=1)))
plot_confusion_matrix(y_test, np.argmax(y_test_predict,axis=1))
fpr, tpr, thresholds = roc_curve(y_test, np.argmax(y_test_predict,axis=1))


# In[44]:


##plotting ROC curve
plt.plot(fpr, tpr, label='ROC curve')
plt.xlabel("False postive rate")
plt.ylabel("True Positive  rate")
plt.title("ROC curve")
plt.show()


# In[45]:


### Creating the hyperparameter grid   and running Random forest  with gridsearchcv for hyperparamter tunning
parameters = {'min_samples_leaf': [100,200],'max_depth': [100], 'bootstrap': [True]}
forest = RandomForestClassifier(max_depth=15, n_estimators=100,class_weight="balanced")
grid = GridSearchCV(forest, parameters, n_jobs=-1)
grid.fit(x_train,y_train)
y_train_predict = grid.predict_proba(x_train)
y_test_predict = grid.predict_proba(x_test)
##printing confusion matrix,precsion, recall matrix and roc_auc score
print(roc_auc_score(y_train,np.argmax(y_train_predict,axis=1)))
print(roc_auc_score(y_test,np.argmax(y_test_predict,axis=1)))
plot_confusion_matrix(y_test, np.argmax(y_test_predict,axis=1))
fpr, tpr, thresholds = roc_curve(y_test, np.argmax(y_test_predict,axis=1))


# In[46]:


plt.plot(fpr, tpr, label='ROC curve')
plt.xlabel("False postive rate")
plt.ylabel("True Postive rate")
plt.title("ROC curve")
plt.show()


# In[47]:


##reading test data
df_test =  pd.read_csv("test.csv");


# In[48]:


df_test.head()


# In[49]:


#DROPPING ID_code for predicting
df_test =  df_test.drop(["ID_code"],axis=1)


# In[50]:


####################################TAKE AWAY FROM DIFFERNT MODELS####################################
#1.From all the models that we have logistic regression was having better auc_roc score
#2.From all the models that we have tried logistic regression was having better precision and recall
#3.From all the models that we have tried logistic regression was having better speed so we would accept logistic 
#regression for use on production enviroment
#######################################################################################################


# In[51]:


lr = LogisticRegression(penalty='l2',C=optimal,class_weight='balanced')
lr.fit(x_train,y_train)


# In[52]:


predict = lr.predict_proba(df_test)

