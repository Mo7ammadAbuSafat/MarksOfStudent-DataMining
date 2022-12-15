#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Mohammad AbuSafat & Haya AbuRaed                                                                          Dr. Nasha't Jallad
                                                 # Data Mining First Assignment


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the dataset
data = pd.read_csv("studetsPereformance.csv")

#Top 5 rows
data.head()


# In[2]:


#data shape
data.shape


# In[3]:


#data describe
data.describe()


# In[4]:


#Checking with null values
data.isnull().sum()


# In[5]:


#fill null values(I wrote two methods for solving the problem that Two adjacent values are null)
data.fillna(method ='ffill' ,inplace=True)
data.fillna(method ='bfill' ,inplace=True)

#Checking null values again
data.isnull().sum()


# In[6]:


#remove duplicate rows
data.drop_duplicates(inplace=True)

#Remove irrelevant attributes.
#we will drop race/ethnicity and lunch  attribute becouse it does not affect our study
data = data.drop(['race/ethnicity'] , axis=1) 
data = data.drop(['lunch'] , axis=1)
data


# In[7]:


#find Correlation
data.corr()


# In[8]:


#Remove correlated attributes
data = data.drop(['reading score'] , axis=1) #because it has correlation value greater than 0.8

#ensure correlation
data.corr()


# In[9]:


data.nunique()


# In[10]:


####  Start remove noise  ####

#remove noise from gender
#check if there is a noise
values = ['female','male']
filter = data['gender'].isin(values)
filter.value_counts()


# In[11]:


#remove noise
data = data.drop(index= data[~filter].index)

#ensure that noise is removed
filter = data['gender'].isin(values)
filter.value_counts()


# In[12]:


#remove noise from parental level of education
#check if there is a noise
values = ['bachelor\'s degree','some college','associate\'s degree','master\'s degree','high school','some high school']
filter = data['parental level of education'].isin(values)
filter.value_counts()


# In[13]:


#remove noise
data = data.drop(index= data[~filter].index)

#check if there is a noise
filter = data['parental level of education'].isin(values)
filter.value_counts()


# In[14]:


#remove noise from test preparation course
#check if there is a noise
values = ['none','completed']
filter = data['test preparation course'].isin(values)
filter.value_counts()


# In[15]:


#remove noise by set noise value equal none
data.loc[(~filter), 'test preparation course' ] = 'none'

#check if there is a noise
filter = data['test preparation course'].isin(values)
filter.value_counts()


# In[16]:


#remove noise from math score and writing score
#check if there is a noise
atts = ["math score", "writing score"]
plt.figure(figsize=(20,10))
data[atts].boxplot()
plt.title("atts show")
plt.show()


# In[17]:


#outlier values is under 30, so we'll set it with a value of 30 
filter1 = data['math score'] < 30
filter1.value_counts()


# In[18]:


filter2 = data['writing score']<30
filter2.value_counts()


# In[19]:


data.loc[(filter1), 'math score'] = 30
data.loc[(filter2), 'writing score'] = 30

#check that we get rid of the noise
atts = ["math score", "writing score"]
plt.figure(figsize=(20,10))
data[atts].boxplot()
plt.title("atts show")
plt.show()


# In[20]:


#show math score diagram
data['math score'].hist()


# In[21]:


#show writing score diagram
data['writing score'].hist()


# In[22]:


#Apply discretization on writing score attribute
bins=[29,39,49,59,69,79,89,100]
cat_labels=[35,45,55, 65, 75, 85, 95]
data['writing score'] = pd.cut(data['writing score'], bins, labels= cat_labels)
data.head()


# In[23]:


#Apply discretization on math score attribute
bins=[29,39,49,59,69,79,89,100]
cat_labels=[35,45,55, 65, 75, 85, 95]
data['math score'] = pd.cut(data['math score'], bins, labels= cat_labels)
data.head()


# In[24]:


#show writing score diagram
data['writing score'].hist()


# In[25]:


#show math score diagram
data['math score'].hist()


# In[26]:


#Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split as tt
trainSet, testSet = tt(data, test_size = 0.2)
trainSet.to_csv('trainSet.csv')
testSet.to_csv('testSet.csv')


# In[27]:


#Define the same data to set it as numerical values
import copy #to copy data
Dnumerical = copy.copy(data)


# In[28]:


#Gender Transformation
Dnumerical.loc[(Dnumerical['gender'] == 'male'), 'gender'] = 1
Dnumerical.loc[(Dnumerical['gender'] == 'female'), 'gender'] = 0
Dnumerical.head()


# In[29]:


#parental level of education Transformation
Dnumerical.loc[(Dnumerical['parental level of education'] == 'bachelor\'s degree'), 'parental level of education'] = 4
Dnumerical.loc[(Dnumerical['parental level of education'] == 'some college'), 'parental level of education'] = 2
Dnumerical.loc[(Dnumerical['parental level of education'] == 'associate\'s degree'), 'parental level of education'] = 3
Dnumerical.loc[(Dnumerical['parental level of education'] == 'master\'s degree'), 'parental level of education'] = 5
Dnumerical.loc[(Dnumerical['parental level of education'] == 'high school'), 'parental level of education'] = 1
Dnumerical.loc[(Dnumerical['parental level of education'] == 'some high school'), 'parental level of education'] = 0
Dnumerical.head()


# In[30]:


#test preparation course Transformation
Dnumerical.loc[(Dnumerical['test preparation course'] == 'none'), 'test preparation course'] = 0
Dnumerical.loc[(Dnumerical['test preparation course'] == 'completed'), 'test preparation course'] = 1
Dnumerical.head()


# In[31]:


#Rating Transformation
Dnumerical.loc[(Dnumerical['Rating'] == 'Fail'), 'Rating'] = 0
Dnumerical.loc[(Dnumerical['Rating'] == 'Acceptance'), 'Rating'] = 1
Dnumerical.loc[(Dnumerical['Rating'] == 'Not good'), 'Rating'] = 2
Dnumerical.loc[(Dnumerical['Rating'] == 'Good'), 'Rating'] = 3
Dnumerical.loc[(Dnumerical['Rating'] == 'Very good'), 'Rating'] = 4
Dnumerical.loc[(Dnumerical['Rating'] == 'Excellent'), 'Rating'] = 5

#convert_dtypes to numeric
Dnumerical["math score"] = pd.to_numeric(Dnumerical["math score"])
Dnumerical["writing score"] = pd.to_numeric(Dnumerical["writing score"])
Dnumerical["Rating"] = Dnumerical["Rating"].astype("category")
Dnumerical.head() #All columns have been converted


# In[ ]:





# In[ ]:





# In[32]:


data.head()


# In[33]:


#Split the Dnumerical into training and testing sets
trainSetNumeric, testSetNumeric = tt(Dnumerical, test_size = 0.2)

#clasification
trainD= trainSetNumeric.drop(['Rating'], axis=1); #train data
trainT= trainSetNumeric.Rating;                   #tarain target
testD= testSetNumeric.drop(['Rating'], axis=1);   #test data
testT= testSetNumeric.Rating;                     #test target


# In[34]:


#Make sure that it has been splittted
trainD.head(12)


# In[35]:


#Make sure that it has been splittted (cont.)
testD.head(12)


# In[36]:


#Make sure that it has been splittted (cont.)
trainT.head(12)


# In[37]:


#Make sure that it has been splittted (cont.)
testT.head(12)


# In[38]:


#classification algorithms

# 1- Apply K-Nearest Neighbors Algorithm

from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn import metrics
import seaborn as sns

#build model
model1= knn(n_neighbors=5)
model1.fit(trainD,trainT)

#predict
pred= model1.predict(testD)

#compare results
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(testT, pred)
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("pred")
plt.ylabel("testT")
plt.show()
print('-'*50)
print ("Test accurecy : ",metrics.accuracy_score(testT,pred))


# In[39]:


# 2- Apply naive bayes Algorithm

from sklearn.naive_bayes import GaussianNB

#build model
nb = GaussianNB()
nb.fit(trainD,trainT)

#predict
pred = nb.predict(testD)

#compare results
cm = confusion_matrix(testT, pred)
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("pred")
plt.ylabel("testT")
plt.show()
print('-'*50)
print("Test Accuracy: {}%".format(round(nb.score(testD,testT)*100,2)))


# In[40]:


#clustering algorithms

# 1- KMeans clustering
attributes= Dnumerical.drop(['Rating'], axis=1)
labels = Dnumerical.Rating

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

#build model
from sklearn.cluster import KMeans
model= KMeans(n_clusters=5)
model.fit(attributes)

#predict
pred = model.predict(attributes)

#compare results
from sklearn import metrics
contingecyMatrix = metrics.cluster.contingency_matrix(labels, pred)
print (contingecyMatrix)


# In[41]:


# To view and compare results
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels, pred)
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("pred")
plt.ylabel("labels")
plt.show()
ari= metrics.cluster.adjusted_rand_score(labels, pred)
print('-'*50)
print("Test Accuracy:" , ari)


# In[42]:


# 2- hieratchy cluster

#import and build hieratchy cluster model
from scipy.cluster.hierarchy import linkage, dendrogram
target_col = Dnumerical["Rating"]
feat = Dnumerical.drop(['Rating'], axis=1)

#Convert them into ndarrays
x = feat.to_numpy(dtype ='float32')
y = target_col.to_numpy()

# Calculate the linkage: mergings
mergings = linkage(x, method = 'ward')

# Plot the dendrogram
dendrogram(mergings, labels = y, leaf_rotation = 120, leaf_font_size = 6)
from matplotlib import pyplot as plt
plt.figure(figsize=(300, 150))
plt.show()


# In[43]:


# Predict and compare results
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='ward')
pred=cluster.fit_predict(x)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(target_col, pred)
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("pred")
plt.ylabel("target_col")
plt.show()
ari= metrics.cluster.adjusted_rand_score(target_col, pred)
print('-'*50)
print("Test Accuracy:" , ari)


# In[44]:


#Define the same data with nominal values
Dnominal = copy.copy(data)

#writing score and math score Transformation
bins=[29,39,49,59,69,79,89,100]
cat_labels=['G','F','E', 'D', 'C', 'B', 'A']
Dnominal['writing score'] = pd.cut(Dnominal['writing score'], bins, labels= cat_labels)
Dnominal['math score'] = pd.cut(Dnominal['math score'], bins, labels= cat_labels)
Dnominal.head() #All columns have been converted


# In[45]:


#association_rules

# 1- apriori association_rule algorithm :

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth, association_rules

#deal with Transaction
te = TransactionEncoder()
#transform the Transaction data set to 2D array
te_ary = te.fit(Dnominal).transform(Dnominal) 
print(te_ary)
print('-'*50)
#convert the array into DataFrame
dd = pd.DataFrame(te_ary, columns=te.columns_)
#get the frequent itemset
frequent_itemsets_apriori = apriori(dd, min_support=0.005, use_colnames=True)
#get the association_rules from frequent itemset 
rules_apriori = association_rules(frequent_itemsets_apriori, metric="lift", min_threshold=1.2)
print("frequent_itemsets: \n",frequent_itemsets_apriori)
print('-'*50)
print("rules: \n",rules_apriori)


# In[46]:


p1 = pd.DataFrame(frequent_itemsets_apriori)
p1.plot(kind="bar")
print('-'*50)
p2 = pd.DataFrame(rules_apriori)
p2.plot(kind="bar")


# In[47]:


# 2- fpgrowth association_rule algorithm :

#deal with Transaction
te = TransactionEncoder()
#transform the Transaction data set to 2D array
te_ary = te.fit(Dnominal).transform(Dnominal) 
print(te_ary)
print('-'*50)
#convert the array into DataFrame
dd = pd.DataFrame(te_ary, columns=te.columns_)
#get the frequent itemset
frequent_itemsets_fpgrowth = fpgrowth(dd, min_support=0.005, use_colnames=True)
#get the association_rules from frequent itemset
rules_fpgrowth = association_rules(frequent_itemsets_fpgrowth, metric="lift", min_threshold=1.2)
print("frequent_itemsets: \n",frequent_itemsets_fpgrowth)
print('-'*50)
print("rules: \n",rules_fpgrowth)


# In[48]:


#comparing :
p1 = pd.DataFrame(frequent_itemsets_fpgrowth)
p1.plot(kind="bar")  #run in interactive window to show the diagrame
p2 = pd.DataFrame(rules_fpgrowth)
p2.plot(kind="bar")  #run in interactive window to show the diagrame


# In[49]:


# I hope you like it, thank you

