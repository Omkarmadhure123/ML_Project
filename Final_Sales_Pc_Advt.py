#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.stats import zscore


# In[3]:


sales=pd.read_csv("Sales_2021.csv")
sales


# In[3]:


sales.isnull().sum()


# In[4]:


sales.hist()


# In[4]:


sales.shape


# In[5]:


sales.describe()


# In[6]:


import statistics as s
s.harmonic_mean(sales['Sales'])
s.harmonic_mean(sales['Advt'])


# In[7]:


sales.info()


# In[8]:


sales.kurtosis()


# In[9]:


sales.skew()


# In[10]:



sales.cov()


# In[11]:



sales.corr()


# In[ ]:





# In[12]:


sales.plot(kind='box')


# In[ ]:





# In[2]:


# sales.plot(kind='hist',layout=(3,1),figsize=(10,20))
sales


# In[14]:


import statsmodels.api as sm
from statsmodels.formula.api import ols


# In[15]:


model=ols('Sales~Advt',data=sales).fit()
model1=sm.stats.anova_lm(model)
model1


# In[16]:


print(model.summary())


# In[17]:


sales['pre1']=model.predict()


# In[18]:


sales


# In[19]:


sales['res1']=sales['Sales'].values-sales['pre1'].values


# In[20]:


sales


# In[21]:


from scipy.stats import zscore
sales['zscore']=zscore(sales['res1'])


# In[22]:


sales


# In[23]:


sales[sales['zscore'] > 1.96]


# In[24]:


sales[sales['zscore'] < -1.96]


# In[25]:


#applying dummy
sales['dummy']=sales['res1']        


# In[26]:


b=sales['dummy']
for i in range(0,len(b)):
    if (np.any(sales['zscore'].values[i] > 1.96)):
        sales['dummy'].values[i]=0
    else:
        sales['dummy'].values[i]=1
sales


# In[27]:


sales.head(18)


# In[28]:


x=sales[['Advt','dummy']]
y=sales['Sales']
y


# In[29]:


x


# In[30]:


plt.scatter(sales['res1'],y)
plt.xlabel('res1')
plt.ylabel('sales')
plt.show()


# In[31]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[32]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
y_train


# In[33]:


x_train1=sm.add_constant(x_train)


# In[34]:


from sklearn import linear_model 
model=sm.OLS(y_train,x_train1).fit()
print(model.summary())


# In[35]:


regr=LinearRegression()


# In[36]:


regr.fit(x_train,y_train)


# In[37]:


print('intercept: ',regr.intercept_)


# In[38]:


print('coefficient: ',regr.coef_)


# In[45]:


y_pred=regr.predict(x_test)
y_pred


# In[ ]:


from sklearn import metrics


# In[ ]:


print('mean absolute error :',metrics.mean_absolute_error(y_test,y_pred))


# In[ ]:


print('mean square error: ',metrics.mean_squared_error(y_test,y_pred))
print('root mean square error: ',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# # stepwise

# In[ ]:


import statsmodels.api as sm
from statsmodels.formula.api import ols

model2=ols('y~x',data=sales).fit()
model3=sm.stats.anova_lm(model2)

print(model2.summary())
print(model3)


# # Forward regression

# In[ ]:


pip install mlxtend


# In[ ]:


from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.linear_model import LinearRegression


# In[ ]:


log=LinearRegression()
sfs1=sfs(log,k_features = 2,forward=True,verbose = 2,scoring='neg_mean_squared_error')
sfs1=sfs1.fit(x,y)


# In[ ]:


features_name=list(sfs1.k_feature_names_)
print(features_name)


# # Backward Regression

# In[ ]:


log = LinearRegression()
sfs1=sfs(log,k_features =2,forward =False,verbose =2,scoring = 'neg_mean_squared_error')
sfs1 = sfs1.fit(x,y)


# In[ ]:


features_name=list(sfs1.k_feature_names_)
print(features_name)


# # Homo and heteroscardecity

# In[ ]:


from statsmodels.stats.diagnostic import het_breuschpagan
model=ols('y~x',data=sales).fit()
_,pvalue,_,_= het_breuschpagan(model.resid,model.model.exog)
print(pvalue)

if pvalue>0.05:
    print('it is Heteroscardecity')
else:
    print('it is Hemoscardecity')


# # Chi2_Test

# In[ ]:


from scipy.stats import chi2_contingency
df1=sales['Advt']
df2=sales['PC']

stat,p,dof,expected = chi2_contingency(df1,df2)
alpha=0.05
print('p value is '+str(p))
if p<= alpha:
    print('dependent(reject H0)')
else:
    print('Independent(H0 holds True)')


# In[ ]:


from scipy.stats import chi2_contingency
df1=sales['Advt']
df2=sales['PC']

stat,p,dof,expected = chi2_contingency(df1,df2)

alpha=0.05
print('p value is'+str(p))
print(p)
if p<=alpha:
    print('Dependent(reject H0)')
else:
    print('Independent(H0 holds True)')


# In[ ]:


from scipy.stats import chi2_contingency
df1=sales['Advt']
df2=sales['PC']

stat,p,dof,expected=chi2_contingency(df1,df2)
print('p value is'+str(p))
alpha=0.05
if p<=alpha:
    print('Dependent(reject H0)')
else:
    print('Independent(H0 holds True)')


# In[ ]:




