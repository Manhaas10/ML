#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('Downloads/concrete_data.csv')
x=data.iloc[:,:-1]
y=data.iloc[:,-1]
print(x)
for i in range(x.shape[1]):
    plt.scatter(x.iloc[:,i],y)
    plt.show()


# In[2]:


df=pd.DataFrame(data)
df.insert(0,'intercept',1)
xt=df[['intercept','cement','blast_furnace_slag','fly_ash','water','superplasticizer','coarse_aggregate','fine_aggregate ','age']].values
yt=df['concrete_compressive_strength'].values.reshape(-1,1)
# values
alpha=0.0000001
it=10000
g=[]
# cost_func
def cost_func(x,y,tita):
    m=len(y)
    pred_y=x.dot(tita)
    cost=(1/2*m)*np.mean(np.square(y-pred_y))
    return cost
# gradient_d
def gradient(x,y,alpha,it,g):
#     print(x.shape[1])
    tita=np.zeros((x.shape[1],1))
    m=len(y)
    costs=np.zeros(it)
    for i in range(it):
        pred_y=x.dot(tita)
        tita=tita-(1/m)*alpha*(x.T.dot(pred_y-y))
        g.append(tita[1])
        costs[i]=cost_func(x,y,tita)
    return tita,costs
# plot training_curve
tita,costs=gradient(xt,yt,alpha,it,g)
plt.plot(range(it),costs)
plt.show()


# In[3]:


plt.plot(g,costs)
plt.show()


# In[4]:


def mse(x,y,tita):
    yp=x.dot(tita)
    ans=np.mean(np.square(yp-y))
    ans2=np.mean(abs(yp-y))
    return ans,ans2
ans,ans2=mse(xt,yt,tita)
print('mean squared error:',ans)
print('mean absolute error:',ans2)

