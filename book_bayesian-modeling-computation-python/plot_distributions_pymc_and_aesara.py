#!/usr/bin/env python
# coding: utf-8

# ## Helpful Links
# 
# https://www.pymc.io/projects/docs/en/latest/learn/core_notebooks/pymc_aesara.html
# 
# https://stackoverflow.com/questions/45464924/python-calculating-pdf-from-a-numpy-array-distribution
# 
# 

# In[1]:


# import aesara
# import aesara.tensor as at
import pymc as pm
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats


# In[2]:


# aesara.__version__


# In[3]:


pm.__version__


# In[4]:


a = np.random.normal(loc=0, scale=1, size=1_000)


# In[5]:


fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(a, color="C0", bins=15)
ax.set(title="Samples from a normal distribution using numpy", ylabel="count");


# In[6]:


y = pm.Normal.dist(mu=0, sigma=2)
y.type


# In[7]:


# aesara.dprint(y)


# In[8]:


pm.draw(y, draws=1_000)[:5]


# In[ ]:





# In[9]:


fig, ax = plt.subplots(figsize=(8,6))
samples = pm.draw(y, draws=10000)
# x = np.linspace(np.min(samples), np.max(samples), 1000)
ax.hist(samples, bins=100);


# In[10]:


samples = pm.draw(y, draws=10000)
kde = scipy.stats.gaussian_kde(np.array(samples))
fig, ax = plt.subplots(figsize=(8,6))
x = np.linspace(np.min(samples), np.max(samples), 10000)
ax.plot(x, kde(x));


# In[ ]:




