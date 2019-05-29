#!/usr/bin/env python
# coding: utf-8

# # Example

# In[2]:


from VoiceClassifier import train,predict_names


# In[3]:


train = train(['anuragD30.raw','anupamD30.raw','animeshD30.raw','amanD30.raw','deepakbD30.raw'],
             ['anu','anup','ani','aman','dee'],epochs = 20)


# In[4]:


print(predict_names('x.raw',train))
print(predict_names('y.raw',train))
print(predict_names('an.raw',train))
print(predict_names('am.raw',train))
print(predict_names('dee.raw',train))


# In[ ]:




