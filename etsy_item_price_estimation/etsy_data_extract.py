
# coding: utf-8

# In[7]:


import datetime
import seaborn as sns
import pandas as pd
from etsy_interface import Etsy
import pickle
import time

e = Etsy('zdipkqu8fxsomaoriwj1x98a', 'rxv4sdcs5g') # gotten from signing up at etsy.com/developers


# In[8]:


total_listing_data = []
for i in range(0, 500):
    index = 100*i
    result = e.show_listings('100', str(index))
    total_listing_data.append(result)
    if i in range(9,500,10):
        time.sleep(1)



with open("total_listing_data.txt", "wb") as fp:
    pickle.dump(total_listing_data, fp)


# In[12]:


with open("total_listing_data.txt", "rb") as fp2:
    total_listing_data_loaded = pickle.load(fp2)


# In[34]:





# In[35]:


user_id_dict = {}
for i in range(0, len(total_listing_data_loaded)):
    for j in range(0, len(total_listing_data_loaded[i]['results'])):
        if 'user_id' not in total_listing_data_loaded[i]['results'][j].keys():
            continue
        key = str(total_listing_data_loaded[i]['results'][j]['user_id'])
        if key not in user_id_dict.keys():
            user_id_dict[key] = 1
len(user_id_dict.keys())


# In[47]:


e2 = Etsy('zdipkqu8fxsomaoriwj1x98a', 'rxv4sdcs5g')
user_info = {}
counter = 0
for single_key in user_id_dict.keys():
    key_int=int(single_key)
    element=e2.get_user_info(key_int)
    user_info[single_key] = element
    counter += 1
    if counter % 10 == 0:
        time.sleep(1)
    if counter % 100 == 0:
        filename = 'user_info_' + str(counter) + '.txt'
        with open(filename, "wb") as fp3:
            pickle.dump(user_info, fp3)


# In[48]:


with open('user_info_final', "wb") as fp3:
            pickle.dump(user_info, fp3)


# In[50]:


with open('user_info_final', "rb") as fp4:
    user_info=pickle.load(fp4)


# In[51]:


print(len(user_info.keys()))

