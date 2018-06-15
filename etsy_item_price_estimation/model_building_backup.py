
# coding: utf-8

# In[1]:


import pickle

with open("total_listing_data.txt", "rb") as fp:
    total_listing_data_loaded = pickle.load(fp)
    
with open("user_info_complete.txt", "rb") as fp2:
    user_info_dictionary = pickle.load(fp2)


# In[2]:


import pandas as pd
import numpy as np
from collections import OrderedDict



# In[3]:





# In[4]:





# In[3]:


#print(total_listing_data_loaded[0]['params'])
#print(total_listing_data_loaded[0]['type'])
#print(len(total_listing_data_loaded[0]['results']))
print(total_listing_data_loaded[0]['results'][0].keys())
print(type(total_listing_data_loaded[0]['results'][0]['used_manufacturer']))
print(total_listing_data_loaded[0]['results'][0]['used_manufacturer'])


# In[54]:


listing_id = []
user_id = []
category_id = [] # this is useless
category_id1 = []
category_id2 = []
category_id3 = []
title_list = []
description_list = []
price_list = []
quantity_list = []
tags_list = []
material_list = []
item_views_list = []
item_fav_list = []
who_made_list = []
when_made_list = []
feedback_count_list = []
seller_score_list = []

item_weight_list = []
item_length_list = []
item_width_list = []
item_height_list = []

item_weight_indicator = []
item_length_indicator = []
item_width_indicator = []
item_height_indicator = []

style1_list = []
style2_list = []

non_taxable_list = []
is_customizable_list = []
is_digital_download_list = []



#sku is probably not helpful at all
no_user_id_count = 0
max1=0
max2=0
category_NameToNum = []
category_NumToName = []
categorySet1 = {}
categorySet2 = {}
categorySet3 = {}
categorySet4 = {}
categorySet5 = {}
categorySet6 = {}
category_NameToNum.append(categorySet1)
category_NameToNum.append(categorySet2)
category_NameToNum.append(categorySet3)

category_NumToName.append(categorySet4)
category_NumToName.append(categorySet5)
category_NumToName.append(categorySet6)

style1_NameToNum = {}
style1_NumToName = {}
style1_counter = 0

style2_NameToNum = {}
style2_NumToName = {}
style2_counter = 0

material_NameToNum = {}
material_NumToName = {}
material_counter = 0

who_made_NameToNum = {}
who_made_NumToName = {}
who_made_counter = 0

when_made_NameToNum = {}
when_made_NumToName = {}
when_made_counter = 0

counters = [0,0,0]
for i in range(0,len(total_listing_data_loaded)):
    results = total_listing_data_loaded[i]['results']
    count = total_listing_data_loaded[i]['count']
    params = total_listing_data_loaded[i]['params']
    responseType = total_listing_data_loaded[i]['type']
    pagination = total_listing_data_loaded[i]['pagination']
    for j in range(0, len(results)):
        if ('category_id' not in results[j].keys()) or ('price' not in results[j].keys())         or (results[j]['currency_code']!='USD') or ('category_path' not in results[j].keys()):
            no_user_id_count += 1
            #print(no_user_id_count, i, j, results[j]['currency_code'])
            continue
        
        if len(results[j]['category_path'])>max1:
            max1 = len(results[j]['category_path'])
        if results[j]['style'] != None and len(results[j]['style'])>max2:
            max2 = len(results[j]['style'])
        listing_id.append(results[j]['listing_id'])
        user_id.append(results[j]['user_id'])
        category_id.append(results[j]['category_id'])
        title_list.append(results[j]['title'])
        description_list.append(results[j]['description'])
        price_list.append(float(results[j]['price']))
        quantity_list.append(results[j]['quantity'])
        tags_list.append(results[j]['tags'])
        for k in range(0, 3):
            if k<len(results[j]['category_path']):
                category_index = 0
                if results[j]['category_path'][k] not in category_NameToNum[k].keys():
                    counters[k] += 1
                    category_index = counters[k]
                    #print(k)
                    category_NameToNum[k][results[j]['category_path'][k]] = counters[k]
                    category_NumToName[k][str(counters[k])] = results[j]['category_path'][k]
                else:
                    category_index = category_NameToNum[k][results[j]['category_path'][k]]

                if k == 0:
                    category_id1.append(category_index)
                elif k == 1:
                    category_id2.append(category_index)
                else:
                    category_id3.append(category_index)  
            else:
                if k == 0:
                    category_id1.append(0)
                elif k == 1:
                    category_id2.append(0)
                else:
                    category_id3.append(0)
        
        if 'materials' not in results[j].keys():
            material_list.append(0)
        elif len(results[j]['materials'])<=0:
            material_list.append(0)
        else:
            if results[j]['materials'][0].lower() not in material_NameToNum.keys():
                material_counter += 1
                material_list.append(material_counter)
                material_NameToNum[results[j]['materials'][0].lower()] = material_counter
                material_NumToName[str(material_counter)] = results[j]['materials'][0].lower()
            else:
                material_list.append(material_NameToNum[results[j]['materials'][0].lower()])
            
        #print(results[j]['materials'])
        item_views_list.append(results[j]['views'])
        item_fav_list.append(results[j]['num_favorers'])
        
        if results[j]['who_made'] == None:
            who_made_list.append(0)
        elif results[j]['who_made'] not in who_made_NameToNum.keys():
            who_made_counter = who_made_counter + 1
            #print(who_made_counter)
            who_made_list.append(who_made_counter)
            who_made_NameToNum[results[j]['who_made']] = who_made_counter
            who_made_NumToName[str(who_made_counter)] = results[j]['who_made']
        else:
            who_made_list.append(who_made_NameToNum[results[j]['who_made']])
        
        if results[j]['when_made'] == None:
            when_made_list.append(0)
        elif results[j]['when_made'] not in when_made_NameToNum.keys():
            when_made_counter = when_made_counter + 1
            when_made_list.append(when_made_counter)
            when_made_NameToNum[results[j]['when_made']] = when_made_counter
            when_made_NumToName[str(when_made_counter)] = results[j]['when_made']
        else:
            when_made_list.append(when_made_NameToNum[results[j]['when_made']])
        
        if results[j]['item_weight'] == None:
            item_weight_indicator.append(0)
            item_weight_list.append(0)
        else:
            item_weight_indicator.append(1)
            item_weight_list.append(float(results[j]['item_weight']))
        
        scale_factor = 1
        if results[j]['item_dimensions_unit'] == 'mm':
            scale_factor = 0.0393701
            
        if results[j]['item_length'] == None:
            item_length_indicator.append(0)
            item_length_list.append(0)
        else:
            item_length_indicator.append(1)
            #print(results[j]['item_length'])
            #print(type(results[j]['item_length']))
            item_length_list.append(float(results[j]['item_length'])*scale_factor)
        
        if results[j]['item_width'] == None:
            item_width_list.append(0)
            item_width_indicator.append(0)
        else:
            item_width_list.append(float(results[j]['item_width'])*scale_factor)
            item_width_indicator.append(1)
            
        if results[j]['item_height'] == None:
            item_height_indicator.append(0)
            item_height_list.append(0)
        else:
            item_height_indicator.append(1)
            item_height_list.append(float(results[j]['item_height'])*scale_factor)
        
        if results[j]['style'] == None:
            style1_list.append(0)
            style2_list.append(0)
        elif len(results[j]['style']) == 1:
            style2_list.append(0)
            if results[j]['style'][0] not in style1_NameToNum.keys():
                style1_counter += 1
                style1_NameToNum[results[j]['style'][0]] = style1_counter
                style1_NumToName[int(style1_counter)] = results[j]['style'][0]
                style1_list.append(style1_counter)
            else:
                style1_list.append(style1_NameToNum[results[j]['style'][0]]) 
        else:
            if results[j]['style'][0] not in style1_NameToNum.keys():
                style1_counter += 1
                style1_NameToNum[results[j]['style'][0]] = style1_counter
                style1_NumToName[int(style1_counter)] = results[j]['style'][0]
                style1_list.append(style1_counter)
            else:
                style1_list.append(style1_NameToNum[results[j]['style'][0]])
            
            if results[j]['style'][1] not in style2_NameToNum.keys():
                style2_counter += 1
                style2_NameToNum[results[j]['style'][1]] = style2_counter
                style2_NumToName[int(style2_counter)] = results[j]['style'][1]
                style2_list.append(style2_counter)
            else:
                style2_list.append(style2_NameToNum[results[j]['style'][1]])
            

        if results[j]['non_taxable']:
            non_taxable_list.append(1)
        else:
            non_taxable_list.append(0)
        if results[j]['is_customizable']:
            is_customizable_list.append(1)
        else:
            is_customizable_list.append(0)
        
        if results[j]['is_digital']:
            is_digital_download_list.append(1)
        else:
            is_digital_download_list.append(0)
        print(results[j]['quantity'])
        #print(item_weight_list[len(item_weight_list)-1], item_length_list[len(item_length_list)-1], item_width_list[len(item_width_list)-1], item_height_list[len(item_height_list)-1])
        
print(no_user_id_count)
print(max2)


# In[55]:


#linking the seller info:
for i in user_id:
    feedback_count_list.append(user_info_dictionary[str(i)]['results'][0]['feedback_info']['count'])
    if user_info_dictionary[str(i)]['results'][0]['feedback_info']['score'] == None:
        seller_score_list.append(0)
    else:
        seller_score_list.append(user_info_dictionary[str(i)]['results'][0]['feedback_info']['score'])
    #print(feedback_count_list[len(feedback_count_list)-1], seller_score_list[len(seller_score_list)-1])



# In[ ]:


#title_list = []
#description_list = []
#price_list = []
#tags_list = []
#item_views_list = []
#item_fav_list = []


data = OrderedDict([ ('category_id1', category_id1),
          ('category_id2', category_id2),
          ('category_id3',  category_id3),
          ('quantity', quantity_list),
          ('material', material_list),
          ('who_made', who_made_list),
          ('when_made', when_made_list),
          ('item_weight', item_weight_list),
          ('item_weight_indicator', item_weight_indicator),
          ('item_length', item_length_list),
          ('item_length_indicator', item_length_indicator),
          ('item_width', item_width_list),
          ('item_width_indicator', item_width_indicator),
          ('item_height', item_height_list),
          ('item_height_indicator', item_height_indicator),
          ('style_1', style1_list),
          ('style_2', style2_list),
          ('non_taxable', non_taxable_list),
          ('is_customizable', is_customizable_list),
          ('is_digital_download', is_digital_download_list),
          ('seller_feedback_count', feedback_count_list),
          ('seller_score', seller_score_list),
          ('price', price_list)
                   ])
df = pd.DataFrame.from_dict(data)


# In[84]:


import matplotlib.pyplot as plt

#print(df)
print(df.iloc[0])
#print(title_list[1:10])
#plt.scatter(df['category_id1'], df['price'], marker='o')
#plt.show()


# In[87]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

df = df.sample(frac=1).reset_index(drop=True)
train_df_len = int(len(df)/5*4)
train_df = df.iloc[0:train_df_len]
test_df = df.iloc[train_df_len:len(df)]

train_x = train_df.values

RF_regr = RandomForestRegressor(random_state=0)
RF_regr.fit(X, y)
print(train_x[0,:])


# In[77]:


regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(X, y)
print(regr.feature_importances_)
print(regr.predict([[0, 0, 0, 0]]))
print(df.iloc[0])

