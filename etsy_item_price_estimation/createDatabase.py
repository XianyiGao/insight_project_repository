
# coding: utf-8

# In[27]:


import pickle
import pandas as pd
import numpy as np
from collections import OrderedDict
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
import pandas as pd


def loadBinary():
    with open("total_listing_data", "rb") as fp:
        listing_data = pickle.load(fp)

    with open("user_info_complete", "rb") as fp2:
        user_info = pickle.load(fp2)
        
    return listing_data, user_info



def feature_extract(total_listing_data_loaded, user_info_dictionary):
    # initiating features
    listing_id = []
    user_id    = []
    # three layers of item category
    category_id1 = []
    category_id2 = []
    category_id3 = []
    title_list   = []
    description_list = []
    price_list       = []
    quantity_list    = []
    tags_list        = []
    material_list    = []
    who_made_list    = []
    when_made_list   = []
    feedback_count_list = []
    seller_score_list   = []
    item_weight_list    = []
    item_length_list    = []
    item_width_list     = []
    item_height_list    = []
    style1_list = []
    style2_list = []
    non_taxable_list         = []
    is_customizable_list     = []
    is_digital_download_list = []
    # following two features are only available after posting, so not considered
    # item_views_list = []
    # item_fav_list = []
    ###############################################
    # indicators are those features that 'tell' machine learning model about whether
    # the corresponding value for the corresponding feature is provided or missing.
    # Indicators are only needed for continuous values. ML generally picks missing
    # data automatically for categorical values with a separate label.
    seller_score_indicator = []
    item_weight_indicator  = []
    item_length_indicator  = []
    item_width_indicator   = []
    item_height_indicator  = []


    # counting invalid data (e.g. price is not USD)
    invalide_data_count = 0
    max1=0
    max2=0

    # extract and organize features, prepare for later machine learning model
    for i in range(0,len(total_listing_data_loaded)):
        results      = total_listing_data_loaded[i]['results']
        count        = total_listing_data_loaded[i]['count']
        params       = total_listing_data_loaded[i]['params']
        responseType = total_listing_data_loaded[i]['type']
        pagination   = total_listing_data_loaded[i]['pagination']
        for j in range(0, len(results)):
            # first get rid of invalid postings
            if ('category_id' not in results[j].keys()) or ('price' not in results[j].keys())             or (results[j]['currency_code']!='USD') or ('category_path' not in results[j].keys())             or float(results[j]['price'])>1000:
                invalide_data_count += 1
                continue

            if len(results[j]['category_path'])>max1:
                max1 = len(results[j]['category_path'])
            if results[j]['style'] != None and len(results[j]['style'])>max2:
                max2 = len(results[j]['style'])
            listing_id.append(results[j]['listing_id'])
            user_id.append(results[j]['user_id'])
            title_list.append(results[j]['title'])
            description_list.append(results[j]['description'])
            price_list.append(float(results[j]['price']))
            quantity_list.append(results[j]['quantity'])
            tags_list.append(results[j]['tags'])
            # handle the three layers of category
            for k in range(0, 3):
                if k<len(results[j]['category_path']):
                    if k == 0:
                        category_id1.append(results[j]['category_path'][k])
                    elif k == 1:
                        category_id2.append(results[j]['category_path'][k])
                    else:
                        category_id3.append(results[j]['category_path'][k])
                else:
                    if k == 0:
                        category_id1.append('none')
                    elif k == 1:
                        category_id2.append('none')
                    else:
                        category_id3.append('none')

            if 'materials' not in results[j].keys():
                material_list.append('none')
            elif len(results[j]['materials'])<=0:
                material_list.append('none')
            else:
                material_list.append(results[j]['materials'][0].lower())


            #item_views_list.append(results[j]['views'])
            #item_fav_list.append(results[j]['num_favorers'])

            if results[j]['who_made'] == None:
                who_made_list.append('none')
            else:
                who_made_list.append(results[j]['who_made'])

            if results[j]['when_made'] == None:
                when_made_list.append('none')
            else:
                when_made_list.append(results[j]['when_made'])

            if results[j]['item_weight'] == None:
                item_weight_indicator.append(0)
                item_weight_list.append(0)
            else:
                item_weight_indicator.append(1)
                item_weight_list.append(float(results[j]['item_weight']))

            scale_factor = 1.0
            # note that different listings may have different dimension unit
            if results[j]['item_dimensions_unit'] == 'mm':
                # convert mm to in
                scale_factor = 0.0393701

            if results[j]['item_length'] == None:
                item_length_indicator.append(0)
                item_length_list.append(0)
            else:
                item_length_indicator.append(1)
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
                style1_list.append('none')
                style2_list.append('none')
            elif len(results[j]['style']) == 1:
                style2_list.append(0)
                style1_list.append(results[j]['style'][0]) 
            else:
                style1_list.append(results[j]['style'][0])
                style2_list.append(results[j]['style'][1])

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

    #linking the seller info:
    for i in user_id:
        feedback_count_list.append(user_info_dictionary[str(i)]['results'][0]['feedback_info']['count'])
        if user_info_dictionary[str(i)]['results'][0]['feedback_info']['score'] == None:
            seller_score_list.append(0)
            seller_score_indicator.append(0)
        else:
            seller_score_list.append(user_info_dictionary[str(i)]['results'][0]['feedback_info']['score'])
            seller_score_indicator.append(1)
    
    # putting everything in a dataframe
    data = OrderedDict([ ('title', title_list),
          ('category_id1', category_id1),
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
          ('seller_score_indicator', seller_score_indicator),
          ('price', price_list)
                   ])
    df = pd.DataFrame.from_dict(data)
    return df

def createPSQL(df):
    username = 'postgres'
    password = 'thisisforetsyapp'     # change this
    host     = 'localhost'
    port     = '5432'            # default port that postgres listens on
    db_name  = 'etsy_db'
    
    engine = create_engine( 'postgresql://{}:{}@{}:{}/{}'.format(username, password, host, port, db_name) )
    print(engine.url)
    if not database_exists(engine.url):
        create_database(engine.url)
    print('connected to database = ', database_exists(engine.url))
    print('loading data to sql, please wait ...')
    df.to_sql('etsy_item_table', engine, if_exists='replace')
    print('completed!')


# In[28]:


if __name__ == '__main__':
    total_listing_data_loaded, user_info_dictionary = loadBinary()
    df = feature_extract(total_listing_data_loaded, user_info_dictionary)
    createPSQL(df)

