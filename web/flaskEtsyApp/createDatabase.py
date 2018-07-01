import pickle
import collections
import sqlalchemy
import sqlalchemy_utils
import pandas as pd


def load_binary(data_filename):
    """
    Loads data file from previously saved file. The data
    include all information about the Etsy items or user info.
    """
    with open(data_filename, 'rb') as data_file:
        loaded_data = pickle.load(data_file)
    return loaded_data

def feature_extract(total_listing_data_loaded, user_info_dictionary):
    """
    Extracts features from raw data pulled from Etsy API.
    Results are combined and returned as a dataframe.
    """

    listing_id = []
    user_id = []
    # three hierarchical layers of item categories
    category_id1 = []
    category_id2 = []
    category_id3 = []
    # features about posted item
    title_list = []
    description_list = []
    material_list = []
    who_made_list = []
    when_made_list = []
    item_weight_list = []
    item_length_list = []
    item_width_list = []
    item_height_list = []
    style1_list = []
    style2_list = []
    tags_list = []
    quantity_list = []
    # Following indicators are those features that 'tell' machine learning
    # model about whether the corresponding value for the corresponding
    # feature is provided or missing. Indicators are only needed for
    # continuous values. ML generally picks missing data automatically for
    # categorical values with a separate label.
    seller_score_indicator = []
    item_weight_indicator = []
    item_length_indicator = []
    item_width_indicator = []
    item_height_indicator = []
    # features about seller
    feedback_count_list = []
    seller_score_list = []
    # other features
    non_taxable_list = []
    is_customizable_list = []
    is_digital_download_list = []
    # listed price of items
    price_list = []
    
    # counting invalid data (e.g. price is not USD)
    invalide_data_count = 0
    max1=0
    max2=0

    # extract and organize features, prepare for later machine learning model
    for page_element in total_listing_data_loaded:
        results      = page_element['results']
        count        = page_element['count']
        params       = page_element['params']
        responseType = page_element['type']
        pagination   = page_element['pagination']
        for each_item in results:
            # first get rid of invalid postings, item price that is 
            # greater than $1000. About 99% of postings extracted
            # has price less than $1000. This project only considers
            # items with price less than or equal $1000.
            if (('category_id' not in each_item.keys()) 
                or ('price' not in each_item.keys())
                or (each_item['currency_code']!='USD')
                or ('category_path' not in each_item.keys())
                or (float(each_item['price'])>1000)):
                invalide_data_count += 1
                continue

            if len(each_item['category_path'])>max1:
                max1 = len(each_item['category_path'])
            if each_item['style'] != None and len(each_item['style'])>max2:
                max2 = len(each_item['style'])
            listing_id.append(each_item['listing_id'])
            user_id.append(each_item['user_id'])
            title_list.append(each_item['title'])
            description_list.append(each_item['description'])
            price_list.append(float(each_item['price']))
            quantity_list.append(each_item['quantity'])
            tags_list.append(each_item['tags'])
            # handle the three layers of category
            for k in range(0, 3):
                if k<len(each_item['category_path']):
                    if k == 0:
                        category_id1.append(each_item['category_path'][k])
                    elif k == 1:
                        category_id2.append(each_item['category_path'][k])
                    else:
                        category_id3.append(each_item['category_path'][k])
                else:
                    if k == 0:
                        category_id1.append('none')
                    elif k == 1:
                        category_id2.append('none')
                    else:
                        category_id3.append('none')

            if 'materials' not in each_item.keys():
                material_list.append('none')
            elif len(each_item['materials'])<=0:
                material_list.append('none')
            else:
                material_list.append(each_item['materials'][0].lower())


            #item_views_list.append(results[j]['views'])
            #item_fav_list.append(results[j]['num_favorers'])

            if each_item['who_made'] == None:
                who_made_list.append('none')
            else:
                who_made_list.append(each_item['who_made'])

            if each_item['when_made'] == None:
                when_made_list.append('none')
            else:
                when_made_list.append(each_item['when_made'])

            if each_item['item_weight'] == None:
                item_weight_indicator.append(0)
                item_weight_list.append(0)
            else:
                item_weight_indicator.append(1)
                item_weight_list.append(float(each_item['item_weight']))

            scale_factor = 1.0
            # note that different listings may have different dimension unit
            if each_item['item_dimensions_unit'] == 'mm':
                # convert mm to in
                scale_factor = 0.0393701

            if each_item['item_length'] == None:
                item_length_indicator.append(0)
                item_length_list.append(0)
            else:
                item_length_indicator.append(1)
                item_length_list.append(float(each_item['item_length'])*scale_factor)

            if each_item['item_width'] == None:
                item_width_list.append(0)
                item_width_indicator.append(0)
            else:
                item_width_list.append(float(each_item['item_width'])*scale_factor)
                item_width_indicator.append(1)

            if each_item['item_height'] == None:
                item_height_indicator.append(0)
                item_height_list.append(0)
            else:
                item_height_indicator.append(1)
                item_height_list.append(float(each_item['item_height'])*scale_factor)

            if each_item['style'] == None:
                style1_list.append('none')
                style2_list.append('none')
            elif len(each_item['style']) == 1:
                style2_list.append(0)
                style1_list.append(each_item['style'][0]) 
            else:
                style1_list.append(each_item['style'][0])
                style2_list.append(each_item['style'][1])

            if each_item['non_taxable']:
                non_taxable_list.append(1)
            else:
                non_taxable_list.append(0)
            if each_item['is_customizable']:
                is_customizable_list.append(1)
            else:
                is_customizable_list.append(0)

            if each_item['is_digital']:
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
    data = collections.OrderedDict([ ('title', title_list),
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

def createPSQL(df, username, password, host, port, db_name):
    """
    Create postgreSQL db, using the processed dataframe
    """
    
    engine = sqlalchemy.create_engine(
        'postgresql://{}:{}@{}:{}/{}'.format(username, password,
                                             host, port, db_name)
        )
    print(engine.url)
    if not sqlalchemy_utils.database_exists(engine.url):
        sqlalchemy_utils.create_database(engine.url)
    print('connected to database = ', sqlalchemy_utils.database_exists(engine.url))
    print('loading data to sql, please wait ...')
    df.to_sql('etsy_item_table', engine, if_exists='replace')
    print('completed!')


if __name__ == '__main__':
    db_username = 'postgres'
    db_password = 'thisisforetsyapp'
    db_host = 'localhost'
    db_port = '5432'   # default port that postgres listens on
    db_etsy_name = 'etsy_db'

    listing_data_filename = 'total_listing_data'
    total_listing_data_loaded = load_binary(listing_data_filename)
    user_info_filename = 'user_info_complete'
    user_info_dictionary = load_binary(user_info_filename)
    df = feature_extract(total_listing_data_loaded, user_info_dictionary)
    createPSQL(df, db_username, db_password, db_host, db_port, db_etsy_name)

