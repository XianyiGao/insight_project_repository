"""
This module can be used to load and process the raw data extracted
from Etsy API. The organized data is then pushed to PostgreSQL database.
"""

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

def find_catgory(index, cat_path):
    """
    Finds the corresponding category string from the index
    """
    category = 'none'
    if index < len(cat_path):
        category = cat_path[index]
    return category

def text_corrected(text_str):
    """
    Check if the text value is available
    """
    if text_str is None:
        fixed_value = 'none'
    else:
        fixed_value = text_str
    return fixed_value

def get_indicator_value(indicator_value):
    """
    Gives indicator value of 0 or 1 based on whether the data is missing.
    """
    result = 1
    if indicator_value is None:
        result = 0
    return result

def numerical_corrected(numerical_value, factor):
    """
    Gives corrected numerical value for complete or missing data.
    """
    if numerical_value is None:
        result = 0
    else:
        result = float(numerical_value) * factor
    return result

def style_corrected(style_value, style_type):
    """
    Gives corrected value for item style, also handles missing data.
    """
    if style_value is None:
        result = 'none'
    elif len(style_value) == 1:
        if style_type == 1:
            result = style_value[0]
        else:
            result = 0
    else:
        if style_type == 1:
            result = style_value[0]
        else:
            result = style_value[1]
    return result

def bool_corrected(bool_value):
    """
    Gives corresponding numerical value for boolean features.
    """
    if bool_value:
        result = 1
    else:
        result = 0
    return result

def feature_extract(listing_data, user_info_dictionary):
    """
    Extracts features from raw data pulled from Etsy API.
    Results are combined and returned as a dataframe.

    This function uses a lot of variables because each one
    is a different feature that needs to be handled
    differently. As there are many types of features
    for this problem, this function is also very lengthy.
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
    # extract and organize features, prepare for later machine learning model
    for page_element in listing_data:
        results = page_element['results']
        for each_item in results:
            # First get rid of invalid postings and item price that is
            # greater than $1000. Most of postings extracted
            # has price less than $1000. This project only considers
            # items with price less than or equal $1000.
            if (('category_id' not in each_item.keys())
                    or ('price' not in each_item.keys())
                    or (each_item['currency_code'] != 'USD')
                    or ('category_path' not in each_item.keys())
                    or (float(each_item['price']) > 1000)):
                invalide_data_count += 1
                continue
            listing_id.append(each_item['listing_id'])
            user_id.append(each_item['user_id'])
            title_list.append(each_item['title'])
            description_list.append(each_item['description'])
            price_list.append(float(each_item['price']))
            quantity_list.append(each_item['quantity'])
            tags_list.append(each_item['tags'])
            # Handle the three layers of category.
            # Since there are only three layers at most,
            # I simply used category_id1, category_id2,
            # category_id3 instead of an array/list.
            category_path = each_item['category_path']
            category_id1.append(find_catgory(0, category_path))
            category_id2.append(find_catgory(1, category_path))
            category_id3.append(find_catgory(2, category_path))
            # material of the item
            if ('materials' not in each_item.keys()
                    or len(each_item['materials']) <= 0):
                material_list.append('none')
            else:
                # get the first one if there are more than one.
                # convert to lower case letters to avoid confusion.
                material_list.append(each_item['materials'][0].lower())

            who_made_list.append(text_corrected(each_item['who_made']))
            when_made_list.append(text_corrected(each_item['when_made']))
            scale_factor = 1.0   # this is used for later unit changing
            item_weight_indicator.append(
                get_indicator_value(each_item['item_weight']))
            item_weight_list.append(
                numerical_corrected(each_item['item_weight'], scale_factor))
            # note that different listings may have different dimension unit
            if each_item['item_dimensions_unit'] == 'mm':
                # convert mm to in
                scale_factor = 0.0393701
            # calculate and insert iten length, width, and height
            # pay special attention to the missing data
            item_length_indicator.append(
                get_indicator_value(each_item['item_length']))
            item_length_list.append(
                numerical_corrected(each_item['item_length'], scale_factor))
            item_width_indicator.append(
                get_indicator_value(each_item['item_width']))
            item_width_list.append(
                numerical_corrected(each_item['item_width'], scale_factor))
            item_height_indicator.append(
                get_indicator_value(each_item['item_height']))
            item_height_list.append(
                numerical_corrected(each_item['item_height'], scale_factor))
            # extract style and other features
            style1_list.append(style_corrected(each_item['style'], 1))
            style2_list.append(style_corrected(each_item['style'], 2))
            non_taxable_list.append(bool_corrected(each_item['non_taxable']))
            is_customizable_list.append(
                bool_corrected(each_item['is_customizable']))
            is_digital_download_list.append(
                bool_corrected(each_item['is_digital']))
    # linking the seller info:
    for i in user_id:
        user_id_string = str(i)
        user_info_result = user_info_dictionary[user_id_string]['results'][0]
        user_feedback_info = user_info_result['feedback_info']
        feedback_count_list.append(user_feedback_info['count'])
        seller_score_indicator.append(
            get_indicator_value(user_feedback_info['score']))
        seller_score_list.append(text_corrected(user_feedback_info['score']))
    # putting everything in a dataframe
    data = collections.OrderedDict([
        ('title', title_list),
        ('category_id1', category_id1),
        ('category_id2', category_id2),
        ('category_id3', category_id3),
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
        ('price', price_list)])
    feature_df = pd.DataFrame.from_dict(data)
    return feature_df

def create_psql(etsy_data_df, database_setting):
    """
    Create postgreSQL db, using the processed dataframe
    """
    engine = sqlalchemy.create_engine(
        'postgresql://{}:{}@{}:{}/{}'.format(database_setting['username'],
                                             database_setting['password'],
                                             database_setting['host'],
                                             database_setting['port'],
                                             database_setting['db_name'])
    )
    print(engine.url)
    if not sqlalchemy_utils.database_exists(engine.url):
        sqlalchemy_utils.create_database(engine.url)
    print('connected to database = ', sqlalchemy_utils.database_exists(engine.url))
    print('loading data to sql, please wait ...')
    etsy_data_df.to_sql('etsy_item_table', engine, if_exists='replace')
    print('completed!')


if __name__ == '__main__':
    DB_DIC = {
        'username' : 'postgres',
        'password' : 'thisisforetsyapp',
        'host' : 'localhost',
        'port' : '5432',   # default port that postgres listens on
        'db_name' : 'etsy_db'
    }
    LISTING_DATA_FILENAME = 'total_listing_data'
    USER_INFO_FILENAME = 'user_info_complete'
    total_listing_data_loaded = load_binary(LISTING_DATA_FILENAME)
    info_dictionary = load_binary(USER_INFO_FILENAME)
    feature_dataframe = feature_extract(total_listing_data_loaded, info_dictionary)
    create_psql(feature_dataframe, DB_DIC)
