"""
This python module controls the backend of the html views and provide model
loading, prediction, and other processing.
"""


import io
import os
import json
import pickle
import flask
import psycopg2

import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from django.utils.safestring import mark_safe

from flask_etsy_app import app
from flask_etsy_app import tfidf_model
from flask_etsy_app import feature_engineering


# Define global variables: Although most of the times global variables
# are not recommendated for python scripts to prevent accidental misuse,
# following ones are important for pre-loading the machine learning model
# so that it does not block the loading of the webpage each time users
# visiting the page. In other words, these variables need to stay active
# and be ready to use anytime the hosting of the page is available.
APP_ROOT = os.path.dirname(os.path.abspath(__file__))  # refers to application top
APP_STATIC = os.path.join(APP_ROOT, 'static')  # define static path
# load pretrained machine learing model
with open(os.path.join(APP_STATIC, 'machine_learning_model'), 'rb') as model_file:
    ML_MODEL = pickle.load(model_file)
# load title list and obtain tokenized words
with open(os.path.join(APP_STATIC, 'title_list'), 'rb') as title_file:
    title_list = pickle.load(title_file)
TITLE_TOKENIZED, TFIDF_VECTORIZER = tfidf_model.tfidf_transform(title_list, 500)
# load feature names and labels
PATH_FEATURE = os.path.join(APP_STATIC, 'feature_dictionary')
FEATURE_ENG_INST = feature_engineering.FeatureLabels(PATH_FEATURE)
CATEGORY1_FEATURES = FEATURE_ENG_INST.get_cat1_features()
CATEGORY2_FEATURES = FEATURE_ENG_INST.get_cat2_features()
CATEGORY3_FEATURES = FEATURE_ENG_INST.get_cat3_features()
MATERIAL_FEATURES = FEATURE_ENG_INST.get_material_features()
WHO_MADE_FEATURES = FEATURE_ENG_INST.get_who_features()
WHEN_MADE_FEATURES = FEATURE_ENG_INST.get_when_features()
STYLE1_FEATURES = FEATURE_ENG_INST.get_style1_features()
STYLE2_FEATURES = FEATURE_ENG_INST.get_style2_features()
FEATURE_LABELS = FEATURE_ENG_INST.get_feature_labels()
# predicted value list: this is for image passing
preds = []


# output page handling
@app.route('/output')
def etsy_output():
    """
    Obtain arguments from input page and use machine learning model to
    predict the item price. The result is shown in the output page.
    """
    title = flask.request.args.get('title_description')
    category1 = flask.request.args.get('category1')
    category2 = flask.request.args.get('category2')
    category3 = flask.request.args.get('category3')
    quantity = flask.request.args.get('quantity')
    material = flask.request.args.get('material')
    who_made = flask.request.args.get('who_made')
    when_made = flask.request.args.get('when_made')
    item_weight = flask.request.args.get('item_weight')
    item_length = flask.request.args.get('item_length')
    item_width = flask.request.args.get('item_width')
    item_height = flask.request.args.get('item_height')
    style1 = flask.request.args.get('style1')
    style2 = 'none'
    is_taxable = flask.request.args.get('is_taxable')
    is_customizable = flask.request.args.get('is_customizable')
    is_digital_downloadable = flask.request.args.get('is_digital_downloadable')
    feedback_count = flask.request.args.get('feedback_count')
    seller_score = flask.request.args.get('seller_score')

    # direct to error page if there is no title or no category 1
    if title is None or len(title) == 0 or category1 == 'none':
        return flask.render_template('error.html')
    # convert the title to feature vector and include to feature_values
    title_splitted = TFIDF_VECTORIZER.transform([title])
    feature_values = title_splitted.toarray()[0].tolist()
    # check values and append values from other features.
    # note that different features need different checking methods
    feature_values.append(int_value_corrected(quantity))
    feature_values.append(float_value_corrected(item_weight))
    feature_values.append(float_indicator_corrected(item_weight))
    feature_values.append(float_value_corrected(item_length))
    feature_values.append(float_indicator_corrected(item_length))
    feature_values.append(float_value_corrected(item_width))
    feature_values.append(float_indicator_corrected(item_width))
    feature_values.append(float_value_corrected(item_height))
    feature_values.append(float_indicator_corrected(item_height))
    feature_values.append(string_corrected(is_taxable, 0))
    feature_values.append(string_corrected(is_customizable, 1))
    feature_values.append(string_corrected(is_digital_downloadable, 1))
    feature_values.append(int_value_corrected(feedback_count))
    seller_score_fixed = int_value_corrected(seller_score)
    feature_values.append(seller_score_fixed)
    # following appends indicator for seller_score
    if seller_score_fixed == 0:
        feature_values.append(0)
    else:
        feature_values.append(1)
    # add one-hot features
    feature_values = one_hot_corrected(feature_values, CATEGORY1_FEATURES, category1, 12)
    feature_values = one_hot_corrected(feature_values, CATEGORY2_FEATURES, category2, 12)
    feature_values = one_hot_corrected(feature_values, CATEGORY3_FEATURES, category3, 12)
    feature_values = one_hot_corrected(feature_values, MATERIAL_FEATURES, material, 8)
    feature_values = one_hot_corrected(feature_values, WHO_MADE_FEATURES, who_made, 8)
    feature_values = one_hot_corrected(feature_values, WHEN_MADE_FEATURES, when_made, 9)
    feature_values = one_hot_corrected(feature_values, STYLE1_FEATURES, style1, 7)
    feature_values = one_hot_corrected(feature_values, STYLE2_FEATURES, style2, 7)
    # organize and input features to the machine learning model
    feature_tuple = tuple(feature_values)
    feature_data = [feature_tuple]
    feature_df = pd.DataFrame.from_records(feature_data, columns=FEATURE_LABELS)
    # obtain predicted price
    predicted_price = ML_MODEL.predict(feature_df)
    predicted_price = round(predicted_price.tolist()[0], 2)
    # this needs to go global for image data changing.
    # image will show up in a separate URL which is refered from a different html page
    global preds
    preds = []
    for pred in ML_MODEL.estimators_:
        preds.append(pred.predict(feature_df)[0])
    figure_name = ''
    for element in preds:
        figure_name = figure_name + str(int(element))
    return flask.render_template('output.html', predicted_price=predicted_price,
                                 figureName=figure_name)

def is_number(string_num):
    """
    Check whether the string is a number string
    """
    try:
        float(string_num)
        return True
    except ValueError:
        return False

def check_value(one_value):
    """
    Check whether there is any value passed in. It works for all different types of object
    """
    checking = True
    if (one_value is None or len(one_value) == 0 or
            (not is_number(one_value)) or int(float(one_value)) == 0):
        checking = False
    return checking

def int_value_corrected(int_value):
    """
    Correct and return int value
    """
    if not check_value(int_value):
        int_value_fixed = 0
    else:
        int_value_fixed = int(float(int_value))
    return int_value_fixed

def float_value_corrected(float_value):
    """
    Correct and return float value
    """
    if not check_value(float_value):
        float_value_fixed = 0
    else:
        float_value_fixed = float(float_value)
    return float_value_fixed

def float_indicator_corrected(float_value_ind):
    """
    Correct and return float indicator value
    """
    if not check_value(float_value_ind):
        float_value_ind_fixed = 0
    else:
        float_value_ind_fixed = 1
    return float_value_ind_fixed

def string_corrected(string_value, value_type):
    """
    Convert 'yes' and 'no' to numerical. 'yes' = 1 and 'no' = 0 for type1
    'yes' = 0 and 'no' = 1 for type0
    """
    if string_value == 'yes':
        value_fixed = 1
    else:
        value_fixed = 0
    if value_type == 0:
        value_fixed = 1 - value_fixed
    return value_fixed

def one_hot_corrected(content, feature_list, current_feature, index):
    """
    Correct and return one-hot feature value
    """
    for i in feature_list:
        if i[index:] == current_feature:
            content.append(1)
        else:
            content.append(0)
    return content

# this route is for image showing
@app.route('/fig/<figureName>')
def fig(figureName):
    # grab variable from the global variable preds
    global preds
    percent = 100
    # following can be customized to show a fixed percentile of the
    # predicted price data
    boundary_down = np.percentile(preds, (100 - percent) / 2.)
    boundary_up = np.percentile(preds, 100 - (100 - percent) / 2.)
    # plot the figure
    plt.figure(figsize=(8, 5))
    density = stats.kde.gaussian_kde(preds)
    x_values = np.arange(boundary_down, boundary_up, .1)
    plt.plot(x_values, density(x_values))
    plt.xlabel('Price (USD)')
    plt.ylabel('Density')
    plt.title('Estimated Price Distribution of Your Item')
    img = io.BytesIO()
    plt.savefig(img)
    img.seek(0)
    return flask.send_file(img, mimetype='image/png')

# both routes should go to home page
@app.route('/')
@app.route('/etsyhome')
def etsy_home():
    """
    Home page of the web app. This page requires user's input
    about the item
    """
    username = 'postgres'
    password = 'thisisforetsyapp'
    host = 'localhost'
    db_name = 'etsy_db'
    db_table = 'etsy_item_table'

    # Connect to make queries using psycopg2
    con = None
    con = psycopg2.connect(database=db_name, user=username, host=host, password=password)

    # query
    sql_query = 'SELECT category_id1 FROM %s;' % (db_table)
    etsy_data_category1 = pd.read_sql_query(sql_query, con)
    unique_category1 = sorted(etsy_data_category1['category_id1'].unique().tolist())
    category_dic_tol = {}
    category_dic2 = {}
    # pulling out category info from database
    for item1 in unique_category1:
        sql_query2 = "SELECT category_id2 FROM %s WHERE category_id1='%s';" % (db_table, item1)
        etsy_data_category2 = pd.read_sql_query(sql_query2, con)
        unique_category2 = sorted(etsy_data_category2['category_id2'].unique().tolist())
        for item2 in unique_category2:
            sql_query3 = "SELECT category_id3 FROM %s WHERE category_id1='%s' AND category_id2='%s';" % (db_table, item1, item2)
            etsy_data_category3 = pd.read_sql_query(sql_query3, con)
            unique_category3 = sorted(etsy_data_category3['category_id3'].unique().tolist())
            new_item = item1 + item2
            category_dic2[new_item] = unique_category3
        category_dic_tol[item1] = unique_category2
    unique_category2 = ['none']
    unique_category3 = ['none']
    # assign the top most materials in the data to pass to html for drawdown menu
    materials = ['others', 'cotton', 'wood', 'glass', 'paper', 'silver',
                 'metal', 'ceramic', 'brass', 'vinyl', 'leather', 'fabric',
                 'stainless steel', 'polyester', 'sterling silver', 'plastic',
                 'illustrator', 'cotton fabric', 'gold', 'copper', 'gemstone',
                 'acrylic', 'cardstock', 'ink', 'aluminum', 'canvas', 'silk',
                 'linen', 'pewter', 'ribbon', 'paint', 'nylon', 'stone',
                 'steel', 'card stock', 'crystal', 'wool', 'rayon', 'pearl',
                 'white gold', 'burlap', 'yarn', 'silicone', 'lace', 'rubber']
    materials = sorted(materials)
    sql_query4 = 'SELECT when_made FROM %s;' % (db_table)
    etsy_when = pd.read_sql_query(sql_query4, con)
    unique_etsy_when = sorted(etsy_when['when_made'].unique().tolist())
    unique_etsy_when.pop()
    unique_etsy_when.pop()
    # assign the top most style values in the data to pass to html for drawdown menu
    style = ['Modern', 'Boho', 'Traditional', 'High Fashion', 'Cottage Chic',
             'Minimalist', 'Rustic', 'Retro', 'Art Deco', 'Fantasy', 'Beach',
             'Abstract', 'Hipster', 'Victorian', 'Woodland', 'Preppy',
             'Mid Century', 'Goth', 'Folk', 'Kawaii', 'Hippie',
             'Country Western', 'Athletic', 'Zen', 'Primitive', 'Historical',
             'Steampunk', 'Asian', 'Rocker', 'Nautical']
    style = sorted(style)
    return flask.render_template('etsyHome.html', category1=unique_category1,
                                 categoryLayer1=mark_safe(json.dumps(category_dic_tol)),
                                 categoryLayer2=mark_safe(json.dumps(category_dic2)),
                                 category2=unique_category2, category3=unique_category3,
                                 materials=materials, when_made=unique_etsy_when, style=style)
