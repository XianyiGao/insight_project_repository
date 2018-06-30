from flask import render_template, request, send_file
from flaskEtsyApp import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
import json
from django.utils.safestring import mark_safe 
import pickle
import os
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from flaskEtsyApp import TFIDF
import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from scipy import stats
from io import BytesIO
from flaskEtsyApp.featureEngineering import title_NLP_TFIDF_features, other_features, category1_features, category2_features, category3_features, material_features, whoMade_features, whenMade_features, style1_features, style2_features, feature_labels


APP_ROOT = os.path.dirname(os.path.abspath(__file__))   # refers to application_top
APP_STATIC = os.path.join(APP_ROOT, 'static')
   
with open(os.path.join(APP_STATIC, 'machine_learning_model'), "rb") as fp:
    ML_model = pickle.load(fp)

with open(os.path.join(APP_STATIC, 'title_list'), "rb") as fp:
    title_list = pickle.load(fp)

title_splitted_total, tfidf_model = TFIDF.tfidf_transform(title_list, 500)
preds = []
print('all data loaded')






@app.route('/output')
def etsy_output():
   title = request.args.get('title_description')
   category1 = request.args.get('category1')
   category2 = request.args.get('category2')
   category3 = request.args.get('category3')
   quantity  = request.args.get('quantity')
   material  = request.args.get('material')
   who_made  = request.args.get('who_made')
   when_made = request.args.get('when_made')
   item_weight = request.args.get('item_weight')
   item_length = request.args.get('item_length')
   item_width  = request.args.get('item_width')
   item_height = request.args.get('item_height')
   style1      = request.args.get('style1')
   style2      = 'none'
   is_taxable  = request.args.get('is_taxable')
   is_customizable = request.args.get('is_customizable')
   is_digital_downloadable = request.args.get('is_digital_downloadable')
   feedback_count  = request.args.get('feedback_count')
   seller_score    = request.args.get('seller_score')
   
   # TODO: make sure people input number in the input field
   if title==None or len(title)==0 or category1=='none':
      return render_template("error.html")
   
   
   title_splitted = tfidf_model.transform([title])
   feature_values = title_splitted.toarray()[0].tolist()
   if quantity==None or len(quantity)==0 or (not is_number(quantity)) or int(float(quantity))==0:
       feature_values.append(0)
   else:
       feature_values.append(int(float(quantity)))

   if item_weight==None or len(item_weight)==0 or (not is_number(item_weight)) or int(float(item_weight))==0:
       feature_values.append(0)
       feature_values.append(0)
   else:
       feature_values.append(float(item_weight))
       feature_values.append(1)

   if item_length==None or len(item_length)==0 or (not is_number(item_length)) or int(float(item_length))==0:
       feature_values.append(0)
       feature_values.append(0)
   else:
       feature_values.append(float(item_length))
       feature_values.append(1)

   if item_width==None or len(item_width)==0 or (not is_number(item_width)) or int(float(item_width))==0:
       feature_values.append(0)
       feature_values.append(0)
   else:
       feature_values.append(float(item_width))
       feature_values.append(1)

   if item_height==None or len(item_height)==0 or (not is_number(item_height)) or int(float(item_height))==0:
       feature_values.append(0)
       feature_values.append(0)
   else:
       feature_values.append(float(item_height))
       feature_values.append(1)

   if is_taxable=='yes':
       feature_values.append(0)
   else:
       feature_values.append(1)

   if is_customizable=='yes':
       feature_values.append(1)
   else:
       feature_values.append(0)

   if is_digital_downloadable=='yes':
       feature_values.append(1)
   else:
       feature_values.append(0)

   if feedback_count==None or len(feedback_count)==0 or (not is_number(feedback_count)) or int(float(feedback_count))==0:
       feature_values.append(0)
   else:
       feature_values.append(int(float(feedback_count)))

   if seller_score==None or len(seller_score)==0 or (not is_number(seller_score)) or int(float(seller_score))==0:
       feature_values.append(0)
       feature_values.append(0)
   else:
       feature_values.append(int(float(seller_score)))
       feature_values.append(1)

   for i in category1_features:
       if i[12:]==category1:
           feature_values.append(1)
       else:
           feature_values.append(0)

   for i in category2_features:
       if i[12:]==category2:
           feature_values.append(1)
       else:
           feature_values.append(0)

   for i in category3_features:
       if i[12:]==category3:
           feature_values.append(1)
       else:
           feature_values.append(0)

   for i in material_features:
       if i[8:]==material:
           feature_values.append(1)
       else:
           feature_values.append(0)

   for i in whoMade_features:
       if i[8:]==who_made:
           feature_values.append(1)
       else:
           feature_values.append(0)
   
   for i in whenMade_features:
       if i[9:]==when_made:
           feature_values.append(1)
       else:
           feature_values.append(0)

   for i in style1_features:
       if i[7:]==style1:
           feature_values.append(1)
       else:
           feature_values.append(0)   

   for i in style2_features:
       if i[7:]==style2:
           feature_values.append(1)
       else:
           feature_values.append(0)
   feature_tuple = tuple(feature_values)
   feature_data = [feature_tuple]
   feature_df = pd.DataFrame.from_records(feature_data, columns=feature_labels)
   predicted_price = ML_model.predict(feature_df)
   predicted_price = round(predicted_price.tolist()[0], 2)
   global preds
   preds = []
   
   for pred in ML_model.estimators_:
       preds.append(pred.predict(feature_df)[0])
   figureName=''
   for i in range(len(preds)):
       figureName = figureName + str(int(preds[i]))
   
   
   #err_down, err_up, mean_value = pred_ints(ML_model, feature_df, percentile=10)
   return render_template("output.html", predicted_price=predicted_price, figureName = figureName)
'''def pred_ints(model, x, percentile=95):
    preds = []
    for pred in model.estimators_:
        preds.append(pred.predict(x)[0])
    err_down = np.percentile(preds, (100 - percentile) / 2.)
    err_up = np.percentile(preds, 100 - (100 - percentile) / 2.)
    mean_value = np.mean(preds)
    return err_down, err_up, mean_value '''

@app.route('/fig/<figureName>')
def fig(figureName):
    global preds
    percent = 100
    print('length of the array: ', len(preds))
    boundary_down = np.percentile(preds, (100 - percent) / 2.)
    boundary_up = np.percentile(preds, 100 - (100 - percent) / 2.)
    plt.figure(figsize=(8,5))
   
    density = stats.kde.gaussian_kde(preds)
    x = np.arange(boundary_down, boundary_up, .1)
    plt.plot(x, density(x))
    plt.xlabel('Price (USD)')
    plt.ylabel('Density')
    plt.title('Density Plot for Estimated Price Distribution')
    img = BytesIO()
    plt.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='image/png')

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

@app.route('/')
@app.route('/etsyhome')
def etsy_home():
    username = 'postgres'
    password = 'thisisforetsyapp'
    host     = 'localhost'
    port     = '5432'
    db_name  = 'etsy_db'
    db_table = 'etsy_item_table'

    # Connect to make queries using psycopg2
    con = None
    con = psycopg2.connect(database = db_name, user = username, host=host, password = password)

    # query:
    sql_query = "SELECT category_id1 FROM " + db_table + ";"
    etsy_data_category1 = pd.read_sql_query(sql_query,con)
    unique_category1 = sorted(etsy_data_category1['category_id1'].unique().tolist())
    category_dic_tol = {}    
    category_dic2 = {}
    for item1 in unique_category1:
        sql_query2 = "SELECT category_id2 FROM " + db_table + " WHERE category_id1='" + item1 + "';"
        etsy_data_category2 = pd.read_sql_query(sql_query2,con)
        unique_category2 = sorted(etsy_data_category2['category_id2'].unique().tolist())
        for item2 in unique_category2:
            sql_query3 = "SELECT category_id3 FROM " + db_table + " WHERE category_id1='" + item1 + "' and category_id2='" + item2 + "';"
            etsy_data_category3 = pd.read_sql_query(sql_query3,con)
            unique_category3 = sorted(etsy_data_category3['category_id3'].unique().tolist())
            newItem = item1+item2
            category_dic2[newItem] = unique_category3
        category_dic_tol[item1] = unique_category2
    unique_category2 = ['none']
    unique_category3 = ['none']
    materials = ['others', 'cotton', 'wood', 'glass', 'paper', 'silver', 'metal', 'ceramic', 'brass', 'vinyl', 'leather', 'fabric', 'stainless steel', 'polyester', 'sterling silver', 'plastic', 'illustrator', 'cotton fabric', 'gold', 'copper', 'gemstone', 'acrylic', 'cardstock', 'ink', 'aluminum', 'canvas', 'silk', 'linen', 'pewter', 'ribbon', 'paint', 'nylon', 'stone', 'steel', 'card stock', 'crystal', 'wool', 'rayon', 'pearl', 'white gold', 'burlap', 'yarn', 'silicone', 'lace', 'rubber']
    materials = sorted(materials)
    sql_query4 = "SELECT when_made FROM " + db_table + ";"
    etsy_when = pd.read_sql_query(sql_query4,con)
    unique_etsy_when = sorted(etsy_when['when_made'].unique().tolist())
    unique_etsy_when.pop()
    unique_etsy_when.pop()
    style = ['Modern', 'Boho', 'Traditional', 'High Fashion', 'Cottage Chic', 'Minimalist', 'Rustic', 'Retro', 'Art Deco', 'Fantasy', 'Beach', 'Abstract', 'Hipster', 'Victorian', 'Woodland', 'Preppy', 'Mid Century', 'Goth', 'Folk', 'Kawaii', 'Hippie', 'Country Western', 'Athletic', 'Zen', 'Primitive', 'Historical', 'Steampunk', 'Asian', 'Rocker', 'Nautical']
    style = sorted(style)
    return render_template("etsyHome.html", category1 = unique_category1, categoryLayer1 = mark_safe(json.dumps(category_dic_tol)), categoryLayer2 = mark_safe(json.dumps(category_dic2)), category2 = unique_category2, category3 = unique_category3, materials=materials, when_made=unique_etsy_when, style=style)
