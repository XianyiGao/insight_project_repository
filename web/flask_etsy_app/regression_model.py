"""
This module does data plotting, regression model training and testing.
"""


import math
import pickle

import psycopg2
import sklearn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import tfidf_model


def get_data(db_dic):
    """
    Get and return data from PostgreSQL db. The returned data is
    in dataframe format.
    """
    con = None # Connect to make queries using psycopg2
    con = psycopg2.connect(database=db_dic['db_name'], user=db_dic['username'],
                           host=db_dic['host'], password=db_dic['password'])
    # query to extract data
    sql_query = 'SELECT * FROM %s;' % (db_dic['db_table'])
    etsy_data_from_sql = pd.read_sql_query(sql_query, con)
    return etsy_data_from_sql

def plot_category_count(df_category):
    """
    Plot the item distribution accros different categories
    """
    plt.figure(figsize=(14, 6), dpi=80, facecolor='w', edgecolor='k')
    fig = sns.countplot(y=df_category['category_id1'])
    return fig

def plot_who_made(df_who_made):
    """
    Plot item price vs. who made the item
    """
    df_plot = df_who_made[['who_made', 'price']]
    plt.figure(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    fig = sns.stripplot(x='who_made', y='price', data=df_plot, jitter=True)
    return fig

def plot_when_made(df_when_made):
    """
    Plot item price vs. when the item was made
    """
    df_plot = df_when_made[['when_made', 'price']]
    plt.figure(figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
    fig = sns.stripplot(y='when_made', x='price', data=df_plot, jitter=True)
    return fig

def plot_seller_score(df_seller_score):
    """
    Plot item price vs the seller score
    """
    df_plot = df_seller_score[['seller_score', 'price']]
    plt.figure(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    fig = plt.scatter(df_plot['seller_score'], df_plot['price'])
    plt.xlabel('seller score rating')
    plt.ylabel('item price')
    return fig

def num_onehot_text_features(df_title, features_data, cat_col, df_onehot):
    """
    Combine categorical, numerical, and text features. As
    data process is different for them, each one need to be processed
    separately.
    """
    non_cat_col = list(features_data.columns.drop(cat_col))
    df_others = features_data[non_cat_col]
    df_without_title = pd.concat([df_others, df_onehot], axis=1)
    df_organized = pd.concat([df_title, df_without_title], axis=1)
    # shuffle the feature matrix with random order
    df_organized = df_organized.sample(frac=1, random_state=321).reset_index(drop=True)
    # deal with missing data; indicators are also applied to deal with missing data
    # when neccessary
    df_organized.fillna(0, inplace=True)
    train_df_len = int(len(df_organized)/5*4)
    train_df = df_organized.iloc[0:train_df_len]
    test_df = df_organized.iloc[train_df_len:len(features_data)]
    splitted_df_dic = {
        'train_df' : train_df,
        'test_df' : test_df
    }
    return splitted_df_dic

def data_preprocessing(df_features, df_title):
    """
    This data preprocessing does training and testing data splitting.
    """
    print('processing data...')
    categorical_col = ['category_id1', 'category_id2', 'category_id3',
                       'material', 'who_made', 'when_made',
                       'style_1', 'style_2']
    # Feature engineering: extracting most important features and
    # getting rid of noisy features or relatively less important ones
    indexing = 0
    for cat_name in categorical_col:
        indexing += 1
        values = df_features[cat_name]
        counts = pd.value_counts(values)
        # make sure there are more than 50 items in the category
        mask = values.isin(counts[counts > 50].index)
        temp = pd.get_dummies(values[mask])
        column_names = temp.columns
        # fix the column names to include one-hot features
        column_names_fixed = [cat_name + x for x in column_names]
        temp.columns = column_names_fixed
        if indexing == 1:
            df_cat_onehot = temp
        else:
            df_cat_onehot = pd.concat([df_cat_onehot, temp], axis=1)
    # organize features (numerical, categorical, and text), split to train and test
    splitted_df = num_onehot_text_features(df_title, df_features,
                                           categorical_col, df_cat_onehot)
    train_df = splitted_df['train_df']
    test_df = splitted_df['test_df']
    features = list(train_df.columns)
    features.remove('price')
    # splitting label and features
    train_x = train_df[features]
    train_y = train_df['price']
    test_x = test_df[features]
    test_y = test_df['price']
    # combine results to a dictionary to return
    result_dic = {
        'train_x' : train_x,
        'train_y' : train_y,
        'test_x' : test_x,
        'test_y' : test_y
    }
    return result_dic

def save_model(model_file_name, rfr_model):
    """
    Save the current trained machine learning model to
    a binary file. This is useful since we do not need
    to train the model each time we use it.
    """
    with open(model_file_name, 'wb') as data_file:
        pickle.dump(rfr_model, data_file)

def train_create_model(train_x, train_y):
    """
    Regression model training
    """
    print('training and creating model...')
    # initialize random forest regressor
    rf_regr = sklearn.ensemble.RandomForestRegressor(
        random_state=0, n_estimators=500, n_jobs=-1)
    rf_regr.fit(train_x, train_y)
    model_name = 'machine_learning_model'
    save_model(model_name, rf_regr)
    print('completed!')
    return model_name

def load_model(is_trained_mode, model_name=None, train_x=None, train_y=None):
    """
    Load the pretrained random forest regression model
    or train and load the model
    """
    # if the model is not trained and saved
    if not is_trained_mode:
        model_name = train_create_model(train_x, train_y)
    with open(model_name, 'rb') as data_file:
        rfr_model = pickle.load(data_file)
    return rfr_model

def calculate_rmse(rfr_model, test_x, test_y):
    """
    Calculate the root mean square error and return
    predicted item price
    """
    predicted_price = rfr_model.predict(test_x)
    plt.scatter(predicted_price, test_y)
    plt.xlabel('predicted value')
    plt.ylabel('item price')
    plt.show()
    rmse = math.sqrt(sklearn.metrics.mean_squared_error(
        test_y, predicted_price))
    rmse_pred_dic = {
        'predicted_price' : predicted_price,
        'rmse' : rmse
    }
    return rmse_pred_dic

def analysis_category_removed(test_x, test_y, predicted_price):
    """
    This is for analysis perpose, checking how the prediction works
    for different categories of items.
    """
    test_y_organized = []
    predicted_price_organized = []
    for i in range(0, len(test_y)):
        if (test_x.iloc[i][527] != 1 and test_x.iloc[i][516] != 1 and
                test_x.iloc[i][522] != 1 and test_x.iloc[i][526] != 1):
            test_y_organized.append(test_y.iloc[i])
            predicted_price_organized.append(predicted_price[i])
    # plot new figure with some categories removed
    plt.scatter(predicted_price_organized, test_y_organized)
    plt.xlabel('predicted value')
    plt.xlim([0, 1000])
    plt.ylim([0, 1000])
    plt.ylabel('item price')
    plt.show()
    rmse = math.sqrt(
        sklearn.metrics.mean_squared_error(test_y_organized,
                                           predicted_price_organized)
    )
    analysis_dic = {
        'predicted_price_organized' : predicted_price_organized,
        'test_y_organized' : test_y_organized,
        'rmse' : rmse
    }
    return analysis_dic

def training_set_analysis(train_x, train_y, rfr_model):
    """
    Analysis on training dataset based on how well model performs
    """
    predicted_price_x = rfr_model.predict(train_x)
    plt.scatter(predicted_price_x, train_y)
    plt.xlabel('predicted value')
    plt.ylabel('item price')
    plt.show()
    rmse = math.sqrt(sklearn.metrics.mean_squared_error(
        train_y, predicted_price_x))
    train_analysis_dic = {
        'predicted_price_x' : predicted_price_x,
        'rmse' : rmse
    }
    return train_analysis_dic

if __name__ == '__main__':
    DB_LOAD_DIC = {
        'username' : 'postgres',
        'password' : 'thisisforetsyapp',
        'host' : 'localhost',
        'port' : '5432',   # default port that postgres listens on
        'db_name' : 'etsy_db',
        'db_table' : 'etsy_item_table'
    }
    extracted_data = get_data(DB_LOAD_DIC)
    title_list = list(extracted_data['title'])
    del extracted_data['title']
    del extracted_data['index']
    title_splitted, tfidf_vectorizer = tfidf_model.tfidf_transform(title_list, 500)
    title_array = title_splitted.toarray()
    title_dataframe = pd.DataFrame(title_array)
    result_processed_dic = data_preprocessing(extracted_data, title_dataframe)
    train_data_x = result_processed_dic['train_x']
    train_data_y = result_processed_dic['train_y']
    test_data_x = result_processed_dic['test_x']
    test_data_y = result_processed_dic['test_y']
    is_trained = True # if the model has been trained and save before.
    saved_model_name = 'machine_learning_model_saved'
    # load the model
    rfrmodel = load_model(is_trained, saved_model_name, train_data_x, train_data_y)
    # prediction and analysis
    test_result_dic = calculate_rmse(rfrmodel, test_data_x, test_data_y)
    predicted_pr = test_result_dic['predicted_price']
    print(test_result_dic['rmse'])
    analysis_result_dic = analysis_category_removed(
        test_data_x, test_data_y, predicted_pr)
    print(analysis_result_dic['rmse'])
    train_analy_dic = training_set_analysis(train_data_x, train_data_y, rfrmodel)
    print(train_analy_dic['rmse'])
