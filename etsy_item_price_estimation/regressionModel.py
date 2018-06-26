
# coding: utf-8

# In[14]:


import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import TFIDF
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from pandas import get_dummies
import pickle
from sklearn.metrics import mean_squared_error
from math import sqrt

def getData():
    username = 'postgres'
    password = 'thisisforetsyapp'
    host     = 'localhost'
    port     = '5432'            # default port that postgres listens on
    db_name  = 'etsy_db'
    db_table = 'etsy_item_table'

    # Connect to make queries using psycopg2
    con = None
    con = psycopg2.connect(database = db_name, user = username, host=host, password = password)

    # query:
    sql_query = "SELECT * FROM " + db_table + ";"
    etsy_data_from_sql = pd.read_sql_query(sql_query,con)
    
    title_list = list(etsy_data_from_sql['title'])
    del etsy_data_from_sql['title']
    del etsy_data_from_sql['index']
    
    return etsy_data_from_sql, title_list


def plotCategoryCount(df):
    fig2 = plt.figure(figsize=(14, 6), dpi= 80, facecolor='w', edgecolor='k')
    fig  = sns.countplot(y=df['category_id1'])
    fig.figure.savefig("category_dist.png")
    return fig
    
def plotWhoMade(df):
    data_plot = df[['who_made', 'price']]
    fig2      = plt.figure(figsize=(8, 6), dpi= 80, facecolor='w', edgecolor='k')
    fig       = sns.stripplot(x="who_made", y="price", data=data_plot, jitter=True)
    fig.figure.savefig("WhoMadeVsPrice.png")
    return fig

def plotWhenMade(df):
    data_plot2 = df[['when_made', 'price']]
    fig2       = plt.figure(figsize=(12, 6), dpi= 80, facecolor='w', edgecolor='k')
    fig        = sns.stripplot(y="when_made", x="price", data=data_plot2, jitter=True)
    fig.figure.savefig("WhenMadeVsPrice.png")
    return fig

def plotSellerScore(df):
    data_plot3 = df[['seller_score', 'price']]
    fig2       = plt.figure(figsize=(8, 6), dpi= 80, facecolor='w', edgecolor='k')
    fig        = plt.scatter(data_plot3['seller_score'], data_plot3['price'])
    plt.xlabel('seller score rating')
    plt.ylabel('item price')
    fig.figure.savefig("SellerScoreVsPrice.png")


def data_preprocessing(df, df_title):
    print('processing data...')
    categorical_col = ['category_id1', 'category_id2', 'category_id3', 'material', 'who_made', 'when_made', 'style_1', 'style_2']
    indexing = 0
    for cat_name in categorical_col:
        indexing += 1
        values = df[cat_name]
        counts = pd.value_counts(values)
        mask = values.isin(counts[counts > 50].index)
        temp = get_dummies(values[mask])
        column_names = temp.columns
        column_names_fixed = [cat_name+x for x in column_names]
        temp.columns = column_names_fixed
        print(len(column_names_fixed))
        if indexing == 1:
            df_cat_oneHot = temp
        else:
            df_cat_oneHot = pd.concat([df_cat_oneHot, temp], axis=1)

    non_categorical_col = list(df.columns.drop(categorical_col))
    print(non_categorical_col)
    df_others = df[non_categorical_col]
    df_without_title = pd.concat([df_others, df_cat_oneHot], axis=1)
    df_organized = pd.concat([df_title, df_without_title], axis=1)


    df_organized = df_organized.sample(frac=1).reset_index(drop=True)
    df_organized.fillna(0, inplace=True)
    train_df_len = int(len(df_organized)/5*4)
    train_df = df_organized.iloc[0:train_df_len]
    test_df = df_organized.iloc[train_df_len:len(df)]


    features = list(train_df.columns)
    features.remove('price')

    train_x = train_df[features]
    train_y = train_df['price']

    test_x = test_df[features]
    test_y = test_df['price']
    return train_x, train_y, test_x, test_y

def train_create_model(train_x, train_y):
    print('training and creating model...')
    RF_regr = RandomForestRegressor(random_state=0, n_estimators=500, n_jobs=-1)
    RF_regr.fit(train_x, train_y)
    print('done')

    print("saving model to 'machine_learning_model'...")
    with open("machine_learning_model", "wb") as fp:
        pickle.dump(RF_regr, fp)
    print('completed!')
    return 'machine_learning_model'

def load_model(model_name):
    with open(model_name, "rb") as fp:
        RF_regr_new = pickle.load(fp)
    return RF_regr_new


def calculate_rmse(RF_regr_new, test_x, test_y):
    predicted_price = RF_regr_new.predict(test_x)
    plt.scatter(predicted_price, test_y)
    plt.xlabel('predicted value')
    plt.ylabel('item price')
    plt.show()

    rms = sqrt(mean_squared_error(test_y, predicted_price))
    return predicted_price, rms

def analysis(test_x, test_y, predicted_price):
    print(test_x.columns[515:537])
    category_bad = []
    summing = np.array([0] * 22)
    for i in range(0, len(test_y)):
        if abs(test_y.iloc[i]-predicted_price[i]) > 300:
            summing = np.add(summing, test_x.iloc[i][515:537])
    print(summing)

def analysis_category_removed(test_x, test_y, predicted_price):
    test_y_organized = []
    predicted_price_organized = []

    for i in range(0, len(test_y)):
        if test_x.iloc[i][527] != 1 and test_x.iloc[i][516] != 1 and test_x.iloc[i][522] != 1 and test_x.iloc[i][526] != 1:
            test_y_organized.append(test_y.iloc[i])
            predicted_price_organized.append(predicted_price[i])

    plt.scatter(predicted_price_organized, test_y_organized)
    plt.xlabel('predicted value')
    plt.xlim([0, 1000])
    plt.ylim([0,1000])
    plt.ylabel('item price')
    plt.show()

    rms = sqrt(mean_squared_error(test_y_organized, predicted_price_organized))
    return rms, predicted_price_organized, test_y_organized

def trainingSetAnalysis(train_x, train_y, RF_regr_new):
    predicted_price_x = RF_regr_new.predict(train_x)
    plt.scatter(predicted_price_x, train_y)
    plt.xlabel('predicted value')
    plt.ylabel('item price')
    plt.show()

    rms = sqrt(mean_squared_error(train_y, predicted_price_x))
    return rms, predicted_price_x

if __name__=='__main__':
    df, title_list = getData()
    #plotCategoryCount(df)
    #plotWhoMade(df)
    #plotWhenMade(df)
    #plotSellerScore(df)
    title_splitted, tfidf_model = TFIDF.tfidf_transform(title_list, 500)
    title_array = title_splitted.toarray()
    df_title = pd.DataFrame(title_array)
    train_x, train_y, test_x, test_y = data_preprocessing(df, df_title)
    
    """
    Note that this model training can take quite some time. However,
    when the model has been trained once, it saves to a binary file.
    Therefore, just need to load the model from file directly for
    later use. Please comment out "model_name = train_create_model(train_x, train_y)"
    if you have run it once and have model saved already.
    """
    #model_name = train_create_model(train_x, train_y)
    rfrModel=load_model("machine_learning_model_500wordfeatures")
   
    
    

    predicted_price, rms = calculate_rmse(rfrModel, test_x, test_y)
    print(rms)
    rms_category_removed, predicted_price_organized, test_y_organized = analysis_category_removed(test_x, test_y, predicted_price)
    print(rms_category_removed)
    rms_training, predicted_price_x = trainingSetAnalysis(train_x, train_y, rfrModel)
    print(rms_training)

    #data_dic = {}
    #data_dic['train_x'] = train_x
    #data_dic['train_y'] = train_y
    #data_dic['test_x'] = test_x
    #data_dic['test_y'] = test_y
    #with open("analysis_data_backup", "wb") as fp:
    #    pickle.dump(data_dic, fp)

