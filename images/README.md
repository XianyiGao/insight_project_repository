# Package Usage

```
from etsy_item_price_estimation import create_database
from etsy_item_price_estimation import etsy_data_extract
from etsy_item_price_estimation import etsy_interface
from etsy_item_price_estimation import regression_model
from etsy_item_price_estimation import tfidf_model
```
## Some of these modules that can also be run as the main python file:

1. `python etsy_data_extract.py`

This etsy_data_extract.py can simply be run to extract data from Etsy API. It will extract and save the raw data locally on your computer. Data will be stored as two files: 'total_listing_data' and 'user_info_complete'.

As Etsy API provided restriction on the amount of API calls in a unit time, this data extraction script will take quite some time to obtain a large set of data. Please refer to the code documentation for details.

Alternatively, you can use pre-extracted data 'user_info_complete' file in this directory. 'total_listing_data' can be downloaded from the link: http://bit.ly/total_listing

2. `python create_database.py`

This create_database.py can process all the raw data obtained from etsy_data_extract.py and create a database table in PostgreSQL. Please make sure your PostgreSQL is installed and running properly in the corresponding device for the module to work.

Instructions on how to install and host PostgreSQL can be found online:
https://wiki.postgresql.org/wiki/Detailed_installation_guides

3. `python regression_model.py`

This regression_model.py can train the tfidf model and regression model. It also provide functions to save the trained models.

A pre-trained Random Forest Regression model can be downloaded through the link:
http://bit.ly/machine_learning_model_Gao

## Other modules that cannot run as the main python file:

1. etsy_interface.py

This module is acting as an interface to connect to Etsy API.

2. tfidf_model.py

This module implements TF-IDF transformation for text features.

