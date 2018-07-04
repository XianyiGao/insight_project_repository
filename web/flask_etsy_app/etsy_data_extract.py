"""
This is for data extraction using Etsy API calls.

There may be limitation on the total numbers of API calls allowed in a given time.
Please refer to Etsy API for more details:
https://www.etsy.com/developers/documentation/getting_started/api_basics
"""


import time
import pickle
import etsy_interface


def extract_listings(etsy_connector):
    """
    Extracts all current active listings available from Etsy API.
    """
    total_listing_data = []
    # 500 is the maximum number related to the page offset
    for i in range(500):
        index = 100 * i   # set offset paramter for Etsy pagination
        max_per_page = '100'  # set maximum numbers of listings per page
        result = etsy_connector.show_listings(max_per_page, str(index))
        total_listing_data.append(result)
        # sleep for 1 second on every 10 calls due to API limitation
        if i in range(9, 500, 10):
            time.sleep(1)
    return total_listing_data

def store_data(file_name, data):
    """
    Save data to a binary file.
    """
    with open(file_name, 'wb') as data_file:
        pickle.dump(data, data_file)

def find_user_id(item_listing_data):
    """
    Finds all user ids and store them as a big dictionary.
    """
    user_id_dict = {}
    for item in item_listing_data:
        temp_results = item['results']
        for id_element in temp_results:
            if 'user_id' not in id_element.keys():
                continue
            key = str(id_element['user_id'])
            if key not in user_id_dict.keys():
                user_id_dict[key] = 1
    return user_id_dict

def extract_user_info(user_id_dict, etsy_connector):
    """
    Extracts all user information from Etsy using all the
    user ids.
    """
    user_info = {}   # put user info into a dictionary with id as key
    counter = 0
    for id_key in user_id_dict.keys():
        if id_key not in user_info.keys():
            key_int = int(id_key)
            try:
                element = etsy_connector.get_user_info(key_int)
            except Exception as error:
                print(error)
                continue
            user_info[id_key] = element
            counter += 1
            # sleep for 1 second on every 10 API calls due to API limitation
            if counter % 10 == 0:
                time.sleep(1)
    return user_info


if __name__ == '__main__':
    print('Please wait until you see the messsage: '
          '"All data are extracted!"\n'
          'This can take very long time ...')

    # define the API keys
    AUTHEN_KEY = 'zdipkqu8fxsomaoriwj1x98a'
    AUTHEN_SECRET = 'rxv4sdcs5g'
    etsy_con = etsy_interface.Etsy(AUTHEN_KEY, AUTHEN_SECRET)

    # extract item listing data
    # and save data as a binary file first
    listing_data = extract_listings(etsy_con)
    listing_data_filename = 'total_listing_data'
    store_data(listing_data_filename, listing_data)

    # organize and clean user id data
    user_id_dictionary = find_user_id(listing_data)
    # extract user info data from different API calls
    user_info_data = extract_user_info(user_id_dictionary, etsy_con)
    user_info_filename = 'user_info_complete'
    store_data(user_info_filename, user_info_data)
    print('All data are extracted!')
