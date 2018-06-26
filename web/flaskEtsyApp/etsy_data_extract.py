import datetime
import seaborn as sns
import pandas as pd
from etsy_interface import Etsy
import pickle
import time
import sys


def extract_listings(e):
    '''
    Extract all current active listings
    '''
    total_listing_data = []
    for i in range(0, 500):
        index = 100*i
        result = e.show_listings('100', str(index))
        total_listing_data.append(result)
        # sleep for 1 second on every 10 calls due to API limitation
        if i in range(9,500,10):
            time.sleep(1)
    with open("total_listing_data", "wb") as fp:
        pickle.dump(total_listing_data, fp)
    return total_listing_data


def find_user_id(total_listing_data_loaded):
    # this user dictionary is for the conviinence of referring users
    user_id_dict = {}
    for i in range(0, len(total_listing_data_loaded)):
        for j in range(0, len(total_listing_data_loaded[i]['results'])):
            if 'user_id' not in total_listing_data_loaded[i]['results'][j].keys():
                continue
            key = str(total_listing_data_loaded[i]['results'][j]['user_id'])
            if key not in user_id_dict.keys():
                user_id_dict[key] = 1
    return user_id_dict

def extract_user_info(user_id_dict, e2):
    user_info = {}
    counter = 0
    for single_key in user_id_dict.keys():
        try:
            if single_key not in user_info.keys():
                key_int=int(single_key)
                element=e2.get_user_info(key_int)
                user_info[single_key] = element
                counter += 1
                # sleep for 1 second on every 10 API calls
                if counter % 10 == 0:
                    time.sleep(1)
                # record a backup in every 100 API calls in case of sudden API restriction
                if counter % 100 == 0:
                    filename = 'user_info_' + str(counter)
                    with open(filename, "wb") as fp3:
                        pickle.dump(user_info, fp3)
        except Exception as error:
            print(error)
    # store the complete data in binary file first
    with open('user_info_complete', "wb") as fp3:
                pickle.dump(user_info, fp3)


            
if __name__=='__main__':
    '''
    Data extracting using Etsy API calls.
    There may be limitation on the total numbers of API calls allowed in a given time.
    Please refer to Etsy API for more details:
    https://www.etsy.com/developers/documentation/getting_started/api_basics
    '''
    print("Please wait until you see the messsage: All data are extracted! This can take very long time ...")
    authen_key1 = 'zdipkqu8fxsomaoriwj1x98a'
    authen_secret1 = 'rxv4sdcs5g'
    e = Etsy(authen_key1, authen_secret1)

    # extract and save data as a binary file first
    listing_data = extract_listings(e)

    user_id_dictionary = find_user_id(listing_data)

    # use another key to extract data due to the limitation of API calls
    authen_key2 = 'dnbju23vywp4369ccw4ugcut'
    authen_secret2 = '5ac0qkonp8'
    e2 = Etsy(authen_key2, authen_secret2)

    # extract and save data as a binary file first
    extract_user_info(user_id_dictionary, e2)
    print('All data are extracted!')

