import sys
sys.path.append("../")

import pandas as pd
import pymongo

from keys import *


def save_dict_to_mlab(results_dict):
    """
    Saves an input dictionary to an mlab database.
    """
    uri = f"mongodb://{mlab_api['username']}:{mlab_api['password']}@ds261277.mlab.com:61277/wiki_scrapper"    

    client = pymongo.MongoClient(uri)
    db = client.get_default_database()
    data_inserter = db["pre_calculated"]

    data_inserter.insert_one(results_dict)

    client.close()

