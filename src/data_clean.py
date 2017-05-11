#data clean up and preparation. The data is already quite clean but I'm sticking with this naming convention. mongodb already made. copying code to create the database in case a new one must be instated.

import pandas as pd
import numpy as np
import json
import pymongo
from copy import copy
import datetime
from bson.son import SON

# complete_df = pd.read_json('../../data/complete.json')
# complete_list=list(complete_df['programs'])
# client=pymongo.MongoClient()
# programs_db=client.programs_database
'''below code used to populate database'''
# for i in xrange(len(complete_list)):
#     programs_db.programs.insert_one(complete_list[i])


def get_composers():
    '''
    Output: List of composers from mongodb
    '''
    pipeline=[{"$unwind": "$works"},{"$group": {"_id": "$works.composerName", "count": {"$sum": 1}}},{"$sort": SON([("count", -1), ("_id", -1)])}]
    return list(programs_db.programs.aggregate(pipeline))

def make_top_composers(n, top_composers=None):
    '''
    Input:  integer. number of top composers.
            list of composers (list of dicts as per data)
    Output: list of top n composers

    Ignores the top composer of all, None, which shows up for intermissions.

    Uses mongodb.
    '''
    if top_composers is None:
        top_composers = get_composers()
    return [x['_id'] for x in top_composers[1:(n+1)]]

def create_works_concert_dfs():
    '''
    Output: two pandas dataframes

    Uses existing database to make these dataframes for easier handling. concerts is most useful for extracting dates. works important for extracting musical piece specific features.
    '''
    works_df=pd.io.json.json_normalize(complete_list, 'works', ['concerts','orchestra','programID', 'season'])
    concert_date_df=pd.io.json.json_normalize(complete_list, 'concerts', ['works','orchestra','programID', 'season'])
    return works_df, concert_date_df

def composers_in_list(works, n=10, composers=None):
    '''
    Input: list of dicts
    Ouput: count of people in composers

    checks if a selected item is in this list of top composers as derived above. works is a list of dictionaries each representing the data for a particular piece of music within the program.
    '''
    if composers is None:
        composers = make_top_composers(n)
    count=0
    works_length=len(works)
    for x in works:
        if 'composerName' in x.keys():
            if x['composerName'] in composers:
                count+=1
    if works_length>0:
        return count,count/float(works_length)
    else:
        return 0,0
