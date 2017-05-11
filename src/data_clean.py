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
    concert_date_df['Date']=pd.to_datetime(concert_date_df['Date'])
    return works_df, concert_date_df

def composers_in_list(works, n=10, composers=None):
    '''
    Input: list of dicts
    Ouput: Int, Float, count of people in composers and fraction of those composers in list

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

"""

The following section introduces CEI (coincident economic indicator) data which could be potentially used to generate a label for the featurized data.

"""

nyc_df=pd.read_csv('../data/nyc_cei.txt', header=0, names=['Date','New York', 'New Jersey', 'NYC','Drop1','Drop2','Drop3'], delim_whitespace=True)
nyc_df.drop(['Drop1','Drop2','Drop3'], axis=1, inplace=True)
nyc_df['Date']=pd.to_datetime(nyc_df['Date'])
nyc_df=nyc_df.set_index('Date')

'''

That was fun.

'''

def shift_by_months(m=0):
    '''
    Input: integer
    Output: dataframe

    Requires that cei dataframe, nyc_df, already created.
    '''
    if m==0:
        return nyc_df.drop(['New Jersey'], axis=1, inplace=True)
    else:
        nyc_df['New York -%s' %m]=nyc_df['New York'].shift(-m)
        nyc_df['NYC -%s' %m]=nyc_df['NYC'].shift(-m)
        output_df = nyc_df[nyc_df['NYC -%s' %m].notnull()]
        output_df.drop(['New Jersey'], axis=1, inplace = True)
        return output_df

def add_columns_by_date(m=0):
    '''
    Input: Integer, number of months to shift CEI data if it all
    Output: DataFrame, concert data with CEI added.

    Uses previous function to generate dataframes and update them with CEI data as merged by dates.
    '''
    works_df, cd_df = create_works_concert_dfs()
    nyc_df=shift_by_months(m)
    for i, date in enumerate(nyc_df.index):
        cd_df.loc[(cd_df['Date'].dt.year==date.year) & (cd_df['Date'].dt.month==date.month), 'nyc_cei']=nyc_df['NYC'][i]
        cd_df.loc[(cd_df['Date'].dt.year==date.year) & (cd_df['Date'].dt.month==date.month), 'new_york_state_cei']=nyc_df['New York'][i]
        if m != 0:
            cd_df.loc[(cd_df['Date'].dt.year==date.year) & (cd_df['Date'].dt.month==date.month), 'nyc_cei_%sm' %m]=nyc_df['NYC -%s' %m][i]
            cd_df.loc[(cd_df['Date'].dt.year==date.year) & (cd_df['Date'].dt.month==date.month), 'new_york_state_cei_%sm']=nyc_df['New York -%s' %m][i]
    output_df=cd_df[cd_df['nyc_cei']].fillna(method='ffill')
    return output_df

def run_prep():
    '''
    Call this to produce featurized data by running functions in order and with default settings.
    '''
    pass

#intended features: categorical composers, fraction "small" ensembles, fraction "big" ensembles, normalized small to total identified sizes
