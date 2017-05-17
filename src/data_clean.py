#data clean up and preparation. The data is already quite clean but I'm sticking with this naming convention. mongodb already made. copying code to create the database in case a new one must be instated.

import pandas as pd
import numpy as np
import json
import pymongo
from copy import copy
import datetime
from bson.son import SON

def run_if_first_time():
    '''below code used to populate database'''
    for i in xrange(len(complete_list)):
        programs_db.programs.insert_one(complete_list[i])

def get_composers():
    '''
    Output: List of composers from mongodb
    '''
    pipeline=[{"$unwind": "$works"},{"$group": {"_id": "$works.composerName", "count": {"$sum": 1}}},{"$sort": SON([("count", -1), ("_id", -1)])}]
    return list(programs_db.programs.aggregate(pipeline))

def get_works():
    '''
    Output: List of works by title from mongodb
    '''
    pipeline=[{"$unwind": "$works"},{"$group": {"_id": "$works.workTitle", "count": {"$sum": 1}}},{"$sort": SON([("count", -1), ("_id", -1)])}]
    return list(programs_db.programs.aggregate(pipeline))

def make_top_composers(n, top_composers=None):
    '''
    Input:  integer. number of top composers.
            list of composers (list of dicts as per created)
    Output: list of top n composers

    Ignores the top composer of all, None, which shows up for intermissions.

    Uses mongodb.
    '''
    if top_composers is None:
        top_composers = get_works()
    return [x['_id'] for x in top_composers[1:(n+1)]]

def make_top_works(w, top_works=None):
    '''
    Input:  integer. number of top works.
            list of works (list of dicts as per created)
    Output: list of top n works

    Ignores the top work of all, None, which shows up for intermissions.

    Uses mongodb.
    '''
    if top_works is None:
        top_works = get_works()
    return [x['_id'] for x in top_works[1:(w+1)]]

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
    works_length=0
    for x in works:
        if 'composerName' in x.keys():
            works_length+=1
            if x['composerName'] in composers:
                count+=1
    if works_length>0:
        return count,count/float(works_length)
    else:
        return 0,0

def works_in_list(works, w=100, works_list=None):
    '''
    Input: list of dicts
    Ouput: Int, Float, count of people in composers and fraction of those composers in list

    checks if a selected item is in this list of top composers as derived above. works is a list of dictionaries each representing the data for a particular piece of music within the program.
    '''
    if works_list is None:
        works_list = make_top_works(n)
    count=0
    works_length=0
    for x in works:
        if 'workTitle' in x.keys():
            works_length+=1
            if type(x['workTitle'])==unicode:
                if x['workTitle'] in works_list:
                    count+=1
    if works_length>0:
        return count,count/float(works_length)
    else:
        return 0,0

def composer_unconventionality(composer, composers_dict):
    '''
    Input: Str or Unicode, Dictionary
    Output: Integer, 1/composer count * max composer count
    Takes in the output from get_composers after it's been transformed into a dictionary for the given composers as keys
    '''
    return np.max(composers_dict.values)*1/float(composers_dict['composer'])

def work_unconventionality(worktitle, works_dict):
    '''
    Input: Str or Unicode, Dictionary
    Output: Integer, 1/worktitle count * max worktitle count
    Takes in the output from get_works after it's been transformed into a dictionary for the given works as keys
    '''
    return np.max(works_dict.values)*1/float(works_dict['worktitle'])

def smaller_ensembles(works_list):
    count=0
    works_length=0
    small_ensemble_keyword_list=['QUINTET', 'CONCERTO', 'QUARTET', 'PIANO', 'HARP', 'SOLO', 'DUO', 'ENSEMBLE']
    for x in works_list:
        if 'workTitle' in x.keys():
            works_length+=1
            for keyword in small_ensemble_keyword_list:
                if keyword in x['workTitle']:
                    count+=1
    if works_length>0:
        return count,count/float(works_length)
    else:
        return 0,0

"""

The following section introduces CEI (coincident economic indicator) data which could be potentially used to generate a label for the featurized data.

"""

# nyc_df=pd.read_csv('../data/nyc_cei.txt', header=0, names=['Date','New York', 'New Jersey', 'NYC','Drop1','Drop2','Drop3'], delim_whitespace=True)
# nyc_df.drop(['Drop1','Drop2','Drop3'], axis=1, inplace=True)
# nyc_df['Date']=pd.to_datetime(nyc_df['Date'])
# nyc_df=nyc_df.set_index('Date')

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


# def add_columns(n=10, w=100, m=0):
#     '''
#     Input: Integer, Integer: n top composers, m months shift back for CEI data if it all
#     Output: DataFrame, concert data with CEI added.
#
#     Uses previous function to generate dataframes and update them with CEI data as merged by dates.
#     '''
#
#     works_df, cd_df = create_works_concert_dfs()
#
#
#     '''Fix Date column to make it datetime compatible'''
#     cd_df['Date']=pd.to_datetime(cd_df['Date'])
#
#     '''Create internal lists'''
#
#     composers = make_top_composers(n)
#     worktitles= make_top_works(w)
#
#     '''
#     COLUMNS TO ADD
#     '''
#     '''
#     Number of Works
#     '''
#
#     cd_df['num_works']=[len(x) for x in cd_df['works']]
#
#     '''
#     Top composer count columns
#     '''
#
#     # cd_df['top_%s_composer_count' %n]=[composers_in_list(x,n,composers)[0] for x in cd_df['works']]
#     # cd_df['top_%s_composer_fract' %n]=[composers_in_list(x,n,composers)[1] for x in cd_df['works']]
#
#     # programmer's note, could refactor this function to only run one list comp, but dataset is small enough that this isn't currently a hindrance.
#
#     '''
#     ********* Top workTitles count columns to be added later ********
#     '''
#
#     '''
#     Individual Top Composer columns
#     '''
#
#     for composer in composers:
#         cd_df['%s_count' %composer]=[composers_in_list(x,1,composer)[0] for x in cd_df['works']]
#         cd_df['%s_fract' %composer]=[composers_in_list(x,1,composer)[1] for x in cd_df['works']]
#
#     '''
#     Individual Top Work Title columns
#     '''
#
#     for worktitle in worktitles:
#         cd_df['%s_count' %worktitle]=[works_in_list(x,1,worktitle)[0] for x in cd_df['works']]
#         cd_df['%s_fract' %worktitle]=[works_in_list(x,1,worktitle)[1] for x in cd_df['works']]
#
#     '''
#     columns for smaller ensembles
#     '''
#
#     cd_df['smaller_ensembles_count']=[smaller_ensembles(x)[0] for x in cd_df['works']]
#     cd_df['smaller_ensembles_fraction']=[smaller_ensembles(x)[1] for x in cd_df['works']]
#
#     '''
#     Date in months from date min
#     '''
#
#     cd_df['days']=(cd_df['Date']-cd_df['Date'].min())/np.timedelta64(1,'D')
#
#     '''
#     CEI columns
#     '''
#
#     nyc_df=shift_by_months(m)
#
#     for i, date in enumerate(nyc_df.index):
#         cd_df.loc[(cd_df['Date'].dt.year==date.year) & (cd_df['Date'].dt.month==date.month), 'nyc_cei']=nyc_df['NYC'][i]
#         cd_df.loc[(cd_df['Date'].dt.year==date.year) & (cd_df['Date'].dt.month==date.month), 'new_york_state_cei']=nyc_df['New York'][i]
#         if m != 0:
#             cd_df.loc[(cd_df['Date'].dt.year==date.year) & (cd_df['Date'].dt.month==date.month), 'nyc_cei_%sm' %m]=nyc_df['NYC -%s' %m][i]
#             cd_df.loc[(cd_df['Date'].dt.year==date.year) & (cd_df['Date'].dt.month==date.month), 'new_york_state_cei_%sm' %m]=nyc_df['New York -%s' %m][i]
#     output_df=cd_df[cd_df['nyc_cei'].notnull()].fillna(method='ffill')
#
#     return output_df


if __name__=='__main__':

    complete_df = pd.read_json('../data/complete.json')
    complete_list=list(complete_df['programs'])
    client=pymongo.MongoClient()
    programs_db=client.programs_database

    nyc_df=pd.read_csv('../data/nyc_cei.txt', header=0, names=['Date','New York', 'New Jersey', 'NYC','Drop1','Drop2','Drop3'], delim_whitespace=True)
    nyc_df.drop(['Drop1','Drop2','Drop3'], axis=1, inplace=True)
    nyc_df['Date']=pd.to_datetime(nyc_df['Date'])
    nyc_df=nyc_df.set_index('Date')

    # program_concerts_df=add_columns(n=10,w=50,m=6)
    # values_only_df=program_concerts_df.drop(['Location', 'Time','Venue', 'eventType','season','programID','works','orchestra'], axis=1)
    # nyc_employment_df=pd.read_excel('../data/ACPSA-DataForADP.xlsx')

#intended features: categorical composers, which season are we win
