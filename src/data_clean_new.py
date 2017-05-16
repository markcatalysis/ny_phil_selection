#old data_clean and processing has been deprecated in favor of this less bulky one. turning this one into a class object.

import pandas as pd
import numpy as np
import json
import pymongo
from copy import copy
import datetime
from bson.son import SON

class data_clean(object):
    def __init__():
        self.works_dict={}
        self.composers_dict={}
        self.complete_df=pd.read_json('../data/complete.json')
        self.complete_list= list(self.complete_df['programs'])
        self._client=pymongo.MongoClient()
        self._programs_db=self._client.programs_database
        self.works_max=0
        self.composers_max=0
        self.run

    # def populate_mongodb(self):
    #     '''below code used to populate database'''
    #     for i in xrange(len(complete_list)):
    #         programs_db.programs.insert_one(complete_list[i])

    def run(self):
        self.get_composers()
        self.get_works()
        self.composers_max=np.max(self.composers_dict.values)
        self.works_max=np.max(self.works_dict.values)

    def get_composers(self):
        '''
        Creates a dict of composers and counts from mongodb
        '''
        pipeline=[{"$unwind": "$works"},{"$group": {"_id": "$works.composerName", "count": {"$sum": 1}}},{"$sort": SON([("count", -1), ("_id", -1)])}]
        temp_list=list(self._programs_db.programs.aggregate(pipeline))
        self.composers_dict={x[0]['_id']:x[1]['count'] for x in temp_list}

    def get_works(self):
        '''
        Creates a dict of works and counts from mongodb
        '''
        pipeline=[{"$unwind": "$works"},{"$group": {"_id": "$works.workTitle", "count": {"$sum": 1}}},{"$sort": SON([("count", -1), ("_id", -1)])}]
        temp_list=list(self._programs_db.programs.aggregate(pipeline))
        self.works_dict={x[0]['_id']:x[1]['count'] for x in temp_list}

    def composer_unconventionality(self, composer):
        '''
        Input: Str or Unicode, Dictionary
        Output: Integer, 1/composer count * max composer count
        Takes in the output from get_composers after it's been transformed into a dictionary for the given composers as keys
        '''
        return self.composers_max/float(composers_dict['composer'])

    def worktitle_unconventionality(self, worktitle):
        '''
        Input: Str or Unicode, Dictionary
        Output: Integer, 1/worktitle count * max worktitle count
        Takes in the output from get_works after it's been transformed into a dictionary for the given works as keys
        '''
        if works_dict=None:
            works_dict=self.works_dict
        return self.works_max/float(works_dict['worktitle'])

    def unconventionality(self, works_list_from_programs):
        '''
        Input: List
        Output: Int
        Takes in a list of dictionaries for a particular philharmonic program and calculates the unconventionality as defined by worktitle_unconventionality*composer_unconventionality
        '''
        unconventionality_list=[]
        for x in works_list:
            if 'workTitle' in x.keys() and 'composerName' in x.keys:
                unconventionality_list.append(worktitle_unconventionality(x['workTitle'])*composer_unconventionality(x['composerName']))
        if len(unconventionality_list)>0:
            return np.mean(unconventionality_list)
        else:
            return 0


    def create_programs_dataframe(self):
        '''
        Output processed dataframes. Each row corresponds to a program or season. Uses first concert date as representive date for program.
        '''
        programs_df=pd.io.json.json_normalize(self.complete_list, 'program_ID', ['concerts','works','orchestra', 'season']).set_index('programID')
        programs_df['unconventionality']=[unconventionality(works_list) for works_list in programs_df['works']]
        # season_df=pd.io.json.json_normalize(self.complete_list, 'season', ['works','orchestra','programID', 'season']).set_index('season')
        # season_df['unconventionality']=[unconventionality(works_list) for works_list in programs_df['works']]
        programs_df['Date']=pd.to_datetime(programs_df['concerts'].iloc[0]['Date'])
        return programs_df



# programs_df['Date']=pd.to_datetime(concert_date_df['Date'])

# def create_works_concert_dfs():
#     '''
#     Output: two pandas dataframes
#
#     Uses existing database to make these dataframes for easier handling. concerts is most useful for extracting dates. works important for extracting musical piece specific features.
#     '''
#     works_df=pd.io.json.json_normalize(complete_list, 'works', ['concerts','orchestra','programID', 'season'])
#     concert_date_df=pd.io.json.json_normalize(complete_list, 'concerts', ['works','orchestra','programID', 'season'])
#     concert_date_df['Date']=pd.to_datetime(concert_date_df['Date'])
#     return works_df, concert_date_df
#
# def composers_in_list(works, n=10, composers=None):
#     '''
#     Input: list of dicts
#     Ouput: Int, Float, count of people in composers and fraction of those composers in list
#
#     checks if a selected item is in this list of top composers as derived above. works is a list of dictionaries each representing the data for a particular piece of music within the program.
#     '''
#     if composers is None:
#         composers = make_top_composers(n)
#     count=0
#     works_length=0
#     for x in works:
#         if 'composerName' in x.keys():
#             works_length+=1
#             if x['composerName'] in composers:
#                 count+=1
#     if works_length>0:
#         return count,count/float(works_length)
#     else:
#         return 0,0
#
# def works_in_list(works, w=100, works_list=None):
#     '''
#     Input: list of dicts
#     Ouput: Int, Float, count of people in composers and fraction of those composers in list
#
#     checks if a selected item is in this list of top composers as derived above. works is a list of dictionaries each representing the data for a particular piece of music within the program.
#     '''
#     if works_list is None:
#         works_list = make_top_works(n)
#     count=0
#     works_length=0
#     for x in works:
#         if 'workTitle' in x.keys():
#             works_length+=1
#             if type(x['workTitle'])==unicode:
#                 if x['workTitle'] in works_list:
#                     count+=1
#     if works_length>0:
#         return count,count/float(works_length)
#     else:
#         return 0,0
