#old data_clean and processing has been deprecated in favor of this less bulky one. turning this one into a class object.

import pandas as pd
import numpy as np
import json
import pymongo
from copy import copy
import datetime
from bson.son import SON

class data_clean(object):
    def __init__(self):
        self.works_dict={}
        self.composers_dict={}
        self.complete_df=pd.read_json('../data/complete.json')
        self.complete_list= list(self.complete_df['programs'])
        self._client=pymongo.MongoClient()
        self._programs_db=self._client.programs_database
        self.works_max=0
        self.composers_max=0
        self.programs_df=[]

    # def populate_mongodb(self):
    #     '''below code used to populate database'''
    #     for i in xrange(len(complete_list)):
    #         programs_db.programs.insert_one(complete_list[i])

    def run(self):
        self.get_composers()
        self.get_works()
        self.create_programs_dataframe()

    def get_composers(self):
        """
        Creates a dict of composers and counts from mongodb

        Note: drops first element in temporary list which is always Composer: None, WorkTitle: None.
        """
        pipeline=[{"$unwind": "$works"},{"$group": {"_id": "$works.composerName", "count": {"$sum": 1}}},{"$sort": SON([("count", -1), ("_id", -1)])}]
        temp_list=list(self._programs_db.programs.aggregate(pipeline))[1:]
        self.composers_max=temp_list[0]['count']
        self.composers_dict={x['_id']:x['count'] for x in temp_list}

    def get_works(self):
        """
        Creates a dict of works and counts from mongodb

        Note: drops first element in temporary list which is always Composer: None, WorkTitle: None.
        """
        pipeline=[{"$unwind": "$works"},{"$group": {"_id": "$works.workTitle", "count": {"$sum": 1}}},{"$sort": SON([("count", -1), ("_id", -1)])}]
        temp_list=list(self._programs_db.programs.aggregate(pipeline))[1:]
        self.works_max=temp_list[0]['count']
        temp_dict={}
        for x in temp_list:
            if type(x['_id'])==unicode:
                k,v=x['_id'], x['count']
                temp_dict[k]=v
        self.works_dict=temp_dict

    def composer_unconventionality(self, composer):
        """
        Input: Str or Unicode, Dictionary
        Output: Integer, 1/composer count * max composer count
        Takes in the output from get_composers after it's been transformed into a dictionary for the given composers as keys
        """
        return 1/float(self.composers_dict[composer])

    def worktitle_unconventionality(self, workTitle):
        """
        Input: Str or Unicode, Dictionary
        Output: Integer, 1/worktitle count * max worktitle count
        Takes in the output from get_works after it's been transformed into a dictionary for the given works as keys
        """
        if type(workTitle)==unicode:
            return 1/float(self.works_dict[workTitle])
        else:
            return 1

    def unconventionality(self, works_list_from_programs):
        """
        Input: List
        Output: Int
        Takes in a list of dictionaries for a particular philharmonic program and calculates the unconventionality as defined by worktitle_unconventionality*composer_unconventionality
        """
        unconventionality_list=[]
        for x in works_list_from_programs:
            if ('workTitle' in x.keys()) and ('composerName' in x.keys()):
                unconventionality_list.append(self.worktitle_unconventionality(x['workTitle'])*self.composer_unconventionality(x['composerName']))
        if len(unconventionality_list)>0:
            return np.mean(unconventionality_list)
        else:
            return 0


    def create_programs_dataframe(self):
        """
        Output processed dataframes. Each row corresponds to a program or season. Uses first concert date as representive date for program.
        """
        programs_df=pd.DataFrame(self.complete_list)
        programs_df['unconventionality']=[self.unconventionality(works_list) for works_list in programs_df['works']]
        # season_df=pd.io.json.json_normalize(self.complete_list, 'season', ['works','orchestra','programID', 'season']).set_index('season')
        # season_df['unconventionality']=[unconventionality(works_list) for works_list in programs_df['works']]
        programs_df['Date']=[x[0]['Date'] for x in programs_df['concerts']]
        programs_df['Date']=pd.to_datetime(programs_df['Date'])
        programs_df=programs_df.join(programs_df.groupby('season').mean(), how='outer', on='season', lsuffix='_by_program', rsuffix='_by_season')
        self.programs_df = programs_df

    def df(self):
        return self.programs_df.drop(['concerts', 'orchestra','id','programID','works'], axis=1)

class econ_data(object):
    '''
    Load econ data. Considering removing dowjones and sp500 as data only stretches back to 2007. I realize this is hardcoding these imports but each data source requires its own special treatment.
    '''
    def __init__(self):
        self.cei=[]
        self.acpsa=[]
        self.nasdaq=[]
        self.volatility_index=[]
        self.dowjones=[]
        self.sp500=[]
        self.data_matrix=[]

    def load_econ_data(self):
        # monthly Coincident Economic Index data
        self.cei=pd.read_csv('../data/nyc_cei.txt', header=0, names=['DATE','New York', 'New Jersey', 'NYC','Drop1','Drop2','Drop3'], delim_whitespace=True)
        self.cei.drop(['Drop1','Drop2','Drop3','New Jersey'], axis=1, inplace=True)
        self.cei['DATE']=pd.to_datetime(self.cei['DATE'])

        # acpsa data by year, Arts and Culture Production Satellite Account
        temp_acpsa=pd.read_excel('../data/ACPSA-DataForADP.xlsx', sheetname=1)
        self.acpsa=temp_acpsa[temp_acpsa.where(temp_acpsa['FIPS, State']=='36 New York')['Industry code'].isin([34, 35, 36])]
        self.acpsa.drop(['FIPS, State', 'Industry name'], axis=1, inplace=True)
        self.acpsa['DATE']=pd.to_datetime(self.acpsa['Year'])

        # nasdaq and below, daily but varying years covered, fillna will be needed later
        # NASDAQ, Dow Jones, Standard and Poor's 500, Chicago Board Option Exchange Volatlity Index
        # data sourced from https://fred.stlouisfed.org/
        self.nasdaq=pd.read_csv('../data/NASDAQCOM.csv')
        self.dowjones=pd.read_csv('../data/DJIA.csv')
        self.sp500=pd.read_csv('../data/SP500.csv')
        self.volatility_index=pd.read_csv('../data/VIXCLS.csv')

    def make_data_matrix(self):
        '''
        Using internal state data pulled from all over to create a full data matrix.
        '''
        dfs_to_merge=[self.nasdaq, self.dowjones, self.sp500, self.volatility_index, self.acpsa, self.cei]
        mergedf=dfs_to_merge.pop(0)
        for df in dfs_to_merge:
            mergedf=pd.merge(mergedf, df, how='outer', on=['DATE','DATE'])
        mergedf['DATE']=pd.to_datetime(mergedf['DATE'])
        self.data_matrix=mergedf.fillna(method='ffill').fillna(0)
        self.data_matrix.index=self.data_matrix.set_index('DATE')

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
