#old data_clean and processing has been deprecated in favor of this less bulky one. turning this one into a class object.

import pandas as pd
import numpy as np
import json
import pymongo
from copy import copy
import datetime
from bson.son import SON
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

'''
Notes on Usage:

data_clean object is used to process the ny philharmonic data and produce a y label (treating it as the endogenous variable)

econ_data object is used to process the economics data being brought in. must be tuned for any additional data since there's no easy generic way to intuit relevant dates and values.

fit_the_data() is a function designed as a pipeline for revealing relevant scores. will likely break this off into its own py file.
'''

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
        # programs_df=programs_df.set_index('Date')
        programs_df=programs_df.join(programs_df.groupby('season').mean(), how='outer', on='season', lsuffix='_by_program', rsuffix='_by_season')
        self.programs_df = programs_df

    def df(self):
        return self.programs_df.drop(['concerts', 'orchestra','id','programID','works'], axis=1)

    def programs(self):
        p_df=self.programs_df.drop(['concerts', 'orchestra','id','programID','works'], axis=1)
        return p_df[['Date', 'unconventionality_by_program']].set_index('Date')

    def seasons(self):
        p_df=self.programs_df.drop(['concerts', 'orchestra','id','programID','works'], axis=1)
        return p_df.groupby('season').first().drop('unconventionality_by_program', axis=1).set_index('Date')

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

    def isnumber(self, x):
        '''
        Input: element
        Output: mask for that element
        '''
        try:
            float(x)
            return True
        except:
            return False


    def make_data_matrix(self):
        '''
        Using internal state data pulled from all over to create a full data matrix.
        '''
        dfs_to_merge = [self.nasdaq, self.dowjones, self.sp500, self.volatility_index, self.acpsa, self.cei]
        mergedf=dfs_to_merge.pop(0)
        for df in dfs_to_merge:
            mergedf=pd.merge(mergedf, df, how='outer', on=['DATE','DATE'])
        mergedf['DATE']=pd.to_datetime(mergedf['DATE'])
        self.data_matrix=mergedf
        self.data_matrix=self.data_matrix.set_index('DATE')
        self.data_matrix=self.data_matrix[self.data_matrix.applymap(self.isnumber)].fillna(method='ffill').fillna(0)



def fit_the_data(dc=None, econ=None):
    '''
    Input: Object, Object  --  fitted data_clean() and econ_data() class objects. if an input is not supplied, produces it for you.
    '''

    if dc is None:
        dc=data_clean()
        dc.run()
    if econ is None:
        econ=econ_data()
        econ.load_econ_data()
        econ.make_data_matrix()

    '''
    reshape relevant X and y data for individual programs and for whole seasons.
    '''

    X_base_df=econ.data_matrix
    y_seasons_df=dc.seasons()
    y_programs_df=dc.programs()
    X_dates=X_base_df.index.date
    y_seasons_df.index.date

    for i, date in enumerate(y_seasons_df.index.date):
        if date in X_dates:
            X_base_df.loc[X_base_df.index.date==date, 'unconventionality']=y_seasons_df['unconventionality_by_season'][i]
    X=X_base_df[X_base_df['unconventionality'].notnull()]
    y=X.pop('unconventionality')
    # X_seasons=X_base_df[X_base_df['unconventionality'].notnull()]
    # y_seasons=X_seasons.pop('unconventionality')

    # reset the base dataframe
    # run for individual programs

    # X_base_df=econ.data_matrix
    # for i, date in enumerate(y_programs_df.index.date):
    #     if date in X_dates:
    #         X_base_df.loc[X_base_df.index.date==date, 'unconventionality']=y_programs_df['unconventionality_by_program'][i]
    # X_programs=X_base_df[X_base_df['unconventionality'].notnull()]
    # y_programs=X_programs.pop('unconventionality')

    X['ones']=np.ones(X.shape[0])
    X=X.reset_index().drop('DATE', axis=1)
    X=StandardScaler().fit_transform(X)
    X_train,X_test,y_train,y_test=train_test_split(X,y)
    lr=LinearRegression()
    lr.fit(X_train,y_train)
    return lr.score(X_test,y_test)
    # return X
