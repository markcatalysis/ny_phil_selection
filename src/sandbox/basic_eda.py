import pandas as pd
import numpy as np
import json
import pymongo
from copy import copy
import datetime

'''
I've recently gotten into the McElroy brothers' media, so expect commenting to reflect that such as soft humor including "good good boy" and other catch phrases.
'''

complete_df = pd.read_json('../../data/complete.json')
complete_df.columns
complete_list=list(complete_df['programs'])
c_df=pd.DataFrame(complete_list)
c_df.iloc[0]['works']
c_df.iloc[0]
np.argmax([len(x) for x in c_df.concerts])
c_df.concerts[6548]

# dataframe above will be helpful for EDA later but for now let's set up a mongodb for easier data calling and faster retrieval

# starting that mongodb like a good good boy

with open('../../data/complete.json') as data_file:
    complete_dict=json.load(data_file)

# don't do the json.loads() method it is a bad idea and will kill your computer memory you bad bad boy
# type(complete_dict)
# len(complete_dict['programs'])

programs_list=complete_dict['programs']

# programs_list==complete_list
# yay...
# programs_list[0]

# '''
# setting up mongo database
# '''
# turns out i have the list available from the original dataframe load if i wanted it... woops. welp! let's continue.

# len(complete_list)
# client=pymongo.MongoClient()
# programs_db=client.programs_database
# for i in xrange(len(complete_list)):
#     programs_db.programs.insert_one(programs_list[i])
#
# '''
# mongodb now populated and available as needed
# '''

'''
The collection programs_db.programs now has all the individual programs inserted in as documents and are in no particular order but can be organized by date should one be so inclined. if order matters, another way to call similar data is above in the dataframe c_df

We will need to reshape this data in order to be able to create labels and features. The first step will be to create an expanded dataframe terminating with workTitle or composerName but is featurized by all the rest of the corresponding relevant database info corresponding to that performance.
'''

# programs_list[0]
# programs_db.programs.find_one({})
# len(programs_db.programs.distinct('works.ID'))
# len(programs_db.programs.distinct('works.composerName'))

# complete_list[0]['works'][0].keys()
# pd.io.json.json_normalize(complete_list, 'works', ['concerts','orchestra','programID', 'season'])

remapped_df=pd.io.json.json_normalize(complete_list, 'works', ['concerts','orchestra','programID', 'season'])
# remapped_df.season.unique()

concert_date_df=pd.io.json.json_normalize(complete_list, 'concerts', ['works','orchestra','programID', 'season'])
type(concert_date_df.Date[0])

'''
*****************
Taking a quick EDA detour to see if NYPhil data tracks with economic indicators. This will help me direct my feature engineering regardless.
*****************
'''
x_list=[0,1,5,3]
nyc_df=pd.read_csv('../../data/nyc_cei.txt', header=0, names=['Date','New York', 'New Jersey', 'NYC','Drop1','Drop2','Drop3'], delim_whitespace=True)
nyc_df.drop(['Drop1','Drop2','Drop3'], axis=1, inplace=True)
nyc_df['Date']=pd.to_datetime(nyc_df['Date'])
nyc_df=nyc_df.set_index('Date')
nyc_df
# creating following simple features from the concert_date_df dataframe to close basic model loop: Month of Event, nyc_cei, new_york_state_cei, Concert Contains Top Composers (to be decided), len of works list

cd_df=copy(concert_date_df)
cd_df.columns
cd_df['Date']=pd.to_datetime(cd_df['Date'])
cd_df['Date'][0].month

# Make list/column for cd_df for matching year and month filled with NYC and New York DataFrame

for i, date in enumerate(nyc_df.index):
    cd_df.loc[(cd_df['Date'].dt.year==date.year) & (cd_df['Date'].dt.month==date.month), 'nyc_cei']=nyc_df['NYC'][i]
    cd_df.loc[(cd_df['Date'].dt.year==date.year) & (cd_df['Date'].dt.month==date.month), 'new_york_state_cei']=nyc_df['New York'][i]

concert_with_cei_df = cd_df[cd_df['nyc_cei'].notnull()]
concert_with_cei_df.head()
