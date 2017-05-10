import pandas as pd
import numpy as np
import json
import pymongo
from copy import copy
import datetime
import collections import Counter
from bson.son import SON
import pprint

'''
I've recently gotten into the McElroy brothers' media, so expect commenting to reflect that such as soft humor including "good good boy" and other catch phrases.
'''

complete_df = pd.read_json('../../data/complete.json')
complete_df.columns
complete_list=list(complete_df['programs'])
c_df=pd.DataFrame(complete_list)
# c_df.iloc[0]['works']
# c_df.iloc[0]
# np.argmax([len(x) for x in c_df.concerts])
# c_df.concerts[6548]

'''
dataframe above will be helpful for EDA later but for now let's set up a mongodb for easier data calling and faster retrieval

starting that mongodb like a good good boy. i will use this for database calls.
'''

with open('../../data/complete.json') as data_file:
    complete_dict=json.load(data_file)

# don't do the json.loads() method it is a bad idea and will kill your computer memory you bad bad boy
# type(complete_dict)
# len(complete_dict['programs'])

# programs_list=complete_dict['programs']
# programs_list==complete_list
# yay...
# len(complete_list)
client=pymongo.MongoClient()
programs_db=client.programs_database
'''below code used to populate database'''
# for i in xrange(len(complete_list)):
#     programs_db.programs.insert_one(complete_list[i])
#
'''
mongodb now populated and available as needed

The collection programs_db.programs now has all the individual programs inserted as documents in the programs collection organized by season.

We will need to reshape this our concert data from the program data in order to be able to make labels and features that are date dependent. The first step will be to create an expanded dataframe terminating with composerName or workTitle but is featurized by all the rest of the corresponding relevant database info corresponding to that performance.
'''

# programs_list[0]
# programs_db.programs.find_one({})
# len(programs_db.programs.distinct('works.ID'))

'''
Making a list of unique composers in case I need this later, although "None" shows up at the top...
2750 composers in total
'''

composers=programs_db.programs.distinct('works.composerName')
# pprint.pprint(list(programs_db.programs.aggregate(pipeline)))

'''
Creating a list of top 100 most frequent composers. top_composers is a list of dictionaries containing (1) composers ranked by frequency and (2) the count of appearances. First making a pipeline to define mongodb call.
'''

pipeline=[{"$unwind": "$works"},{"$group": {"_id": "$works.composerName", "count": {"$sum": 1}}},{"$sort": SON([("count", -1), ("_id", -1)])}]
top_composers=list(programs_db.programs.aggregate(pipeline))
top_100_composers=[x['_id'] for x in top_composers[1:101]]

'''
Create works and concert event dataframes instead of program dataframe. concert_date_df will be used much more later on.
'''

works_df=pd.io.json.json_normalize(complete_list, 'works', ['concerts','orchestra','programID', 'season'])
concert_date_df=pd.io.json.json_normalize(complete_list, 'concerts', ['works','orchestra','programID', 'season'])

'''
*****************
Taking a quick EDA detour to see if NYPhil data tracks with economic indicators. This will help me direct my feature engineering regardless. First we bring in a new data source, the nyc_cei.txt, table of three month by month economic indicators in NYC, New Jersey, and NY State.
*****************
'''

nyc_df=pd.read_csv('../../data/nyc_cei.txt', header=0, names=['Date','New York', 'New Jersey', 'NYC','Drop1','Drop2','Drop3'], delim_whitespace=True)
nyc_df.drop(['Drop1','Drop2','Drop3'], axis=1, inplace=True)
nyc_df['Date']=pd.to_datetime(nyc_df['Date'])
nyc_df=nyc_df.set_index('Date')

'''
We can now use this dataframe to populate columns that can act as labels or features for the concert data.

I will now create the following simple features based on concert_date_df to complete a minimal model: nyc_cei, new_york_state_cei, fraction of works in containing Top Composers (to be decided), len of works list.
'''

cd_df=copy(concert_date_df)
cd_df.columns
cd_df['Date']=pd.to_datetime(cd_df['Date'])
cd_df['Date'][0].month
unicorn
# Make list/column for cd_df for matching year and month filled with NYC and New York DataFrame

for i, date in enumerate(nyc_df.index):
    cd_df.loc[(cd_df['Date'].dt.year==date.year) & (cd_df['Date'].dt.month==date.month), 'nyc_cei']=nyc_df['NYC'][i]
    cd_df.loc[(cd_df['Date'].dt.year==date.year) & (cd_df['Date'].dt.month==date.month), 'new_york_state_cei']=nyc_df['New York'][i]

concert_with_cei_df = cd_df[cd_df['nyc_cei'].notnull()]
'''Adding works length as column as per previously prescribed'''
concert_with_cei_df['num_works']=[len(x) for x in concert_with_cei_df['works']]

'''The concert_with_cei_df will be further modified with count of top 100 composer in program and fraction of 100 composers in works'''

def composers_in_list(works):
    '''
    checks if a selected item is in this list of top composers as derived above
    '''
    count=0
    works_length=len(works)
    for x in works:
        if 'composerName' in x.keys():
            if x['composerName'] in top_100_composers:
                count+=1
    return count,count/float(works_length)

# composers_in_list(concert_with_cei_df['works'][11156])
