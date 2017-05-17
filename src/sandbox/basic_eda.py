import pandas as pd
import numpy as np
import json
import pymongo
from copy import copy
import datetime
from bson.son import SON
import pprint
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
'''
I've recently gotten into the McElroy brothers' media, so expect commenting to reflect that such as soft humor including "good good boy" and other catch phrases.
'''

complete_df = pd.read_json('../../data/complete.json')
# complete_df.columns
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
concert_date_df.head()
'''
*****************
Taking a quick EDA detour to see if NYPhil data tracks with economic indicators. This will help me direct my feature engineering regardless. First we bring in a new data source, the nyc_cei.txt, table of three month by month economic indicators in NYC, New Jersey, and NY State.
*****************
'''

nyc_df=pd.read_csv('../../data/nyc_cei.txt', header=0, names=['Date','New York', 'New Jersey', 'NYC','Drop1','Drop2','Drop3'], delim_whitespace=True)
nyc_df.drop(['Drop1','Drop2','Drop3'], axis=1, inplace=True)
nyc_df['Date']=pd.to_datetime(nyc_df['Date'])
nyc_df=nyc_df.set_index('Date')

'''Adding columns for cei data set back to 6 months to account for lag in response to data'''
nyc_df['New York -12']=nyc_df['New York'].shift(-12)
nyc_df['NYC -12']=nyc_df['NYC'].shift(-12)
'''
We can now use this dataframe to populate columns that can act as labels or features for the concert data.

I will now create the following simple features based on concert_date_df to complete a minimal model: nyc_cei, new_york_state_cei, fraction of works in containing Top Composers (to be decided), len of works list.
'''

cd_df=copy(concert_date_df)
cd_df['Date']=pd.to_datetime(cd_df['Date'])
cd_df

'''Making column for cd_df for corresponding NYC and New York cei data.'''
for i, date in enumerate(nyc_df.index):
    cd_df.loc[(cd_df['Date'].dt.year==date.year) & (cd_df['Date'].dt.month==date.month), 'nyc_cei']=nyc_df['NYC'][i]
    cd_df.loc[(cd_df['Date'].dt.year==date.year) & (cd_df['Date'].dt.month==date.month), 'new_york_state_cei']=nyc_df['New York'][i]
    cd_df.loc[(cd_df['Date'].dt.year==date.year) & (cd_df['Date'].dt.month==date.month), 'nyc_cei_12m']=nyc_df['NYC -12'][i]
    cd_df.loc[(cd_df['Date'].dt.year==date.year) & (cd_df['Date'].dt.month==date.month), 'new_york_state_cei_12m']=nyc_df['New York -12'][i]
concert_with_cei_df = cd_df[cd_df['nyc_cei'].notnull()]
'''Adding works length as column as per previously prescribed'''
concert_with_cei_df['num_works']=[len(x) for x in concert_with_cei_df['works']]
'''The concert_with_cei_df will be further modified with count of top 100 composer in program and fraction of 100 composers in works'''

def composers_in_list(works):
    '''
    checks if a selected item is in this list of top composers as derived above. works is a list of dictionaries each representing the data for a particular piece of music within the program.
    '''
    count=0
    works_length=len(works)
    for x in works:
        if 'composerName' in x.keys():
            if x['composerName'] in top_100_composers:
                count+=1
    if works_length>0:
        return count,count/float(works_length)
    else:
        return 0,0

top_10_composers=[x['_id'] for x in top_composers[1:11]]

def composers_in_10_list(works):
    '''
    checks if a selected item is in this list of top composers as derived above. works is a list of dictionaries each representing the data for a particular piece of music within the program.
    '''
    count=0
    works_length=len(works)
    for x in works:
        if 'composerName' in x.keys():
            if x['composerName'] in top_10_composers:
                count+=1
    if works_length>0:
        return count,count/float(works_length)
    else:
        return 0,0

concert_with_cei_df['top_composer_count']=[composers_in_list(x)[0] for x in concert_with_cei_df['works']]
concert_with_cei_df['top_composer_fract']=[composers_in_list(x)[1] for x in concert_with_cei_df['works']]
concert_with_cei_df['top_10_composer_count']=[composers_in_10_list(x)[0] for x in concert_with_cei_df['works']]
concert_with_cei_df['top_10_composer_fract']=[composers_in_10_list(x)[1] for x in concert_with_cei_df['works']]
'''composer based columns added'''

"""let's inspect this df a bit. will rename it into something more wieldy."""
big_df=copy(concert_with_cei_df)
big_df=big_df.drop(['Location', 'Time','Venue', 'eventType','season','programID','works','orchestra'], axis=1)
sm=scatter_matrix(big_df, diagonal='kde', figsize=(6, 6), alpha=.2)
plt.tight_layout
[s.xaxis.label.set_rotation(45) for s in sm.reshape(-1)]
[s.yaxis.label.set_rotation(45) for s in sm.reshape(-1)]
[s.get_yaxis().set_label_coords(-0.3,0.5) for s in sm.reshape(-1)]
[s.set_xticks(()) for s in sm.reshape(-1)]
[s.set_yticks(()) for s in sm.reshape(-1)]
plt.show()

big_df.columns
X=big_df[['top_composer_fract', 'top_composer_count', 'top_10_composer_fract', 'top_10_composer_count','num_works']].fillna(value=0)
X['ones']=np.ones((X.shape[0],1))
y=big_df[['new_york_state_cei_12m']].fillna(value=0)

X_train,X_test,y_train,y_test=train_test_split(X,y)
lr=LinearRegression()
lr.fit(X_train,y_train)
lr.score(X_test,y_test)
'''
Score was ~0.0008 R^2. Pretty rough.

We're going to engineer more features! Contains small pieces: concerto, quartet, quintet, trio, duo. No idea if those are in there but we'll build a function for it. Also will generate a function for top 10 composers only. Result of top 10... nearly no gain in signal. Maybe no signal? Interesting.

Score improvement to ~0.0027, which suggest slightly more signal in top 10 but not much. To avoid p-hacking let's focus on non-composer rate features. Composers will be visited again later with regard to dummifying and dropping columns. Don't forget to add column of 1s!
'''

def smaller_ensembles(works_list):
    count=0
    small_ensemble_keyword_list=['QUINTET', 'CONCERTO', 'QUARTET', 'PIANO', 'HARP', 'SOLO', 'DUO', 'ENSEMBLE']
    works_length=len(works_list)
    for x in works_list:
        if 'workTitle' in x.keys():
            for keyword in small_ensemble_keyword_list:
                if keyword in x['workTitle']:
                    count+=1
    if works_length>0:
        return count,count/float(works_length)
    else:
        return 0,0

[smaller_ensembles(x) for x in concert_with_cei_df['works']]

# works_df=pd.io.json.json_normalize(complete_list, 'works', ['concerts','orchestra','programID', 'season'])
# 'QUINTET' in works_df.iloc[2].workTitle
