import pandas as pd
import numpy as np
import json
import pymongo

'''
I've recently gotten into the McElroy brothers' media, so expect commenting to reflect that such as soft humor including "good good boy" and other catch phrases.
'''

complete_df = pd.read_json('../../data/complete.json')
complete_df.head()
complete_list=list(complete_df['programs'])
c_df=pd.DataFrame(complete_list)
c_df.iloc[0]['works']
c_df.iloc[0]
'''
dataframe above will be helpful for EDA later but for now let's set up a mongodb for easier data calling and faster retrieval
'''


#starting that mongodb like a good good boy

with open('../../data/complete.json') as data_file:
    complete_dict=json.load(data_file)
# json.loads(complete_dict)
# don't do the json.loads() method it is a bad idea and will kill your computer memory you bad bad boy
# type(complete_dict)
# len(complete_dict['programs'])
programs_list=complete_dict['programs']
programs_list[0]
# turns out i have the list available from the original dataframe load if i wanted it... woops. welp! let's continue.

client=pymongo.MongoClient()
programs_db=client.programs_database
for i in xrange(len(complete_list)):
    programs_db.programs.insert_one(programs_list[i])

'''
The collection programs_db.programs now has all the individual programs inserted in as documents and are in no particular order but can be organized by date should one be so inclined. if order matters, another way to call similar data is above in the dataframe c_df

We will need to reshape this data in order to be able to create labels and features. The first step will be to create an expanded dataframe terminating with workTitle or composerName but is featurized by all the rest of the corresponding relevant database info corresponding to that performance.
'''
programs_list[0]
programs_db.programs.distinct('works')
len(programs_db.programs.distinct('works.ID'))
