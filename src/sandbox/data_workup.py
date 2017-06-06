from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from pprint import pprint
import cPickle as pickle
import seaborn as sns

import os
os.chdir('src')

run data_clean

state_employment_df=pd.read_excel('../data/ACPSA-DataForADP.xlsx', sheetname=1)
ny_arts_employment_df=state_employment_df[state_employment_df.where(state_employment_df['FIPS, State']=='36 New York')['Industry code'].isin([34, 35, 36])]
ny_arts_employment_df.drop(['FIPS, State', 'Industry name'], axis=1)
ny_arts_employment_df.shape

# values_only_df.columns
y=values_only_df.new_york_state_cei
X=values_only_df.drop(['Date','nyc_cei_6m', 'nyc_cei', 'new_york_state_cei', 'new_york_state_cei_6m'], axis=1)
X['ones']=np.ones(X.shape[0])
# pprint(X.columns)
X_standard=StandardScaler().fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X_standard,y)

lr=LinearRegression()
lr.fit(X_train,y_train)
lr.score(X_test,y_test)

lr.coef_[np.argpartition(np.absolute(lr.coef_),-10)[-10:]]
X.columns[np.argpartition(np.absolute(lr.coef_),-10)[-10:]]
plt.plot(X['days'],y)
plt.plot(X['days'],lr.predict(X_standard))
plt.legend()
plt.savefig('../presentation/ny_state_cei_with_days')
plt.show()
# values_only_df.columns
# np.max(X['days'])

plt.show()
plt.savefig('../presentation/nyc_cei_with_days')
y=values_only_df.new_york_state_cei
X=values_only_df.drop(['Date','nyc_cei_6m', 'nyc_cei', 'new_york_state_cei', 'new_york_state_cei_6m'], axis=1)
X['ones']=np.ones(X.shape[0])
X_standard=StandardScaler().fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X_standard,y)
lr.score(X_test,y_test)

# with days 0.82042109296942156
# without days 0.025627018384981715

#[['days','ones']]

#score to beat
#with n=10
#nyc_cei base score only time and ones 0.82170387012973989
# current with all features: 0.82243052902996461
#nyc_cei_6m 0.81709142238659505
# current with all features: 0.81792110542641727

#updated scores to beat
#with n=50 and composer count commented out
#nyc_cei base score only time and ones 0.81877982143582251
#nyc_cei with all features: 0.82327963441095975

'''
New high score... yay...
'''

#MORE play counts added
#baseline with no features: 0.81878244937070133
#added features: 0.82234015735596777
#FUDGE MONKEY not much there but hey more features. Let's get to finding latent features.


'''
Implementing hierarchal-like clustering to identify commonly played pieces.
'''

debra=DBSCAN(eps=0.2, algorithm='brute', metric='cosine')
debra.fit(X)

core_samples_mask = np.zeros_like(debra.labels_, dtype=bool)
core_samples_mask[debra.core_sample_indices_] = True
labels = debra.labels_
labels
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)

# pprint(values_only_df.drop(['Date','nyc_cei_6m', 'nyc_cei', 'new_york_state_cei', 'new_york_state_cei_6m'], axis=1).iloc[labels==2])

'''Fascinating... with eps=0.5, there are only 3 clusters, one super cluser and two smaller ones. The programs labeled 2 in this clustering have 0 values all across the board. Nothing appears in the most common composers or pieces. Those labeled 1 have a higher percentage of small ensemble pieces but low overlap with the other more common composers. Interesting. Checking what the works are for the programs in cluster 2.'''

program_concerts_df['works'].iloc[labels==2]

'''All of the ones with label == 2 are Handel's Messiah! Excellent... I'm pickle those results. Although I could probably search for them to recreate the indices, I find this to be an amusing list that could be of use later.'''

handels_labels=labels==2
with open('Messiah.csv', 'w') as f:
    pickle.dumps(handels_labels)
