from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression


from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


# program_concerts_df['works']def make_top_works(w, top_works=None):

run data_clean

values_only_df.shape
y=values_only_df.nyc_cei
X=values_only_df.drop(['Date','nyc_cei_6m', 'nyc_cei', 'new_york_state_cei', 'new_york_state_cei_6m'], axis=1)
X['ones']=np.ones(X.shape[0])
X=StandardScaler().fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X,y)

lr=LinearRegression()
lr.fit(X_train,y_train)
lr.score(X_test,y_test)

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

debra=DBSCAN(eps=0.5, algorithm='brute', metric='cosine')
debra.fit(X)

core_samples_mask = np.zeros_like(debra.labels_, dtype=bool)
core_samples_mask[debra.core_sample_indices_] = True
labels = debra.labels_
labels
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
