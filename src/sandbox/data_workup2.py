# from pandas.tools.plotting import scatter_matrix
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt
# from sklearn.cluster import DBSCAN
# from sklearn.preprocessing import StandardScaler
# from pprint import pprint
# import cPickle as pickle
import os
os.chdir('src')

import numpy as np
import pandas as pd
from data_clean_new import data_clean
from data_clean_new import econ_data
from data_clean_new import model_fit
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
import datetime
from pprint import pprint

# np.mean(mf.logistic())
mf=model_fit(d=60)
mf.logistic()
# removing feature for days since
# d=0 d_shift=0 (0.58333333333333337, 0.58333333333333337, 0.58333333333333337)
# d=21 d_shift=0 (0.54545454545454541, 0.5, 0.59999999999999998)
# d=28, d_shift=0 (0.66666666666666663, 0.66666666666666663, 0.66666666666666663)
# d=60, d_shift=0 (0.60869565217391308, 0.58333333333333337, 0.63636363636363635)
# d=120 (0.66666666666666663, 0.66666666666666663, 0.66666666666666663)
# d=180 (0.58333333333333337, 0.58333333333333337, 0.58333333333333337)
# d=360 (0.52173913043478259, 0.5, 0.54545454545454541)


# all features
# d=0 d_shift=0 (0.71999999999999986, 0.75, 0.69230769230769229)
# d=21 d_shift=0 (0.66666666666666663, 0.58333333333333337, 0.77777777777777779)
# d=28 d_shift=0 (0.75, 0.75, 0.75)
# d=60 d_shift=0 (0.80000000000000004, 0.83333333333333337, 0.76923076923076927)
# d=120 d_shift=0 (0.75, 0.75, 0.75)
# d=180 d_shift=0 (0.60869565217391308, 0.58333333333333337, 0.63636363636363635)
# d=365 d_shift=0 (0.71999999999999986, 0.75, 0.69230769230769229)
# d=730 d_shift=0 (0.66666666666666663, 0.66666666666666663, 0.66666666666666663)

zip(list(mf.econ.data_matrix.columns), list(mf.logr.coef_[0]))
mf.randomforest()
#removed days since data feature
# d=28 (0.64000000000000001, 0.66666666666666663, 0.61538461538461542)
# d=60 (0.7857142857142857, 0.91666666666666663, 0.6875) (although it got higher than this on subsequent tries)
# d=120 (0.61538461538461531, 0.66666666666666663, 0.5714285714285714) (although it got lower, all 0.5 a couple times... clearly i need more trees?)
mf.econ.data_matrix['Year']

# dc=data_clean()
# dc.run()
# dc.seasons().columns[0]

# econ=econ_data()
# econ.load_econ_data()
# econ.make_data_matrix()
# columns_1=econ.data_matrix.columns
# pprint(zip(range(len(columns_1)),list(columns_1)))
# X_base_df=econ.data_matrix
# y_seasons_df=dc.seasons()
# y_programs_df=dc.programs()
# X_dates=X_base_df.index.date
# y_seasons_df.index.date


pca=PCA()
pca.fit(StandardScaler().fit_transform(mf.X), mf.y)
pca.components_
pca.explained_variance_


nmf=NMF()
nmf.fit(mf.X[mf.X>=0], mf.y)
nmf.components_
nmf.reconstruction_err_


# X.reset_index().drop('DATE', axis=1)

# for i, date in enumerate(y_seasons_df.index.date):
#     if date in X_dates:
#         X_base_df.loc[X_base_df.index.date==date, 'unconventionality']=y_seasons_df['unconventionality_by_season'][i]
# X_seasons=X_base_df[X_base_df['unconventionality'].notnull()]
# y_seasons=X_seasons.pop('unconventionality')
# y_seasons
