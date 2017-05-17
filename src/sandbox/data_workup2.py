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
from data_clean_new import fit_the_data
import datetime

dc=data_clean()
dc.run()
econ=econ_data()
econ.load_econ_data()
econ.make_data_matrix()

# X_base_df=econ.data_matrix
# y_seasons_df=dc.seasons()
# y_programs_df=dc.programs()
# X_dates=X_base_df.index.date
# y_seasons_df.index.date

X=fit_the_data()
# X.reset_index().drop('DATE', axis=1)
X
X.applymap(np.isreal)

# for i, date in enumerate(y_seasons_df.index.date):
#     if date in X_dates:
#         X_base_df.loc[X_base_df.index.date==date, 'unconventionality']=y_seasons_df['unconventionality_by_season'][i]
# X_seasons=X_base_df[X_base_df['unconventionality'].notnull()]
# y_seasons=X_seasons.pop('unconventionality')
# y_seasons
