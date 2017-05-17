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
# import os
# os.chdir('src')

import numpy as np
import pandas as pd
from data_clean_new import data_clean
from data_clean_new import econ_data



dc=data_clean()
dc.run()
p_df=dc.df()
p_df

econ=econ_data()
econ.load_econ_data()
econ.make_data_matrix()
econ.data_matrix
