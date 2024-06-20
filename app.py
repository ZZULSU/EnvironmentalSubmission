import pandas as pd

data = pd.read_csv('second_threshold_30_6000_data/yield_piecewise_X_6000_threshold30.csv')
data = data.dropna()
data.to_csv('second_threshold_30_6000_data/yield_piecewise_X_6000_threshold30.csv', index=False)