import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load your data into a pandas DataFrame
data = pd.read_csv('yield_piecewise_X.csv')