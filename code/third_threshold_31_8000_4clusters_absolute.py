import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from scipy import stats

# Load your data into a pandas DataFrame
data = pd.read_csv('../third_threshold_31_6000_data/yield_piecewise_X_6000_threshold31.csv')
data = data.dropna()
data.to_csv('../third_threshold_31_6000_data/yield_piecewise_X_6000_threshold31', index=False)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')  # Within cluster sum of squares
plt.show()


k = 4 # Example: You determine 3 clusters from the elbow method
kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
clusters = kmeans.fit_predict(scaled_data)

plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=clusters, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
plt.title('Clustered Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()


clusters = kmeans.fit_predict(scaled_data)

# Add the cluster assignments to the original DataFrame
data['Cluster'] = clusters

# Save the DataFrame with cluster assignments to a new CSV file (optional)
data.to_csv('../third_threshold_31_6000_data/clustered_data_X.csv', index=False)


# Load your dataset into a Pandas DataFrame
df = pd.read_csv('../third_threshold_31_6000_data/clustered_data_X.csv')

# Separate the DataFrame based on the cluster column
cluster_0 = df[df['Cluster'] == 0]
cluster_1 = df[df['Cluster'] == 1]
cluster_2 = df[df['Cluster'] == 2]
cluster_3 = df[df['Cluster'] == 3]

# List of all variables (assuming they are numeric columns)
variables = df.columns.difference(['Cluster'])  # Exclude the 'cluster' column

# Plot box plots for each variable
for var in variables:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Cluster', y=var, data=df)
    plt.title(f'Box Plot of {var} by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel(var)
    plt.grid(True)
    plt.show()

# Load the first dataset
df1 = pd.read_csv('../third_threshold_31_6000_data/clustered_data_X.csv')
# Load the third dataset
df2 = pd.read_csv('../third_threshold_31_6000_data/yield_piecewise_Y_6000.csv')

# Copy the specific column from the first dataset
# Replace 'column_name' with the actual column name you want to copy
column_to_copy = df1['Cluster']

# Add this column to the third dataset as the end column
df2['Cluster'] = column_to_copy

# Save the modified third dataset to a new CSV file
df2.to_csv('../third_threshold_31_6000_data/yield_piecewise_Y_6000_with_cluster.csv', index=False)

data = pd.read_csv('../third_threshold_31_6000_data/yield_piecewise_Y_6000_with_cluster.csv')
data = data.dropna()

yield_1980 = data.iloc[:, 0]
yield_2000 = data.iloc[:, 1]

increase_absolute = yield_2000 - yield_1980
data['difference'] = increase_absolute

data.drop(data.columns[[0, 1]], axis=1, inplace=True)

data.to_csv('../third_threshold_31_6000_data/increase_rate_with_cluster_6000.csv', index=False)

cluster_0 = data[data['Cluster'] == 0]
cluster_3 = data[data['Cluster'] == 3]


# Calculate means of the 'difference' columns
mean_df0 = cluster_0['difference'].mean()
mean_df3 = cluster_3['difference'].mean()

print(f"Mean of 'difference' in first DataFrame: {mean_df0}")
print(f"Mean of 'difference' in third DataFrame: {mean_df3}")

# Perform an independent samples t-test
t_statistic, p_value = stats.ttest_ind(cluster_0['difference'], cluster_3['difference'])

print(f"t-statistic: {t_statistic}")
print(f"p-value: {p_value}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("There is a statistically significant difference between the two means.")
else:
    print("There is no statistically significant difference between the two means.")