import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

# LOAD DATA
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# CLEAN DATA
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

# KEEP ONLY NUMERIC-USEFUL FEATURES (simple version)
features = df[["tenure", "MonthlyCharges", "TotalCharges"]]

# SCALE DATA (VERY IMPORTANT for clustering)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# CREATE CLUSTERS
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(scaled_features)

# SEE RESULTS
print(df["Cluster"].value_counts())

# SHOW SIMPLE VISUALIZATION
plt.scatter(df["MonthlyCharges"], df["TotalCharges"], c=df["Cluster"])
plt.xlabel("Monthly Charges")
plt.ylabel("Total Charges")
plt.title("Customer Segments")
plt.show()

print("\nCluster Insights (basic interpretation):")
print("Cluster 0–3 represent different customer types based on spending and tenure.")
print("Use this to target marketing campaigns differently for each group.")