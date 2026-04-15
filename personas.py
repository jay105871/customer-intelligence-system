import pandas as pd

# LOAD DATA
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# CLEAN
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

# FEATURES
features = df[["tenure", "MonthlyCharges", "TotalCharges"]]

# SCALE + CLUSTER (same logic as before but simplified)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

scaler = StandardScaler()
scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(scaled)

# ADD CHURN INFO
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# GROUP ANALYSIS
summary = df.groupby("Cluster").agg({
    "tenure": "mean",
    "MonthlyCharges": "mean",
    "TotalCharges": "mean",
    "Churn": "mean",
    "customerID": "count"
}).rename(columns={"customerID": "count"})

print(summary)

print("\nBUSINESS SUMMARY:")
print("Cluster 2 = High value / high risk → retention focus")
print("Cluster 0 = New users at risk → onboarding focus")
print("Cluster 1 = Loyal users → upsell focus")
print("Cluster 3 = average users → general marketing")