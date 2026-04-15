import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -------------------------
# 1. LOAD DATA
# -------------------------
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# -------------------------
# 2. CLEAN DATA
# -------------------------
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# -------------------------
# 3. FEATURE ENGINEERING
# -------------------------
df_ml = df.copy()
df_ml = df_ml.drop("customerID", axis=1)
df_ml = pd.get_dummies(df_ml)

# -------------------------
# 4. CHURN MODEL (PREDICTION)
# -------------------------
X = df_ml.drop("Churn", axis=1)
y = df_ml["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

df["Churn_Probability"] = model.predict_proba(X)[:, 1]

# -------------------------
# 5. CUSTOMER SEGMENTATION
# -------------------------
features = df[["tenure", "MonthlyCharges", "TotalCharges"]]

scaler = StandardScaler()
scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(scaled)

# -------------------------
# 6. BUSINESS PERSONAS
# -------------------------
summary = df.groupby("Cluster").agg({
    "tenure": "mean",
    "MonthlyCharges": "mean",
    "Churn": "mean",
    "customerID": "count"
}).rename(columns={"customerID": "Customer_Count"})

print("\nCLUSTER SUMMARY:")
print(summary)

# -------------------------
# 7. FINAL BUSINESS OUTPUT TABLE
# -------------------------
final_output = df[[
    "customerID",
    "tenure",
    "MonthlyCharges",
    "Churn_Probability",
    "Cluster"
]]

final_output.to_csv("customer_insights_output.csv", index=False)

print("\nDONE: Output saved as customer_insights_output.csv")