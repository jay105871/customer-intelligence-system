import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# -------------------------
# PAGE CONFIG (UI UPGRADE)
# -------------------------
st.set_page_config(
    page_title="Customer Intelligence System",
    layout="wide"
)

st.title("📊 Customer Intelligence Dashboard")
st.markdown("End-to-end churn prediction + segmentation system")

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# -------------------------
# SIDEBAR (NEW UI FEATURE)
# -------------------------
st.sidebar.header("Filters")

min_tenure = st.sidebar.slider("Minimum Tenure", 0, 72, 0)
df = df[df["tenure"] >= min_tenure]

st.sidebar.write("Rows after filter:", len(df))

# -------------------------
# KPI CARDS (UI IMPROVEMENT)
# -------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", len(df))
col2.metric("Churn Rate", f"{df['Churn'].mean():.2f}")
col3.metric("Avg Monthly Charges", f"{df['MonthlyCharges'].mean():.2f}")

st.divider()

# -------------------------
# CHURN MODEL
# -------------------------
df_model = df.copy()
df_model = df_model.drop("customerID", axis=1)
df_model = pd.get_dummies(df_model)

X = df_model.drop("Churn", axis=1)
y = df_model["Churn"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

df["Churn_Probability"] = model.predict_proba(X)[:, 1]

# -------------------------
# SEGMENTATION
# -------------------------
features = df[["tenure", "MonthlyCharges", "TotalCharges"]]

scaler = StandardScaler()
scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(scaled)

# -------------------------
# LAYOUT IMPROVEMENT (TABS)
# -------------------------
tab1, tab2, tab3 = st.tabs(["📊 Data", "🤖 Predictions", "📈 Visualizations"])

# -------------------------
# TAB 1 - DATA
# -------------------------
with tab1:
    st.subheader("Customer Data")
    st.dataframe(df.head(50), use_container_width=True)

# -------------------------
# TAB 2 - ML OUTPUTS
# -------------------------
with tab2:
    st.subheader("Churn Predictions")
    st.dataframe(df[["customerID", "Churn_Probability", "Cluster"]].head(50),
                 use_container_width=True)

# -------------------------
# TAB 3 - VISUALS
# -------------------------
with tab3:
    st.subheader("Churn Distribution")

    fig1, ax1 = plt.subplots()
    ax1.hist(df["Churn"], bins=2)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["No Churn", "Churn"])
    st.pyplot(fig1)

    st.subheader("Monthly Charges vs Churn Risk")

    fig2, ax2 = plt.subplots()
    ax2.scatter(df["MonthlyCharges"], df["Churn_Probability"], alpha=0.4)
    ax2.set_xlabel("Monthly Charges")
    ax2.set_ylabel("Churn Probability")
    st.pyplot(fig2)