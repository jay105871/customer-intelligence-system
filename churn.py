import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# STEP 1: LOAD DATA
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# STEP 2: CLEAN DATA
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

# STEP 3: CHANGE YES/NO INTO 1/0
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# STEP 4: REMOVE CUSTOMER ID
df = df.drop("customerID", axis=1)

# STEP 5: CONVERT TEXT TO NUMBERS
df = pd.get_dummies(df)

# STEP 6: SPLIT DATA INTO INPUT AND OUTPUT
X = df.drop("Churn", axis=1)
y = df["Churn"]

# STEP 7: TRAIN AND TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# STEP 8: CREATE MODEL
model = RandomForestClassifier()

# STEP 9: TRAIN MODEL
model.fit(X_train, y_train)

# STEP 10: MAKE PREDICTIONS
predictions = model.predict(X_test)

# STEP 11: CHECK ACCURACY
print("Accuracy:", accuracy_score(y_test, predictions))