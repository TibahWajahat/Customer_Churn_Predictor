import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Drop customerID column
df = df.drop("customerID", axis=1)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Replace missing values
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Convert Yes/No columns to numbers
df = df.replace({"Yes":1, "No":0})

# Convert target column
df["Churn"] = df["Churn"].replace({"Yes":1, "No":0})

# Convert remaining categorical columns
df = pd.get_dummies(df)

# Split features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Save column names
joblib.dump(X.columns.tolist(), "columns.pkl")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# Train model
model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)

# Save model
joblib.dump(model, "churn_model.pkl")

print("✅ Model, scaler, and columns saved successfully!")