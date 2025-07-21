import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


data = pd.read_csv("garments_worker_productivity.csv")
print("Dataset loaded successfully!")
print("Shape of dataset:", data.shape)
print(data.head())
# Data Preprocessing
print("\nChecking for missing values...")
print(data.isnull().sum())

# Drop rows with missing values (if any)
data.dropna(inplace=True)

# Convert 'date' column to datetime format and extract month
data['date'] = pd.to_datetime(data['date'])
data['month'] = data['date'].dt.month
data.drop('date', axis=1, inplace=True)

# Encode categorical columns
from sklearn.preprocessing import LabelEncoder
categorical_cols = ['department', 'day', 'quarter']
le = LabelEncoder()

for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

print("\nData after preprocessing:")
print(data.head())

#  Split Data
X = data.drop('actual_productivity', axis=1)
y = data['actual_productivity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Models

print("\nTraining Linear Regression Model...")
lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)

print("\nTraining Random Forest Regressor...")
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

print("\nTraining XGBoost Regressor...")
xgb = XGBRegressor()
xgb.fit(X_train, y_train)
pred_xgb = xgb.predict(X_test)

#  Evaluate Models
def evaluate_model(name, y_test, y_pred):
    print(f"\nModel: {name}")
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))

evaluate_model("Linear Regression", y_test, pred_lr)
evaluate_model("Random Forest", y_test, pred_rf)
evaluate_model("XGBoost", y_test, pred_xgb)

#  Save the best model
import pickle

with open("best_model.pkl", "wb") as f:
    pickle.dump(xgb, f)

print("\n XGBoost model saved as 'best_model.pkl'")
