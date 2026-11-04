import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Load historical meteorological dataset
df = pd.read_csv("solar_historical_data.csv")

# 2. Define Features (X) and Target (y)
X = df[["ghi", "dni", "ambient_temp", "wind_speed"]]
y = df["target_power"]

# 3. Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train the Random Forest Regressor model
model = RandomForestRegressor(
    n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

# 5. Evaluate model performance
predictions = model.predict(X_test)
r2 = r2_score(y_test, predictions)
print(f"Model R² Score: {r2:.4f}")

# 6. Save the trained model to disk
joblib.dump(model, "solar_ml_model.pkl")
print("Model saved successfully as solar_ml_model.pkl")
