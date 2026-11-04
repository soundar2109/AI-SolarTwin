import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

print("1. Loading your historical solar log dataset...")
# Load your uploaded CSV file
df = pd.read_csv("ai_solartwin_log_2026-10-20_to_2026-11-04.csv")

# 2. Define Features (X) and Target (y) based on your exact CSV columns
X = df[["GHI (W/m²)", "Ambient (°C)", "Cell (°C)"]]
y = df["AI Net (W)"]

print("2. Splitting data into training (80%) and testing (20%) sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("3. Training the Random Forest Regressor model...")
model = RandomForestRegressor(
    n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

print("4. Evaluating model performance...")
predictions = model.predict(X_test)
r2 = r2_score(y_test, predictions)
rmse = mean_squared_error(y_test, predictions, squared=False)

print(f"-> Model R² Score: {r2:.4f}")
print(f"-> Root Mean Squared Error (RMSE): {rmse:.4f}")

print("5. Saving the trained model to disk...")
joblib.dump(model, "solar_ml_model.pkl")
print("Success! Model saved as 'solar_ml_model.pkl'.")
