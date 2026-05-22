import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

def train_and_save_surrogate_model(csv_path):
    # 1. Create the output directory
    output_dir = "ml-model"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # 2. Load the Master CSV
    df = pd.read_csv(csv_path)

    # 3. Select Features (Inputs) and Target (Output)
    # Ensure these names match your master_fof_impact_data.csv columns exactly
    features = ['Input_Total_Mass_kg', 'Input_Velocity_km_s', 'Input_Periapsis_km']
    target = 'Output_Bound_Fraction'

    X = df[features]
    y = df[target]

    # 4. Split and Scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Train XGBoost
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        objective='reg:squarederror'
    )
    model.fit(X_train_scaled, y_train)

    # 6. Generate Predictions for the Test Set
    predictions = model.predict(X_test_scaled)

    # 7. Save the Prediction Results to CSV
    # This combines the ground truth with your AI's guess for comparison
    results_df = pd.DataFrame({
        'Actual_Bound_Fraction': y_test,
        'Predicted_Bound_Fraction': predictions,
        'Error': np.abs(y_test - predictions)
    })
    results_df.to_csv(f"{output_dir}/test_predictions.csv", index=False)

    # 8. Save the Model and Scaler
    with open(f"{output_dir}/surrogate_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(f"{output_dir}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # 9. Create and Save the Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.6, color='midnightblue', edgecolors='white')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Physics Prediction')
    plt.xlabel('Actual Bound Fraction (SWIFT SPH)')
    plt.ylabel('Predicted Bound Fraction (XGBoost)')
    plt.title('Surrogate Model Performance')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"{output_dir}/accuracy_plot.png", dpi=300)
    
    print("-" * 50)
    print(f"SUCCESS: All artifacts saved to the '{output_dir}/' folder:")
    print(f" - Model: surrogate_model.pkl")
    print(f" - Scaler: scaler.pkl")
    print(f" - Data: test_predictions.csv")
    print(f" - Plot: accuracy_plot.png")
    print("-" * 50)

if __name__ == "__main__":
    # Point this to your Master CSV file
    train_and_save_surrogate_model("master_training_data.csv")