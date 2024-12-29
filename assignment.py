import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# data = pd.read_csv("AirQualityUCI")
file_path = 'C:/Users/Saarang/Documents/NA-Assignment03/AirQualityUCI.csv' 
data = pd.read_csv(file_path, sep=';', decimal=',', skipfooter=1, engine='python', on_bad_lines='skip')

# Replace the placeholder '-200' with NaN
data.replace(-200, np.nan, inplace=True)
data.head()

# Select the column for interpolation (e.g., Temperature 'T')
column_to_interpolate = 'T'
data[column_to_interpolate] = pd.to_numeric(data[column_to_interpolate], errors='coerce')

# Function to simulate missing data for evaluation
def simulate_missing_data(data, column, missing_ratio=0.1):
    """
    Randomly remove a percentage of data to simulate missing values for evaluation.
    """
    column_data = data[column].copy()
    np.random.seed(42)  # For reproducibility
    missing_indices = np.random.choice(
        column_data.dropna().index,
        size=int(len(column_data.dropna()) * missing_ratio),
        replace=False,
    )
    simulated_data = column_data.copy()
    simulated_data.loc[missing_indices] = np.nan
    return simulated_data, missing_indices

# Simulate missing data
data['Simulated_T'], missing_indices = simulate_missing_data(data, column_to_interpolate)

# Apply interpolation methods
methods = {
    "Linear": lambda x: x.interpolate(method="linear"),
    "Polynomial (Order 2)": lambda x: x.interpolate(method="polynomial", order=2),
    "Spline (Order 3)": lambda x: x.interpolate(method="spline", order=3),
}

interpolated_results = {method: func(data['Simulated_T']) for method, func in methods.items()}

# Plot the results
plt.figure(figsize=(12, 8))

# Plot original data
plt.plot(data[column_to_interpolate], label="Original Data", alpha=0.7)

# Plot simulated missing data
plt.scatter(missing_indices, data.loc[missing_indices, column_to_interpolate],
            color="red", label="Missing Data")

# Plot interpolated results
for method, result in interpolated_results.items():
    plt.plot(result, label=f"Interpolated Data ({method})", alpha=0.7)

plt.title("Comparison of Interpolation Methods")
plt.xlabel("Index")
plt.ylabel(column_to_interpolate)
plt.legend()
plt.grid()
plt.show()

# Evaluate performance (e.g., Mean Squared Error)
def evaluate_interpolation(true_values, interpolated_values, indices):
    """
    Calculate Mean Squared Error (MSE) for interpolated values.
    """
    mse = ((true_values[indices] - interpolated_values[indices]) ** 2).mean()
    return mse

performance = {
    method: evaluate_interpolation(data[column_to_interpolate], result, missing_indices)
    for method, result in interpolated_results.items()
}

print("Performance (MSE):", performance)

