import pandas as pd
import numpy as np

def generate_climate_data():
    dates = pd.date_range(start='1970-01-01', end='2025-12-31', freq='MS')
    n = len(dates)

    months = dates.month.to_numpy()
    years = dates.year.to_numpy()

    seasonal_effect = 10 * np.sin(2 * np.pi * months / 12)

    years_passed = years - 1970
    trend = years_passed * 0.02

    temp = 15 + seasonal_effect + trend + np.random.normal(0, 0.5, n)

    anomaly_indices = np.random.choice(n, 5, replace=False)
    temp[anomaly_indices] += np.random.choice([-5, 5], 5)

    df = pd.DataFrame({
        "Date": dates,
        "AverageTemperature": temp
    })

    df.to_csv("data/climate_data.csv", index=False)
    print("✅ Climate dataset generated successfully!")

if __name__ == "__main__":
    generate_climate_data()