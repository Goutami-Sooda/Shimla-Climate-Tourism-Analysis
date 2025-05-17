import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the LULC data and select only the relevant columns
df = pd.read_csv("C:/Users/sooda/Climate Analysis Shimla/project/LULC Analysis/lulc_tourism_data.csv")
df.columns = df.columns.str.strip()

# Select only Year and the three classes
df = df[['Year', 'Cropland %', 'Forest %', 'Urban %']]

# Convert 'Year' to datetime (January 1st of each year)
df['Date'] = pd.to_datetime(df['Year'], format='%Y')
df.set_index('Date', inplace=True)
df = df.drop(columns=['Year'])

# Create monthly datetime index from Jan 2008 to Dec 2024
monthly_index = pd.date_range(start='2008-01-01', end='2024-01-01', freq='MS')

# Reindex to monthly and interpolate linearly
df_monthly = df.reindex(monthly_index)
df_monthly_interpolated = df_monthly.interpolate(method='linear')

# Reset index and clean up
df_final = df_monthly_interpolated.reset_index()
df_final.rename(columns={'index': 'Date'}, inplace=True)

# Keep only required columns and round to 4 decimal places
df_final = df_final[['Date', 'Cropland %', 'Forest %', 'Urban %']]
df_final[['Cropland %', 'Forest %', 'Urban %']] = df_final[['Cropland %', 'Forest %', 'Urban %']].round(4)

# Save to CSV
df_final.to_csv("monthly_lulc_interpolated_only_2008_2024.csv", index=False)

# Optional: Plot the results
df_final.set_index('Date')[['Cropland %', 'Forest %', 'Urban %']].plot(figsize=(12,6), title='LULC Trends (Interpolated Only)')
plt.ylabel('Percentage')
plt.grid()
plt.tight_layout()
plt.show()
