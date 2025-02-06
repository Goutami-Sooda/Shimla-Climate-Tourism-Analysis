import pandas as pd
import numpy as np

# Create the dataset for tourist footfall from 2008 to 2022
data = {
    "Year": [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
    "Indian Tourists footfall": [2061539, 2175314, 2485564, 2818270, 3195332, 2992991, 3193637, 3261152, 3416629, 3318829, 2872013, 3030246, 599202, 951792, 2565269],
    "Foreign Tourists footfall": [112917, 108981, 127737, 134167, 158671, 164006, 156235, 154155, 165476, 162168, 123000, 132608, 21111, 825, 10698],
    "Total Tourists footfall": [2174456, 2284295, 2613301, 2952437, 3354003, 3156997, 3349872, 3415307, 3582105, 3480997, 2995013, 3162854, 620313, 952617, 2575967]
}

# Convert to DataFrame
df_tourism = pd.DataFrame(data)

# Calculate the yearly growth rate for 'Total Tourists footfall' (2009 to 2020)
df_tourism['Growth Rate'] = df_tourism['Total Tourists footfall'].pct_change()

# Calculate the average growth rate from 2009 to 2020 (excluding 2020 as it has a sharp drop)
average_growth_rate = df_tourism.iloc[1:]['Growth Rate'].mean()

# Extrapolate backwards to fill missing years (2004-2007) assuming the growth rate was similar
# Start with the 2008 value and use the reverse growth rate to estimate previous years
df_tourism_extended = df_tourism.copy()

for year in range(2007, 2003, -1):
    last_value = df_tourism_extended[df_tourism_extended['Year'] == year + 1]['Total Tourists footfall'].values[0]
    estimated_value = last_value / (1 + average_growth_rate)  # Reverse growth to get value for the previous year
    df_tourism_extended = df_tourism_extended._append({
        'Year': year,
        'Indian Tourists footfall': np.nan,  # Placeholder, as we do not have specific breakdown
        'Foreign Tourists footfall': np.nan,  # Placeholder
        'Total Tourists footfall': estimated_value,
        'Growth Rate': np.nan  # Not needed for extrapolated rows
    }, ignore_index=True)

# Sort the dataframe by year in ascending order
df_tourism_extended = df_tourism_extended.sort_values(by="Year").reset_index(drop=True)

# Display the result
print(df_tourism_extended)



import pandas as pd
import numpy as np

# Load Monthly LST Data
df_lst = pd.read_csv("D:\\Downloads\\climate_csv_files\\Shimla_monthly.csv")  # Your LST dataset (2 values per month)


# # Load Yearly Tourist Footfall Data
# df_tourism = pd.read_csv("D:\\Downloads\\climate_csv_files\\Shimla2_combined.csv")  # Has 'year' and 'Total Tourists footfall'

# # Drop duplicate years, keeping the first occurrence
# df_tourism = df_tourism.drop_duplicates(subset=['year'], keep='first')

# Merge Tourism Footfall into Monthly Data
df_lst['Total Tourists footfall'] = df_lst['year'].map(df_tourism_extended.set_index('Year')['Total Tourists footfall'])

# 1️⃣ **Even Split of Yearly Footfall**
df_lst["Total Tourists footfall"] = df_lst["Total Tourists footfall"] / 12  # Divide by 12 months

# 2️⃣ **Distribute per Month with Seasonal Weights (Optional)**
# Assume an estimated seasonality pattern (higher in May-June, Dec-Jan)
seasonal_weights = {
    1: 0.10, 2: 0.08, 3: 0.07, 4: 0.06, 5: 0.05, 6: 0.04,
    7: 0.04, 8: 0.05, 9: 0.06, 10: 0.09, 11: 0.11, 12: 0.12
}

# Normalize Weights (Ensure sum = 1)
total_weight = sum(seasonal_weights.values())
seasonal_weights = {k: v / total_weight for k, v in seasonal_weights.items()}

# Apply Weights to Adjust Monthly Tourist Footfall
df_lst["Total Tourists footfall"] = df_lst.apply(
    lambda row: row["Total Tourists footfall"] * seasonal_weights[row["month"]], axis=1
)

# Save Updated Dataset
df_lst.to_csv("monthly_data_with_tourism.csv", index=False)
print(df_lst.tail())


