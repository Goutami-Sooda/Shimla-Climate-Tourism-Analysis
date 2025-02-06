import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas.plotting as pd_plotting
import scipy 

df=pd.read_csv("D:\\Downloads\\climate_csv_files\\Shimla_monthly.csv")


# Convert the 'time' column to datetime format
df['time'] = pd.to_datetime(df['time'], format='%d-%m-%Y', errors='coerce')

# Plotting the Solar Zenith Angle vs LST
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.scatter(df['solze'], df['lst'], color='orange')
plt.title('Solar Zenith Angle vs LST')
plt.xlabel('Solar Zenith Angle (degrees)')
plt.ylabel('Land Surface Temperature (K)')
plt.grid(True)

# Plotting the Solar Azimuth Angle vs LST
plt.subplot(2, 2, 2)
plt.scatter(df['solaz'], df['lst'], color='green')
plt.title('Solar Azimuth Angle vs LST')
plt.xlabel('Solar Azimuth Angle (degrees)')
plt.ylabel('Land Surface Temperature (K)')
plt.grid(True)

# Plotting the LST Uncertainty vs LST
plt.subplot(2, 2, 3)
plt.scatter(df['lst_uncertainty'], df['lst'], color='blue')
plt.title('LST Uncertainty vs LST')
plt.xlabel('LST Uncertainty (K)')
plt.ylabel('Land Surface Temperature (K)')
plt.grid(True)

# Plotting the Local Atmospheric Uncertainty vs LST
plt.subplot(2, 2, 4)
plt.scatter(df['lst_unc_loc_atm'], df['lst'], color='red')
plt.title('Local Atmospheric Uncertainty vs LST')
plt.xlabel('Local Atmospheric Uncertainty (K)')
plt.ylabel('Land Surface Temperature (K)')
plt.grid(True)

# Adjust layout to ensure the plots do not overlap
plt.tight_layout()
plt.show()


# Create the line plot
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
df['time'] = pd.to_datetime(df['time'], format='%d-%m-%Y', errors='coerce')
# Extract the year from the 'time' column
df['year'] = df['time'].dt.year

# Create the line plot
plt.figure(figsize=(12, 6))
plt.plot(df['time'], df['lst'], marker='o', linestyle='-')

# Title and labels
plt.title('Line Plot of LST over Time')
plt.xlabel('Time')
plt.ylabel('LST')

# # Set x-axis ticks to display only the years, remove month/day information
plt.xticks(df['time'][::12], labels=df['year'][::12], rotation=90)  # Show year every 12th month for clarity

# Add gridlines for better readability
plt.grid(True)

# Adjust layout to ensure everything fits nicely
plt.tight_layout()

# Show the plot
plt.show()




# Drop rows with invalid dates
df = df.dropna(subset=['time'])

# Line Plot of LST over Time with Improved X-Axis Labels
plt.figure(figsize=(12, 6))
plt.plot(df['time'], df['lst'], marker='o', linestyle='-')
plt.title('Line Plot of LST over Time')
plt.xlabel('Time')
plt.ylabel('LST')
plt.xticks(rotation=90) 
plt.grid()
plt.show()









#corr map
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure the 'time' column is in datetime format and any non-numeric columns are excluded
df['time'] = pd.to_datetime(df['time'], errors='coerce')

# Select only numeric columns for correlation calculation
numeric_df = df.select_dtypes(include=['number'])

# Compute the Kendall correlation matrix
kendall_corr = numeric_df.corr(method='kendall')

# Display the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(kendall_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Kendall Correlation Map')
plt.tight_layout()
plt.show()



# df['lst_rolling_mean'] = df['lst'].rolling(window=12).mean()
# plt.plot(df['time'], df['lst'], alpha=0.5, label='Original')
# plt.plot(df['time'], df['lst_rolling_mean'], color='red', label='Rolling Mean')


# Drop rows with invalid dates
df = df.dropna(subset=['time'])

# Check the range of years in the dataset
print(df['year'].min(), df['year'].max())

# Group data by year and calculate the average LST
yearly_avg_lst = df.groupby('year')['lst'].mean().reset_index()

# Plot the change in LST over the years
plt.figure(figsize=(10, 6))
plt.plot(yearly_avg_lst['year'], yearly_avg_lst['lst'], marker='o', linestyle='-', color='b')
plt.title('Change in Land Surface Temperature (LST) Over the Years (2004-2020)')
plt.xlabel('Year')
plt.ylabel('Average LST')
plt.grid()
plt.xticks(yearly_avg_lst['year'], rotation=90)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()




# Line Plot of LST over Time within a year
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

# Line Plot of LST over Time with Improved X-Axis Labels
plt.figure(figsize=(12, 6))
plt.plot(df['time'], df['lst'], marker='o', linestyle='-')
plt.title('Line Plot of LST over Time')
plt.xlabel('Time')
plt.ylabel('LST')

# Improve the x-axis label display
ax = plt.gca()  # Get the current axis
ax.xaxis.set_major_locator(MaxNLocator(10))  # Show up to 10 evenly spaced x-axis labels
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Format labels as 'Year-Month'

plt.xticks(rotation=45, ha='right')  # Rotate for better readability
plt.grid()
plt.tight_layout()  # Adjust layout to avoid overlaps
plt.show()



#heatmap of seasonal variation
pivot_data = df.pivot_table(index='year', columns='month', values='lst')
sns.heatmap(pivot_data, cmap='coolwarm')

monthly_avg = df.groupby('month')['lst'].mean()
plt.plot(monthly_avg, marker='o', color='blue')


# 1. Histogram of LST Values
plt.figure(figsize=(10, 6))
plt.hist(df['lst'], bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of LST Values')
plt.xlabel('LST')
plt.ylabel('Frequency')
plt.grid(axis='y')
plt.show()

# 2. Strip Plot of LST with Quality Flag
plt.figure(figsize=(10, 6))
sns.stripplot(x='qual_flag', y='lst', data=df, jitter=True, palette='Set2')
plt.title('Strip Plot of LST with Quality Flag')
plt.xlabel('Quality Flag')
plt.ylabel('LST')
plt.grid(axis='y')
plt.show()

# 3. Box Plot of LST Uncertainty
plt.figure(figsize=(10, 6))
sns.boxplot(x='qual_flag', y='lst_uncertainty', data=df, palette='Set1')
plt.title('Box Plot of LST Uncertainty')
plt.xlabel('Quality Flag')
plt.ylabel('LST Uncertainty')
plt.grid(axis='y')
plt.show()

# 4. Scatter Plot Matrix
pd_plotting.scatter_matrix(df[['lst', 'lst_uncertainty', 'qual_flag']],
                            figsize=(10, 10), color='blue', alpha=0.5, diagonal='kde')
plt.suptitle('Scatter Plot Matrix')
plt.show()

# 5. Line Plot of LST over Time
plt.figure(figsize=(12, 6))
plt.plot(df['time'], df['lst'], marker='o', linestyle='-')
plt.title('Line Plot of LST over Time')
plt.xlabel('Time')
plt.ylabel('LST')
plt.xticks(rotation=45)
plt.grid()
plt.show()

