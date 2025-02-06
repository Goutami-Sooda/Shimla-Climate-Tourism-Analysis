import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

lst_tourist_yearly=pd.read_csv("D:\\Downloads\\climate_csv_files\\Shimla2_combined.csv")



# Select the columns of interest
columns_of_interest = ['year', 'lst', 'Indian Tourists footfall', 'Foreign Tourists footfall', 'Total Tourists footfall']

# Compute the correlation matrix
correlation_matrix = lst_tourist_yearly[columns_of_interest].corr()

# Plot the correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation between Year, LST, and Tourist Footfall')
plt.show()