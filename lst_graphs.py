import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas.plotting as pd_plotting

df=pd.read_csv("D:\\Downloads\\climate_csv_files\\Shimla_monthly.csv")

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