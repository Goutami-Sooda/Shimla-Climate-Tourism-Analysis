import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your LULC data
# Expected columns: ['Year', 'Cropland %', 'Forest %', 'Urban %', ...]
df = pd.read_csv("C:/Users/sooda/Climate Analysis Shimla/project/LULC Analysis/lulc_tourism_data.csv")
df.columns = df.columns.str.strip()

df.set_index(pd.to_datetime(df['Year'], format='%Y'))[['Cropland %', 'Forest %', 'Urban %']].plot(marker='o')
plt.title("Annual LULC Trends (2008–2022)")
plt.ylabel("Percentage")
plt.grid(True)
plt.show()
