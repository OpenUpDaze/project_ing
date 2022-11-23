import pandas as pd

csv_path = 'housing.csv'
data = pd.read_csv(csv_path)
data.head()