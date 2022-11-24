import torch
import pandas as pd
import matplotlib.pyplot as plt
######test####
###呵呵##
csv_path = 'housing.csv'
housing = pd.read_csv(csv_path)
x_pd, y_pd = housing.iloc[:, -3:-2], housing.iloc[:, -2:-1]

x = torch.tensor(x_pd.values)
y = torch.tensor(y_pd.values)



# x轴数据集中表头值如房价，纵轴表示这个值出现的次数
housing.hist(bins=50,figsize=(20,15))
plt.show()

