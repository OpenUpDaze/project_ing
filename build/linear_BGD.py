import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


train_data_path = r'F:\22postgraduate_class\machine_learning\Linear_by_torch\regress_data1.csv'
data = pd.read_csv(train_data_path)
x = data.iloc[:,:-1] # X是所有行，去掉最后一列
y = data.iloc[:,[-1]] # y是所有行，最后一列

x = np.matrix(x.values)  # 其中X.values的数据类型为numpy.ndarray，而括号中的X的数据类型为DataFrame
y = np.matrix(y.values)
w = np.matrix(np.array([0,0]))



x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]])

x_array = x_data.tolist()
y_array = y_data.tolist()



def forward(x):
    return x * w


def cost(x_array,y_array):
    cost = 0
    for x, y in zip(x_array,y_array):
        cost += (forward(x) - y)**2
    return cost/len(x_array)


def gradient(x_array, y_array):
    gra = 0
    for x, y in zip(x_array,y_array):
        gra += (forward(x) - y) * x
    return gra / len(x_array)

w = 0.5
a = 0.01
for epoch in range(1000):
    cost = cost(x_array, y_array)
    w = w - a*gradient(x_array, y_array)
    print(cost)











