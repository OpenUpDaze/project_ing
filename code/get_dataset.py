import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

csv_path = 'housing.csv'
housing = pd.read_csv(csv_path)


#############################数据处理###############################
x_pd, y_pd = housing.iloc[:, -3:-2], housing.iloc[:, -2:-1]

x_data = torch.tensor(x_pd.values, dtype=torch.float32)
y_data = torch.tensor(y_pd.values, dtype=torch.float32)

y_data.data /= 500000   # 归一化y，防止数据过大，并且避免建立计算图

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x_data.data, y_data.data)
ax.set_xlabel('收入/万')
ax.set_ylabel('median_house_value')
plt.show()


###########################建立模型###############################
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel,self).__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self,x):
        y_pred = self.linear(x)
        return y_pred
model = LinearModel()


criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    print(epoch,loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


print('w = ',model.linear.weight.item())
print('b = ',model.linear.bias.item())


#---------------------------画图---------------------------#
w = model.linear.weight.item()
b = model.linear.bias.item()
x = np.linspace(0, 25, 100)
f = x * w + b

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, f, 'r', label='预测值')
ax.scatter(x_data.data, y_data.data)
ax.set_xlabel('收入/万')
ax.set_ylabel('median_house_value')
plt.show()




