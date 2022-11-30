# 循环几次数据，损失就很小了，记下循环损失为nan
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler  # 导入归一化模块
import time

feature_number = 3  # 设置特征数目
out_prediction = 1  # 设置输出数目
learning_rate = 0.001  # 设置学习率0.00001
epochs = 1000 # 设置训练次数
Myseed = 2022  # 设置随机种子分关键，不然每次划分的数据集都不一样，不利于结果复现

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


'''导入数据'''

csv_path = 'housing.csv'
housing = pd.read_csv(csv_path)
x_pd, y_pd = housing.iloc[:, 5:8], housing.iloc[:, -2:-1]

'''对每列（特征）归一化'''


# # feature_range控制压缩数据范围，默认[0,1]
# scaler = MinMaxScaler(feature_range=[0, 1])  # 实例化，调整0,1的数值可以改变归一化范围
#
# X = scaler.fit_transform(x_pd)  # 将标签归一化到0,1之间
# Y = scaler.fit_transform(y_pd)  # 将特征归于化到0,1之间

'''对每列数据执行标准化'''

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()                                  # 实例化
X = scaler.fit_transform(x_pd)                        # 标准化特征
Y = scaler.fit_transform(y_pd)                           # 标准化标签

# x = scaler.inverse_transform(X) # 这行代码可以将数据恢复至标准化之前


'''划分数据集'''
import torch

X = torch.tensor(X, dtype=torch.float32) 
Y = torch.tensor(Y, dtype=torch.float32)

torch_dataset = torch.utils.data.TensorDataset(X, Y) 
torch.manual_seed(seed=Myseed)  
train, test = torch.utils.data.random_split(torch_dataset, [14448, 6192])
train_data = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)

'''训练部分'''


class Model(torch.nn.Module):
    def __init__(self, n_feature, n_output): 
        self.n_feature = n_feature
        self.n_output = n_output
        super(Model, self).__init__()
        self.input_layer = torch.nn.Linear(self.n_feature, 20)  
        self.hidden1 = torch.nn.Linear(20, 16)  
        self.hidden2 = torch.nn.Linear(16, 10)  
        self.predict = torch.nn.Linear(10, self.n_output)  
        self.ReLU = torch.nn.ReLU()  

    def forward(self, x):
        out = self.ReLU(self.input_layer(x))
        out = self.ReLU(self.hidden1(out))
        out = self.ReLU(self.hidden2(out))
        out = self.predict(out)  
        return out


model = Model(n_feature=feature_number, n_output=out_prediction)  

criterion = torch.nn.MSELoss(reduction='mean')  
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


loss_array = []
for epoch in range(epochs):  
    for batch_idx, (data, target) in enumerate(train_data):
        pre = model(data)  
        loss = criterion(pre, target) 
        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step()  
    print('epoch = ', epoch, 'loss = ', loss.item())
    loss_array.append(loss.item())

 #  model.eval()  # 启动测试模式
pre_array = []
test_array = []
for test_x, test_ys in test:
    pre = model(test_x)
    pre_array.append(pre.item())
    test_array.append(test_ys.item())

plt.figure()
plt.scatter(test_array, pre_array, color='red', alpha=1/7)
plt.xlabel('true')
plt.ylabel('prediction')
plt.title('true vs prediction')



print('run time =', time.process_time(), 's')

epoch_array = [i for i in range(epochs)]
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(epoch_array, loss_array, 'r', label='loss')
ax.set_xlabel('epoch')
ax.set_ylabel('loss')
fig.savefig('fig.png')
plt.show()




