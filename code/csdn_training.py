# 循环几次数据，损失就很小了，记下循环损失为nan
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler  # 导入归一化模块
import time

feature_number = 1  # 设置特征数目
out_prediction = 1  # 设置输出数目
learning_rate = 0.00001  # 设置学习率
epochs = 1000  # 设置训练代数


plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


'''导入数据'''

csv_path = 'housing.csv'
housing = pd.read_csv(csv_path)
x_pd, y_pd = housing.iloc[:, 7:8], housing.iloc[:, -2:-1]

'''对每列（特征）归一化'''


# # feature_range控制压缩数据范围，默认[0,1]
# scaler = MinMaxScaler(feature_range=[0, 1])  # 实例化，调整0,1的数值可以改变归一化范围
#
# X = scaler.fit_transform(x_pd)  # 将标签归一化到0,1之间
# Y = scaler.fit_transform(y_pd)  # 将特征归于化到0,1之间

'''对每列数据执行标准化'''

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()  # 实例化
X = scaler.fit_transform(x_pd)  # 标准化特征
Y = scaler.fit_transform(y_pd)  # 标准化标签

# x = scaler.inverse_transform(X) # 这行代码可以将数据恢复至标准化之前


'''划分数据集'''


X = torch.tensor(X, dtype=torch.float32)  # 将数据集转换成torch能识别的格式
Y = torch.tensor(Y, dtype=torch.float32)

torch_dataset = torch.utils.data.TensorDataset(X, Y)  # 组成torch专门的数据库

# 划分训练集测试集与验证集
torch.manual_seed(seed=2022)  # 设置随机种子分关键，不然每次划分的数据集都不一样，不利于结果复现
train_validaion, test = torch.utils.data.random_split(
    torch_dataset,
    [14448, 6192])  # 先将数据集拆分为训练集+验证集（共450组），测试集（56组） # hyd 总长度一定要与数据集中数字数据长度一致 14448+6192 = 20640
train, validation = torch.utils.data.random_split(train_validaion, [14448, 0])  # 再将训练集+验证集拆分为训练集400，测试集50

# 再将训练集划分批次，每batch_size个数据一批（测试集与验证集不划分批次）
train_data = torch.utils.data.DataLoader(train,
                                         batch_size=32,
                                         shuffle=True)

'''训练部分'''




class Model(torch.nn.Module):
    def __init__(self, n_feature, n_output):  # n_feature为特征数目，这个数字不能随便取,n_output为特征对应的输出数目，也不能随便取
        self.n_feature = n_feature
        self.n_output = n_output
        super(Model, self).__init__()
        self.input_layer = torch.nn.Linear(self.n_feature, 20)  # 输入层
        self.hidden1 = torch.nn.Linear(20, 16)  # 1类隐藏层
        self.hidden2 = torch.nn.Linear(16, 10)  # 2类隐藏
        self.predict = torch.nn.Linear(10, self.n_output)  # 输出层
        self.ReLU = torch.nn.ReLU()  # hyd 激活函数

    def forward(self, x):
        '''定义前向传递过程'''
        out = self.ReLU(self.input_layer(x))
        out = self.ReLU(self.hidden1(out))
        out = self.ReLU(self.hidden2(out))
        out = self.predict(out)  # 回归问题最后一层不需要激活函数
        # 除去feature_number与out_prediction不能随便取，隐藏层数与其他神经元数目均可以适当调整以得到最佳预测效果
        return out


model = Model(n_feature=feature_number, n_output=out_prediction)  # 这里直接确定了隐藏层数目以及神经元数目，实际操作中需要遍历

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#optimizer = optim.Adam(model.parameters(), learning_rate)  # 使用Adam算法更新参数
criterion = torch.nn.MSELoss(reduction='mean')  # 误差计算公式，回归问题采用均方误差

loss_array = []
for epoch in range(epochs):  # 整个数据集迭代次数
    for batch_idx, (data, target) in enumerate(train_data):
        logits = model.forward(data)  # 前向计算结果（预测结果）
        loss = criterion(logits, target)  # 计算损失
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 后向传递过程
        optimizer.step()  # 优化权重与偏差矩阵
    print('epoch = ', epoch, 'loss = ', loss.item())
    loss_array.append(loss.item())

print('run time =', time.process_time(), 's')

epoch_array = [i for i in range(epochs)]
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(epoch_array, loss_array, 'r', label='loss')
ax.set_xlabel('epoch')
ax.set_ylabel('loss')
fig.savefig('fig.png')
plt.show()
    # logit = []  # 这个是验证集，可以根据验证集的结果进行调参，这里根据验证集的结果选取最优的神经网络层数与神经元数目
    # target = []
    # model.eval()  # 启动测试模式
    # for data, targets in validation:  # 输出验证集的平均误差
    #     logits = model.forward(data).detach().numpy()
    #     targets = targets.detach().numpy()  # hyd .detach()返回一个new Tensor，只不过不再有梯度,再加.numpy()转化为numpy的格式
    #     target.append(targets[0])
    #     logit.append(logits[0])
    # average_loss = criteon(torch.tensor(logit), torch.tensor(target))
    # print(f'\nTrain Epoch:{epoch} for the Average loss of VAL')

####################### 3 测试模型及可视化##############################


#
# prediction = []
# test_y = []
# model.eval()  # 启动测试模式
# for test_x, test_ys in test:
#     predictions = model(test_x)
#     predictions = predictions.detach().numpy()
#     prediction.append(predictions[0])
#     test_ys.detach().numpy()
#     test_y.append(test_ys[0])
# prediction = scaler.inverse_transform(np.array(prediction).reshape(
#     -1, 1))  # 将数据恢复至归一化之前
# test_y = scaler.inverse_transform(np.array(test_y).reshape(-1, 1))
# # 均方误差计算
# test_loss = criterion(torch.tensor(prediction, dtype=torch.float32), torch.tensor(test_y, dtype=torch.float32))
# print('测试集均方误差：', test_loss.detach().numpy())
#
# # 可视化
# plt.figure()
# plt.scatter(test_y, prediction, color='red')
# plt.plot([0, 52], [0, 52], color='black', linestyle='-')
# plt.xlim([-0.05, 52])
# plt.ylim([-0.05, 52])
# plt.xlabel('true')
# plt.ylabel('prediction')
# plt.title('true vs prection')
# plt.show()



