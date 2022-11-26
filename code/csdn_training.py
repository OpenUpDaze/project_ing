import pandas as pd
'''导入数据'''
data = pd.read_excel('波士顿房价预测.xlsx',header=None,index_col=None)  # 一共506组数据，每组数据13个特征，13个特征对应一个输出
x = data.loc[:, 0:12]  # 将特征数据存储在x中，表格前13列为特征,
y = data.loc[:, 13:13]  # 将标签数据存储在y中，表格最后一列为标签




'''对每列数据执行标准化'''
 
from sklearn.preprocessing import StandardScaler
 
scaler = StandardScaler()  # 实例化
X = scaler.fit_transform(x)  # 标准化特征
Y = scaler.fit_transform(y)  # 标准化标签
 
# x = scaler.inverse_transform(X) # 这行代码可以将数据恢复至标准化之前


'''划分数据集'''
import torch
 
X = torch.tensor(X, dtype=torch.float32)  # 将数据集转换成torch能识别的格式
Y = torch.tensor(Y, dtype=torch.float32)
torch_dataset = torch.utils.data.TensorDataset(X, Y)  # 组成torch专门的数据库
batch_size = 6  # 设置批次大小
 
# 划分训练集测试集与验证集
torch.manual_seed(seed=2021) # 设置随机种子分关键，不然每次划分的数据集都不一样，不利于结果复现
train_validaion, test = torch.utils.data.random_split(
    torch_dataset,
    [450, 56],
)  # 先将数据集拆分为训练集+验证集（共450组），测试集（56组）
train, validation = torch.utils.data.random_split(
    train_validaion, [400, 50])  # 再将训练集+验证集拆分为训练集400，测试集50
 
# 再将训练集划分批次，每batch_size个数据一批（测试集与验证集不划分批次）
train_data = torch.utils.data.DataLoader(train,
                                         batch_size=batch_size,
                                         shuffle=True)


'''训练部分'''
import torch.optim as optim
 
 
feature_number = 13  # 设置特征数目
out_prediction = 1  # 设置输出数目
learning_rate = 0.01  # 设置学习率
epochs = 50  # 设置训练代数
 
 
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_output, n_neuron1, n_neuron2,n_layer):  # n_feature为特征数目，这个数字不能随便取,n_output为特征对应的输出数目，也不能随便取
        self.n_feature=n_feature
        self.n_output=n_output
        self.n_neuron1=n_neuron1
        self.n_neuron2=n_neuron2
        self.n_layer=n_layer
        super(Net, self).__init__()
        self.input_layer = torch.nn.Linear(self.n_feature, self.n_neuron1) # 输入层
        self.hidden1 = torch.nn.Linear(self.n_neuron1, self.n_neuron2) # 1类隐藏层    
        self.hidden2 = torch.nn.Linear(self.n_neuron2, self.n_neuron2) # 2类隐藏
        self.predict = torch.nn.Linear(self.n_neuron2, self.n_output) # 输出层
 
    def forward(self, x):
        '''定义前向传递过程'''
        out = self.input_layer(x)
        out = torch.relu(out) # 使用relu函数非线性激活
        out = self.hidden1(out)
        out = torch.relu(out)
        for i in range(self.n_layer):
            out = self.hidden2(out)
            out = torch.relu(out) 
        out = self.predict( # 回归问题最后一层不需要激活函数
            out
        )  # 除去feature_number与out_prediction不能随便取，隐藏层数与其他神经元数目均可以适当调整以得到最佳预测效果
        return out
 
net = Net(n_feature=feature_number,
                      n_output=out_prediction,
                      n_layer=1,
                      n_neuron1=20,
                      n_neuron2=20) # 这里直接确定了隐藏层数目以及神经元数目，实际操作中需要遍历
optimizer = optim.Adam(net.parameters(), learning_rate)  # 使用Adam算法更新参数
criteon = torch.nn.MSELoss()  # 误差计算公式，回归问题采用均方误差
 
for epoch in range(epochs):  # 整个数据集迭代次数
    net.train() # 启动训练模式
    for batch_idx, (data, target) in enumerate(train_data):
        logits = net.forward(data)  # 前向计算结果（预测结果）
        loss = criteon(logits, target)  # 计算损失
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 后向传递过程
        optimizer.step()  # 优化权重与偏差矩阵
 
    logit = []  # 这个是验证集，可以根据验证集的结果进行调参，这里根据验证集的结果选取最优的神经网络层数与神经元数目
    target = []
    net.eval() # 启动测试模式
    for data, targets in validation:  # 输出验证集的平均误差
        logits = net.forward(data).detach().numpy()
        targets=targets.detach().numpy()
        target.append(targets[0])
        logit.append(logits[0])
    average_loss =  criteon(torch.tensor(logit), torch.tensor(target))
    print('\nTrain Epoch:{} for the Average loss of VAL')


####################### 3 测试模型及可视化##############################
    
import matplotlib.pyplot as plt
import numpy as np
prediction = []
test_y = []
net.eval() # 启动测试模式
for test_x, test_ys in test:
    predictions = net(test_x)
    predictions=predictions.detach().numpy()
    prediction.append(predictions[0])
    test_ys.detach().numpy()
    test_y.append(test_ys[0])
prediction = scaler.inverse_transform(np.array(prediction).reshape(
    -1, 1))  # 将数据恢复至归一化之前
test_y = scaler.inverse_transform(np.array(test_y).reshape(-1, 1))
# 均方误差计算
test_loss = criteon(torch.tensor(prediction ,dtype=torch.float32), torch.tensor(test_y, dtype=torch.float32))
print('测试集均方误差：',test_loss.detach().numpy())
 
# 可视化
plt.figure()
plt.scatter(test_y, prediction, color='red')
plt.plot([0, 52], [0, 52], color='black', linestyle='-')
plt.xlim([-0.05, 52])
plt.ylim([-0.05, 52])
plt.xlabel('true')
plt.ylabel('prediction')
plt.title('true vs prection')
plt.show()