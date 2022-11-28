import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model

'''一元线性回归'''
x = np.linspace(3, 6, 40)
y = 3 * x + 22
y += np.random.rand(40)  # 给数据加点噪声

x, y = x[:, None], y[:, None]  # 因为fit函数需要x和y为矩阵，所以用这样的方式对x和y升维

model = linear_model.LinearRegression()

model.fit(x, y)

# 查看预测结果
x_ = [[3], [4], [5], [6]]
y_ = model.predict(x_)
print(y_)

# 查看w和b，并绘制拟合的直线
w, b = model.coef_, model.intercept_
print("w值为:", model.coef_)
print("b截距值为:", model.intercept_)
y_ = w * x + b  # 拟合的函数
# 数据集绘制,散点图，图像满足函假设函数图像
plt.scatter(x, y)
plt.plot(x, y_, color="red", linewidth=3.0, linestyle="-")
plt.legend(["Data", "func"], loc=0)
plt.show()

