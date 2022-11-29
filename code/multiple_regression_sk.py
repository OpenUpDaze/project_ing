import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# 二次回归
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# x从-3 - 3均匀取值
# x = np.random.uniform(-3, 3 ,size=100)
# X = x.reshape(-1, 1)  #reshape(-1,1)转换成1列
# y是二次方程
x = np.linspace(-3, 3, 100)
X = x.reshape(-1, 1)
y = 0.3 * x ** 2 + 2 * x + 1 + np.random.normal(0, 1, size=100)  # 正态分布的均值，正态分布的标准差
print('x= ', x)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x, y)
ax.set_title('原始数据', fontsize=18)
plt.show()


# 实例化线性模型
lr = LinearRegression()
lr.fit(X, y)
y_predict1 = lr.predict(X)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x, y)
ax.set_title('线性回归', fontsize=18)
ax.plot(x, y_predict1,'r')
plt.show()



poly = PolynomialFeatures(degree=2)
# degree=2 生成2次特征，可以调整

poly.fit(X)
X2 = poly.transform(X)

print('X2 shape ', X2.shape)
print(X2[0:5, :])

# 继续使用线性模型
lr.fit(X2, y)
y_predict2 = lr.predict(X2)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x, y)
ax.set_title('多项式回归', fontsize=18)
ax.plot(x, y_predict2, 'r')
plt.show()

