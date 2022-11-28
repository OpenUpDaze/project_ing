import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

# 读取数据集
datasets_X = []
datasets_Y = []
fr = open('house_price.csv', 'r', encoding='utf-8')
lines = fr.readlines()
for line in lines:
    items = line.strip().split(',')
    datasets_X.append(int(float(items[0])))
    datasets_Y.append(int(float(items[1])))

length = len(datasets_X)
datasets_X = np.array(datasets_X).reshape([length, 1])
datasets_Y = np.array(datasets_Y)

minX = min(datasets_X)
maxX = max(datasets_X)
X = np.arange(minX, maxX).reshape([-1, 1])

poly_reg = PolynomialFeatures(degree=2)  # degree=2表示建立datasets_X的二次多项式特征X_poly。
X_poly = poly_reg.fit_transform(datasets_X)  # 使用PolynomialFeatures构造x的二次多项式X_poly
lin_reg_2 = linear_model.LinearRegression()
lin_reg_2.fit(X_poly, datasets_Y)  # 然后创建线性回归，使用线性模型（linear_model）学习X_poly和y之间的映射关系

print(X_poly)
print(lin_reg_2.predict(poly_reg.fit_transform(X)))
print('Coefficients:', lin_reg_2.coef_)  # 查看回归方程系数(k)
print('intercept:', lin_reg_2.intercept_)  ##查看回归方程截距(b)
print('the model is y={0}+({1}*x)+({2}*x^2)'.format(lin_reg_2.intercept_, lin_reg_2.coef_[0], lin_reg_2.coef_[1]))
# 图像中显示
plt.scatter(datasets_X, datasets_Y, color='red')  # scatter函数用于绘制数据点，这里表示用红色绘制数据点；
# plot函数用来绘制回归线，同样这里需要先将X处理成多项式特征；
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color='blue')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()