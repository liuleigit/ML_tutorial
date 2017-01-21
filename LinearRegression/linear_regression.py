# -*- coding: utf-8 -*-
# @bref :
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import axes3d

pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_seq_items', None)

import seaborn as sns
sns.set_context('notebook')
sns.set_style('white')

data = np.loadtxt('linear_regression_data1.txt', delimiter=',')
#np.ones是构造列向量。 data[:, 0]表示取data所有行的第0列
X = np.c_[np.ones(data.shape[0]), data[:, 0]]
#data[:,1] = [1, 4] ,则np.c_[data[:,1]] = [[1], [4]]
y = np.c_[data[:,1]]
#plt.scatter(X[:,1], y, s=30, c='r', marker='x', linewidths=1)
#plt.xlim(4,24)
#plt.xlabel('Population of City in 10,000s')
#plt.ylabel('Profit in $10,000s')
#plt.show()

#计算损失函数
def computeCost(X, y, theta=[[0], [0]]):
    m = y.size
    J = 0
    h = X.dot(theta)
    print X
    print y
    print h
    J = 1.0/(2*m)*(np.sum(np.square(h - y)))
    return J

print computeCost(X, y)

#梯度下降
def gradientDescent(X, y, theta=[[0], [0]], alpha=0.01, num_iters = 1500):
    m = y.size
    J_history = np.zeros(num_iters)
    for iter in np.arange(num_iters):
        h = X.dot(theta)
        theta = theta - alpha*(1.0/m)*(X.T.dot(h-y))
        J_history[iter] = computeCost(X, y, theta)
    return(theta, J_history)


theta, Cost_J = gradientDescent(X, y)
print type(Cost_J)
print('theta: ', theta.ravel())

#plt.plot(Cost_J)
#plt.ylabel('Cost_J')
#plt.xlabel('Iterations')
#plt.show()

xx = np.arange(5, 23)
yy = theta[0] + theta[1]*xx

#画出自己写的梯度下降的线性回归
plt.scatter(X[:, 1], y, s=30, c='r', marker='x', linewidths=1) #画出样本点
plt.plot(xx,yy, label='Linear regression (Gradient descent)')
#对比scikit-learn 中的线性回归对比
regr = LinearRegression()
# X[:, 1] = [6.11, 5.52...];    reshape(-1, 1)后是[[6.11], [5.52], ...]
regr.fit(X[:, 1].reshape(-1,1), y.ravel())
plt.plot(xx, regr.intercept_ + regr.coef_*xx, label='Linear regression (scikit learn GLM)')

plt.xlim(4,24)
plt.xlabel('Population of city in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.legend(loc=4)
plt.show()





