# -*- coding: utf-8 -*-
# @bref :
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures

pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_seq_items', None)

import seaborn as sns
sns.set_context('notebook')
sns.set_style('white')

def loaddata(file, delimeter):
    data = np.loadtxt(file, delimiter=delimeter)
    print('Dimensions: ', data.shape)
    print(data[1:6, :])
    return(data)

def plotData(data, label_x, label_y, label_pos, label_neg, axes = None):
    neg = data[:, 2] == 0
    pos = data[:, 2] == 1
    if axes == None:
        axes = plt.gca() #gca返回当前的坐标实例
    axes.scatter(data[pos][:, 0], data[pos][:, 1], marker='+', c='k', s=60, linewidth=2, label=label_pos)
    axes.scatter(data[neg][:, 0], data[neg][:, 1], c='y', s=60, label=label_neg)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend(frameon=True, fancybox=True)

data = loaddata('data1.txt', ',')

X = np.c_[np.ones((data.shape[0], 1)), data[:, 0:2]]
y = np.c_[data[:, 2]]
plotData(data, 'Exam 1 score', 'Exam 2 score', 'Pass', 'Fail')
plt.show()

def sigmoid(z):
    return (1 / (1 + np.exp(-z)))

def costFunction(theta, X, y):
    m = y.size
    h = sigmoid(X.dot(theta))
    J = -1.0 * (1.0/m)*(np.log(h).T.dot(y) + np.log(1-h).T.dot(1-y))
    if np.isnan(J[0]):
        return np.inf
    return J[0]


def gradient(theta, X, y):
    m = y.size
    h = sigmoid(X.dot(theta.reshape(-1, 1)))
    grad = (1.0/m)*X.T.dot(h-y)
    return grad.flatten()

initial_theta = np.zeros(X.shape[1])
cost = costFunction(initial_theta, X, y)
grad = gradient(initial_theta, X, y)
print('Cost: \n', cost)
print('Grad: \n', grad)


res = minimize(costFunction, initial_theta, args=(X, y), jac=gradient, options={'maxiter':400})
#print res

def predict(theta, X, threshold=0.5):
    p = sigmoid(X.dot(theta.T)) >= threshold
    return p.astype('int')

#sigmoid(np.array([1, 45, 85]).dot(res.x.T))

print '----------'
print predict(res.x, np.array([1, 45, 85]))

plt.scatter(45, 85, s=60, c='r', marker='v', label='(45, 85)')
plotData(data, 'Exam 1 score', 'Exam 2 score', 'Admitted', 'Not admitted')
x1_min, x1_max = X[:,1].min(), X[:,1].max()
x2_min, x2_max = X[:,2].min(), X[:,2].max()
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0],1)), xx1.ravel(), xx2.ravel()].dot(res.x))
h = h.reshape(xx1.shape)
#contour 画轮廓、等高线、决策边界
plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')
plt.show()

#加正则化项
data2 = loaddata('data2.txt', ',')
y = np.c_[data2[:, 2]]
X =  data2[:, 0:2]

#plotData(data2, 'Microchip Test 1', 'Microchip Test 2', 'y = 1', 'y = 0')
#增加一点多项式特征出来(最高6维)
poly = PolynomialFeatures(6)
XX = poly.fit_transform(data2[:, 0:2])  #将两维的数据映射为多维, 这里计算得XX.shape = (118, 28)

#reg 是惩罚项的参数
def costFunctionReg(theta, reg, *args):
    m = y.size
    h = sigmoid(XX.dot(theta))
    J = -1.0*(1.0/m)*(np.log(h).T.dot(y)+np.log(1-h).T.dot(1-y)) + (reg/(2.0*m))*np.sum(np.square(theta[1:]))

    if np.isnan(J[0]):  # nan: not a number
        return(np.inf)  #inf是一个无限大的正数
    return J[0]


def gradientReg(theta, reg, *args):
    m = y.size
    h = sigmoid(XX.dot(theta.reshape(-1, 1)))
    grad = (1.0 / m) * XX.T.dot(h - y) + (reg / m) * np.r_[[[0]], theta[1:].reshape(-1, 1)]
    return (grad.flatten())

initial_theta = np.zeros(XX.shape[1])
#costFunctionReg(initial_theta, 1, XX, y)

fig, axes = plt.subplots(1, 3, sharey=True, figsize=(17, 5))

#决策边界,分别看正则化系数lanbda取不同值得情况
#lambda = 0, 不惩罚; lambda = 1, 正常惩罚; lambda = 100,正则化太激进
#numpy.ravel()和numpy.flatten()都将多维数组降为一维。ravel返回的是原数组的试图(view),它的修改也会改变原数组; flatten返回拷贝
for i, C in enumerate([0.0, 1.0, 100.0]):
    #最优化cost FunctionReg
    res2 = minimize(costFunctionReg, initial_theta, args=(C, XX, y), jac=gradientReg,options={'maxiter':3000})
    #准确率
    accuracy = 100 * sum(predict(res2.x, XX) == y.ravel()) / y.size
    #对X,y 散列绘图
    plotData(data2, 'Microchip Test 1', 'Microchip Test 2', 'y = 1', 'y = 0', axes.flatten()[i])
    # 画出决策边界
    print 'draw decision boundary'
    x1_min, x1_max = X[:,0].min(), X[:,0].max(),
    x2_min, x2_max = X[:,1].min(), X[:,1].max(),
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    h = sigmoid(poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(res2.x))
    h = h.reshape(xx1.shape)
    axes.flatten()[i].contour(xx1, xx2, h, [0.5], linewidths=1, colors='g');
    axes.flatten()[i].set_title('Train accuracy {}% with Lambda = {}'.format(np.round(accuracy, decimals=2), C))

plt.show()





