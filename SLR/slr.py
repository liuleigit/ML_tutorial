#coding: utf-8
#note : 简单线性回归。利用预测值与真实值差的平方作为评价函数,利用求导可以得到斜率和截距
#http://blog.csdn.net/leiting_imecas/article/details/53000733
import numpy as np
def fitSLR(x, y):
    n = len(x)
    dinominator = 0
    numerator = 0
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    for i in range(0, n):
        numerator += (x[i] - x_mean)*(y[i] - y_mean)
        dinominator += (x[i] - x_mean)**2
    print "numerator:", numerator
    print "dinominator", dinominator
    b1 = numerator/float(dinominator)
    b0 = y_mean/float(x_mean)

    return b0, b1

def predict(x, b0, b1):
    return b0 + x*b1
x = [1, 3, 2, 1, 3]
y = [14, 24, 18, 17, 27]
b0, b1 = fitSLR(x, y)
print "intercept:", b0, " slope:", b1
y_test = predict(6, b0, b1)
print "y_test:", y_test