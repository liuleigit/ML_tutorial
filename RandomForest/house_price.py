# -*- coding: utf-8 -*-
# @bref :
import numpy as np
import pandas as pd

train_df = pd.read_csv('data/train.csv', index_col=0)
test_df = pd.read_csv('data/test.csv', index_col=0)

#对分类器来说,label最好是平滑的,符合高斯分布的。 我们会先把label'平滑化"(正太化)
# 我们使用了log1p, 也就是log(x+1),避免了复值得问题
# 记得预测结果是将数据变回去
prices = pd.DataFrame({"price":train_df["SalePrice"], "log(price + 1)":np.log1p(train_df["SalePrice"])})
#prices['price'].hist().get_figure().savefig('/Users/liulei/prices.png')
prices['log(price + 1)'].hist().get_figure().savefig('/Users/liulei/log1_prices.png') #直方图
#合并数据   方便同时将train 和test统一做预处理
y_train = np.log1p(train_df.pop('SalePrice'))
all_df = pd.concat((train_df, test_df), axis=0)

print all_df.shape

#变量转化, 统一不方便或者补unity的数据
all_df['MSSubClass'] = all_df['MSSubClass'].astype(str)
#--------------把category 变量转变成numerical表达形式
#将category 的变量变成numerical时, 数字本身的大小其实是没有意义的,所以用数字表示并给模型学习会麻烦,所以我们使用one-hot
#pandas 自带get_dummies直接一键将categorical的变量One-hot,注意不处理数值型变量
print pd.get_dummies(all_df['MSSubClass'], prefix='MSSubClass').head()
#把所有的category都one-hot
all_dummy_df = pd.get_dummies(all_df)
print all_dummy_df.head()

#-----------------------处理numerical变量,例如有空缺。 需要知道空缺的意思,是否需要填充
#这里我们使用平均值来填满。  有时可能填充0
print all_dummy_df.isnull().sum().sort_values(ascending=False).head(10)
mean_cols = all_dummy_df.mean()
print mean_cols.head(10)
all_dummy_df = all_dummy_df.fillna(mean_cols)

#-----------------------标准和numerical数据
#这一步并不是必要，但是得看你想要用的分类器是什么。一般来说，regression的分类器都比较傲娇，最好是把源数据给放在一个标准分布内。不要让数据间的差距太大。
#当然不需要把One-Hot的那些0/1数据给标准化。我们的目标应该是那些本来就是numerical的数据
#计算标准分布：(X-X')/s, 使数据更平滑,便于计算。  使用另一种使数据平滑的方法
numeric_cols = all_df.columns[all_df.dtypes != 'object']
numeric_col_means = all_dummy_df.loc[:, numeric_cols].mean()
numeric_col_std = all_dummy_df.loc[:, numeric_cols].std()
all_dummy_df.loc[:, numeric_cols] = (all_dummy_df.loc[:, numeric_cols] - numeric_col_means)/numeric_col_std

dummy_train_df = all_dummy_df.loc[train_df.index]
dummy_test_df = all_dummy_df.loc[test_df.index]
print dummy_train_df.shape, dummy_test_df.shape

#------------------模型一, Ridge regression, 对于多因子的数据集，这种模型可以方便的把所有的var都无脑的放进去
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
X_train = dummy_train_df.values
X_test = dummy_test_df.values
#使用交叉验证的方法来获取模型参数
alphas = np.logspace(-3, 2, 50) #创建等比数列,起始点为10的-3次方,终点是10的2次方,共50个点
test_scores = []
for alpha in alphas:
    clf = Ridge(alpha)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='mean_squared_error'))
    test_scores.append(np.mean(test_score))
#查看cv方法alpha取值对mse的影响
import matplotlib.pyplot as plt
#plt.plot(alphas, test_scores)
#plt.title("Alpha vs CV Error")
#plt.show()

#-------- 模型二  RF
from sklearn.ensemble import RandomForestRegressor
max_features = [.1, .3, .5, .7, .9, .99]
test_scores = []
for max_feat in max_features:
    clf = RandomForestRegressor(n_estimators=200, max_features=max_feat)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=5, scoring='mean_squared_error'))
    test_scores.append(np.mean(test_score))
#plt.plot(max_features, test_scores)
#plt.title("Max Features vs CV Error")
#plt.show()

#-----------集成模型
#前面已经将两个模型的最优参数求出来了
ridge = Ridge(alpha=15)
rf = RandomForestRegressor(n_estimators=500, max_features=.3)
ridge.fit(X_train, y_train)
rf.fit(X_train, y_train)
y_ridge = np.expm1(ridge.predict(X_test))
y_rf = np.expm1(rf.predict(X_test))
#简单求平均
y_final = (y_ridge + y_rf) /2

submission_df = pd.DataFrame(data={'Id':test_df.index, 'ScalePrice':y_final})
print submission_df.head()


