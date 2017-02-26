# -*- coding: utf-8 -*-
# @bref :这是一个城市自行车租赁系统，提供的数据为2年内华盛顿按小时记录的自行车租赁数据，其中训练集由每个月的前19天组成，测试集由20号之后的时间组成（需要我们自己去预测）
#        这是一个连续值预测的问题

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_train = pd.read_csv('kaggle_bike_competition_train.csv')
print df_train.head()
#了解一下字段的名字和类型

#先处理时间,因为它包含的信息非常多,毕竟变化都是随着时间发生的
temp = pd.DatetimeIndex(df_train['datetime'])
df_train['date'] = temp.date
df_train['time'] = temp.time
#时间最小粒度到小时,所以干脆把小时字段取出来作为更简洁的特征
df_train['hour'] = pd.to_datetime(df_train.time, format="%H:%M:%S")
df_train['hour'] = pd.Index(df_train['hour']).hour

#设定一个字段:一周中的第几天; 再设定一个字段dateDays表示离租赁自行车活动第一天多久了
#对时间类的特征做处理,产生一个星期几的类别型特征
df_train['dayofweek'] = pd.DatetimeIndex(df_train.date).dayofweek
#对时间类的特征做处理,产生一个时间长度变量
df_train['dateDays'] = (df_train.date - df_train.date[0]).astype('timedelta64[D]')

#统计一下一周各天的租赁情况,分注册的人和没注册的人
byday = df_train.groupby('dayofweek')
print byday['casual'].sum().reset_index()
print byday['registered'].sum().reset_index()
#上面发现,周末有所不同,所以有必要增加周六、周日的列。其他日子差别不大,所以后面可以删除dayofweek
df_train['Saturday'] = 0 #增加一列,值都为0
df_train.Saturday[df_train.dayofweek==5] = 1 #周六的列中是周六的样本置1
df_train['Sunday'] = 0
df_train.Sunday[df_train.dayofweek==6]=1

#从原始的中删除处理过的字段
df_train = df_train.drop(['datetime', 'casual', 'registered', 'date', 'time', 'dayofweek'], axis=1)
print df_train.dtypes
print df_train.shape

#特征向量化
#对离散值和连续值区分一下,以便后面做不同的特征处理
from sklearn.feature_extraction import DictVectorizer
#把连续值放到一个dict中
featureConCols = ['temp', 'atemp', 'humidity', 'windspeed', 'dateDays', 'hour']
dataFeatureCon = df_train[featureConCols]
dataFeatureCon = dataFeatureCon.fillna('NA')
X_dictCon = dataFeatureCon.T.to_dict().values() #pandas的datafram转成python的dict
#把离散值得属性放到一个dict中
featureCatCols = ['season', 'holiday', 'workingday', 'weather', 'Saturday', 'Sunday']
dataFeatureCat = df_train[featureCatCols]
dataFeatureCat = dataFeatureCat.fillna('NA')
X_dictCat = dataFeatureCat.T.to_dict().values()

#向量化特征
vec = DictVectorizer(sparse=False)
X_vec_cat = vec.fit_transform(X_dictCat)
X_vec_con = vec.fit_transform(X_dictCon)

#标准化连续性特征。 均值为0,方差为1, 有利于收敛和准确性
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X_vec_con) #计算均值和方差
X_vec_con = scaler.transform(X_vec_con)  #执行标准化
#类别特征进行onehot编码
enc = preprocessing.OneHotEncoder()
enc.fit(X_vec_cat)
X_vec_cat = enc.transform(X_vec_cat).toarray()

#特征拼在一起
import numpy as np
X_vec = np.concatenate((X_vec_con, X_vec_cat), axis=1)
print X_vec

#将数据分成两部分: X, y
df_train_target = df_train['count'].values
df_train_data = df_train.drop('count', axis = 1).values

#算法部分
#下面的过程会让你看到,其实应用机器学习算法的过程,多半是在调参, 各种不同的参数会带来不同的结果
#比如正则化系数,比如决策树类的算法的树深和棵树,比如距离判定准则等等
from sklearn import linear_model
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import explained_variance_score

#依旧使用交叉验证的方式(交叉验证集4约占20%)来看模型的效果。我们会试支持向量回归(Support
# vector regression)、岭回归(Ridge regression)和随机森林回归(random forest regression)
#每个模型跑三趟
#切分数据
rs = ShuffleSplit(n_iter=3, test_size=0.2, random_state=0)
cv = rs.split(df_train_data)
#各种模型跑一圈
data_cv = []
for train, test in cv:
    data_cv.append((train, test))

print '岭回归'
for train, test in data_cv:
    ridge = linear_model.Ridge().fit(df_train_data[train], df_train_target[train])
    print ("train score: {0:.3f}, test score:{1:.3f}\n").format(
        ridge.score(df_train_data[train], df_train_target[train]),
        ridge.score(df_train_data[test], df_train_target[test])
    )

print "支持向量回归 /SVR(kernel='rbf', C=10, gamma=.001)"
for train, test in data_cv:
    svr = svm.SVR(kernel='rbf', C = 10, gamma = .001)
    svr = svr.fit(df_train_data[train], df_train_target[train])
    print ("train score: {0:.3f}, test score:{1:.3f}\n").format(
        svr.score(df_train_data[train], df_train_target[train]),
        svr.score(df_train_data[test], df_train_target[test])
    )

print "随机森林回归  Random Forest(n_extimator = 100)"
for train, test in data_cv:
    rfr = RandomForestRegressor(n_estimators=100).fit(df_train_data[train], df_train_target[train])
    print ("train score: {0:.3f}, test score:{1:.3f}\n").format(
        rfr.score(df_train_data[train], df_train_target[train]),
        rfr.score(df_train_data[test], df_train_target[test]))

#不出意料,随机森林回归获得了最佳结果

#支持向量回归并不理想,尝试下grid search
tuned_parameters = [{'kernel':['rbf'], 'gamma':[0.1, 0.01, 0.001],
                     'C':[1, 10, 100, 1000]}]
#                    {'kernel':['linear'], 'C':[1, 10, 100, 1000]}]
scores = ['precision', 'recall']
for score in scores:
    print ("# Tuning hyper-parameters for %s" % score)
    print ()
    #svr = GridSearchCV(svm.SVR(), tuned_parameters, cv=5, scoring='%s_macro' % score)
    svr = GridSearchCV(svm.SVR(), tuned_parameters, cv=5)
    svr.fit(df_train_data[train], df_train_target[train])
    print("Best parameters set found on development set:")
    print()
    print(svr.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in svr.grid_scores_:
        print("%0.3f  (+/-%0.3f) for %r" %(mean_score, scores.std()/2, params))
    print ""



X_train, X_test, y_train, y_test = train_test_split(df_train_data, df_train_target, test_size=0.2, random_state=0)

tuned_parameters = [{'n_estimators':[10, 100, 500]}]
scores = ['r2']
for score in scores:
    print score
    clf = GridSearchCV(RandomForestRegressor(), tuned_parameters, cv=5, scoring=score)
    clf.fit(X_train, y_train)
    print '---- find best parameters----'
    print(clf.best_estimator_)
    print ""

    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"%(mean_score, scores.std() / 2, params))
    print ""

#使用学习曲线查看模型状态是不是过拟合or欠拟合
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                            n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

title = "Learning Curves (Random Forest, n_estimators = 100)"
cv = ShuffleSplit(n_iter=10, test_size=0.2, random_state=0)
cv = cv.split(df_train_data)
estimator = RandomForestRegressor(n_estimators=100)
plot_learning_curve(estimator, title, df_train_data, df_train_target, (0.0, 1.01), cv=cv, n_jobs=4)
plt.show()


#尝试一下缓解过拟合, 未必成功
print "随机森林回归 /Random Forest(n_estimators=200, max_features=0.6, max_depth=15)"
for train, test in cv:
    svc = RandomForestRegressor(n_estimators=200, max_features=0.6, max_depth=15).fit(df_train_data[train], df_train_target[train])
    print("train score: {0:.3f}, test score: {1:.3f}\n".format(
        svc.score(df_train_data[train], df_train_target[train]), svc.score(df_train_data[test], df_train_target[test])))




