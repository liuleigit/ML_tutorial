#-*-coding: UTF-8 -*-
from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO

allElectronicsData = open('decision_tree_test.csv', 'r')
reader = csv.reader(allElectronicsData)
headers = reader.next()

print (headers)

feature_list = []
label_list = []

for row in reader:
    label_list.append(row[len(row) -1])
    row_dict = {}
    for i in range(1, len(row) - 1):
        row_dict[headers[i]] = row[i]
    feature_list.append(row_dict)

print(feature_list)
print(label_list)
#vectorize feature#用于将[{},{}...]格式的数据转换为值为数字的矩阵
vec = DictVectorizer()
dummyX = vec.fit_transform(feature_list).toarray()
print("dummY:" + str(dummyX))

#victorize class labels
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(label_list)
print("dummyY: " + str(dummyY))

#using decision tree for classification
clf = tree.DecisionTreeClassifier(criterion='entropy')
#建模
clf = clf.fit(dummyX, dummyY)
print("clf: "+str(clf))

with open("allElectronicInformatinoGainOri.dot", 'w') as f:
    f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)

oneRowX = dummyX[0, :]
print("oneRowX: " + str(oneRowX))

#假设一条新数据,进行预测
newRowX = oneRowX
newRowX[0] = 1
newRowX[2] = 0
print("newRowX: " + str(newRowX))

newRowX.reshape(-1, 1)
predictedY = clf.predict(newRowX)
print ("predictedY: " + str(predictedY))