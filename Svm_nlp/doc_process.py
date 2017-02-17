# -*- coding: utf-8 -*-
# @Time    : 17/2/17 下午5:37
# @Author  : liulei
# @Brief   : 
# @File    : doc_process.py
# @Software: PyCharm Community Edition

from nlp_util import my_feature_engineer

f = open('1348.txt', 'r')
s = f.read()
print my_feature_engineer.filter_tags(s)
