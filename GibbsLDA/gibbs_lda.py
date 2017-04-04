# -*- coding: utf-8 -*-
# @Time    : 17/4/4 下午6:19
# @Author  : liulei
# @Brief   : 
# @File    : gibbs_lda.py
# @Software: PyCharm Community Edition

#-------------------------------------生成需要使用的文档----------------
import numpy as np
'''
我们今天就用灰常简单的例子来做个简单的演示（更加便于理解）：
词库里一共就五个单词V：money,loan,bank,river,stream
并且一共就两个话题T: T1, T2
假设此刻我们有两个『话题-文字』的分布为：
ϕ1money=1/3, ϕ1loan=1/3, ϕ1bank=1/3
ϕ2bank=1/3, ϕ2stream=1/3, ϕ2river=1/3
'''
vocab = ["money", "loan", "bank", "river", "stream"]
z_1 = np.array([1.0/3, 1.0/3, 1.0/3, .0, .0])
z_2 = np.array([.0, .0, 1.0/3, 1.0/3, 1.0/3])

print np.where(np.array(vocab)=="loan")
#把两个topic分布转化为phi矩阵, 这个矩阵其实就是LDA的目标分布
phi_actual = np.array([z_1, z_2]).T.reshape(len(z_2), 2)
print phi_actual
#使用上面的假定分布生成文章

D = 16 #文章数目
mean_lengh = 10   #每个文档的平均单词数量
#根据泊松分布给每个文档安排句子长度
len_doc = np.random.poisson(mean_lengh, size=D)
T =2 #话题数目
docs = []
orig_topics = []
for i in range(D):
    z = np.random.randint(0,2)
    if z == 0:
        words = np.random.choice(vocab, size=len_doc[i], p=z_1).tolist()
    else:
        words = np.random.choice(vocab, size=len_doc[i], p=z_2).tolist()
    orig_topics.append(z)
    docs.append(words)


#-------------------- gibbs lda-----------------------
w_i = []  #记录每个文档中每个词在vocab中的index
i = [] #记录所有词的index, 0,1,2....
d_i = []
z_i = [] #记录每个文档中每个词的主题index, 随机初始化
counter = 0
#走一遍文档
for doc_idx, doc in enumerate(docs):
    #以及每个文档里的单词
    for word_idx, word in enumerate(doc):
        #找到当前词在V中的位置
        w_i.append(np.where(np.array(vocab)==word)[0][0])
        #并记下i
        i.append(counter)
        #再记录下该单词所属文档的index
        d_i.append(doc_idx)
        #这个单词在这个文档中的随机初始化topic分布
        z_i.append(np.random.randint(0, T))
        counter += 1

#为了前后统一, list变为np array
w_i = np.array(w_i)
d_i = np.array(d_i)
z_i = np.array(z_i)

#初始化两个分布:  WT---单词-话题, DT---doc-topic分布
WT = np.zeros((len(vocab), T))
for idx, word_ in enumerate(vocab): #每个单词
    #np.where(w_i == idx) 得到w_i中第idx个词在w_i中的indexs
    topics = z_i[np.where(w_i == idx)] #这个单词在所有文档中的话题分布
    for t in range(T):
        WT[idx, t] = sum(topics == t)
print z_i
print d_i
DT = np.zeros((D, T))
for idx, doc_ in enumerate(range(D)):
    topics = z_i[np.where(d_i == idx)]
    for t in range(T):
        DT[idx, t] = sum(topics == t)

#保存初始化的分布,用于对比
WT_orig = WT.copy()
DT_orit = DT.copy()

phi_1 = np.zeros((len(vocab), 100))
phi_2 = np.zeros((len(vocab), 100))

#总共跑100次
iters = 100
beta = 1.
alpha = 1.
for step in range(iters):
    for current in i:
        #把D和W分别拿出来
        doc_idx = d_i[current]
        w_idx = w_i[current]
        #并吧两个从总体集合中减去
        DT[doc_idx, z_i[current]] -= 1
        WT[w_idx, z_i[current]] -= 1
        #计算新的D和W的分布
        prob_word = (WT[w_idx, :] + beta) / (WT[:, :].sum(axis=0) + len(vocab) * beta)
        prob_document = (DT[doc_idx, :] + alpha) / (DT.sum(axis=0) + D * alpha)
        # 这其实就是对于每个topic的概率
        prob = prob_word * prob_document

        # 把Z更新（根据刚刚求得的prob）
        z_i[current] = np.random.choice([0, 1], 1, p=prob / prob.sum())[0]

        # 更新计数器
        DT[doc_idx, z_i[current]] += 1
        WT[w_idx, z_i[current]] += 1

        # 记录下Phi的变化
        phi = WT / (WT.sum(axis=0))
        phi_1[:, step] = phi[:, 0]
        phi_2[:, step] = phi[:, 1]

print z_i











