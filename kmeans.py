# -*- coding: utf-8 -*-

import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
from util.Distance import distanceNorm
import random
import json
from sklearn.metrics import accuracy_score

# iris = datasets.load_iris()
# X = iris.data[:, :4]
# print(X)
# k = 3
# print(type(X))

df = pd.read_csv('DP/iris.csv')
df.rename(columns={'Seq': 'id', 'Sepal.Length': 'sl', 'Sepal.Width': 'sw', 'Petal.Length': 'pl',
                   'Petal.Width': 'pw', 'Species':'specie'}, inplace=True)
X  = pd.DataFrame(df,columns=['sl','sw','pl','pw'])
# X  = np.array(X)
print(X.head())
k = 3
iter_times = 100

def kmeans_by_myself(data):
	m = len(data)  # 样本个数
	print(m)
	n = data.shape[1]  # 维度
	cluster_center = np.zeros((k,n))

	# 选择合适的初始聚类中心
	j = np.random.randint(m)  # [0,m]
	cluster_center[0] = data.ix[j,:]
	# print (cluster_center[0])
	dis = np.zeros(m) - 1  # 样本到当前所有聚类中心的最近距离
	for i in range(k - 1):
		for j in range(m):
			d = distanceNorm('2',np.array(cluster_center[i]),data.ix[j,:])
			# d = symmetricalKL(cluster_center[i],data.ix[j,:])
			if (dis[j] < 0) or (dis[j] > d):
			# if (dis[j] > d):
				dis[j] = d
		j = random_select(dis)  # 按照dis加权选择样本j
		i += 1
		# print('print center:', j, data[j])
		cluster_center[i] = data.ix[j,:]
		print("Find Cluster Center: ",i)

	# 聚类
	cluster = np.zeros(m,dtype=np.int) - 1  # 所有样本尚未聚类
	cc = np.zeros((k,n))  # 下一轮的聚类中心
	# cc_text_id = np.zeros((k,), dtype=np.int)
	c_number = np.zeros(k)  # 每个簇的样本数目
	# times = 30
	for t in range(iter_times):
		cc.flat = 0
		c_number.flat = 0
		for i in range(m):
			# if np.random.random()*m > 100:
			# 	continue
			c = nearest(data.ix[i,:],cluster_center)
			# print("i 对应的簇：", i, c)
			cluster[i] = c  # 第i个样本归于第c个簇
			cc[c] += data.ix[i,:]
			c_number[c] += 1
		# print('before for circulation:', cc[0], c_number[0])
		for i in range(k):
			cluster_center[i] = cc[i] / c_number[i]
		print(t,'%.2f%%' % (100 * float(t) / iter_times))
	# print ("迭代过程中的聚类中心：", cluster_center[1])
	return cluster# ,c_number,cc


# def asymmetricKL(P,Q):
# 	return sum(P * log(P / Q))  # calculate the kl divergence between P and Q
#
#
# def symmetricalKL(P,Q):
# 	return (asymmetricKL(P,Q) + asymmetricKL(Q,P)) / 2.00


# 按照dis加权返回0~len(distance)的一个数j
def random_select(distance):
	dis_sum = (np.mat(distance)).sum(axis=1)[0,0]
	# print(type(dis_sum), dis_sum)
	dis = [value / dis_sum for value in distance]
	# print('probability:', dis)
	some_list = [i for i in range(len(dis))]
	return random_pick(some_list,dis)


def random_pick(some_list,probabilities):
	x = random.uniform(0,1)
	cumulative_probability = 0.0
	for item,item_probability in zip(some_list,probabilities):
		cumulative_probability += item_probability
		if x < cumulative_probability:
			break
	return item


def nearest(onedata,cluster_center):
	dis_list = []
	for i in range(len(cluster_center)):
		dis_list.append(distanceNorm('2',onedata,cluster_center[i]))
	return dis_list.index(min(dis_list))



# 绘制数据分布图
# plt.scatter(X[:, 0], X[:, 1], c="red", marker='o', label='see')
# plt.xlabel('petal length')
# plt.ylabel('petal width')
# plt.legend(loc=2)
# plt.show()

# label_pred = kmeans_by_myself(X)
# label_pred = [str(d) for d in label_pred]
# print(','.join(label_pred))


label_pred = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,2,2,2,2,0,2,2,2,2,2,2,0,0,2,2,2,2,0,2,0,2,0,2,2,0,0,2,2,2,2,2,0,2,2,2,2,0,2,2,2,0,2,2,2,0,2,2,0]
l = [1,0,2]
y_train = []
for d in l:
	y_train.extend([d for i in range(50)])
print ('结果集准确率：', accuracy_score(y_train,label_pred))

# estimator = KMeans(X)
# estimator.fit(X)
# label_pred = estimator.labels_ #获取聚类标签
# print(label_pred)

# 绘制数据分布图
# fig = plt.figure()
# axes = fig.add_subplot(111)
# # plt.scatter(X[:, 0], X[:, 1], c="red", marker='o', label='see')
# for i in range(len(label_pred)):
# 	if label_pred[i] == '0':
# 		#  第i行数据，及returnMat[i:,0]及矩阵的切片意思是:i：i+1代表第i行数据,0代表第1列数据
# 		axes.scatter(X[:3],X[:4],color='red') #, label='see'
# 	if label_pred[i] == '1':
# 		axes.scatter(X[:3],X[:4],color='green') #, label='see'
#
# 	if label_pred[i] == '2':
# 		axes.scatter(X[:3],X[:4],color='blue') # , label='see'
# plt.show()
