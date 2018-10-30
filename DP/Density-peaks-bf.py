# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

def distanceNorm(Norm, v1, v2):
	D_value = v1-v2
	if(Norm=='1'):
		counter = np.absolute(D_value)
		counter = np.sum(counter)
	elif Norm == '2':
		counter = np.power(D_value,2)
		counter = np.sum(counter)
		counter = np.sqrt(counter)
	elif Norm == 'Infinity':
		counter = np.absolute(D_value)
		counter = np.max(counter)
	else:
		raise Exception('We will program this later......')
	
	return counter

def chi(x):
	if(x<0):
		return 1
	else:
		return 0

def count_indicater(features,labels,t,dis_method='2'):
	distance = np.zeros((len(labels),len(labels)))
	distance_sort = list()
	density = np.zeros(len(labels))
	distance_higherDensity = np.zeros(len(labels))
	
	for index_i in range(len(labels)):
		for index_j in range(index_i + 1,len(labels)):
			distance[index_i,index_j] = distanceNorm(dis_method,features[index_i],features[index_j])
			distance_sort.append(distance[index_i,index_j])
	distance += distance.T
	
	# distance_sort = np.array(distance_sort)
	# print(len(distance_sort))
	position = int(len(distance_sort) * t / 100)
	cutoff = np.round(sorted(distance_sort)[position*2 + len(labels)],5)
	
	## 获取每个点的密度值
	for index_i in range(len(labels)):
		distance_cutoff_i = distance[index_i] - cutoff
		for index_j in range(len(labels)):
			density[index_i] += chi(distance_cutoff_i[index_j])
	
	## 计算最大的密度的点
	Max = np.max(density)
	MaxIndexList = list()
	# MaxIndexList.extend(list(density).index(Max))
	for index_i in range(len(labels)):
		if(density[index_i] == Max):
			MaxIndexList.extend([index_i])

	## 获取每个点密度比他大且离他最近的点的距离
	Min = 0
	for index_i in range(len(labels)):
		if index_i in MaxIndexList:
			distance_higherDensity[index_i] = np.max(distance[index_i])
			continue
		else:
			Min = np.max(distance[index_i])
		for index_j in range(len(labels)):
			if density[index_i] < density[index_j] and distance[index_i,index_j] < Min:
				Min = distance[index_i,index_j]
			else:
				continue
		distance_higherDensity[index_i] = Min
	return 	density, distance_higherDensity, cutoff
	
def test_method():
	csv_data = pd.read_csv('E:\heting\Graduation_Project_experiment\save_data\iris.csv')
	
	newDf = pd.DataFrame(csv_data,columns=['Sepal.Length','Sepal.Width','Petal.Length','Petal.Width'])
	newDf = np.array(newDf)
	labels = csv_data.Species
	density,distance_higherDensity,cutoff = count_indicater(newDf,labels, 2)
	print(cutoff)
	csv_data['density'] = density
	csv_data['distance_higherDensity'] = distance_higherDensity
	csv_data.to_csv('middledata/Density-peaks-iris-2w.csv',index=0)
	
if __name__ == '__main__':
	# test_method()
	# size_mapping = {'XL': 3,'L': 2,'M': 1}
	# df['size'] = df['size'].map(size_mapping)
	csv_data = pd.read_csv('middledata/Density-peaks-iris-1.5w.csv')
	x = csv_data.density
	y = csv_data.distance_higherDensity
	labels = csv_data.Species
	fig = plt.figure()
	axes = fig.add_subplot(111)
	
	axes.set_title('pw-1.5')
	plt.xlabel('density')
	plt.ylabel('distance_higherDensity')  ## 高局部密度点距离
	
	for i in range(len(labels)):
		if labels[i] == 'setosa':
			#  第i行数据，及returnMat[i:,0]及矩阵的切片意思是:i：i+1代表第i行数据,0代表第1列数据
			axes.scatter(x[i],y[i],color='red')
		if labels[i] == 'versicolor':
			axes.scatter(x[i],y[i],color='green')
		
		if labels[i] == 'virginica':
			axes.scatter(x[i],y[i],color='black')
		
	# axes.scatter(x,y,c='r',marker='o')
	# plt.legend('x1')
	plt.show()
	