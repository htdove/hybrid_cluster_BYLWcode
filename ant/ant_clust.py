from random import random,randint
import pygame
import pandas as pd
import numpy as np
from util.Distance import distanceNorm

N = 30  ## 网格数目N*N
n_ants = 20  # 蚂蚁数目
alpha = 1.5
sigma = 1.7
life = 100  # 蚂蚁的存活时间
moves = 10

vision_range = 2
# dead_ants = []
# alive_ants = []
# items = []

class Ant():
	def __init__(self, size, life):
		self.x = int(random() * size)
		self.y = int(random() * size)
		self.carrying = []
		self.life = life
		alive_ants[self.x][self.y] += 1
	
	def move(self,size):
		if self.life <= 0 and self.carrying == []:
			alive_ants[self.x][setlf.y] -= 1
		else:
			## 随机到下一个位置
			self.x = (self.x + randint(-1,1)) % size
			self.y = (self.y + randint(-1,1)) % size
			# 蚂蚁不携带，但是位置上有数据的情况
			print(self.carrying)
			if self.carrying == [] and dead_ants[self.x][self.y]!=[]:
				foi = f(self.x,self.y,dead_ants[self.x][self.y])
				if foi<=1:
					p = 1
				else:
					p = 1/float(foi ** 2)
				if random()<p: ## 将数据捡起来的概率
					self.carrying = dead_ants[self.x][self.y]
					dead_ants[self.x][self.y] = []
					self.life = life
				else:
					self.life -= 1
			# 蚂蚁携带，位置上无数据的讨论放下的情况
			elif self.carrying != [] and dead_ants[self.x][self.y] == []:
				foi = f(self.x,self.y,self.carrying)
				if foi >= 1:
					p = 1
				else:
					p = foi ** 4
				if random()<p:
					dead_ants[self.x][self.y] = self.carrying
					self.carrying = {}
					self.life = life
				else:
					self.life -= 1
			else:
				self.life -= 1
		
def f(x,y,comparing):
	xmin = x-vision_range
	xmax = x+vision_range+1
	ymin = y-vision_range
	ymax = y+vision_range+1
	foi = 0
	for row in dead_ants[xmin:xmax]:
		for items in row[ymin:ymax]:
			print('70',items, dead_ants[x][y])
			if items!=dead_ants[x][y] and items !=[]:
				print(items[1:-1],comparing[1:-1])
				foi += max(1-distanceNorm('1',items[1:-1],comparing[1:-1])/float(alpha),0)
	return (foi/float(sigma**2))

# df = pd.read_csv('../DP/iris.csv',names=['pl','pw','sl','sw','specie'])
# # print(df.dtypes)
# # items = df.to_dict('index').values
# items = pd.DataFrame(df,columns=['pl','pw','sl','sw'])
# items = np.array(items)
# print(items[:5])
df = pd.read_csv('../DP/iris.csv')
items = np.array(df)
print(items[:5])
classes = {'Iris-virginica': 1,
           'Iris-setosa': 2,
           'Iris-versicolor': 3}

def generate_grid(size, fill):
	return [[fill for _ in range(size)] for _ in range(size)]
## 将数据点全部散落在网格当中
def spreads_items(dead_ants, items):
	i = 0
	j = 0
	for item in items:
		while (dead_ants[i][j] != []):
			i = int(random() * len(dead_ants))
			j = int(random() * len(dead_ants))
		dead_ants[i][j] = item
	return dead_ants
#
alive_ants = generate_grid(N,0)  # 生成网格
dead_ants = spreads_items(generate_grid(N,[]), items)  # 将数据分配在N*N网格上
ants = [Ant(N, life) for _ in range(n_ants)]  # 定义了n_ants只ant类
print(dead_ants)
print(alive_ants)
# print(ants)

MARGIN = 2
WIDTH = 600 / N - MARGIN
HEIGHT = 600 / N - MARGIN
WINDOW_SIZE = [600,600]

BLACK = (0,0,0)
WHITE = (255,255,255)
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)
PINK = (255,000,255)

values = [0,255,128]
colors = []
for r in values:
	for g in values:
		for b in values[:2]:
			colors.append((r,g,b))
pygame.init()

screen = pygame.display.set_mode(WINDOW_SIZE)

clount = 0
while (True):
	for ant in ants:
		alive_ants[ant.x][ant.y] -= 1
		ant.move(N)
		alive_ants[ant.x][ant.x] += 1
	count += 1
	if (count%moves==0):
		for row in range(N):
			for column in range(N):
				color = BLACK
				if dead_ants[row][column] != {}:
					specie = dead_ants[row][column][-1]
					class_n = classes[specie]
					color = colors[int(class_n)]
				if alive_ants[row][column] > 0:
					color = WHITE
				pygame.draw.rect(screen,
				                 color,
				                 [(MARGIN + WIDTH) * column + MARGIN,
				                  (MARGIN + HEIGHT) * row + MARGIN,
				                  WIDTH,
				                  HEIGHT])
				
		clock.tick(30)
	pygame.display.flip()
	