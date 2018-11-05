import numpy as np
import json

# griddata = np.load('antdata/myantresult.npy')
# print(griddata[27][3])
# N = len(griddata)
# # count = 0
# def getclust(m, row, col, vis,tmp):
# 	# global count
# 	# count+=1
# 	# print(count)
# 	if(row>=len(m) or row<0 or col>=len(m[0]) or col<0 or vis[row][col] == 1):
# 		return
# 	if(m[row][col] != {} and vis[row][col] == 0):
# 		vis[row][col] = 1
# 		tmp.append(m[row][col]['id'])
#
# 		getclust(griddata,row + 1,col,vis,tmp)
# 		getclust(griddata,row + 1,col-1,vis,tmp)
# 		getclust(griddata,row + 1,col,vis,tmp)
# 		getclust(griddata,row,col + 1,vis,tmp)
# 		getclust(griddata,row,col - 1,vis,tmp)
# 		getclust(griddata,row - 1,col + 1,vis,tmp)
# 		getclust(griddata,row - 1,col,vis,tmp)
# 		getclust(griddata,row - 1,col - 1,vis,tmp)
#
# vis = [[0 for _ in range(N)] for _ in range(N)]
# allres = []
# for i in range(N):
# 	for j in range(N):
# 		if(griddata[i][j] != {} and vis[i][j] == 0):
# 			tmp = []
# 			getclust(griddata, i, j, vis, tmp)
# 			allres.append(tmp)
#
# with open('antdata/a.json','w') as json_file:
# 	json.dump(allres,json_file,ensure_ascii=False)
# # from itertools import chain
# # c=list(chain(*allres))
# # print(sorted(c))
# for d in allres:
# 	print(d)
#
with open('antdata/a.json','r') as json_file:
	res = json.load(json_file)
print(len(res))


