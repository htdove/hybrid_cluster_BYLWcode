import numpy as np

def distanceNorm(Norm,v1,v2):
	D_value = v1 - v2
	if (Norm == '1'):
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