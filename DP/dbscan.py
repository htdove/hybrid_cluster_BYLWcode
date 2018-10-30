# -*- coding: utf-8 -*-

import numpy as np
import math

def _dist(p,q):
	return math.sqrt(np.power(p - q,2).sum())

def _eps_neighborhood(p,q,eps):
	return _dist(p,q) < eps


if __name__ == '__main__':
	m = np.matrix('1 1.2 0.8 3.7 3.9 3.6 10; 1.1 0.8 1 4 3.9 4.1 10')
	print(m[:,1])