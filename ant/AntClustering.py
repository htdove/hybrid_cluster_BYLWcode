# -*- coding: utf-8 -*-
import numpy as np
from ant import Ant

class AntClustering():
	def __init__(self,
	             data,
	             grid=100,
	             rad=2,
	             antnum=50,
	             iterations=5 * 10 ** 6,
	             alpha=0,
	             sleep=0,
	             dsize=500):
		
		self.size = grid  # Grid size
		self.rad = rad  # How far can ants see?
		self.antnum = antnum  # Number of workers
		self.iterations = iterations
		self.workers = list()  # Worker ant list
		self.d_size = dsize  # Display size
		self.data = data  # dataset
		self.sleep = sleep  # Sleeps before starting
		
		''' Calculates alpha if not provided '''
		if alpha == 0:
			self.alpha = self.calc_alpha()
		else:
			self.alpha = alpha
		print("alpha: " + str(self.alpha))
		
		''' Generates grid '''
		self.grid = np.empty((self.size,self.size),dtype=np.object)
		self._distribute_data(self.grid,self.data)
		print(self.grid)
		# print(self.calc_alpha())
		
		''' Initializes ant agents '''
		self._create_ants(self.antnum,
		                  self.rad,
		                  self.grid,
		                  self.iterations // self.antnum,
		                  self.alpha)
