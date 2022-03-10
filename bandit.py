import numpy as np

class Bandit:
	'''
	@param	k		number of arms
	@param	mu		means for the arms
	@param	sigma	standard deviation for the arms
	'''
	def __init__(self, k: int, means: np.ndarray, sds: np.ndarray):
		self.k = k
		self.means = means
		self.sds = sds
	
	'''
	@param	k		which arm to pull
	'''
	def act(self, k: int) -> dict:
		if k < 0 or k >= self.k:
			raise ValueError(f'Invalid arm index. Received value: {k}')
		return { 'value': np.random.normal(self.means[k], self.sds[k]), 'regret': np.max(self.means) - self.means[k] }
