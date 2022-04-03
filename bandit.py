import numpy as np

class Bandit:
	'''
	@param	k		number of arms
	@param	mu		means for the arms
	@param	sigma	standard deviation for the arms
	'''
	def __init__(self, means: np.ndarray, sds: np.ndarray):
		if (means.size != sds.size):
			raise ValueError(f'Number of means {means.size} does not match number of standard deviations {sds.size}')
		self.means = means
		self.sds = sds
	
	'''
	@param	k		which arm to pull
	'''
	def act(self, k: int) -> tuple:
		if k < 0 or k >= self.means.size:
			raise ValueError(f'Invalid arm index. Received value: {k}')
		return np.random.normal(self.means[k], self.sds[k]), np.max(self.means) - self.means[k]
