import numpy as np

class Bandit:
	'''
	@param	means	means for the gaussian distribution of the arms
	@param	vars	variances for the gaussian distribution of the arms
	'''
	def __init__(self, means: np.ndarray, vars: np.ndarray):
		if (means.size != vars.size):
			raise ValueError(f'Number of means {means.size} does not match number of variances {vars.size}')
		self.means = means
		self.vars = vars
	
	'''
	@param	k		which arm to pull
	returns a 3-tuple: (reward, regret, true mean of the selected arm)
	'''
	def act(self, k: int) -> tuple:
		if k < 0 or k >= self.means.size:
			raise ValueError(f'Invalid arm index. Received value: {k}')
		return np.random.normal(self.means[k], self.vars[k]), np.max(self.means) - self.means[k], self.means[k]
	
	def __str__(self):
		return f'Means: {self.means}, Variances: {self.vars}'
