import numpy as np

from bandit import Bandit

class SuperPolicy:
	def __init__(self, numArms: int, samples: int, T: int):
		self.numArms = numArms
		self.samples = samples
		self.T = T
	
	def step(self):
		pass

	def reset(self) -> None:
		pass

class EpsilonGreedyPolicy(SuperPolicy):
	def __init__(self, numArms: int, samples: int, T: int, epsilon: float):
		super().__init__(numArms, samples, T)
		self.epsilon = epsilon
		self.Q = np.zeros(self.numArms)
		self.N = np.zeros(self.numArms)
	
	def clear(self) -> None:
		self.Q = np.zeros(self.numArms)
		self.N = np.zeros(self.numArms)
	
	def step(self, bandit: Bandit) -> tuple:
		arms = [i for i in range(self.numArms)]

		arm_to_pull = np.argmax(self.Q) if np.random.rand() >= self.epsilon else np.random.choice(arms)
		rew, reg = bandit.act(arm_to_pull)
		
		self.N[arm_to_pull] += 1
		self.Q[arm_to_pull] += ((0.1) * (rew - self.Q[arm_to_pull]))

		return rew, reg

class UCBPolicy(SuperPolicy):
	def __init__(self, numArms: int, samples: int, T: int):
		super().__init__(numArms, samples, T)
		self.Q = np.zeros(self.numArms)
		self.N = np.zeros(self.numArms)

		self.means = np.random.normal(0, 1.0, numArms)
		self.sds = np.full(numArms, 1.0)

		self.values = np.zeros((samples, T))
		self.regrets = np.zeros((samples, T))
	
	def clear(self) -> None:
		self.Q = np.zeros(self.numArms)
		self.N = np.zeros(self.numArms)
	
	def step(self, bandit: Bandit) -> tuple:
		arm_to_pull = np.argmax(self.Q + 2 * (np.sqrt(np.log(np.sum(self.N) + 1) / (self.N + 1e-6))))
		rew, reg = bandit.act(arm_to_pull)
		
		self.N[arm_to_pull] += 1
		self.Q[arm_to_pull] += 0.1 * (rew - self.Q[arm_to_pull])

		return rew, reg
	

class ModifiedUCBPolicy(SuperPolicy):
	def __init__(self, numArms: int, samples: int, T: int):
		super().__init__(numArms, samples, T)
		self.Q = np.zeros(self.numArms)
		self.N = np.zeros(self.numArms)
		self.std = np.zeros(self.numArms)
		self.total_runs = T

		self.means = np.random.normal(0, 1.0, numArms)
		self.sds = np.full(numArms, 1.0)

		self.values = np.zeros((samples, T))
		self.regrets = np.zeros((samples, T))

		

	def clear(self) -> None:
		self.Q = np.zeros(self.numArms)
		self.N = np.zeros(self.numArms)
		self.std = np.zeros(self.numArms)
	
	def step(self, bandit: Bandit) -> tuple:
		arm_to_pull = np.argmax(self.Q + 2 * (np.sqrt(np.log(np.sum(self.N) + 1) / (self.N + 1e-6))) * (  1/ (self.std + 1e-6 ) )  )
		rew, reg = bandit.act(arm_to_pull)
		
		self.N[arm_to_pull] += 1
		self.Q[arm_to_pull] += 0.1 * ( rew - self.Q[arm_to_pull])
		self.std[arm_to_pull] +=  np.sqrt(0.1 * (rew - self.Q[arm_to_pull]) ** 2 + (1 - 0.1) * self.std[arm_to_pull] ** 2)
		return rew, reg
	

class ModifiedUCBPolicywithResourceConstraints(SuperPolicy):
	def __init__(self, numArms: int, samples: int, T: int):
		super().__init__(numArms, samples, T)
		self.Q = np.zeros(self.numArms)
		self.N = np.zeros(self.numArms)
		self.std = np.zeros(self.numArms)
		self.total_runs = T

		self.means = np.random.normal(0, 1.0, numArms)
		self.sds = np.full(numArms, 1.0)

		self.values = np.zeros((samples, T))
		self.regrets = np.zeros((samples, T))

		

	def clear(self) -> None:
		self.Q = np.zeros(self.numArms)
		self.N = np.zeros(self.numArms)
		self.std = np.zeros(self.numArms)
	
	def step(self, bandit: Bandit) -> tuple:
		arm_to_pull = np.argmax(self.Q )
		rew, reg = bandit.act(arm_to_pull)
		
		self.N[arm_to_pull] += 1
		self.Q[arm_to_pull] += 0.1 * ( rew - self.Q[arm_to_pull])
		self.std[arm_to_pull] +=  np.sqrt(0.1 * (rew - self.Q[arm_to_pull]) ** 2 + (1 - 0.1) * self.std[arm_to_pull] ** 2)
		return rew, reg
