import numpy as np

from bandit import Bandit

class SuperPolicy:
	def __init__(self, numArms: int, samples: int, T: int):
		self.numArms = numArms
		self.samples = samples
		self.T = T

	def reset(self) -> None:
		pass

	def run(self):
		for run in range(self.samples):
			self.runHelper(run)

	def runHelper(self, run: int) -> None:
		pass

class EpsilonGreedyPolicy(SuperPolicy):
	def __init__(self, numArms: int, samples: int, T: int, epsilon: float):
		super().__init__(numArms, samples, T)
		self.epsilon = epsilon
		self.values = np.zeros((samples, T))
		self.regrets = np.zeros((samples, T))

	def run(self) -> tuple:
		for run in range(self.samples):
			self.runHelper(run)

		return np.mean(self.values, axis=0)

	# move simulation code to run.py
	def runHelper(self, run: int) -> None:
		arms = [i for i in range(self.numArms)]
		means = np.random.normal(0, 1.0, self.numArms)
		sds = np.full(self.numArms, 1.0)
		bandit = Bandit(means, sds)
		Q = np.zeros(self.numArms)
		N = np.zeros(self.numArms)

		for i in range(self.T):
			arm_to_pull = np.argmax(Q) if np.random.rand() >= self.epsilon else np.random.choice(arms)
			reward = bandit.act(arm_to_pull)

			self.values[run, i] = reward[0]
			self.regrets[run, i] = reward[1]
			Q[arm_to_pull] += 0.1 * (reward[0] - Q[arm_to_pull])
			N[arm_to_pull] += 1

class UCBPolicy(SuperPolicy):
	def __init__(self, numArms: int, samples: int, T: int):
		super().__init__(numArms, samples, T)
		self.values = np.zeros((samples, T))
		self.regrets = np.zeros((samples, T))
		self.means = np.random.normal(0, 1.0, numArms)
		self.sds = np.full(numArms, 1.0)

	def run(self) -> tuple:
		for run in range(self.samples):
			self.runHelper(run)

		return np.mean(self.values, axis=0)

	def runHelper(self, run: int) -> None:
		bandit = Bandit(self.means, self.sds)
		Q = np.zeros(self.numArms)
		N = np.zeros(self.numArms)

		for i in range(self.T):
			arm_to_pull = np.argmax(Q + 2 * (np.sqrt(np.log(i + 1) / (N + 1e-6))))
			reward = bandit.act(arm_to_pull)

			self.values[run, i] = reward[0]
			self.regrets[run, i] = reward[1]
			Q[arm_to_pull] += 0.1 * (reward[0] - Q[arm_to_pull])
			N[arm_to_pull] += 1
